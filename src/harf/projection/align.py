"""Day 8: alignment projection.

Merge-tree projection (self-supervised, no parallel data needed)
----------------------------------------------------------------
Every BPE token in the base vocab was created by a merge (a, b) -> ab, and the base model has learned
embeddings for a, b and ab. We fit a ridge map
        E[ab]  ≈  W · [E[a] ; E[b]] + c
on the base model's own merges (optionally only Arabic-script merges), then initialise each new token
from its own merge pair — in merge order, so children are always initialised before parents.

Parallel anchor (optional, uses bilingual data)
-----------------------------------------------
For target words that become a single new token and have an English translation in a lexicon
(e.g. extracted from FLORES-dev word alignments), blend the projected vector toward the mean embedding
of the English translation:  e = (1-α)·proj + α·mean(E[english pieces]).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..tokenizer.common import backend_json, touches_arabic
from ..utils import log, require_torch, write_json


def base_merge_triples(base_tok, which: str = "arabic", max_pairs: int | None = None, seed: int = 0):
    tj = backend_json(base_tok)
    vocab = tj["model"]["vocab"]
    out = []
    for m in tj["model"]["merges"]:
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        ab = a + b
        if a in vocab and b in vocab and ab in vocab:
            if which == "arabic" and not touches_arabic(ab):
                continue
            out.append((vocab[a], vocab[b], vocab[ab]))
    if max_pairs and len(out) > max_pairs:
        import random
        rng = random.Random(seed)
        out = rng.sample(out, max_pairs)
    return out


def fit_ridge(E, triples, lambdas, val_frac: float = 0.1, seed: int = 0) -> dict:
    """Closed-form ridge with intercept; lambda chosen on held-out merges by mean cosine."""
    torch = require_torch()
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(triples), generator=g)
    t = torch.as_tensor(triples, dtype=torch.long)[idx]
    n_val = max(int(len(t) * val_frac), 1)
    val, tr = t[:n_val], t[n_val:]
    dev = E.device
    E32 = E.float()

    def XY(tt):
        tt = tt.to(dev)
        return torch.cat([E32[tt[:, 0]], E32[tt[:, 1]]], dim=1), E32[tt[:, 2]]

    Xtr, Ytr = XY(tr)
    Xva, Yva = XY(val)
    xm, ym = Xtr.mean(0, keepdim=True), Ytr.mean(0, keepdim=True)
    Xc, Yc = (Xtr - xm).double(), (Ytr - ym).double()
    XtX, XtY = Xc.T @ Xc, Xc.T @ Yc
    scale = torch.trace(XtX) / XtX.shape[0]
    I = torch.eye(XtX.shape[0], dtype=XtX.dtype, device=dev)
    best = None
    results = []
    for lam in lambdas:
        W = torch.linalg.solve(XtX + lam * scale * I, XtY).float()
        pred = (Xva - xm) @ W + ym
        cos = F.cosine_similarity(pred, Yva, dim=1).mean().item()
        results.append({"lambda": lam, "val_cos": cos})
        if best is None or cos > best[1]:
            best = (lam, cos, W)
    lam, cos, W = best
    d = E.shape[1]
    mean_pred = 0.5 * (Xva[:, :d] + Xva[:, d:])
    rnd = torch.randn_like(Yva) * E32.std()
    target_norm = Ytr.norm(dim=1).median().item()
    diag = {
        "n_train": len(tr), "n_val": len(val), "lambda_search": results, "best_lambda": lam,
        "val_cos_projection": cos,
        "val_cos_mean_of_parts": F.cosine_similarity(mean_pred, Yva, dim=1).mean().item(),
        "val_cos_random": F.cosine_similarity(rnd, Yva, dim=1).mean().item(),
        "val_norm_ratio_projection": (((Xva - xm) @ W + ym).norm(dim=1).median() / Yva.norm(dim=1).median()).item(),
        "target_norm_median": target_norm,
    }
    log.info("projection: best λ=%s val_cos=%.3f (mean-of-parts %.3f, random %.3f)", lam, cos,
             diag["val_cos_mean_of_parts"], diag["val_cos_random"])
    return {"W": W, "x_mean": xm, "y_mean": ym, "target_norm": target_norm, "diagnostics": diag}


def apply_projection(E, proj: dict, entries: list[dict], rescale_norm: bool = True):
    """Initialise new rows in merge order. `entries` have 'pair' (byte-level strings) and 'id';
    `vocab_ext` maps strings to ids for both base and new tokens. E must already be resized."""
    torch = require_torch()
    W, xm, ym = proj["W"], proj["x_mean"], proj["y_mean"]
    rows = []
    with torch.no_grad():
        for e in entries:  # merge order == id order -> parents after children
            a_id, b_id = e["pair_ids"]
            x = torch.cat([E[a_id].float(), E[b_id].float()]).unsqueeze(0)
            y = ((x - xm) @ W + ym).squeeze(0)
            if rescale_norm:
                y = y * (proj["target_norm"] / y.norm().clamp_min(1e-6))
            E[e["id"]] = y.to(E.dtype)
            rows.append(y)
    return torch.stack(rows)


def parallel_anchor(E_new_rows, entries, ext_tok, base_tok, E_base, lexicon_path: str, alpha: float):
    """Blend single-token target words toward their English translation's mean embedding."""
    torch = require_torch()
    id2row = {e["id"]: i for i, e in enumerate(entries)}
    n = 0
    with torch.no_grad():
        for line in Path(lexicon_path).read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            tgt, eng = line.split("\t", 1)
            ids = ext_tok.encode(" " + tgt.strip(), add_special_tokens=False)
            if len(ids) != 1 or ids[0] not in id2row:
                continue
            eng_ids = base_tok.encode(" " + eng.strip(), add_special_tokens=False)
            v = E_base[eng_ids].float().mean(0)
            r = id2row[ids[0]]
            E_new_rows[r] = (1 - alpha) * E_new_rows[r] + alpha * v * (E_new_rows[r].norm() / v.norm().clamp_min(1e-6))
            n += 1
    log.info("parallel anchor applied to %d new tokens", n)
    return E_new_rows, n


def nearest_neighbors(E, new_ids, base_n: int, tok, k: int = 5, sample: int = 40) -> list[dict]:
    torch = require_torch()
    import torch.nn.functional as F
    base = F.normalize(E[:base_n].float(), dim=1)
    out = []
    for i in new_ids[:sample]:
        sims = base @ F.normalize(E[i].float(), dim=0)
        top = torch.topk(sims, k)
        out.append({"id": int(i), "token": tok.decode([int(i)]),
                    "neighbors": [(tok.decode([int(j)]), round(float(s), 3)) for s, j in zip(top.values, top.indices)]})
    return out


def write_nn_report(path: Path, nns: list[dict]) -> None:
    lines = ["# Nearest base-vocabulary neighbours of new embeddings", "",
             "Sanity check: neighbours should share letters/morphology or meaning, not be random.", ""]
    for r in nns:
        nb = ", ".join(f"`{t}` ({s})" for t, s in r["neighbors"])
        lines.append(f"- `{r['token']}` → {nb}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
