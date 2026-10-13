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


def base_merge_map(base_tok) -> dict[int, tuple[int, int]]:
    """result_id -> (a_id, b_id) for every base merge (used for chained validation)."""
    tj = backend_json(base_tok)
    vocab = tj["model"]["vocab"]
    out = {}
    for m in tj["model"]["merges"]:
        a, b = m.split(" ", 1) if isinstance(m, str) else m
        if a in vocab and b in vocab and (a + b) in vocab:
            out.setdefault(vocab[a + b], (vocab[a], vocab[b]))
    return out


def fit_ridge(E, triples, lambdas, val_frac: float = 0.1, seed: int = 0, residual: bool = True,
              select_by: str = "mrr", chunk: int = 8192, n_retrieval: int = 2000, chain_map: dict | None = None) -> dict:
    """Closed-form ridge with intercept, memory-safe (statistics accumulated in chunks).

    residual=True  : predict the CORRECTION  E[ab] - mean(E[a], E[b]);  pred = mean + W·x + c.
                     As lambda -> inf this reduces exactly to mean-of-parts init.
    residual=False : predict E[ab] directly (original "projection_direct"; regresses every
                     prediction toward the average embedding -> collapsed, hard-to-distinguish rows).
    select_by="mrr": choose lambda by held-out retrieval (is each prediction nearest to ITS OWN
                     target among 2000 candidates?), which penalises collapse. "cos" = mean cosine.
    Residual mode has NO intercept (a shared offset added to every new row causes collapse), and
    lambda=inf (W=0, i.e. exactly mean-of-parts) is always a candidate, so selection can fall back to it.
    chain_map: if given, selection uses CHAINED validation: the parts a, b are themselves predicted
    from their own merges first, as happens when new tokens are built from other new tokens.
    """
    torch = require_torch()
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(triples), generator=g)
    t = torch.as_tensor(triples, dtype=torch.long)[idx]
    n_val = max(int(len(t) * val_frac), 1)
    val, tr = t[:n_val], t[n_val:]
    dev = E.device
    W_emb = E.detach()
    d = W_emb.shape[1]

    def XY(tt):
        tt = tt.to(dev)
        a, b, ab = W_emb[tt[:, 0]].float(), W_emb[tt[:, 1]].float(), W_emb[tt[:, 2]].float()
        X = torch.cat([a, b], dim=1)
        Y = ab - 0.5 * (a + b) if residual else ab
        return X, Y, ab, 0.5 * (a + b)

    with torch.no_grad():
        sx = torch.zeros(2 * d, dtype=torch.float64, device=dev)
        sy = torch.zeros(d, dtype=torch.float64, device=dev)
        norms = []
        for i in range(0, len(tr), chunk):
            X, Y, ab, _ = XY(tr[i:i + chunk])
            sx += X.sum(0).double()
            sy += Y.sum(0).double()
            norms.append(ab.norm(dim=1).cpu())
        xm, ym = (sx / len(tr)).float(), (sy / len(tr)).float()
        if residual:
            ym = torch.zeros_like(ym)  # no shared offset
        XtX = torch.zeros(2 * d, 2 * d, dtype=torch.float64, device=dev)
        XtY = torch.zeros(2 * d, d, dtype=torch.float64, device=dev)
        for i in range(0, len(tr), chunk):
            X, Y, _, _ = XY(tr[i:i + chunk])
            Xc, Yc = X - xm, Y - ym
            XtX += (Xc.T @ Xc).double()
            XtY += (Xc.T @ Yc).double()
            del X, Y, Xc, Yc
        scale = torch.trace(XtX) / XtX.shape[0]
        I = torch.eye(2 * d, dtype=torch.float64, device=dev)

        Xva, Yva, ABva, MEANva = XY(val)

        def predict(W, X, mean_parts):
            y = (X - xm) @ W + ym
            return y + mean_parts if residual else y

        def retrieval(P, T):
            n = min(n_retrieval, len(P))
            Pn, Tn = F.normalize(P[:n], dim=1), F.normalize(T[:n], dim=1)
            S = Pn @ Tn.T
            ranks = (S > S.diag().unsqueeze(1)).sum(1) + 1
            off = (Pn @ Pn.T).sum() - n
            return {"mrr": (1.0 / ranks.float()).mean().item(), "top1": (ranks == 1).float().mean().item(),
                    "cos": F.cosine_similarity(P, T, dim=1).mean().item(),
                    "mean_pairwise_cos": (off / (n * (n - 1))).item()}

        # chained validation: replace each part by its own prediction when it is itself a merge result
        chain = None
        if chain_map:
            def parts(ids):
                kids = [chain_map.get(int(i)) for i in ids.tolist()]
                has = torch.tensor([k is not None for k in kids], device=dev)
                k1 = torch.tensor([k[0] if k else 0 for k in kids], device=dev)
                k2 = torch.tensor([k[1] if k else 0 for k in kids], device=dev)
                return has, k1, k2
            vd = val.to(dev)
            chain = (parts(vd[:, 0]), parts(vd[:, 1]))
            log.info("chained validation: %.0f%% of val parts are themselves merges",
                     50 * (chain[0][0].float().mean() + chain[1][0].float().mean()).item())

        def chained_X(W):
            outs = []
            for (has, k1, k2), col in zip(chain, (0, 1)):
                real = W_emb[val[:, col].to(dev)].float()
                ka, kb = W_emb[k1].float(), W_emb[k2].float()
                pred = predict(W, torch.cat([ka, kb], 1), 0.5 * (ka + kb))
                outs.append(torch.where(has.unsqueeze(1), pred, real))
            return torch.cat(outs, 1), 0.5 * (outs[0] + outs[1])

        candidates = [(lam, torch.linalg.solve(XtX + lam * scale * I, XtY).float()) for lam in lambdas]
        if residual:
            candidates.append((float("inf"), torch.zeros(2 * d, d, device=dev)))
        results, best = [], None
        for lam, W in candidates:
            m = retrieval(predict(W, Xva, MEANva), ABva)
            if chain is not None:
                Xc2, M2 = chained_X(W)
                mc = retrieval(predict(W, Xc2, M2), ABva)
                m.update({f"chained_{k}": v for k, v in mc.items()})
            results.append({"lambda": lam, **m})
            key = m[f"chained_{select_by}"] if chain is not None else m[select_by]
            if best is None or key > best[0]:
                best = (key, lam, W, m)
        _, lam, W, m = best
        diag = {
            "residual": residual, "select_by": select_by, "n_train": len(tr), "n_val": len(val),
            "lambda_search": results, "best_lambda": lam, "val_projection": m,
            "val_mean_of_parts": retrieval(MEANva, ABva),
            "val_random": retrieval(torch.randn_like(ABva) * W_emb[:1000].float().std(), ABva),
            "val_true_targets_mean_pairwise_cos": retrieval(ABva, ABva)["mean_pairwise_cos"],
        }
    target_norm = torch.cat(norms).median().item()
    diag["target_norm_median"] = target_norm
    log.info("projection(residual=%s): best λ=%s  MRR=%.3f top1=%.3f cos=%.3f pairwise=%.3f | mean-of-parts MRR=%.3f cos=%.3f "
             "pairwise=%.3f | true pairwise=%.3f", residual, lam, m["mrr"], m["top1"], m["cos"], m["mean_pairwise_cos"],
             diag["val_mean_of_parts"]["mrr"], diag["val_mean_of_parts"]["cos"],
             diag["val_mean_of_parts"]["mean_pairwise_cos"], diag["val_true_targets_mean_pairwise_cos"])
    if residual and lam == float("inf"):
        log.warning("Selected lambda=inf: the learned correction does NOT beat plain mean-of-parts on validation.")
    return {"W": W, "x_mean": xm, "y_mean": ym, "residual": residual, "target_norm": target_norm,
            "diagnostics": diag}


def apply_projection(E, proj: dict, entries: list[dict], rescale_norm: bool = False):
    """Initialise new rows in merge order (children before parents). Entries need 'id' and 'pair_ids'."""
    torch = require_torch()
    W, xm, ym = proj["W"], proj["x_mean"], proj["y_mean"]
    rows = []
    with torch.no_grad():
        for e in entries:
            a_id, b_id = e["pair_ids"]
            a, b = E[a_id].float(), E[b_id].float()
            y = ((torch.cat([a, b]).unsqueeze(0) - xm) @ W + ym).squeeze(0)
            if proj.get("residual"):
                y = y + 0.5 * (a + b)
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
    E = E.detach()
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
