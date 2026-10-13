"""Day 7: embedding initialisation strategies for new tokens. Same vocabulary, only init changes.

  random      N(0, σ²) with σ = std of existing rows                       (lower bound)
  mean        mean of the base-token embeddings the new token replaces      (constituent baseline)
  projection  merge-tree ridge projection (+ optional parallel anchor)      (HARF)

Input and output embeddings are handled separately when the model is untied.
"""
from __future__ import annotations

from pathlib import Path

from ..projection.align import (apply_projection, base_merge_map, base_merge_triples, fit_ridge, nearest_neighbors,
                                parallel_anchor, write_nn_report)
from ..tokenizer.common import touches_arabic
from ..utils import log, require_torch, write_json
from .inject import is_tied


def _attach_pair_ids(entries: list[dict], ext_vocab: dict[str, int]) -> None:
    for e in entries:
        e["pair_ids"] = [ext_vocab[e["pair"][0]], ext_vocab[e["pair"][1]]]


def init_random(E, entries, base_n: int, seed: int):
    torch = require_torch()
    g = torch.Generator(device="cpu").manual_seed(seed)
    std = E[:base_n].float().std().item()
    rows = torch.randn(len(entries), E.shape[1], generator=g) * std
    return rows


def init_mean(E, entries):
    torch = require_torch()
    return torch.stack([E[e["base_piece_ids"]].float().mean(0).cpu() for e in entries])


def _mean_pairwise_cos(R, n: int = 1000, seed: int = 0) -> float:
    """Average cosine between distinct rows: high = rows look alike (collapse)."""
    torch = require_torch()
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(R), generator=g)[:n]
    Z = F.normalize(R[idx.to(R.device)].float(), dim=1)
    k = len(Z)
    return ((Z @ Z.T).sum() - k).item() / max(k * (k - 1), 1)


def dev_bpc(model, tok, texts: list[str]) -> float:
    """Bits per character on a small VALIDATION sample (used only to choose lambda)."""
    import math
    from ..evaluation.perplexity import sentence_nll
    nll, _ = sentence_nll(model, tok, texts, batch_size=8, max_len=2048)
    return float(nll.sum() / (math.log(2) * sum(len(t) for t in texts)))


def initialize(model, base_tok, ext_tok, entries: list[dict], strategy: str, cfg: dict, out_dir: Path,
               dev_texts: list[str] | None = None):
    """Initialise new rows in-place on `model`; returns (ids, input_rows, output_rows, diagnostics)."""
    torch = require_torch()
    base_n = len(base_tok)
    _attach_pair_ids(entries, ext_tok.get_vocab())
    ids = [e["id"] for e in entries]
    tied = is_tied(model)
    mats = {"input": model.get_input_embeddings().weight}
    if not tied:
        mats["output"] = model.get_output_embeddings().weight
    rows_out, diag = {}, {"strategy": strategy, "tied": tied}
    pcfg = cfg["projection"]
    for kind, W in mats.items():
        with torch.no_grad():
            if strategy == "random":
                rows = init_random(W, entries, base_n, cfg["seed"] + (kind == "output"))
            elif strategy == "mean":
                rows = init_mean(W, entries)
            elif strategy in ("projection", "projection_direct"):
                pc = dict(pcfg)
                if strategy == "projection_direct":  # original method, kept as an ablation (it collapses)
                    pc.update(residual=False, rescale_norm=True, select_by="cos", chained_validation=False)
                triples = base_merge_triples(base_tok, pc["pairs"], pc["max_pairs"], cfg["seed"])
                if len(triples) < 1000:
                    log.warning("Only %d base merges for projection (pairs=%s). Consider pairs=all.",
                                len(triples), pc["pairs"])
                by_bpc = pc.get("select_by") == "bpc"
                proj = fit_ridge(W, triples, pc["lambdas"], pc["val_frac"], cfg["seed"],
                                 residual=pc.get("residual", True), select_by="mrr" if by_bpc else pc.get("select_by", "mrr"),
                                 chain_map=base_merge_map(base_tok) if pc.get("chained_validation", True) else None)
                if by_bpc and kind == "input" and dev_texts:
                    # choose lambda by actual language-modelling quality on validation sentences
                    model.eval()
                    search = []
                    for lam, Wc in proj["candidates"]:
                        apply_projection(W, {**proj, "W": Wc}, entries, pc.get("rescale_norm", False))
                        b = dev_bpc(model, ext_tok, dev_texts)
                        search.append({"lambda": lam, "dev_bpc": b})
                        log.info("  λ=%s  dev BPC=%.4f", lam, b)
                    best = min(search, key=lambda r: r["dev_bpc"])
                    proj["W"] = dict(proj["candidates"])[best["lambda"]]
                    diag["bpc_lambda_search"] = search
                    diag["best_lambda_by_dev_bpc"] = best["lambda"]
                    log.info("projection: λ chosen by dev BPC = %s (%.4f)", best["lambda"], best["dev_bpc"])
                    if best["lambda"] == float("inf"):
                        log.warning("Dev BPC prefers λ=inf: the learned correction does not help over mean-of-parts.")
                rows = apply_projection(W, proj, entries, pc.get("rescale_norm", False)).cpu()
                diag[f"projection_{kind}"] = proj["diagnostics"]
                if pcfg.get("lexicon") and kind == "input":
                    rows, n = parallel_anchor(rows, entries, ext_tok, base_tok, W, pcfg["lexicon"], pcfg["anchor_alpha"])
                    diag["parallel_anchor_applied"] = n
                torch.save({k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in proj.items()
                            if k not in ("diagnostics", "candidates")},
                           out_dir / f"projection_{kind}.pt")
            else:
                raise ValueError(f"unknown init strategy {strategy!r}")
            idx = torch.as_tensor(ids, device=W.device)
            W[idx] = rows.to(W.device, W.dtype)
        rows_out[kind] = rows.float().cpu()
        diag[f"{kind}_new_mean_pairwise_cos"] = _mean_pairwise_cos(rows)
        ref_ids = [i for t, i in base_tok.get_vocab().items() if i < base_n and touches_arabic(t)]
        diag[f"{kind}_base_arabic_mean_pairwise_cos"] = _mean_pairwise_cos(W[ref_ids].detach())
        diag[f"{kind}_new_norm_median"] = rows.float().norm(dim=1).median().item()
        diag[f"{kind}_base_norm_median"] = W[:base_n].float().norm(dim=1).median().item()
    nns = nearest_neighbors(model.get_input_embeddings().weight, ids, base_n, ext_tok)
    write_nn_report(out_dir / "nearest_neighbors.md", nns)
    write_json(out_dir / "init_diagnostics.json", diag)
    return ids, rows_out["input"], rows_out.get("output"), diag
