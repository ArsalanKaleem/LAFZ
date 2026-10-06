"""Day 14: paired statistics on a fixed evaluation set.

paired_test: paired t-test (as proposed by the source) + Wilcoxon signed-rank (robust to the heavy
tails typical of per-sentence token counts) + bootstrap CI of the mean difference + Cohen's d_z.
paired_bootstrap_chrf: paired bootstrap resampling of corpus-level chrF++ between two systems.
holm: Holm–Bonferroni correction across a family of comparisons.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_test(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    out = {"n": int(len(d)), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
           "ci95_mean_diff": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
           "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else 0.0}
    if np.allclose(d, 0):
        out.update({"t_p": 1.0, "wilcoxon_p": 1.0})
        return out
    out["t_stat"], out["t_p"] = map(float, stats.ttest_rel(b, a))
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(b, a).pvalue)
    except ValueError:
        out["wilcoxon_p"] = 1.0
    return out


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "significant": adj < alpha}
    return out


def _chrf_stats(hyps: list[str], refs: list[str]):
    from sacrebleu.metrics import CHRF
    m = CHRF(word_order=2)
    try:
        st = np.array(m._extract_corpus_statistics(hyps, [refs]), dtype=float)
        return m, st
    except Exception:
        return m, None


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2).corpus_score(hyps, [refs]).score


def paired_bootstrap_chrf(hyp_a: list[str], hyp_b: list[str], refs: list[str], n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    m, sa = _chrf_stats(hyp_a, refs)
    _, sb = _chrf_stats(hyp_b, refs)
    N = len(refs)
    if sa is not None and sb is not None:
        score = lambda st, idx: m._compute_score_from_stats(list(st[idx].sum(0))).score  # noqa: E731
        base_a, base_b = score(sa, np.arange(N)), score(sb, np.arange(N))
        diffs = []
        for _ in range(n):
            idx = rng.integers(0, N, N)
            diffs.append(score(sb, idx) - score(sa, idx))
    else:  # slow but dependency-free fallback
        base_a, base_b = corpus_chrf(hyp_a, refs), corpus_chrf(hyp_b, refs)
        diffs = []
        for _ in range(min(n, 200)):
            idx = rng.integers(0, N, N)
            diffs.append(corpus_chrf([hyp_b[i] for i in idx], [refs[i] for i in idx]) -
                         corpus_chrf([hyp_a[i] for i in idx], [refs[i] for i in idx]))
    diffs = np.array(diffs)
    return {"chrf_a": float(base_a), "chrf_b": float(base_b), "diff": float(base_b - base_a),
            "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
            "p_b_not_better": float((diffs <= 0).mean()), "p_b_not_worse": float((diffs >= 0).mean())}
