"""Days 13–15: aggregate every saved artifact into final tables, significance tests, plots, failure cases.
Nothing here recomputes model outputs — every number traces to a file in runs/<model>/<lang>/eval/."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..text.normalize import is_arabic_char
from ..utils import log, read_json, read_jsonl, write_json
from .stats import holm, paired_bootstrap_chrf, paired_test


def _maybe(path: Path):
    return read_json(path) if path.exists() else None


def final_table(cfg: dict, lang_dir: Path) -> pd.DataFrame:
    ev = lang_dir / "eval"
    comp = _maybe(ev / "compression_results.json") or {"conditions": {}}
    rows = []
    base_ppl = _maybe(ev / "ppl_baseline.json")
    base_lat = _maybe(ev / "latency_baseline.json")
    for name, c in cfg["conditions"].items():
        b = c.get("budget", 0)
        r = {"condition": name, "budget": b, "init": c.get("init"), "lora": c.get("lora")}
        cs = comp["conditions"].get("baseline" if not b else f"v{b}")
        if cs:
            r.update({"scr": cs.get("scr", 0.0), "tokens_per_word": cs["tokens_per_word"],
                      "median_tokens_per_sentence": cs["median_tokens_per_sentence"]})
        p = _maybe(ev / f"ppl_{name}.json")
        if p:
            r.update({"bpc": p["bpc"], "word_ppl": p["word_ppl"]})
            if base_ppl:
                r["bpc_change_pct"] = 100 * (p["bpc"] / base_ppl["bpc"] - 1)
        f = _maybe(ev / f"flores_{name}.json")
        if f:
            for k, v in f.items():
                if isinstance(v, dict) and "chrf++" in v:
                    r[f"chrf++ {k}"] = v["chrf++"]
        en = _maybe(ev / f"ppl_english_{name}.json")
        base_en = _maybe(ev / "ppl_english_baseline.json")
        if en:
            r["english_bpc"] = en["bpc"]
            if base_en:
                r["english_bpc_change_pct"] = 100 * (en["bpc"] / base_en["bpc"] - 1)
        l = _maybe(ev / f"latency_{name}.json")
        if l:
            r["emit_ms_median"] = l["emit_ms_median"]
            r["tokenizer_ms_per_sentence"] = l["encode_per_sentence_ms"]
            if base_lat:
                r["latency_speedup"] = base_lat["emit_total_s"] / l["emit_total_s"]
        if len(r) > 4:
            rows.append(r)
    df = pd.DataFrame(rows)
    out = lang_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "final_results.csv", index=False)
    (out / "final_results.md").write_text(_md_table(df), encoding="utf-8")
    _plots(df, out)
    log.info("final table:\n%s", df.to_string(index=False))
    return df


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no results yet_\n"
    cols = list(df.columns)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else ("" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _plots(df: pd.DataFrame, out: Path) -> None:
    if df.empty or "bpc_change_pct" not in df or "scr" not in df:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    d = df.dropna(subset=["scr", "bpc_change_pct"])
    ax.scatter(100 * d["scr"], d["bpc_change_pct"])
    for _, r in d.iterrows():
        ax.annotate(r["condition"], (100 * r["scr"], r["bpc_change_pct"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(2, color="r", lw=0.6, ls="--", label="+2% (source target)")
    ax.set_xlabel("Sequence compression ratio (%)")
    ax.set_ylabel("Δ bits-per-char vs baseline (%)  ↓ better")
    ax.set_title("Efficiency / quality trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=150)
    plt.close(fig)


def significance(cfg: dict, lang_dir: Path) -> dict:
    """Paired tests vs baseline (per-sentence BPC; per-sentence tokens; chrF++ bootstrap) + Holm, and
    short/long breakdown (split at the median baseline token count)."""
    ev = lang_dir / "eval"
    out: dict = {"bpc": {}, "tokens": {}, "chrf": {}, "by_length": {}}
    base = ev / "ppl_per_sentence_baseline.csv"
    if base.exists():
        b = pd.read_csv(base)
        b_bpc = b["nll"] / (math.log(2) * b["n_chars"])
        for name in cfg["conditions"]:
            p = ev / f"ppl_per_sentence_{name}.csv"
            if name == "baseline" or not p.exists():
                continue
            c = pd.read_csv(p)
            assert (c["id"].values == b["id"].values).all(), "sentence sets differ!"
            c_bpc = c["nll"] / (math.log(2) * c["n_chars"])
            out["bpc"][name] = paired_test(b_bpc.values, c_bpc.values)
            long = b["n_tokens"] > b["n_tokens"].median()
            out["by_length"][name] = {"short": paired_test(b_bpc[~long].values, c_bpc[~long].values),
                                      "long": paired_test(b_bpc[long].values, c_bpc[long].values)}
            if "lang" in b and b["lang"].nunique() > 1:
                out["by_length"][name]["by_lang"] = {
                    str(l): paired_test(b_bpc[b["lang"] == l].values, c_bpc[b["lang"] == l].values)
                    for l in b["lang"].unique()}
        out["bpc_holm"] = holm({k: v.get("wilcoxon_p", 1.0) for k, v in out["bpc"].items()})
    comp = ev / "compression_per_sentence.csv"
    if comp.exists():
        d = pd.read_csv(comp)
        for col in [c for c in d.columns if c.startswith("tok_v")]:
            out["tokens"][col[4:]] = paired_test(d["tok_baseline"].values, d[col].values)
    for key in _flores_directions(ev):
        direction_file = ev / f"flores_baseline_{key}.jsonl"
        if not direction_file.exists():
            continue
        bh = list(read_jsonl(direction_file))
        for name in cfg["conditions"]:
            f = ev / f"flores_{name}_{key}.jsonl"
            if name == "baseline" or not f.exists():
                continue
            ch = list(read_jsonl(f))
            out["chrf"].setdefault(key, {})[name] = paired_bootstrap_chrf(
                [r["hyp"] for r in bh], [r["hyp"] for r in ch], [r["ref"] for r in bh])
    out["pairs"] = pairwise_comparisons(cfg, lang_dir)
    write_json(lang_dir / "reports" / "significance.json", out)
    return out


_DIR_RE = re.compile(r"_([a-z]{3}_[A-Za-z]{4}-[a-z]{3}_[A-Za-z]{4})$")


def _flores_directions(ev: Path) -> list[str]:
    """Translation directions present in eval/ (robust to condition names containing underscores)."""
    return sorted({m.group(1) for f in ev.glob("flores_*.jsonl") if (m := _DIR_RE.search(f.stem))})


DEFAULT_PAIRS = [["baseline_lora", "D"], ["baseline_lora", "D_mean"], ["D_mean", "D"],
                 ["baseline_lora", "D_750"], ["baseline_lora", "D_mean_750"], ["D_mean_750", "D_750"]]


def pairwise_comparisons(cfg: dict, lang_dir: Path) -> dict:
    """Paired tests between specific conditions (a = reference, b = candidate; negative BPC diff = b better).
    Writes reports/comparisons.md. Pairs come from cfg['report']['pairs'] or DEFAULT_PAIRS."""
    ev = lang_dir / "eval"
    pairs = (cfg.get("report") or {}).get("pairs") or DEFAULT_PAIRS
    res: dict = {"bpc": {}, "chrf": {}}
    lines = ["# Pairwise comparisons", "",
             "Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), "
             "Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.", "",
             "## BPC", "", "| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |", "|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        fa, fb = ev / f"ppl_per_sentence_{a}.csv", ev / f"ppl_per_sentence_{b}.csv"
        if not (fa.exists() and fb.exists()):
            continue
        da, db = pd.read_csv(fa), pd.read_csv(fb)
        assert (da["id"].values == db["id"].values).all(), f"sentence sets differ: {a} vs {b}"
        ba = da["nll"] / (math.log(2) * da["n_chars"])
        bb = db["nll"] / (math.log(2) * db["n_chars"])
        t = paired_test(ba.values, bb.values)
        t["bpc_a"] = float(da["nll"].sum() / (math.log(2) * da["n_chars"].sum()))
        t["bpc_b"] = float(db["nll"].sum() / (math.log(2) * db["n_chars"].sum()))
        res["bpc"][f"{a} vs {b}"] = t
    hol = holm({k: v.get("wilcoxon_p", 1.0) for k, v in res["bpc"].items()})
    for k, t in res["bpc"].items():
        t["p_holm"] = hol[k]["p_holm"]
        a, b = k.split(" vs ")
        lines.append(f"| {a} | {b} | {t['bpc_a']:.4f} | {t['bpc_b']:.4f} | {t['mean_diff']:+.4f} | "
                     f"[{t['ci95_mean_diff'][0]:+.4f}, {t['ci95_mean_diff'][1]:+.4f}] | {t['p_holm']:.2g} |")
    keys = _flores_directions(ev)
    for key in keys:
        lines += ["", f"## chrF++ {key}", "", "| a | b | chrF a | chrF b | diff | 95% CI |", "|---|---|---|---|---|---|"]
        for a, b in pairs:
            fa, fb = ev / f"flores_{a}_{key}.jsonl", ev / f"flores_{b}_{key}.jsonl"
            if not (fa.exists() and fb.exists()):
                continue
            ra, rb = list(read_jsonl(fa)), list(read_jsonl(fb))
            r = paired_bootstrap_chrf([x["hyp"] for x in ra], [x["hyp"] for x in rb], [x["ref"] for x in ra])
            res["chrf"].setdefault(key, {})[f"{a} vs {b}"] = r
            lines.append(f"| {a} | {b} | {r['chrf_a']:.2f} | {r['chrf_b']:.2f} | {r['diff']:+.2f} | "
                         f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] |")
    out = lang_dir / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparisons.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("pairwise comparisons written to %s", out / "comparisons.md")
    return res


# ------------------------------------------------------------------------------ failure analysis
def _rep4(text: str) -> float:
    w = text.split()
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not grams else 1 - len(set(grams)) / len(grams)


def _script_mix(text: str, expect_arabic: bool) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    ar = sum(is_arabic_char(c) for c in letters) / len(letters)
    return 1 - ar if expect_arabic else ar


def failure_cases(cfg: dict, lang_dir: Path, top: int = 15) -> Path:
    ev = lang_dir / "eval"
    lines = ["# Failure cases", "", "Auto-flagged; every flag needs a human verdict "
             "(degeneration, repetition, spacing, punctuation, code-switching, semantic error).", ""]
    summary = []
    for f in sorted(ev.glob("flores_*_*.jsonl")):
        m = re.match(r"flores_(.+?)_([a-z]{3}_[A-Za-z]{4}-[a-z]{3}_[A-Za-z]{4})$", f.stem)
        if not m:
            continue
        cond, key = m.groups()
        rows = list(read_jsonl(f))
        to_ar = not key.endswith("eng_Latn")
        for r in rows:
            r["rep4"] = _rep4(r["hyp"])
            r["script_mix"] = _script_mix(r["hyp"], to_ar)
            r["len_ratio"] = len(r["hyp"]) / max(len(r["ref"]), 1)
            r["badness"] = r["rep4"] + r["script_mix"] + abs(math.log(max(r["len_ratio"], 1e-3)))
        summary.append({"condition": cond, "direction": key, "empty": sum(not r["hyp"] for r in rows),
                        "mean_rep4": float(np.mean([r["rep4"] for r in rows])),
                        "mean_script_mix": float(np.mean([r["script_mix"] for r in rows])),
                        "median_len_ratio": float(np.median([r["len_ratio"] for r in rows]))})
        lines += [f"## {cond} — {key}", ""]
        for r in sorted(rows, key=lambda r: -r["badness"])[:top]:
            lines += [f"- **#{r['i']}** rep4={r['rep4']:.2f} mix={r['script_mix']:.2f} len×{r['len_ratio']:.2f}",
                      f"  - ref: {r['ref']}", f"  - hyp: {r['hyp']}", "  - verdict: "]
        lines.append("")
    out = lang_dir / "reports" / "failure_cases.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(lang_dir / "reports" / "failure_summary.csv", index=False)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
