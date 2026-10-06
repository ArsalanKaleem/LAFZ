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
    for direction_file in sorted(ev.glob("flores_baseline_*.jsonl")):
        key = direction_file.stem.replace("flores_baseline_", "")
        bh = list(read_jsonl(direction_file))
        for name in cfg["conditions"]:
            f = ev / f"flores_{name}_{key}.jsonl"
            if name == "baseline" or not f.exists():
                continue
            ch = list(read_jsonl(f))
            out["chrf"].setdefault(key, {})[name] = paired_bootstrap_chrf(
                [r["hyp"] for r in bh], [r["hyp"] for r in ch], [r["ref"] for r in bh])
    write_json(lang_dir / "reports" / "significance.json", out)
    return out


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
