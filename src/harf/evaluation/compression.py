"""Day 10: compression on identical sentences (tokenizer-only; no GPU).

SCR (fixed definition used everywhere):  SCR = 1 - Σ tokens_ext / Σ tokens_base   over the eval set.
Also reports per-sentence SCR distribution, tokens/word, median tokens/sentence, paired tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.stats import paired_test
from ..tokenizer.common import load_tokenizer
from ..utils import log, write_json


def count_tokens(tok, texts: list[str]) -> np.ndarray:
    return np.array([len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]])


def evaluate_compression(base_tok, tokenizers: dict[str, object], rows: list[dict], out_dir: Path) -> pd.DataFrame:
    """tokenizers: label -> tokenizer (label e.g. 'v2000'). Sentence boundaries are fixed by `rows`."""
    texts = [r["text"] for r in rows]
    df = pd.DataFrame({"id": [r["id"] for r in rows], "lang": [r.get("lang") for r in rows],
                       "n_chars": [len(t) for t in texts], "n_words": [len(t.split()) for t in texts]})
    df["tok_baseline"] = count_tokens(base_tok, texts)
    summary = {"definition": "SCR = 1 - sum(tokens_ext)/sum(tokens_base)", "n_sentences": len(df), "conditions": {}}
    for label, tok in tokenizers.items():
        df[f"tok_{label}"] = count_tokens(tok, texts)
    for label in ["baseline"] + list(tokenizers):
        col = df[f"tok_{label}"]
        s = {"total_tokens": int(col.sum()), "median_tokens_per_sentence": float(col.median()),
             "tokens_per_word": float(col.sum() / df["n_words"].sum()),
             "chars_per_token": float(df["n_chars"].sum() / col.sum())}
        if label != "baseline":
            s["scr"] = float(1 - col.sum() / df["tok_baseline"].sum())
            per = 1 - col / df["tok_baseline"].clip(lower=1)
            s["per_sentence_scr_median"] = float(per.median())
            s["per_sentence_scr_iqr"] = [float(per.quantile(0.25)), float(per.quantile(0.75))]
            s["paired_test_vs_baseline"] = paired_test(df["tok_baseline"].to_numpy(), col.to_numpy())
            s["by_lang"] = {str(l): float(1 - g[f"tok_{label}"].sum() / g["tok_baseline"].sum())
                            for l, g in df.groupby("lang")}
        summary["conditions"][label] = s
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "compression_per_sentence.csv", index=False)
    write_json(out_dir / "compression_results.json", summary)
    pd.DataFrame([{"condition": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))}}
                  for k, v in summary["conditions"].items()]).to_csv(out_dir / "compression_results.csv", index=False)
    _plot(df, list(tokenizers), out_dir)
    log.info("compression: %s", {k: round(v.get("scr", 0), 4) for k, v in summary["conditions"].items()})
    return df


def _plot(df: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    data = [1 - df[f"tok_{l}"] / df["tok_baseline"].clip(lower=1) for l in labels]
    try:
        axes[0].boxplot(data, tick_labels=labels)
    except TypeError:
        axes[0].boxplot(data, labels=labels)
    axes[0].set_title("Per-sentence SCR")
    axes[0].set_ylabel("1 - ext/base")
    for l in labels:
        axes[1].scatter(df["tok_baseline"], df[f"tok_{l}"], s=4, alpha=0.4, label=l)
    m = df["tok_baseline"].max()
    axes[1].plot([0, m], [0, m], "k--", lw=0.8)
    axes[1].set_xlabel("baseline tokens")
    axes[1].set_ylabel("extended tokens")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "compression.png", dpi=150)
    plt.close(fig)


def tokenizers_for_budgets(lang_dir: Path, conditions: dict) -> dict:
    """One tokenizer per distinct budget (conditions with the same budget share it)."""
    out = {}
    for name, c in conditions.items():
        b = c.get("budget", 0)
        if b and f"v{b}" not in out and (lang_dir / name / "tokenizer").exists():
            out[f"v{b}"] = load_tokenizer(str(lang_dir / name / "tokenizer"))
    return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))
