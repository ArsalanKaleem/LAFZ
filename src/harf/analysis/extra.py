"""Extra analyses for the paper's mechanism/novelty claims.

ceiling      : compression ceiling imposed by the INHERITED pre-tokenizer. Merges never cross pre-token
               boundaries, so with unlimited new vocabulary a merge-based extension can at best turn each
               pre-token into one token:  ceiling SCR = 1 - (#pre-tokens / #base tokens).
loops        : does fragmentation predict degenerate (looping) generation? Item-level correlation between
               the reference's fertility under the BASE tokenizer and repetition in each system's output.
lora_norms   : how much did LoRA change the shared weights? ||(alpha/r) B A||_F per module, summed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..utils import log, read_jsonl, write_json

DEFAULT_CEILING_LANGS = [
    "eng_Latn", "rus_Cyrl", "urd_Arab", "snd_Arab", "pnb_Arab", "pbt_Arab", "pes_Arab", "arb_Arab",
    "hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu", "mal_Mlym", "sin_Sinh", "amh_Ethi", "tha_Thai",
]


# ----------------------------------------------------------------------------- 1. pre-tokenizer ceiling
def pretoken_ceiling(tok, sentences: list[str]) -> dict:
    backend = tok.backend_tokenizer
    norm, pre = backend.normalizer, backend.pre_tokenizer
    n_tok = n_pre = n_words = n_chars = 0
    for s in sentences:
        n_tok += len(tok.encode(s, add_special_tokens=False))
        t = norm.normalize_str(s) if norm is not None else s
        n_pre += len(pre.pre_tokenize_str(t))
        n_words += len(s.split())
        n_chars += len(s)
    return {"tokens_per_word": n_tok / max(n_words, 1), "pretokens_per_word": n_pre / max(n_words, 1),
            "tokens_per_char": n_tok / max(n_chars, 1), "ceiling_scr": 1 - n_pre / max(n_tok, 1),
            "n_sentences": len(sentences)}


def ceiling_table(tok, flores_dir: str, langs: list[str], split: str, out_dir: Path,
                  achieved: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    from ..text.scripts import script_for_flores
    from ..tokenizer.repair import repair_pretokenizer
    rows = []
    for code in langs:
        p = Path(flores_dir) / split / f"{code}.{split}"
        if not p.exists():
            log.warning("missing %s", p)
            continue
        sents = p.read_text(encoding="utf-8").splitlines()
        r = {"lang": code, **pretoken_ceiling(tok, sents)}
        sc = script_for_flores(code)
        if sc is not None and sc.marks:
            rep = pretoken_ceiling(repair_pretokenizer(tok, sc), sents)
            r["pretokens_per_word_repaired"] = rep["pretokens_per_word"]
            # ceiling relative to the ORIGINAL tokenization (repair changes pre-tokens, not the baseline)
            r["ceiling_scr_repaired"] = 1 - rep["pretokens_per_word"] / r["tokens_per_word"]
        for label, scr in (achieved or {}).get(code, {}).items():
            r[f"achieved_{label}"] = scr
            r[f"share_of_ceiling_{label}"] = scr / r["ceiling_scr"] if r["ceiling_scr"] > 0 else float("nan")
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("ceiling_scr")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "pretokenizer_ceiling.csv", index=False)
    lines = ["# Pre-tokenizer ceiling", "",
             "Merges cannot cross pre-token boundaries, so `ceiling_scr` = 1 − pre-tokens/tokens is the most any "
             "merge-based vocabulary extension can compress, however many tokens it adds. `ceiling_scr_repaired` is the "
             "same bound after the script-scoped pre-tokenizer repair.", "",
             "| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    (out_dir / "pretokenizer_ceiling.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


# ----------------------------------------------------------------------------- 2. loops vs fertility
def rep4(text: str) -> float:
    w = text.split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 0.0 if not g else 1 - len(set(g)) / len(g)


def loops_vs_fertility(base_tok, ev: Path, conditions: list[str], direction: str, out_dir: Path,
                       loop_threshold: float = 0.5) -> dict:
    res: dict = {"direction": direction, "loop_threshold": loop_threshold, "conditions": {}}
    lines = [f"# Looping vs fragmentation ({direction})", "",
             "Fertility = base-tokenizer tokens per word of the REFERENCE sentence (a property of the content, "
             f"identical for all systems). Loop = output with ≥{loop_threshold:.0%} repeated 4-grams.", "",
             "| condition | Spearman ρ (fertility, rep4) | p | loop rate Q1 (least fragmented) | Q2 | Q3 | Q4 (most) |",
             "|---|---|---|---|---|---|---|"]
    for cond in conditions:
        f = ev / f"flores_{cond}_{direction}.jsonl"
        if not f.exists():
            continue
        rows = list(read_jsonl(f))
        fert = np.array([len(base_tok.encode(" " + r["ref"], add_special_tokens=False)) / max(len(r["ref"].split()), 1)
                         for r in rows])
        rp = np.array([rep4(r["hyp"]) for r in rows])
        rho, p = stats.spearmanr(fert, rp)
        q = pd.qcut(fert, 4, labels=False, duplicates="drop")
        rates = [float((rp[q == k] >= loop_threshold).mean()) for k in range(4)]
        res["conditions"][cond] = {"spearman_rho": float(rho), "p": float(p), "loop_rate_by_quartile": rates,
                                   "loop_rate": float((rp >= loop_threshold).mean())}
        lines.append(f"| {cond} | {rho:+.3f} | {p:.2g} | " + " | ".join(f"{x:.0%}" for x in rates) + " |")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"loops_vs_fertility_{direction}.json", res)
    (out_dir / f"loops_vs_fertility_{direction}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res


# ----------------------------------------------------------------------------- 3. LoRA update norms
def lora_update_norms(adapter_dir: Path) -> dict:
    """Frobenius norm of the effective update (alpha/r)·B·A per module; summed by module type and layer."""
    import torch
    from safetensors.torch import load_file
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    by_type, by_layer, total_sq = {}, {}, 0.0
    for k, A in sd.items():
        if ".lora_A." not in k:
            continue
        B = sd[k.replace(".lora_A.", ".lora_B.")]
        n2 = float(torch.linalg.matrix_norm(scale * (B.float() @ A.float())) ** 2)
        total_sq += n2
        mtype = k.split(".lora_A.")[0].split(".")[-1]
        layer = next((p for p in k.split(".") if p.isdigit()), "?")
        by_type[mtype] = by_type.get(mtype, 0.0) + n2
        by_layer[layer] = by_layer.get(layer, 0.0) + n2
    return {"total_norm": math.sqrt(total_sq),
            "by_module_type": {k: math.sqrt(v) for k, v in sorted(by_type.items())},
            "by_layer": {k: math.sqrt(v) for k, v in sorted(by_layer.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1)}}
