"""Day 3: per-sentence tokenization audit (tokens/word, tokens/char, fragmentation) + inspection sheet."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from ..text.scripts import get_script
from ..utils import log, sha256_file, write_json


def _is_target_word(w: str, script) -> bool:
    return any(script.is_char(c) and c.isalpha() for c in w)


def audit_sentences(tok, rows: list[dict], script=None) -> pd.DataFrame:
    script = script or get_script("arabic")
    texts = [r["text"] for r in rows]
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    word_cache: dict[str, int] = {}

    def word_len(w: str) -> int:
        if w not in word_cache:
            word_cache[w] = len(tok.encode(" " + w, add_special_tokens=False))
        return word_cache[w]

    recs = []
    for r, ids in zip(rows, enc):
        t = r["text"]
        words = t.split()
        tw = [w for w in words if _is_target_word(w, script)]
        frag = sum(word_len(w) > 1 for w in tw)
        recs.append({
            "id": r["id"], "lang": r.get("lang"), "n_chars": len(t), "n_bytes": len(t.encode("utf-8")),
            "n_words": len(words), "n_target_words": len(tw), "n_tokens": len(ids),
            "n_fragmented_target_words": frag,
            "tokens_in_target_words": sum(word_len(w) for w in tw),
        })
    return pd.DataFrame(recs)


def summarize(df: pd.DataFrame) -> dict:
    tot = df.sum(numeric_only=True)
    return {
        "n_sentences": int(len(df)),
        "total_tokens": int(tot["n_tokens"]),
        "tokens_per_word": float(tot["n_tokens"] / max(tot["n_words"], 1)),
        "tokens_per_char": float(tot["n_tokens"] / max(tot["n_chars"], 1)),
        "chars_per_token": float(tot["n_chars"] / max(tot["n_tokens"], 1)),
        "bytes_per_token": float(tot["n_bytes"] / max(tot["n_tokens"], 1)),
        "median_tokens_per_sentence": float(np.median(df["n_tokens"])),
        "target_word_fertility": float(tot["tokens_in_target_words"] / max(tot["n_target_words"], 1)),
        "fragmented_target_word_rate": float(tot["n_fragmented_target_words"] / max(tot["n_target_words"], 1)),
    }


def inspection_sheet(tok, rows: list[dict], n: int, seed: int) -> str:
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    out = ["# Tokenization inspection sheet", "",
           "Pieces are separated by `|`. `�` marks a token that is a partial UTF-8 character "
           "(the tokenizer split a single letter into bytes).", "",
           "Check: punctuation (۔ ، ؟), numerals (۰-۹ / 0-9), mixed English, ZWNJ, diacritics.", ""]
    for r in sample:
        ids = tok.encode(r["text"], add_special_tokens=False)
        pieces = [tok.decode([i]) for i in ids]
        out += [f"## {r['id']}  ({len(ids)} tokens, {len(r['text'].split())} words)", "",
                f"<div dir=\"rtl\">{r['text']}</div>", "", "`" + "|".join(pieces).replace("`", "'") + "`", "",
                "Notes: ", ""]
    return "\n".join(out)


def run_audit(tok, rows: list[dict], out_dir: Path, name: str, seed: int, inspect_n: int = 75,
              freeze: bool = False, force: bool = False, script=None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_per_sentence.csv"
    if freeze and csv_path.exists() and not force:
        raise SystemExit(f"{csv_path} is frozen. Refusing to overwrite (use --force only if you really mean it).")
    df = audit_sentences(tok, rows, script)
    df.to_csv(csv_path, index=False)
    summ = {"overall": summarize(df)}
    for lang, g in df.groupby("lang"):
        summ[str(lang)] = summarize(g)
    summ["per_sentence_sha256"] = sha256_file(csv_path)
    write_json(out_dir / f"{name}_metrics.json", summ)
    if inspect_n:
        (out_dir / f"{name}_inspection.md").write_text(inspection_sheet(tok, rows, inspect_n, seed), encoding="utf-8")
    log.info("%s: %s", name, summ["overall"])
    return summ
