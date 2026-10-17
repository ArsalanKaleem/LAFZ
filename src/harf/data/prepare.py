"""Day 2: raw text -> cleaned, deduplicated, document-level train/valid/test splits + data card.

Outputs (per language) in <processed>/<lang>/:
  train.jsonl   documents  {id, doc_id, lang, text, source, license}
  valid.jsonl   sentences  {id, doc_id, lang, text}
  test.jsonl    sentences
  data_card.json / data_card.md
"""
from __future__ import annotations

import glob
import hashlib
import os
import random
from collections import Counter
from pathlib import Path

from ..text.scripts import script_for_lang
from ..text.normalize import (script_letter_ratio, char_inventory, looks_like_other_language, marker_counts,
                              normalize, split_sentences)
from ..utils import log, open_text, read_jsonl, stable_hash01, write_json, write_jsonl


def iter_raw_docs(patterns: list[str], doc_mode: str = "paragraph", text_field: str = "text"):
    seen_files = set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            if os.path.isdir(path) or path in seen_files:
                continue
            seen_files.add(path)
            base = os.path.relpath(path)
            if ".jsonl" in path or path.endswith(".json"):
                for i, rec in enumerate(read_jsonl(path)):
                    text = rec.get(text_field)
                    if not isinstance(text, str):
                        continue
                    yield {"doc_id": f"{base}:{rec.get('id', i)}", "text": text,
                           "source": rec.get("source", base), "license": rec.get("license")}
            else:
                with open_text(path) as f:
                    content = f.read()
                if doc_mode == "line":
                    chunks = content.split("\n")
                else:
                    chunks = [c for c in content.replace("\r\n", "\n").split("\n\n")]
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        yield {"doc_id": f"{base}:{i}", "text": chunk, "source": base, "license": None}


def prepare_language(cfg: dict, lang: str) -> dict:
    dcfg = cfg["data"]
    seed = str(cfg["seed"])
    out_dir = Path(cfg["paths"]["processed"]) / lang
    patterns = cfg["paths"]["raw"].get(lang, [f"data/raw/{lang}/**/*"])
    script = script_for_lang(cfg, lang)
    min_ratio = dcfg.get("min_script_ratio", dcfg.get("min_arabic_ratio", 0.6))
    norm_counts: Counter = Counter()
    drops: Counter = Counter()
    seen_hashes = set()
    splits = {"train": [], "valid": [], "test": []}

    for doc in iter_raw_docs(patterns, dcfg["doc_mode"], dcfg["text_field"]):
        drops["docs_in"] += 1
        text = normalize(doc["text"], lang, canonicalize_letters=dcfg["canonicalize_letters"],
                         canonicalize_heh=dcfg["canonicalize_heh"], strip_html=dcfg["strip_html"],
                         strip_urls=dcfg["strip_urls"], counter=norm_counts)
        if len(text) < dcfg["min_doc_chars"]:
            drops["too_short"] += 1
            continue
        if script_letter_ratio(text, script) < min_ratio:
            drops["low_script_ratio"] += 1
            continue
        if dcfg.get("purity_filter") and looks_like_other_language(text, lang):
            drops["other_language_markers"] += 1
            continue
        h = hashlib.sha1("".join(text.split()).encode()).hexdigest()
        if h in seen_hashes:
            drops["duplicate_doc"] += 1
            continue
        seen_hashes.add(h)
        r = stable_hash01(doc["doc_id"], salt=seed)
        split = "test" if r < dcfg["test_frac"] else "valid" if r < dcfg["test_frac"] + dcfg["valid_frac"] else "train"
        splits[split].append({**doc, "text": text, "lang": lang})

    if not splits["train"]:
        raise SystemExit(f"No usable documents for {lang} from {patterns}")

    # train: documents. valid/test: sentences, with leakage removal against train sentences.
    train_sents = set()
    for d in splits["train"]:
        train_sents.update(split_sentences(d["text"]))
    rng = random.Random(cfg["seed"])
    eval_sets = {}
    for split in ("valid", "test"):
        rows, seen = [], set()
        for d in splits[split]:
            for j, s in enumerate(split_sentences(d["text"], dcfg["min_sentence_chars"])):
                if s in train_sents:
                    drops[f"{split}_leak_removed"] += 1
                    continue
                if s in seen:
                    continue
                seen.add(s)
                rows.append({"id": f"{split}-{d['doc_id']}-{j}", "doc_id": d["doc_id"], "lang": lang, "text": s})
        if len(rows) > dcfg["max_eval_sentences"]:
            rows = sorted(rng.sample(rows, dcfg["max_eval_sentences"]), key=lambda r: r["id"])
        eval_sets[split] = rows

    if not eval_sets["valid"] or not eval_sets["test"]:
        raise SystemExit(f"{lang}: empty valid/test split after leakage removal ({dict(drops)}). "
                         "Increase valid_frac/test_frac or add more distinct documents.")
    if len(eval_sets["valid"]) < 200 or len(eval_sets["test"]) < 200:
        log.warning("%s: few eval sentences (valid=%d, test=%d). Increase valid_frac/test_frac.",
                    lang, len(eval_sets["valid"]), len(eval_sets["test"]))

    train_rows = [{"id": f"train-{i}", **d} for i, d in enumerate(splits["train"])]
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", eval_sets["valid"])
    write_jsonl(out_dir / "test.jsonl", eval_sets["test"])

    all_train_text = [d["text"] for d in train_rows]
    sd_m, ur_m = 0, 0
    for t in all_train_text:
        a, b = marker_counts(t)
        sd_m += a
        ur_m += b
    card = {
        "language": lang,
        "sources": sorted({d["source"] for d in train_rows})[:200],
        "licenses": sorted({str(d.get("license")) for d in train_rows}),
        "counts": {
            "train_docs": len(train_rows),
            "train_chars": sum(len(t) for t in all_train_text),
            "train_words": sum(len(t.split()) for t in all_train_text),
            "valid_sentences": len(eval_sets["valid"]),
            "test_sentences": len(eval_sets["test"]),
        },
        "filtering": dict(drops),
        "normalization_counts": dict(norm_counts),
        "normalization_settings": {k: dcfg[k] for k in ("canonicalize_letters", "canonicalize_heh",
                                                        "strip_html", "strip_urls", "purity_filter")},
        "marker_letters_in_train": {"sindhi_markers": sd_m, "urdu_markers": ur_m},
        "char_inventory_top": char_inventory(all_train_text[:50000]),
        "seed": cfg["seed"],
    }
    write_json(out_dir / "data_card.json", card)
    _write_card_md(out_dir / "data_card.md", card)
    log.info("%s: %s", lang, card["counts"])
    return card


def _write_card_md(path: Path, card: dict) -> None:
    lines = [f"# Data card — {card['language']}", "", "## Counts"]
    lines += [f"- {k}: {v:,}" for k, v in card["counts"].items()]
    lines += ["", "## Filtering"] + [f"- {k}: {v:,}" for k, v in card["filtering"].items()]
    lines += ["", "## Normalization (counts of changes)"] + [f"- {k}: {v:,}" for k, v in card["normalization_counts"].items()]
    lines += ["", f"Settings: `{card['normalization_settings']}`", "",
              f"Marker letters in train: {card['marker_letters_in_train']}", "",
              "## Sources / licenses", "", f"Licenses: {card['licenses']}", ""]
    lines += [f"- {s}" for s in card["sources"][:50]]
    lines += ["", "## Top characters", "", "| char | codepoint | name | share |", "|---|---|---|---|"]
    for r in card["char_inventory_top"][:80]:
        ch = r["char"] if not r["char"].isspace() else "␠"
        lines.append(f"| {ch} | {r['codepoint']} | {r['name']} | {r['share']:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_split(cfg: dict, lang: str, split: str) -> list[dict]:
    path = Path(cfg["paths"]["processed"]) / lang / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `harf prepare` first.")
    return list(read_jsonl(path))


def load_split_multi(cfg: dict, split: str) -> list[dict]:
    rows = []
    for lang in cfg["lang"].split("+"):
        rows.extend(load_split(cfg, lang, split))
    return rows
