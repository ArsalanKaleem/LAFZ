"""Day 6 (tokenizer half): write an extended tokenizer by appending vocab entries + merges.

Existing IDs never change. New IDs start right after the largest existing ID (including added/special
tokens, e.g. Qwen's <|endoftext|> etc.)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from tokenizers import Tokenizer

from ..utils import log, write_json
from .common import backend_json, load_tokenizer

CONTROL_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return x**2  # comment",
    "Prices rose 3.5% in 2024; see https://example.com for details.",
    "Mixed: Hello world 12345 !!! ??? ...",
    "日本語のテキスト and émojis 🙂",
]


def extend_tokenizer(base_tok, manifest: dict, n_new: int, out_dir: Path, verify_texts: list[str]) -> dict:
    entries = manifest["tokens"][:n_new]
    if len(entries) < n_new:
        raise SystemExit(f"Manifest has only {len(manifest['tokens'])} tokens, budget {n_new} requested.")
    tj = backend_json(base_tok)
    model = tj["model"]
    vocab, merges = model["vocab"], model["merges"]
    string_merges = bool(merges) and isinstance(merges[0], str)
    max_id = max(list(vocab.values()) + [t["id"] for t in tj.get("added_tokens", [])])
    next_id = max_id + 1
    # Make the model vocab contiguous by registering added/special tokens at their existing ids.
    # Otherwise loaders that (re-)add special tokens assign id = len(model vocab), which after
    # extension collides with our new ids (observed with transformers>=5 TokenizersBackend).
    for t in tj.get("added_tokens", []):
        if t["content"] not in vocab:
            vocab[t["content"]] = t["id"]
    if len(set(vocab.values())) != next_id:
        log.warning("Base id space has gaps (%d ids used of %d).", len(set(vocab.values())), next_id)
    if next_id != len(base_tok):
        log.warning("len(base_tok)=%d but max id+1=%d; using the latter.", len(base_tok), next_id)
    new_ids = []
    for e in entries:
        if e["token"] in vocab:
            raise RuntimeError(f"Token already in vocab: {e['token']!r}")
        a, b = e["pair"]
        if a not in vocab or b not in vocab:
            raise RuntimeError(f"Merge components missing for {e['token']!r}: {a!r}, {b!r}")
        vocab[e["token"]] = next_id
        e["id"] = next_id
        new_ids.append(next_id)
        next_id += 1
        merges.append(f"{a} {b}" if string_merges else [a, b])

    new_backend = Tokenizer.from_str(json.dumps(tj))
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    base_tok.save_pretrained(out_dir)
    for stale in ("vocab.json", "merges.txt"):  # slow-tokenizer files would be inconsistent
        (out_dir / stale).unlink(missing_ok=True)
    new_backend.save(str(out_dir / "tokenizer.json"))

    probe_text = " " + entries[0]["text"].strip() if entries else ""
    probe_ids = new_backend.encode(probe_text, add_special_tokens=False).ids if entries else []
    write_json(out_dir / "harf_meta.json", {
        "expected_len": next_id, "n_new": len(entries), "first_new_id": new_ids[0] if new_ids else None,
        "probe": {"text": probe_text, "ids": probe_ids} if entries else None,
    })
    write_json(out_dir / "new_tokens.json", entries)

    ext = load_tokenizer(str(out_dir))
    report = verify_extension(base_tok, ext, new_ids, verify_texts)
    write_json(out_dir / "verification.json", report)
    return report


def verify_extension(base_tok, ext_tok, new_ids: list[int], texts: list[str]) -> dict:
    rep: dict = {"len_base": len(base_tok), "len_ext": len(ext_tok), "n_new": len(new_ids)}
    # 1. existing ids unchanged
    bv, ev = base_tok.get_vocab(), ext_tok.get_vocab()
    rep["existing_ids_unchanged"] = all(ev.get(t) == i for t, i in bv.items())
    # 2. non-Arabic control text tokenized identically
    rep["control_identical"] = all(base_tok.encode(t, add_special_tokens=False) ==
                                   ext_tok.encode(t, add_special_tokens=False) for t in CONTROL_TEXTS)
    # 3. lossless round trip on target text
    enc = ext_tok(texts, add_special_tokens=False)["input_ids"]
    bad = [t for t, ids in zip(texts, enc) if ext_tok.decode(ids) != t]
    rep["roundtrip_failures"] = len(bad)
    rep["roundtrip_examples"] = bad[:5]
    # 4. compression and usage of new tokens
    base_n = sum(len(x) for x in base_tok(texts, add_special_tokens=False)["input_ids"])
    ext_n = sum(len(x) for x in enc)
    new_set = set(new_ids)
    used = {i for ids in enc for i in ids if i in new_set}
    rep.update({"base_tokens": base_n, "ext_tokens": ext_n, "scr": 1 - ext_n / max(base_n, 1),
                "new_tokens_used": len(used), "new_token_usage_rate": len(used) / max(len(new_ids), 1)})
    ok = rep["existing_ids_unchanged"] and rep["control_identical"] and rep["roundtrip_failures"] == 0
    rep["ok"] = bool(ok)
    (log.info if ok else log.error)("extension verification: %s", {k: v for k, v in rep.items() if k != "roundtrip_examples"})
    if not ok:
        raise RuntimeError(f"Tokenizer extension failed verification: {rep}")
    return rep
