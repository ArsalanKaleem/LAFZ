"""Script-scoped pre-tokenizer repair for PRETRAINED models.

GPT-2-style word regexes match words as \\p{L}+, but abugidas write vowels as combining marks (Mn/Mc),
so every word is split at every vowel sign and no merge can ever cross those splits (Regmi et al., 2026
diagnose this for tokenizers trained from scratch). Here we repair the inherited regex of an EXISTING
model, scoped to ONE script: only that script's combining marks are added to the word class (and removed
from the punctuation classes). Text without that script's marks is pre-tokenized exactly as before, so the
non-interference guarantee is preserved. The vocabulary and merges are unchanged by the repair itself.
"""
from __future__ import annotations

import json

from tokenizers import Tokenizer

from ..text.scripts import Script
from .common import backend_json

# (original fragment, template) — {M} is replaced by the script's combining marks.
_REPLACEMENTS = [
    ("\\p{L}+", "[\\p{L}{M}]+"),                      # word class: letters (+ target-script marks)
    ("[^\\r\\n\\p{L}\\p{N}]", "[^\\r\\n\\p{L}\\p{N}{M}]"),  # optional word prefix must not eat a mark
    ("[^\\s\\p{L}\\p{N}]", "[^\\s\\p{L}\\p{N}{M}]"),        # punctuation run must not eat a mark
]


def _escape_for_class(chars: str) -> str:
    return "".join("\\" + c if c in "\\]^-[" else c for c in chars)


def _patch_pattern(pattern: str, marks: str) -> tuple[str, int]:
    m = _escape_for_class(marks)
    n = 0
    for old, new in _REPLACEMENTS:
        if old in pattern:
            pattern = pattern.replace(old, new.replace("{M}", m))
            n += 1
    return pattern, n


def _walk(node, marks: str, counter: list):
    if isinstance(node, dict):
        if node.get("type") == "Split" and isinstance(node.get("pattern"), dict) and "Regex" in node["pattern"]:
            new, n = _patch_pattern(node["pattern"]["Regex"], marks)
            node["pattern"]["Regex"] = new
            counter[0] += n
            if "\\p{L}+" in new:
                counter[1] += 1
        for v in node.values():
            _walk(v, marks, counter)
    elif isinstance(node, list):
        for v in node:
            _walk(v, marks, counter)


def repaired_backend_json(tok, script: Script) -> dict:
    if not script.marks:
        raise ValueError(f"Script {script.name} has no combining marks; nothing to repair.")
    tj = backend_json(tok)
    counter = [0, 0]
    _walk(tj.get("pre_tokenizer"), script.marks, counter)
    if counter[0] == 0 or counter[1]:
        raise RuntimeError("Unsupported pre-tokenizer pattern: could not find the GPT-2-style word class "
                           "\\p{L}+ to repair (or some occurrence was left unpatched).")
    return tj


def repair_pretokenizer(tok, script: Script):
    """Return a NEW HF tokenizer with the same vocab/merges/special tokens and a script-scoped repaired regex."""
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer.from_str(json.dumps(repaired_backend_json(tok, script)))
    new = PreTrainedTokenizerFast(tokenizer_object=backend)
    for attr in ("bos_token", "eos_token", "pad_token", "unk_token"):
        val = getattr(tok, attr, None)
        if val is not None:
            setattr(new, attr, val)
    if getattr(tok, "chat_template", None):
        new.chat_template = tok.chat_template
    new.harf_repaired_script = script.name
    return new
