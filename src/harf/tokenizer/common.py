"""Byte-level BPE helpers and robust tokenizer loading."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ARABIC_LEAD_BYTES = set(range(0xD8, 0xDE))  # UTF-8 lead bytes of U+0600..U+077F


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


def bl_to_bytes(token: str) -> bytes | None:
    """Byte-level token string (e.g. 'Ġک') -> raw bytes; None if not a byte-level string."""
    u2b = unicode_to_bytes()
    try:
        return bytes(u2b[c] for c in token)
    except KeyError:
        return None


def bl_to_text(token: str, errors: str = "replace") -> str:
    b = bl_to_bytes(token)
    return token if b is None else b.decode("utf-8", errors=errors)


def text_to_bl(text: str) -> str:
    b2u = bytes_to_unicode()
    return "".join(b2u[b] for b in text.encode("utf-8"))


def touches_arabic(token: str) -> bool:
    """True iff the byte-level token contains a UTF-8 lead byte of the Arabic block.

    If every new merge's result satisfies this, any text without Arabic-block characters is
    provably tokenized exactly as before (strict non-interference)."""
    b = bl_to_bytes(token)
    return b is not None and any(x in ARABIC_LEAD_BYTES for x in b)


def touches_script(token: str, script) -> bool:
    """True iff the byte-level token contains a UTF-8 signature of `script` (see text/scripts.py)."""
    b = bl_to_bytes(token)
    return b is not None and script.touches(b)


def is_byte_level(tok_json: dict) -> bool:
    s = json.dumps(tok_json.get("pre_tokenizer")) + json.dumps(tok_json.get("decoder"))
    return "ByteLevel" in s


def load_tokenizer(path_or_name: str, revision: str | None = None):
    """Load a HF tokenizer. For HARF-extended tokenizers, verify the extension survived loading and
    fall back to PreTrainedTokenizerFast(tokenizer_file=...) if a class-specific loader rebuilt it."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    kw = {"revision": revision} if revision and not Path(str(path_or_name)).exists() else {}
    tok = AutoTokenizer.from_pretrained(path_or_name, **kw)
    meta_path = Path(str(path_or_name)) / "harf_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        probe = meta.get("probe")  # (text, expected_ids)
        ok = len(tok) == meta["expected_len"]
        if ok and probe:
            ok = tok.encode(probe["text"], add_special_tokens=False) == probe["ids"]
        if not ok:
            fast = PreTrainedTokenizerFast(tokenizer_file=str(Path(path_or_name) / "tokenizer.json"))
            for attr in ("bos_token", "eos_token", "pad_token", "unk_token"):
                val = getattr(tok, attr, None)
                if val is not None:
                    setattr(fast, attr, val)
            if getattr(tok, "chat_template", None):
                fast.chat_template = tok.chat_template
            tok = fast
            if len(tok) != meta["expected_len"]:
                raise RuntimeError(f"Extended tokenizer at {path_or_name} did not load correctly "
                                   f"(len={len(tok)}, expected {meta['expected_len']}).")
    return tok


def backend_json(tok) -> dict:
    return json.loads(tok.backend_tokenizer.to_str())
