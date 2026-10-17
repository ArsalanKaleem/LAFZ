"""Writing-system registry used everywhere a step depends on the target script.

For each script we derive, from Unicode code-point ranges:
  * `is_char(ch)`      : is this character in the script?
  * `marks`            : its combining marks (Unicode Mn/Mc) — what the pre-tokenizer repair keeps attached
  * `signatures`       : UTF-8 byte prefixes that occur ONLY in text containing this script.

Non-interference guarantee: every new token must contain a signature. A merge whose result contains a
signature can only fire on text that contains that signature, so any text WITHOUT characters of the
target script is tokenized exactly as by the base tokenizer.

Signatures are taken per 64-code-point UTF-8 block and only for blocks lying ENTIRELY inside the script's
ranges, so a block shared with a neighbouring script never becomes a signature.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

# Code-point ranges per script (inclusive).
SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    # Arabic signatures are kept identical to HARF v1 (UTF-8 lead bytes D8..DD = U+0600..U+077F) for
    # reproducibility of the Urdu/Sindhi results.
    "arabic": [(0x0600, 0x077F)],
    "devanagari": [(0x0900, 0x097F)],
    "bengali": [(0x0980, 0x09FF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "gujarati": [(0x0A80, 0x0AFF)],
    "tamil": [(0x0B80, 0x0BFF)],
    "telugu": [(0x0C00, 0x0C7F)],
    "kannada": [(0x0C80, 0x0CFF)],
    "malayalam": [(0x0D00, 0x0D7F)],
    "sinhala": [(0x0D80, 0x0DFF)],
    "thai": [(0x0E00, 0x0E7F)],
    "ethiopic": [(0x1200, 0x137F)],
}
# Extra code points recognised as "in script" for text statistics (not used for signatures).
EXTRA_CHARS: dict[str, list[tuple[int, int]]] = {
    "arabic": [(0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "ethiopic": [(0x1380, 0x139F), (0x2D80, 0x2DDF)],
}
# FLORES-200 script suffix -> registry name
FLORES_SCRIPT = {"Arab": "arabic", "Deva": "devanagari", "Beng": "bengali", "Guru": "gurmukhi",
                 "Gujr": "gujarati", "Taml": "tamil", "Telu": "telugu", "Knda": "kannada",
                 "Mlym": "malayalam", "Sinh": "sinhala", "Thai": "thai", "Ethi": "ethiopic"}


@dataclass(frozen=True)
class Script:
    name: str
    ranges: tuple[tuple[int, int], ...]
    extra: tuple[tuple[int, int], ...] = ()
    signatures: frozenset = field(default_factory=frozenset)
    marks: str = ""

    def is_char(self, ch: str) -> bool:
        cp = ord(ch)
        return any(a <= cp <= b for a, b in self.ranges + self.extra)

    def touches(self, b: bytes) -> bool:
        return any(sig in b for sig in self.signatures)


def _block_signatures(ranges) -> frozenset:
    sigs = set()
    for a, b in ranges:
        for start in range(a - a % 64, b + 1, 64):
            end = start + 63
            if start < a or end > b:
                continue  # block only partly inside the script -> not a safe signature
            if end < 0x80:
                continue
            if end < 0x800:
                sigs.add(bytes([0xC0 | (start >> 6)]))
            elif end < 0x10000:
                sigs.add(bytes([0xE0 | (start >> 12), 0x80 | ((start >> 6) & 0x3F)]))
    return frozenset(sigs)


@lru_cache(maxsize=None)
def get_script(name: str) -> Script:
    if name not in SCRIPT_RANGES:
        raise KeyError(f"Unknown script {name!r}; known: {sorted(SCRIPT_RANGES)}")
    ranges = tuple(SCRIPT_RANGES[name])
    marks = "".join(chr(cp) for a, b in ranges for cp in range(a, b + 1)
                    if unicodedata.category(chr(cp)) in ("Mn", "Mc"))
    return Script(name, ranges, tuple(EXTRA_CHARS.get(name, ())), _block_signatures(ranges), marks)


def script_for_lang(cfg: dict, lang: str) -> Script:
    return get_script((cfg.get("scripts") or {}).get(lang, "arabic"))


def script_for_flores(code: str) -> Script | None:
    name = FLORES_SCRIPT.get(code.split("_")[-1])
    return get_script(name) if name else None
