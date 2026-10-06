"""Script-aware text handling for Urdu and Sindhi.

Design rule: never silently transform the script. Every transformation here is opt-in or
"safe" (NFC, invisible formatting characters, tatweel), and is counted so it can be reported.

Key orthographic facts that make Urdu/Sindhi normalisation dangerous if done naively:
  * Sindhi kaf is ڪ (U+06AA). The Urdu kaf ک (U+06A9) is a *different letter* in Sindhi (kheh /kʰ/).
  * Sindhi uses Arabic yeh ي (U+064A); Urdu uses Farsi yeh ی (U+06CC).
  * Urdu gol-heh ہ (U+06C1) vs do-chashmi heh ھ (U+06BE) vs Arabic heh ه (U+0647) are frequently confused.
  * ZWNJ (U+200C) is meaningful in Urdu typing and must be kept.
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter

ARABIC_RANGES = [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]

# Heuristic marker letters (used for purity filtering / auditing, not for language ID claims).
SINDHI_MARKERS = set("ٻڀٽٿڄڃڇڊڌڍڏڙڪڳڱڻڦ")
URDU_MARKERS = set("ٹڈڑے")

TATWEEL = "\u0640"
BIDI_AND_BOM = dict.fromkeys(map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff"))

# Opt-in letter canonicalisation (per language!).
CANON = {
    "ur": {"\u064a": "\u06cc", "\u0649": "\u06cc", "\u0643": "\u06a9"},   # ي ى -> ی ; ك -> ک
    "sd": {"\u0649": "\u064a", "\u06cc": "\u064a", "\u0643": "\u06aa"},   # ى ی -> ي ; ك -> ڪ  (NOT ک!)
}
URDU_HEH = {"\u0647": "\u06c1"}  # ه -> ہ (opt-in, separately)

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")
HTML_TAG_RE = re.compile(r"<[^<>]{1,200}>")
WS_RE = re.compile(r"[ \t\u00a0\u2000-\u200b\u3000]+")
SENT_SPLIT_RE = re.compile(r"(?<=[۔؟!?])\s+|(?<=\.)\s+(?=\S)|\n+")


def is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(a <= cp <= b for a, b in ARABIC_RANGES)


def arabic_letter_ratio(text: str) -> float:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(is_arabic_char(c) for c in letters) / len(letters)


def normalize(text: str, lang: str, *, canonicalize_letters: bool = False, canonicalize_heh: bool = False,
              strip_html: bool = True, strip_urls: bool = True, counter: Counter | None = None) -> str:
    c = counter if counter is not None else Counter()
    if strip_html:
        new = HTML_TAG_RE.sub(" ", text)
        c["html_tags"] += text != new
        text = html.unescape(new)
    if strip_urls:
        text, n = URL_RE.subn(" ", text)
        c["urls"] += n
    nfc = unicodedata.normalize("NFC", text)
    c["nfc_changed"] += nfc != text
    text = nfc
    c["tatweel"] += text.count(TATWEEL)
    text = text.replace(TATWEEL, "")
    before = len(text)
    text = text.translate(BIDI_AND_BOM)
    c["bidi_bom"] += before - len(text)
    if canonicalize_letters and lang in CANON:
        for src, dst in CANON[lang].items():
            n = text.count(src)
            if n:
                c[f"canon_{ord(src):04X}->{ord(dst):04X}"] += n
                text = text.replace(src, dst)
    if canonicalize_heh and lang == "ur":
        for src, dst in URDU_HEH.items():
            c["canon_heh"] += text.count(src)
            text = text.replace(src, dst)
    lines = [WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def split_sentences(text: str, min_chars: int = 1) -> list[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s and len(s.strip()) >= min_chars]


def marker_counts(text: str) -> tuple[int, int]:
    return sum(ch in SINDHI_MARKERS for ch in text), sum(ch in URDU_MARKERS for ch in text)


def looks_like_other_language(text: str, lang: str) -> bool:
    sd, ur = marker_counts(text)
    if lang == "sd":
        return ur > 2 and sd == 0
    if lang == "ur":
        return sd > 2 and ur == 0
    return False


def char_inventory(texts, top: int = 120) -> list[dict]:
    cnt = Counter()
    for t in texts:
        cnt.update(t)
    total = sum(cnt.values()) or 1
    rows = []
    for ch, n in cnt.most_common(top):
        rows.append({"char": ch, "codepoint": f"U+{ord(ch):04X}",
                     "name": unicodedata.name(ch, "<unnamed>"), "category": unicodedata.category(ch),
                     "count": n, "share": round(n / total, 6)})
    return rows
