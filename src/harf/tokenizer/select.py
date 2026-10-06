"""Day 5: select new subwords.

Strategy `continued` (default, recommended)
-------------------------------------------
Continue BPE *on top of the base tokenizer's own segmentation* of target-script words:
  1. pre-tokenize the training corpus with the base pipeline -> word frequency table
  2. segment each word with the base BPE model                 -> base pieces
  3. greedily merge the most frequent adjacent pair, N times   -> N new tokens, N new merges

Properties (these are what make the extension *correct*, not just plausible):
  * Exact budget: every merge creates exactly one new ID (pairs whose result already exists are skipped).
  * Faithful: appended merges have lower priority than all base merges, so the extended tokenizer
    = base segmentation followed by the new merges — exactly what was simulated here.
  * Prefix-consistent: the first 2k merges of the 8k run *are* the 2k run -> clean vocab-size ablation.
  * Non-interfering: every new token contains an Arabic-block lead byte, so text without Arabic-block
    characters (English, code, numbers) is tokenized identically to the base tokenizer.
  * Each new token has a binary merge tree (a, b) down to base tokens -> used by embedding init.
  * Each merge's frequency = tokens saved on the training word table -> exact training-set SCR curve.

Strategy `target_bpe` (guide-style ablation)
--------------------------------------------
Rank strings from an independently trained target BPE by freq x (base_pieces - 1) and add each as a
left-to-right merge chain over its base pieces (intermediate pieces also consume IDs).
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from typing import Iterable

from ..utils import log
from .common import backend_json, bl_to_text, touches_arabic


# ----------------------------------------------------------------------------- word table
def word_counts(base_tok, texts: Iterable[str], min_word_freq: int, max_unique: int) -> Counter:
    backend = base_tok.backend_tokenizer
    norm, pre = backend.normalizer, backend.pre_tokenizer
    cnt: Counter = Counter()
    for t in texts:
        if norm is not None:
            t = norm.normalize_str(t)
        for piece, _ in pre.pre_tokenize_str(t):
            cnt[piece] += 1
    cnt = Counter({w: c for w, c in cnt.items() if c >= min_word_freq and touches_arabic(w)})
    if len(cnt) > max_unique:
        cnt = Counter(dict(cnt.most_common(max_unique)))
    return cnt


def base_segment(base_tok, words: Iterable[str]) -> dict[str, list[str]]:
    model = base_tok.backend_tokenizer.model
    return {w: [t.value for t in model.tokenize(w)] for w in words}


# ----------------------------------------------------------------------------- continued BPE
def continued_bpe(base_tok, texts: Iterable[str], budget: int, min_pair_freq: int = 5,
                  min_word_freq: int = 2, max_unique: int = 2_000_000) -> dict:
    base_vocab = base_tok.backend_tokenizer.get_vocab(with_added_tokens=True)
    wc = word_counts(base_tok, texts, min_word_freq, max_unique)
    seg = base_segment(base_tok, wc)
    words = [list(seg[w]) for w in wc]
    freqs = [wc[w] for w in wc]
    base_total = sum(len(s) * f for s, f in zip(words, freqs))
    log.info("continued BPE: %d unique target words, %d base tokens (weighted)", len(words), base_total)

    pair_count: Counter = Counter()
    pair_words: dict[tuple[str, str], set[int]] = defaultdict(set)
    for i, (sym, f) in enumerate(zip(words, freqs)):
        for a, b in zip(sym, sym[1:]):
            pair_count[(a, b)] += f
            pair_words[(a, b)].add(i)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)

    new_tokens: list[dict] = []
    created: set[str] = set()
    banned: set[tuple[str, str]] = set()
    saved = 0
    while len(new_tokens) < budget and heap:
        negc, pair = heapq.heappop(heap)
        c = pair_count.get(pair, 0)
        if c != -negc or pair in banned:
            continue  # stale entry
        if c < min_pair_freq:
            break
        merged = pair[0] + pair[1]
        if merged in base_vocab or merged in created or not touches_arabic(merged):
            banned.add(pair)
            continue
        created.add(merged)
        saved += c
        new_tokens.append({"token": merged, "pair": list(pair), "freq": int(c), "cum_saved": int(saved)})
        # apply merge in affected words, update counts incrementally
        changed: Counter = Counter()
        for i in pair_words.pop(pair, set()):
            sym, f = words[i], freqs[i]
            if len(sym) < 2:
                continue
            for a, b in zip(sym, sym[1:]):
                changed[(a, b)] -= f
            out, j = [], 0
            while j < len(sym):
                if j < len(sym) - 1 and sym[j] == pair[0] and sym[j + 1] == pair[1]:
                    out.append(merged)
                    j += 2
                else:
                    out.append(sym[j])
                    j += 1
            words[i] = out
            for a, b in zip(out, out[1:]):
                changed[(a, b)] += f
                pair_words[(a, b)].add(i)
        for p, d in changed.items():
            if d == 0:
                continue
            pair_count[p] += d
            if pair_count[p] <= 0:
                pair_count.pop(p, None)
            elif p not in banned:
                heapq.heappush(heap, (-pair_count[p], p))
        pair_count.pop(pair, None)

    _annotate(new_tokens, base_vocab)
    stats = {"strategy": "continued", "unique_words": len(wc), "base_tokens_weighted": base_total,
             "n_new": len(new_tokens), "train_scr_at_n": {}}
    for n in (1000, 2000, 4000, 8000, len(new_tokens)):
        if 0 < n <= len(new_tokens):
            stats["train_scr_at_n"][str(n)] = new_tokens[n - 1]["cum_saved"] / max(base_total, 1)
    return {"tokens": new_tokens, "stats": stats}


# ----------------------------------------------------------------------------- target BPE chains
def target_bpe_chains(base_tok, target_tok, texts: list[str], budget: int) -> dict:
    """target_tok: tokenizers.Tokenizer trained by train_bpe.train_target_bpe."""
    base_vocab = base_tok.backend_tokenizer.get_vocab(with_added_tokens=True)
    model = base_tok.backend_tokenizer.model
    tv = target_tok.get_vocab()
    id2tok = {i: t for t, i in tv.items()}
    freq: Counter = Counter()
    for enc in target_tok.encode_batch(texts):
        freq.update(enc.ids)
    cands = []
    for tid, f in freq.items():
        s = id2tok[tid]
        if s in base_vocab or not touches_arabic(s) or not bl_to_text(s).strip():
            continue
        pieces = [t.value for t in model.tokenize(s)]
        if len(pieces) < 2:
            continue
        cands.append((f * (len(pieces) - 1), s, pieces, f))
    cands.sort(key=lambda x: (-x[0], x[1]))
    created: dict[str, dict] = {}
    new_tokens: list[dict] = []
    for score, s, pieces, f in cands:
        chain, cur, reachable = [], pieces[0], True
        for p in pieces[1:]:
            nxt = cur + p
            if nxt in base_vocab:  # base never builds it this way -> chain would be unreachable
                reachable = False
                break
            if nxt not in created:
                chain.append((cur, p, nxt))
            cur = nxt
        if not reachable or len(new_tokens) + len(chain) > budget:
            continue
        for a, b, nxt in chain:
            created[nxt] = {"token": nxt, "pair": [a, b], "freq": int(f), "cum_saved": None}
            new_tokens.append(created[nxt])
        if len(new_tokens) >= budget:
            break
    _annotate(new_tokens, base_vocab)
    return {"tokens": new_tokens, "stats": {"strategy": "target_bpe", "n_candidates": len(cands),
                                            "n_new": len(new_tokens)}}


# ----------------------------------------------------------------------------- annotation
def _annotate(new_tokens: list[dict], base_vocab: dict[str, int]) -> None:
    pair_of = {t["token"]: t["pair"] for t in new_tokens}

    def expand(tok: str) -> list[str]:
        if tok in pair_of:
            a, b = pair_of[tok]
            return expand(a) + expand(b)
        return [tok]

    for rank, t in enumerate(new_tokens):
        t["rank"] = rank
        t["text"] = bl_to_text(t["token"])
        t["base_pieces"] = expand(t["token"])
        t["base_piece_ids"] = [base_vocab[p] for p in t["base_pieces"]]
