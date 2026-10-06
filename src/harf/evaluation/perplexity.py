"""Day 11: held-out language-modelling quality, comparable ACROSS tokenizers.

Token-level perplexity is NOT comparable between tokenizers (fewer, longer tokens trivially change it).
We therefore report per-sentence total negative log-likelihood (nats) of the same text and normalise by
units every condition shares:
    bits per character (BPC)  = NLL / (ln 2 · #chars)          <- primary
    word-level perplexity     = exp(NLL / #words)
    token-level perplexity    = exp(NLL / #tokens)              <- reported, flagged non-comparable
Each sentence is scored after one fixed start token (BOS if the model has one, else EOS), which is
tokenized identically in every condition."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import log, require_torch, write_json


def sentence_nll(model, tok, texts: list[str], batch_size: int = 16, max_len: int = 512):
    torch = require_torch()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = [[start] + tok.encode(t, add_special_tokens=False)[: max_len - 1] for t in texts]
    truncated = sum(len(tok.encode(t, add_special_tokens=False)) > max_len - 1 for t in texts)
    if truncated:
        log.warning("%d sentences truncated at %d tokens — NLL is then partial; raise eval.ppl_max_len.", truncated, max_len)
    order = np.argsort([len(e) for e in enc])
    nll = np.zeros(len(texts))
    ntok = np.zeros(len(texts), dtype=int)
    device = model.get_input_embeddings().weight.device
    V = len(tok)
    with torch.inference_mode():
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            L = max(len(enc[j]) for j in idx)
            ids = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, j in enumerate(idx):
                ids[r, : len(enc[j])] = torch.tensor(enc[j])
                att[r, : len(enc[j])] = 1
            ids, att = ids.to(device), att.to(device)
            logits = model(input_ids=ids, attention_mask=att).logits
            for r, j in enumerate(idx):  # per row in fp32 to keep memory at L x V
                n = len(enc[j]) - 1
                # Only real token ids compete: padded embedding rows (>= len(tok)) are excluded for every condition.
                ce = torch.nn.functional.cross_entropy(logits[r, :n, :V].float(), ids[r, 1:n + 1], reduction="sum")
                nll[j] = ce.item()
                ntok[j] = n
            del logits
    return nll, ntok


def evaluate_perplexity(model, tok, rows: list[dict], label: str, out_dir: Path, batch_size: int, max_len: int) -> dict:
    texts = [r["text"] for r in rows]
    nll, ntok = sentence_nll(model, tok, texts, batch_size, max_len)
    df = pd.DataFrame({"id": [r["id"] for r in rows], "lang": [r.get("lang") for r in rows], "nll": nll,
                       "n_tokens": ntok, "n_chars": [len(t) for t in texts], "n_words": [len(t.split()) for t in texts]})
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"ppl_per_sentence_{label}.csv", index=False)
    s = _summ(df)
    s["by_lang"] = {str(l): _summ(g) for l, g in df.groupby("lang")}
    s["label"] = label
    write_json(out_dir / f"ppl_{label}.json", s)
    log.info("ppl[%s]: BPC=%.4f wordPPL=%.2f", label, s["bpc"], s["word_ppl"])
    return s


def _summ(df: pd.DataFrame) -> dict:
    N = df["nll"].sum()
    return {"bpc": float(N / (math.log(2) * df["n_chars"].sum())),
            "word_ppl": float(math.exp(N / df["n_words"].sum())),
            "token_ppl_noncomparable": float(math.exp(N / df["n_tokens"].sum())),
            "total_nll": float(N), "n_sentences": int(len(df))}
