"""Day 4: train an independent target-language byte-level BPE that reuses the BASE tokenizer's
normalizer + pre-tokenizer + decoder, so its pieces live in the same byte-level space.

This is (a) the guide's Day-4 artifact, (b) an upper bound on achievable fragmentation reduction,
and (c) the candidate source for the `target_bpe` selection strategy (ablation vs `continued`)."""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from ..utils import log, write_json
from .common import backend_json, is_byte_level


def train_target_bpe(base_tok, texts, vocab_size: int, min_frequency: int, out_dir: Path) -> Tokenizer:
    tj = backend_json(base_tok)
    if tj["model"]["type"] != "BPE" or not is_byte_level(tj):
        raise SystemExit("HARF currently supports byte-level BPE base tokenizers (Qwen2/2.5, Llama 3.x, GPT-2 style).")
    tj["added_tokens"] = []
    tj["post_processor"] = None
    tj["model"] = {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                   "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "vocab": {}, "merges": []}
    tok = Tokenizer.from_str(json.dumps(tj))
    tok.model = models.BPE()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, show_progress=True,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[])
    tok.train_from_iterator(texts, trainer=trainer)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    model = json.loads(tok.to_str())["model"]
    write_json(out_dir / "train_stats.json", {"vocab_size": tok.get_vocab_size(), "n_merges": len(model["merges"]),
                                              "min_frequency": min_frequency})
    log.info("target BPE: vocab=%d merges=%d", tok.get_vocab_size(), len(model["merges"]))
    return tok


def as_hf(tok: Tokenizer):
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_object=tok)
