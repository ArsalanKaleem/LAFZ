"""Offline test doubles: a tiny tokenizer with the *same pipeline* as Qwen2.5
(NFC normalizer, Qwen split regex + ByteLevel pre-tokenizer, ByteLevel decoder, special tokens after vocab).
No downloads needed, so the whole tokenizer half of HARF is testable in CI."""
import random
from pathlib import Path

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
QWEN_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")
ENGLISH = ("the quick brown fox jumps over the lazy dog while the cat sleeps in the warm sun and people "
           "read books write code compute numbers and travel between cities every day of the year").split()


def _sentences(path):
    text = (ROOT / "examples" / "data" / path).read_text(encoding="utf-8")
    return [s for s in text.replace("۔", "۔\n").replace(". ", ".\n").replace("؟", "؟\n").split("\n") if s.strip()]


@pytest.fixture(scope="session")
def ur_sents():
    return _sentences("ur_sample.txt")


@pytest.fixture(scope="session")
def sd_sents():
    return _sentences("sd_sample.txt")


@pytest.fixture(scope="session")
def toy_base_tok(tmp_path_factory, ur_sents):
    from transformers import PreTrainedTokenizerFast
    rng = random.Random(0)
    corpus = [" ".join(rng.choice(ENGLISH) for _ in range(12)) for _ in range(3000)]
    corpus += ur_sents[:3]  # base knows a little Arabic script, like a real multilingual LM
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, min_frequency=2, special_tokens=[],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    hf.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    d = tmp_path_factory.mktemp("toy_base")
    hf.save_pretrained(d)
    from harf.tokenizer.common import load_tokenizer
    return load_tokenizer(str(d))


@pytest.fixture(scope="session")
def train_texts(ur_sents, sd_sents):
    rng = random.Random(1)
    out = []
    for _ in range(40):
        s = ur_sents + sd_sents
        rng.shuffle(s)
        out.extend(s)
    return out
