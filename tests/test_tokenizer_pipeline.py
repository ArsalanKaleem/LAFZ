import pandas as pd

from harf.text.normalize import looks_like_other_language, normalize, split_sentences
from harf.tokenizer.audit import audit_sentences, summarize
from harf.tokenizer.common import bl_to_text, load_tokenizer, text_to_bl, touches_arabic
from harf.tokenizer.extend import CONTROL_TEXTS, extend_tokenizer
from harf.tokenizer.select import continued_bpe, target_bpe_chains
from harf.tokenizer.train_bpe import train_target_bpe


def test_normalize_is_language_specific():
    # Arabic kaf ك must become Urdu ک in Urdu but Sindhi ڪ in Sindhi; Sindhi ک (kheh) must survive.
    assert normalize("كتاب", "ur", canonicalize_letters=True) == "کتاب"
    assert normalize("كتاب", "sd", canonicalize_letters=True) == "ڪتاب"
    assert normalize("کان", "sd", canonicalize_letters=True) == "کان"
    assert normalize("ab\u200fc\u0640d", "ur") == "abcd"          # bidi mark + tatweel removed
    assert "\u200c" in normalize("می\u200cشود", "ur")             # ZWNJ kept
    assert normalize("کتاب", "ur") == "کتاب"                       # default: no canonicalisation


def test_purity_and_split():
    assert looks_like_other_language("ٹ ڈ ڑ ے ہے", "sd")
    assert not looks_like_other_language("ڪراچي سنڌ جو شهر آهي", "sd")
    assert split_sentences("پہلا جملہ۔ دوسرا جملہ؟ تیسرا") == ["پہلا جملہ۔", "دوسرا جملہ؟", "تیسرا"]


def test_bytelevel_roundtrip():
    s = " ڪراچي"
    assert bl_to_text(text_to_bl(s)) == s and touches_arabic(text_to_bl(s))
    assert not touches_arabic(text_to_bl(" hello"))


def test_continued_bpe_extension(tmp_path, toy_base_tok, train_texts, ur_sents, sd_sents):
    res = continued_bpe(toy_base_tok, train_texts, budget=200, min_pair_freq=2, min_word_freq=1)
    toks = res["tokens"]
    assert 0 < len(toks) <= 200
    assert all(touches_arabic(t["token"]) for t in toks)
    # prefix consistency: smaller budget == prefix of larger budget
    res50 = continued_bpe(toy_base_tok, train_texts, budget=50, min_pair_freq=2, min_word_freq=1)
    assert [t["token"] for t in res50["tokens"]] == [t["token"] for t in toks[:50]]
    # merge trees expand to base tokens
    vocab = toy_base_tok.get_vocab()
    assert all(all(p in vocab for p in t["base_pieces"]) for t in toks)
    assert all("".join(t["base_pieces"]) == t["token"] for t in toks)

    texts = ur_sents + sd_sents
    rep = extend_tokenizer(toy_base_tok, res, len(toks), tmp_path / "ext", texts)
    assert rep["ok"] and rep["control_identical"] and rep["roundtrip_failures"] == 0
    assert rep["scr"] > 0.2, rep
    ext = load_tokenizer(str(tmp_path / "ext"))
    assert len(ext) == len(toy_base_tok) + len(toks)
    # special tokens untouched
    assert ext.convert_tokens_to_ids("<|endoftext|>") == toy_base_tok.convert_tokens_to_ids("<|endoftext|>")
    for t in CONTROL_TEXTS:
        assert ext.encode(t) == toy_base_tok.encode(t)
    # the simulation is faithful: training-table savings match real encoding on training words
    base_n = sum(len(x) for x in toy_base_tok(train_texts, add_special_tokens=False)["input_ids"])
    ext_n = sum(len(x) for x in ext(train_texts, add_special_tokens=False)["input_ids"])
    assert base_n - ext_n == toks[-1]["cum_saved"]


def test_target_bpe_strategy(tmp_path, toy_base_tok, train_texts, ur_sents):
    tgt = train_target_bpe(toy_base_tok, train_texts, vocab_size=800, min_frequency=2, out_dir=tmp_path / "bpe")
    res = target_bpe_chains(toy_base_tok, tgt, train_texts, budget=150)
    assert 0 < len(res["tokens"]) <= 150
    rep = extend_tokenizer(toy_base_tok, res, len(res["tokens"]), tmp_path / "ext2", ur_sents)
    assert rep["ok"] and rep["scr"] > 0


def test_audit(toy_base_tok, ur_sents):
    rows = [{"id": str(i), "lang": "ur", "text": t} for i, t in enumerate(ur_sents)]
    df = audit_sentences(toy_base_tok, rows)
    s = summarize(df)
    assert isinstance(df, pd.DataFrame) and s["tokens_per_word"] > 1
    assert 0 <= s["fragmented_target_word_rate"] <= 1
