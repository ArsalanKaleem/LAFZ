"""Multi-script guarantees: signatures, scoped pre-tokenizer repair, non-interference across scripts."""
import random
from pathlib import Path

import pytest

from harf.analysis.extra import ceiling_table
from harf.text.scripts import get_script, script_for_flores
from harf.tokenizer.extend import extend_tokenizer
from harf.tokenizer.repair import repair_pretokenizer
from harf.tokenizer.select import continued_bpe

ROOT = Path(__file__).resolve().parents[1]


def _sents(name):
    text = (ROOT / "examples" / "data" / name).read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        for end in ("।", "።", "."):
            line = line.replace(end + " ", end + "\n")
        out += [s.strip() for s in line.split("\n") if s.strip()]
    return out


def _corpus(sents, reps=40, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(reps):
        s = sents[:]
        rng.shuffle(s)
        out += s
    return out


SAMPLES = {"tamil": "தமிழ் மொழி மிகவும் பழமையானது", "devanagari": "भारत एक विशाल देश है",
           "arabic": "پاکستان ایک خوبصورت ملک ہے", "ethiopic": "ኢትዮጵያ ትልቅ አገር ናት"}


def test_signatures_are_script_exclusive():
    for a, text in SAMPLES.items():
        for b in SAMPLES:
            assert get_script(b).touches(text.encode()) == (a == b), (a, b)
    assert script_for_flores("tam_Taml").name == "tamil" and script_for_flores("eng_Latn") is None


def test_repair_is_scoped_to_one_script(toy_base_tok):
    rep = repair_pretokenizer(toy_base_tok, get_script("tamil"))
    p0, p1 = toy_base_tok.backend_tokenizer.pre_tokenizer, rep.backend_tokenizer.pre_tokenizer
    ta = SAMPLES["tamil"]
    assert len(p1.pre_tokenize_str(ta)) == len(ta.split()) < len(p0.pre_tokenize_str(ta))
    for other in [SAMPLES["devanagari"], SAMPLES["arabic"], "Hello, world! 123", "naïve café"]:
        assert p0.pre_tokenize_str(other) == p1.pre_tokenize_str(other)
        assert rep.encode(other, add_special_tokens=False) == toy_base_tok.encode(other, add_special_tokens=False)
    assert rep.decode(rep.encode(ta, add_special_tokens=False)) == ta


@pytest.mark.parametrize("script,fname", [("tamil", "ta_sample.txt"), ("devanagari", "hi_sample.txt")])
def test_repair_raises_achievable_compression(tmp_path, toy_base_tok, script, fname):
    sc = get_script(script)
    sents = _sents(fname)
    corpus = _corpus(sents)
    plain = continued_bpe(toy_base_tok, corpus, budget=300, min_pair_freq=2, min_word_freq=1, script=sc)
    rep_tok = repair_pretokenizer(toy_base_tok, sc)
    repd = continued_bpe(rep_tok, corpus, budget=300, min_pair_freq=2, min_word_freq=1, script=sc)
    n = min(len(plain["tokens"]), len(repd["tokens"]))
    r_plain = extend_tokenizer(toy_base_tok, plain, n, tmp_path / "plain", sents, script_name=script)
    r_rep = extend_tokenizer(rep_tok, repd, n, tmp_path / "rep", sents, reference_tok=toy_base_tok, script_name=script)
    # both verified (incl. other scripts unchanged); same budget; repair compresses more
    assert r_plain["ok"] and r_rep["ok"] and r_rep["n_control_texts"] >= 9
    assert r_rep["scr"] > r_plain["scr"], (r_rep["scr"], r_plain["scr"])


def test_amharic_and_pashto_extend(tmp_path, toy_base_tok):
    for script, fname in (("ethiopic", "am_sample.txt"), ("arabic", "ps_sample.txt")):
        sents = _sents(fname)
        res = continued_bpe(toy_base_tok, _corpus(sents), budget=150, min_pair_freq=2, min_word_freq=1,
                            script=get_script(script))
        rep = extend_tokenizer(toy_base_tok, res, len(res["tokens"]), tmp_path / script, sents, script_name=script)
        assert rep["ok"] and rep["scr"] > 0.2


def test_ceiling_table_reports_repaired_ceiling(tmp_path, toy_base_tok):
    d = tmp_path / "flores" / "devtest"
    d.mkdir(parents=True)
    (d / "tam_Taml.devtest").write_text("\n".join(_sents("ta_sample.txt")), encoding="utf-8")
    (d / "urd_Arab.devtest").write_text(SAMPLES["arabic"], encoding="utf-8")
    df = ceiling_table(toy_base_tok, str(tmp_path / "flores"), ["tam_Taml", "urd_Arab"], "devtest", tmp_path / "o")
    ta = df[df.lang == "tam_Taml"].iloc[0]
    assert ta["ceiling_scr_repaired"] > ta["ceiling_scr"]
