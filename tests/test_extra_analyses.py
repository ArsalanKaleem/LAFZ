import json
import random

from harf.analysis.extra import ceiling_table, loops_vs_fertility, pretoken_ceiling, rep4


def test_pretoken_ceiling_and_table(tmp_path, toy_base_tok, ur_sents):
    r = pretoken_ceiling(toy_base_tok, ur_sents)
    assert 0 < r["ceiling_scr"] < 1 and r["pretokens_per_word"] <= r["tokens_per_word"]
    d = tmp_path / "flores" / "devtest"; d.mkdir(parents=True)
    (d / "urd_Arab.devtest").write_text("\n".join(ur_sents), encoding="utf-8")
    (d / "eng_Latn.devtest").write_text("the cat sat on the mat\nhello world", encoding="utf-8")
    df = ceiling_table(toy_base_tok, str(tmp_path / "flores"), ["eng_Latn", "urd_Arab", "tam_Taml"], "devtest",
                       tmp_path / "out", achieved={"urd_Arab": {"v2000": 0.3}})
    assert set(df["lang"]) == {"eng_Latn", "urd_Arab"}            # missing language skipped
    u = df[df.lang == "urd_Arab"].iloc[0]
    assert abs(u["share_of_ceiling_v2000"] - 0.3 / u["ceiling_scr"]) < 1e-9


def test_loops_vs_fertility(tmp_path, toy_base_tok, ur_sents):
    assert rep4("a b c d a b c d a b c d") > 0.5 and rep4("a b c d e") == 0
    ev = tmp_path / "eval"; ev.mkdir()
    rng = random.Random(0)
    refs = [rng.choice(ur_sents) for _ in range(80)]
    for cond, loopy in (("baseline", True), ("D", False)):
        with open(ev / f"flores_{cond}_eng_Latn-urd_Arab.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(refs):
                hyp = (r.split()[0] + " x y z ") * 6 if loopy and len(r) > 40 else r
                f.write(json.dumps({"i": i, "src": "s", "ref": r, "hyp": hyp}, ensure_ascii=False) + "\n")
    res = loops_vs_fertility(toy_base_tok, ev, ["baseline", "D"], "eng_Latn-urd_Arab", tmp_path / "rep")
    assert res["conditions"]["baseline"]["loop_rate"] > res["conditions"]["D"]["loop_rate"] == 0
    assert (tmp_path / "rep" / "loops_vs_fertility_eng_Latn-urd_Arab.md").exists()
