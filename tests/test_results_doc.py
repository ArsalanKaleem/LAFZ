"""`harf document` on a toy end-to-end run: every section that has artifacts must appear with numbers."""
import json
import random
import shutil
from pathlib import Path

from harf.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_document(tmp_path, monkeypatch, toy_base_tok, ur_sents):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(ROOT / "configs", "configs")
    toy_base_tok.save_pretrained("toy_base")
    Path("configs/t_q.yaml").write_text("extends: qwen2.5-1.5b.yaml\nmodel: {name: toy_base, revision: null}\n"
                                        "multilang_config: configs/t_m.yaml\n")
    Path("configs/t_m.yaml").write_text("extends: multilang.yaml\nmodel: {name: toy_base, revision: null}\n")
    rng = random.Random(0)
    words = "شہر پانی کتاب زندگی بازار موسم حکومت تعلیم اسکول دریا".split()
    docs = [" ".join(rng.sample(ur_sents, 3)) + " " + " ".join(rng.choice(words) for _ in range(6)) + "۔" for _ in range(600)]
    Path("data/raw/ur").mkdir(parents=True)
    Path("data/raw/ur/docs.txt").write_text("\n\n".join(docs), encoding="utf-8")
    common = ["--config", "configs/t_q.yaml", "--lang", "ur", "--set", "tokenizer.max_budget=150",
              "--set", "conditions.D.budget=100", "--set", "data.valid_frac=0.15", "--set", "data.test_frac=0.15",
              "--log-level", "ERROR"]
    for cmd in (["prepare"], ["audit"], ["select"], ["extend", "--condition", "D"], ["eval-compression"]):
        main(cmd + common)
    ev = Path("runs/toy_base/ur/eval")
    refs = [rng.choice(ur_sents) for _ in range(120)]
    for cond, loop_p in (("baseline", 0.9), ("baseline_lora", 0.7), ("D", 0.3)):
        with open(ev / f"flores_{cond}_eng_Latn-urd_Arab.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(refs):
                hyp = ("الف ب ج د " * 5) if rng.random() < loop_p else r
                f.write(json.dumps({"i": i, "src": "s", "ref": r, "hyp": hyp}, ensure_ascii=False) + "\n")
    rep = Path("runs/toy_base/ur/reports"); rep.mkdir(parents=True, exist_ok=True)
    (rep / "human_eval_baseline_lora_vs_D_eng_Latn-urd_Arab.results.json").write_text(json.dumps(
        {"a": "baseline_lora", "b": "D", "n_rated": 100, "preferred": {"baseline_lora": 12, "D": 51, "tie": 37},
         "fluency": {"mean_baseline_lora": 1.59, "mean_D": 2.22}, "adequacy": {"mean_baseline_lora": 1.28, "mean_D": 2.06},
         "preference_sign_test_p": 7.4e-07}))
    main(["document", "--config", "configs/t_q.yaml", "--log-level", "ERROR"])
    doc = Path("docs/RESULTS.md").read_text(encoding="utf-8")
    for needle in ("## Urdu (`ur`)", "### Compression (test split)", "### All conditions", "### Human evaluation",
                   "51 / 12".replace("51 / 12", "12 / 51 / 37"), "### Looping at matched length", "Decomposition vs",
                   "Source: `runs/toy_base/ur/eval/compression_results.json`", "## Pre-tokenizer ceiling"):
        assert needle in doc, needle
