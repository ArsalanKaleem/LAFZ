"""End-to-end CLI test of every CPU step on a toy Qwen-style tokenizer (no GPU, no downloads)."""
import json
import random
import shutil
from pathlib import Path

from harf.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_pipeline(tmp_path, monkeypatch, toy_base_tok, ur_sents):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(ROOT / "configs", "configs")
    toy_base_tok.save_pretrained("toy_base")
    rng = random.Random(0)
    vocab = "شہر پانی کتاب زندگی بازار موسم حکومت تعلیم اسکول دریا کسان فصل".split()
    docs = [" ".join(rng.sample(ur_sents, 3)) + " " + " ".join(rng.choice(vocab) for _ in range(6)) + "۔"
            for _ in range(600)]
    Path("data/raw/ur").mkdir(parents=True)
    Path("data/raw/ur/docs.txt").write_text("\n\n".join(docs), encoding="utf-8")
    common = ["--config", "configs/debug.yaml", "--lang", "ur", "--set", "model.name=toy_base",
              "--set", "model.revision=null", "--set", "tokenizer.max_budget=150",
              "--set", "conditions.A.budget=50", "--set", "conditions.F.budget=150",
              "--set", "data.valid_frac=0.15", "--set", "data.test_frac=0.15", "--log-level", "ERROR"]
    for cmd in (["prepare"], ["audit"], ["select"], ["extend", "--condition", "A"],
                ["extend", "--condition", "F"], ["eval-compression"], ["report"]):
        main(cmd + common)
    run = Path("runs/toy_base/ur")
    comp = json.loads((run / "eval/compression_results.json").read_text())
    assert comp["conditions"]["v150"]["scr"] >= comp["conditions"]["v50"]["scr"] > 0
    assert json.loads((run / "A/tokenizer/verification.json").read_text())["ok"]
    assert (run / "reports/final_results.csv").exists()
    assert (Path("data/processed/ur/data_card.md")).exists()
