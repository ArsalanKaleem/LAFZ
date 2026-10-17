"""Blinded human evaluation: pairwise A/B sheet from FLORES predictions, and scoring.

make_sheet  -> reports/human_eval_<a>_vs_<b>.csv   (rater fills fluency_1/2, adequacy_1/2 (1-5), preferred (1/2/tie))
               reports/human_eval_<a>_vs_<b>.key.json (which output is which; do not open while rating)
score_sheet -> reports/human_eval_<a>_vs_<b>.results.json
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
from scipy import stats

from ..utils import read_json, read_jsonl, write_json


def make_sheet(lang_dir: Path, a: str, b: str, direction: str, n: int = 100, seed: int = 0) -> Path:
    ev = lang_dir / "eval"
    ra = list(read_jsonl(ev / f"flores_{a}_{direction}.jsonl"))
    rb = list(read_jsonl(ev / f"flores_{b}_{direction}.jsonl"))
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(ra)), min(n, len(ra))))
    rows, key = [], {}
    for k, i in enumerate(idx):
        flip = rng.random() < 0.5
        o1, o2 = (rb[i]["hyp"], ra[i]["hyp"]) if flip else (ra[i]["hyp"], rb[i]["hyp"])
        rows.append({"item": k, "source": ra[i]["src"], "reference": ra[i]["ref"], "output_1": o1, "output_2": o2,
                     "fluency_1": "", "adequacy_1": "", "fluency_2": "", "adequacy_2": "", "preferred": "", "notes": ""})
        key[str(k)] = {"flores_i": i, "output_1": b if flip else a, "output_2": a if flip else b}
    out = lang_dir / "reports" / f"human_eval_{a}_vs_{b}_{direction}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel shows Urdu/Sindhi correctly
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(out.with_suffix(".key.json"), {"a": a, "b": b, "direction": direction, "seed": seed, "items": key})
    return out


def score_sheet(sheet: Path) -> dict:
    key = read_json(sheet.with_suffix(".key.json"))
    a, b = key["a"], key["b"]
    sc = {a: {"fluency": [], "adequacy": []}, b: {"fluency": [], "adequacy": []}}
    pref = {a: 0, b: 0, "tie": 0}
    pairs = {"fluency": [], "adequacy": []}
    with open(sheet, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            k = key["items"][str(r["item"])]
            try:
                vals = {m: {k["output_1"]: float(r[f"{m}_1"]), k["output_2"]: float(r[f"{m}_2"])}
                        for m in ("fluency", "adequacy")}
            except ValueError:
                continue  # unrated row
            for m in ("fluency", "adequacy"):
                sc[a][m].append(vals[m][a]); sc[b][m].append(vals[m][b])
                pairs[m].append((vals[m][a], vals[m][b]))
            p = str(r.get("preferred", "")).strip().lower()
            if p in ("1", "2"):
                pref[k[f"output_{p}"]] += 1
            elif p:
                pref["tie"] += 1
    res = {"a": a, "b": b, "n_rated": len(pairs["fluency"]), "preferred": pref}
    for m in ("fluency", "adequacy"):
        if pairs[m]:
            x = np.array(pairs[m])
            res[m] = {f"mean_{a}": float(x[:, 0].mean()), f"mean_{b}": float(x[:, 1].mean()),
                      "wilcoxon_p": float(stats.wilcoxon(x[:, 1], x[:, 0]).pvalue) if (x[:, 1] != x[:, 0]).any() else 1.0}
    decided = pref[a] + pref[b]
    if decided:
        res["preference_sign_test_p"] = float(stats.binomtest(pref[b], decided).pvalue)
    write_json(sheet.with_suffix(".results.json"), res)
    return res
