"""Collect EVERY saved number into one Markdown file (docs/RESULTS.md), with the source file of each table.

Nothing here trains or evaluates a model: it only reads artifacts under runs/ and data/processed/, and
recomputes cheap CPU analyses (looping vs length) from saved FLORES predictions. Rerun after each session.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..tokenizer.common import load_tokenizer
from ..utils import git_commit, log, read_json, read_jsonl
from .extra import loops_by_steps, loops_vs_fertility
from .report import final_table

MULTILANG = {"ps", "am", "hi", "ta"}
LANG_NAMES = {"ur": "Urdu", "sd": "Sindhi", "ps": "Pashto", "am": "Amharic", "hi": "Hindi", "ta": "Tamil"}


def _f(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _pct(v):
    return "" if v is None else f"{100 * v:.1f}%"


def _md_df(df: pd.DataFrame, nd=4) -> list[str]:
    if df is None or df.empty:
        return ["_none_", ""]
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(_f(v, nd) for v in r) + " |")
    return out + [""]


def _src(p: Path) -> str:
    return f"<sub>Source: `{p.as_posix()}`</sub>"


def _section_language(cfg: dict, lang: str, lang_dir: Path, base_tok) -> list[str]:
    L = [f"## {LANG_NAMES.get(lang, lang)} (`{lang}`)", ""]
    ev, rep = lang_dir / "eval", lang_dir / "reports"

    card = Path(cfg["paths"]["processed"]) / lang / "data_card.json"
    if card.exists():
        c = read_json(card)
        L += ["### Data", ""] + [f"- {k}: {v:,}" for k, v in c["counts"].items()]
        L += [f"- filtering: {c.get('filtering')}", "", _src(card), ""]

    ba = lang_dir / "_shared" / "baseline_audit" / "baseline_metrics.json"
    if ba.exists():
        m = read_json(ba)["overall"]
        L += ["### Baseline tokenization (validation split)", ""]
        L += [f"- {k}: {_f(v) if isinstance(v, float) else v}" for k, v in m.items()] + ["", _src(ba), ""]

    comp = ev / "compression_results.json"
    if comp.exists():
        c = read_json(comp)
        rows = []
        for k, v in c["conditions"].items():
            pt = v.get("paired_test_vs_baseline", {})
            rows.append({"tokenizer": k, "SCR": v.get("scr"), "tokens/word": v["tokens_per_word"],
                         "chars/token": v["chars_per_token"], "median tokens/sent": v["median_tokens_per_sentence"],
                         "per-sentence SCR median": v.get("per_sentence_scr_median"), "Wilcoxon p": pt.get("wilcoxon_p")})
        L += ["### Compression (test split)", "", f"Definition: {c['definition']}; n = {c['n_sentences']} sentences.", ""]
        L += _md_df(pd.DataFrame(rows)) + [_src(comp), ""]

    try:
        df = final_table(cfg, lang_dir)
        if not df.empty:
            metric = [c for c in df.columns if c in ("bpc", "emit_ms_median", "english_bpc") or c.startswith("chrf")]
            if metric:  # keep the reference row and any condition that was actually evaluated
                df = df[(df["condition"] == "baseline") | df[metric].notna().any(axis=1)]
            else:       # nothing evaluated yet: keep conditions that were at least built
                df = df[(df["condition"] == "baseline") | df["condition"].map(lambda c: (lang_dir / c).exists())]
            keep = [c for c in df.columns if df[c].notna().any()]
            L += ["### All conditions", ""] + _md_df(df[keep]) + [_src(rep / "final_results.csv"), ""]
    except Exception as e:  # never let one language break the document
        L += [f"_final table failed: {e}_", ""]

    for name, title in (("comparisons.md", "Pairwise comparisons"), ("comparisons_english.md", "English retention: paired tests")):
        p = rep / name
        if p.exists():
            body = p.read_text(encoding="utf-8").split("\n")
            body = [ln for ln in body if not ln.startswith("# ")]
            L += [f"### {title}", ""] + body + [_src(p), ""]

    he = sorted(rep.glob("human_eval_*.results.json"))
    if he:
        L += ["### Human evaluation (blinded)", "", "| file | a | b | n | preferred a / b / tie | fluency a → b | adequacy a → b | sign-test p |",
              "|---|---|---|---|---|---|---|---|"]
        for p in he:
            r = read_json(p)
            a, b = r["a"], r["b"]
            fl, ad = r.get("fluency", {}), r.get("adequacy", {})
            pr = r["preferred"]
            L.append(f"| {p.name} | {a} | {b} | {r['n_rated']} | {pr.get(a)} / {pr.get(b)} / {pr.get('tie')} | "
                     f"{_f(fl.get(f'mean_{a}'), 2)} → {_f(fl.get(f'mean_{b}'), 2)} | {_f(ad.get(f'mean_{a}'), 2)} → "
                     f"{_f(ad.get(f'mean_{b}'), 2)} | {_f(r.get('preference_sign_test_p'), 6)} |")
        L += ["", "Raters: record who rated each language in the paper.", ""]

    rs = rep / "flores_equal_chars.md"
    if rs.exists():
        body = [ln for ln in rs.read_text(encoding="utf-8").split("\n") if not ln.startswith("# ")]
        L += ["### FLORES at an equal character budget"] + [ln.replace("## ", "#### ", 1) if ln.startswith("## ") else ln for ln in body] + [_src(rs), ""]

    fs = rep / "failure_summary.csv"
    if fs.exists():
        L += ["### Generation failure summary (FLORES outputs)", ""] + _md_df(pd.read_csv(fs), 3) + [_src(fs), ""]

    tgt = cfg["flores_codes"].get(lang)
    direction = f"eng_Latn-{tgt}"
    found = {f.stem[len("flores_"):-len(direction) - 1]: f for f in ev.glob(f"flores_*_{direction}.jsonl")}
    order = list(cfg["conditions"])
    files = {c: found[c] for c in sorted(found, key=lambda c: order.index(c) if c in order else len(order))}
    if files:
        res = loops_vs_fertility(base_tok, ev, list(files), direction, rep)
        L += [f"### Looping vs per-word fragmentation ({direction})", "",
              "| condition | Spearman ρ | p | loop rate Q1 → Q4 (least → most fragmented) |", "|---|---|---|---|"]
        for c, r in res["conditions"].items():
            L.append(f"| {c} | {r['spearman_rho']:+.3f} | {r['p']:.2g} | " + " → ".join(_pct(x) for x in r["loop_rate_by_quartile"]) + " |")
        L.append("")
        toks = {}
        for c in files:
            cd = lang_dir / c / "tokenizer"
            if (cfg["conditions"].get(c) or {}).get("budget"):
                if cd.exists():
                    toks[c] = load_tokenizer(str(cd))
            else:
                toks[c] = base_tok
        rows = {c: list(read_jsonl(files[c])) for c in toks}
        if "baseline_lora" in rows:
            ls = loops_by_steps(rows, toks)
            b = ls["bins"]
            heads = [f"{b[i]}–{b[i + 1] if b[i + 1] < 10_000 else '∞'}" for i in range(len(b) - 1)]
            L += [f"### Looping at matched length ({direction})", "",
                  "Loop rate (n) by the number of tokens each system's own tokenizer needs to write the reference.", "",
                  "| condition | " + " | ".join(heads) + " | overall |", "|" + "---|" * (len(heads) + 2)]
            for c, t in ls["table"].items():
                cells = [f"{_pct(r)} ({n})" if n >= ls["min_n"] else f"– ({n})" for r, n in zip(t["rate"], t["n"])]
                L.append(f"| {c} | " + " | ".join(cells) + f" | {_pct(t['overall'])} |")
            L += ["", "Decomposition vs `baseline_lora` (reweighting to its length distribution; approximate):", "",
                  "| condition | ref loop rate | at ref lengths | actual | reduction | from per-step looping | from shorter sequences | share per-step | coverage |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for c, d in ls["decomposition"].items():
                if toks[c] is base_tok:
                    continue  # decomposition only makes sense for extended-vocabulary systems
                flag = "" if d["coverage_of_ref"] >= 0.8 else " ⚠ low coverage: unreliable"
                L.append(f"| {c} | {_pct(d['ref_loop_rate'])} | {_pct(d['at_ref_lengths'])} | {_pct(d['loop_rate'])} | "
                         f"{_pct(d['reduction_total'])} | {_pct(d['reduction_per_step'])} | {_pct(d['reduction_from_length'])} | "
                         f"{_pct(d['share_per_step'])} | {_pct(d['coverage_of_ref'])}{flag} |")
            L += ["", "Coverage = share of the reference system's sentences in length bins both systems occupy. "
                  "Below 100%, the two parts need not sum exactly to the total.", ""]

    ln = rep / "lora_update_norms.json"
    if ln.exists():
        n = read_json(ln)
        types = sorted({t for v in n.values() for t in v["by_module_type"]})
        L += ["### LoRA update size ‖ΔW‖", "", "| condition | total | " + " | ".join(types) + " |", "|" + "---|" * (len(types) + 2)]
        for c, v in n.items():
            L.append(f"| {c} | {v['total_norm']:.2f} | " + " | ".join(f"{v['by_module_type'].get(t, 0):.2f}" for t in types) + " |")
        L += ["", _src(ln), ""]

    fd = sorted(ev.glob("fragdecode_*.json"))
    for p in fd:
        r = read_json(p)
        L += [f"### Forced fragmentation of English output ({r['direction']}, n={r['n']})", "",
              "| constraint | chrF++ | mean rep4 | loop rate | median len ratio | output tokens/word |", "|---|---|---|---|---|---|"]
        for k, v in r["runs"].items():
            L.append(f"| {k} | {v['chrf++']:.2f} | {v['mean_rep4']:.3f} | {_pct(v['loop_rate'])} | {v['median_len_ratio']:.2f} | {v['tokens_per_word_output']:.2f} |")
        L += ["", _src(p), ""]

    tr = []
    for c in cfg["conditions"]:
        s = lang_dir / c / "lora" / "train_summary.json"
        if s.exists():
            v = read_json(s)
            tr.append({"condition": c, "steps": v["steps"], "final loss (per token)": v["final_loss"],
                       "trainable params": v["trainable_params"], "wall time (h)": v["wall_s"] / 3600})
        d = lang_dir / c / "init" / "init_diagnostics.json"
        if d.exists() and tr and tr[-1]["condition"] == c:
            tr[-1]["λ (dev BPC)"] = read_json(d).get("best_lambda_by_dev_bpc")
    if tr:
        L += ["### Training runs", "", "Per-token losses are not comparable across tokenizers.", ""] + _md_df(pd.DataFrame(tr), 3)
    return L


def build_results_doc(out_path: Path = Path("docs/RESULTS.md"), runs_root: Path = Path("runs"),
                      model_slug: str = "Qwen2.5-1.5B", qwen_cfg: str = "configs/qwen2.5-1.5b.yaml",
                      multi_cfg: str = "configs/multilang.yaml") -> Path:
    mdir = runs_root / model_slug
    langs = sorted([p.name for p in mdir.iterdir() if p.is_dir() and p.name in LANG_NAMES],
                   key=lambda x: list(LANG_NAMES).index(x)) if mdir.exists() else []
    base_cfg = load_config(qwen_cfg)
    base_tok = load_tokenizer(base_cfg["model"]["name"], base_cfg["model"].get("revision"))
    L = ["# HARF — all results", "",
         f"Generated {_dt.datetime.now():%Y-%m-%d %H:%M} by `harf document` · git `{(git_commit() or 'unknown')[:10]}` · "
         f"model `{base_cfg['model']['name']}` @ `{base_cfg['model'].get('revision')}`", "",
         "Every table lists its source file. Regenerate after each session; do not edit numbers by hand.", "",
         "Contents: " + ", ".join(f"[{LANG_NAMES[l]}](#{LANG_NAMES[l].lower()}-{l})" for l in langs) +
         ", [Pre-tokenizer ceiling](#pre-tokenizer-ceiling)", ""]
    for lang in langs:
        cfg = load_config(multi_cfg if lang in MULTILANG else qwen_cfg, lang=lang)
        try:
            L += _section_language(cfg, lang, mdir / lang, base_tok)
        except Exception as e:
            log.exception("section %s failed", lang)
            L += [f"## {LANG_NAMES[lang]}", "", f"_section failed: {e}_", ""]
    ceil = mdir / "analysis" / "pretokenizer_ceiling.csv"
    L += ["## Pre-tokenizer ceiling", ""]
    if ceil.exists():
        df = pd.read_csv(ceil)
        keep = [c for c in df.columns if df[c].notna().any()]
        L += ["`ceiling_scr` = 1 − pre-tokens/tokens: the most any merge-based extension can compress under the "
              "inherited pre-tokenizer; `ceiling_scr_repaired` = the same after the script-scoped repair.", ""]
        L += _md_df(df[keep], 3) + [_src(ceil), ""]
    else:
        L += ["_Run `harf ceiling` first._", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path
