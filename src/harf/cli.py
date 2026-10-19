"""HARF command line.

  harf env                                   Day 1  record environment
  harf smoke                                 Day 1  base model loads + generates Urdu/Sindhi
  harf prepare                               Day 2  corpus -> splits + data card
  harf audit                                 Day 3  frozen baseline tokenization audit
  harf train-bpe                             Day 4  independent target BPE (+ held-out fragmentation)
  harf select                                Day 5  merge list (continued BPE) -> candidate_tokens_{2k,4k,8k}.json
  harf extend    --condition C               Day 6  extended tokenizer (+ verification)
  harf inject    --condition C               Day 6-8 resize + initialise embeddings (+ projection diagnostics)
  harf train-lora --condition D              Day 9  LoRA + new rows
  harf eval-compression                      Day 10 all budgets at once
  harf eval-ppl     --condition C            Day 11
  harf eval-flores  --condition C            Day 12
  harf eval-latency --condition C            Day 13
  harf report                                Day 14 final table + significance + failure cases
  harf run --condition D [--steps ...]       any condition end-to-end (skips finished steps)

Common flags: --config, --lang, --set key=value (repeatable), --force
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import cond_dir, lang_dir, langs_of, load_config, save_resolved, shared_dir
from .utils import env_info, log, read_json, set_seed, setup_logging, write_json

UR_PROMPT = "پاکستان کا سب سے بڑا شہر"
SD_PROMPT = "سنڌ جي گاديءَ جو هنڌ"


# ----------------------------------------------------------------------------- helpers
def _base_tok(cfg):
    from .tokenizer.common import load_tokenizer
    return load_tokenizer(cfg["model"]["name"], cfg["model"].get("revision"))


def _train_texts(cfg):
    from .data.prepare import load_split_multi
    return [r["text"] for r in load_split_multi(cfg, "train")]


def _eval_rows(cfg, split=None):
    from .data.prepare import load_split_multi
    return load_split_multi(cfg, split or cfg["eval"]["split"])


def _script(cfg):
    from .text.scripts import script_for_lang
    return script_for_lang(cfg, cfg["lang"].split("+")[0])


def _work_tok(cfg):
    """Tokenizer the extension is BUILT on: the base tokenizer, or its script-repaired variant."""
    base = _base_tok(cfg)
    if cfg["tokenizer"].get("repair_pretokenizer"):
        from .tokenizer.repair import repair_pretokenizer
        return repair_pretokenizer(base, _script(cfg))
    return base


def _manifest_path(cfg):
    suffix = "_repaired" if cfg["tokenizer"].get("repair_pretokenizer") else ""
    return shared_dir(cfg) / f"merges_{cfg['tokenizer']['strategy']}{suffix}.json"


# ----------------------------------------------------------------------------- commands
def cmd_env(cfg, args):
    info = env_info()
    info["config"] = cfg
    write_json(Path(cfg["paths"]["runs"]) / "environment.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "config"}, indent=2, ensure_ascii=False))


def cmd_smoke(cfg, args):
    from .model.inject import build_condition_model
    model, tok = build_condition_model(cfg, None)
    res = {}
    for name, prompt in (("ur", UR_PROMPT), ("sd", SD_PROMPT)):
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=40, do_sample=False)
        res[name] = {"prompt": prompt, "prompt_tokens": ids["input_ids"].shape[1],
                     "output": tok.decode(out[0], skip_special_tokens=True)}
        print(f"[{name}] {res[name]['output']}")
    write_json(Path(cfg["paths"]["runs"]) / "smoke_test.json", res)


def cmd_prepare(cfg, args):
    from .data.prepare import prepare_language
    for lang in langs_of(cfg):
        prepare_language(cfg, lang)


def cmd_audit(cfg, args):
    from .tokenizer.audit import run_audit
    run_audit(_base_tok(cfg), _eval_rows(cfg, "valid"), shared_dir(cfg) / "baseline_audit", "baseline",
              cfg["seed"], freeze=True, force=args.force, script=_script(cfg))


def cmd_train_bpe(cfg, args):
    from .tokenizer.audit import run_audit
    from .tokenizer.train_bpe import as_hf, train_target_bpe
    out = shared_dir(cfg) / "target_bpe"
    tok = train_target_bpe(_base_tok(cfg), _train_texts(cfg), cfg["tokenizer"]["target_bpe_vocab"],
                           cfg["tokenizer"]["min_pair_freq"], out)
    run_audit(as_hf(tok), _eval_rows(cfg, "valid"), out, "target_bpe_standalone", cfg["seed"], inspect_n=30,
              script=_script(cfg))


def cmd_select(cfg, args):
    from .tokenizer.select import continued_bpe, target_bpe_chains
    tc = cfg["tokenizer"]
    base = _work_tok(cfg)
    texts = _train_texts(cfg)
    if tc["strategy"] == "continued":
        res = continued_bpe(base, texts, tc["max_budget"], tc["min_pair_freq"], tc["min_word_freq"],
                            tc["max_unique_words"], script=_script(cfg))
    else:
        from tokenizers import Tokenizer
        p = shared_dir(cfg) / "target_bpe" / "tokenizer.json"
        if not p.exists():
            raise SystemExit("Run `harf train-bpe` first for strategy=target_bpe.")
        res = target_bpe_chains(base, Tokenizer.from_file(str(p)), texts, tc["max_budget"], script=_script(cfg))
    res["base_model"] = cfg["model"]["name"]
    res["repaired_pretokenizer"] = bool(tc.get("repair_pretokenizer"))
    res["base_len"] = len(base)
    write_json(_manifest_path(cfg), res)
    for b in sorted({c["budget"] for c in cfg["conditions"].values() if c.get("budget")}):
        if b > len(res["tokens"]):
            log.warning("budget %d > %d learned merges; candidate file for it not written.", b, len(res["tokens"]))
            continue
        sub = [{k: t[k] for k in ("rank", "token", "text", "pair", "freq", "base_pieces")} | {"language": cfg["lang"]}
               for t in res["tokens"][:b]]
        tag = f"{b // 1000}k" if b % 1000 == 0 else str(b)
        write_json(shared_dir(cfg) / f"candidate_tokens_{tag}.json", sub)
    log.info("selected %d tokens; stats=%s", len(res["tokens"]), res["stats"])


def cmd_extend(cfg, args):
    from .tokenizer.extend import extend_tokenizer
    budget = cfg["cond"]["budget"]
    if not budget:
        log.info("condition %s has no new vocabulary; nothing to extend.", cfg["condition"])
        return
    if not _manifest_path(cfg).exists():
        cmd_select(cfg, args)
    manifest = read_json(_manifest_path(cfg))
    if len(manifest["tokens"]) < budget:
        raise SystemExit(f"Only {len(manifest['tokens'])} merges reach min_pair_freq={cfg['tokenizer']['min_pair_freq']}, "
                         f"but condition {cfg['condition']} asks for {budget}. Add data or lower tokenizer.min_pair_freq "
                         "(then re-run `harf select`). Not silently truncating: budgets must be what the table says.")
    texts = [r["text"] for r in _eval_rows(cfg, "valid")]
    extend_tokenizer(_work_tok(cfg), manifest, budget, cond_dir(cfg) / "tokenizer", texts,
                     reference_tok=_base_tok(cfg), script_name=_script(cfg).name)


def cmd_inject(cfg, args):
    from .model.embedding_init import initialize
    from .model.inject import inject_vocab, load_base_model, roundtrip_check, save_rows
    from .tokenizer.common import load_tokenizer
    if not cfg["cond"]["budget"]:
        return
    cdir = cond_dir(cfg)
    if not (cdir / "tokenizer").exists():
        cmd_extend(cfg, args)
    ext = load_tokenizer(str(cdir / "tokenizer"))
    entries = read_json(cdir / "tokenizer" / "new_tokens.json")
    model = load_base_model(cfg, dtype="float32")
    inject_vocab(model, len(ext))
    init_dir = cdir / "init"
    init_dir.mkdir(exist_ok=True)
    n_dev = cfg["projection"].get("dev_sentences", 200)
    dev_texts = [r["text"] for r in _eval_rows(cfg, "valid")][:n_dev]  # VALID split only; test stays untouched
    ids, rin, rout, diag = initialize(model, _base_tok(cfg), ext, entries, cfg["cond"]["init"], cfg, init_dir,
                                      dev_texts=dev_texts)
    save_rows(cdir / "new_rows.pt", ids, rin, rout, {"strategy": cfg["cond"]["init"], **{
        k: v for k, v in diag.items() if not isinstance(v, dict)}})
    assert roundtrip_check(cfg, cdir, model)
    # generation smoke test after injection (output quality is NOT expected to be good before LoRA)
    ids_in = ext(UR_PROMPT if cfg["lang"].startswith("ur") else SD_PROMPT, return_tensors="pt").to(model.device)
    gen = model.generate(**ids_in, max_new_tokens=20, do_sample=False,
                         suppress_tokens=list(range(len(ext), model.get_input_embeddings().weight.shape[0])) or None)
    write_json(init_dir / "smoke_generation.json", {"output": ext.decode(gen[0], skip_special_tokens=True)})


def cmd_train_lora(cfg, args):
    from .training.lora import train_lora
    if not cfg["cond"]["lora"]:
        return
    cdir = cond_dir(cfg)
    if cfg["cond"]["budget"] and not (cdir / "new_rows.pt").exists():
        cmd_inject(cfg, args)
    train_lora(cfg, cdir)


def cmd_eval_compression(cfg, args):
    from .evaluation.compression import evaluate_compression, tokenizers_for_budgets
    toks = tokenizers_for_budgets(lang_dir(cfg), cfg["conditions"])
    for name, c in cfg["conditions"].items():  # repaired variants get their own column
        rep = (c.get("overrides") or {}).get("tokenizer", {}).get("repair_pretokenizer")
        if rep and (lang_dir(cfg) / name / "tokenizer").exists():
            from .tokenizer.common import load_tokenizer
            toks[f"v{c['budget']}_repaired"] = load_tokenizer(str(lang_dir(cfg) / name / "tokenizer"))
    if not toks:
        raise SystemExit("No extended tokenizers yet. Run `harf extend --condition A` (and E, F) first.")
    evaluate_compression(_base_tok(cfg), toks, _eval_rows(cfg), lang_dir(cfg) / "eval")


def _cond_model(cfg, instruct=False):
    from .model.inject import build_condition_model
    name, c = cfg["condition"], cfg["cond"]
    cdir = None if name == "baseline" else cond_dir(cfg)
    if c.get("budget") and not (cdir / "new_rows.pt").exists():
        raise SystemExit(f"{name}: no new_rows.pt — run `harf inject --condition {name}` first "
                         "(refusing to silently evaluate the baseline under this name).")
    if c.get("lora") and not (cdir / "lora" / "adapter").exists():
        raise SystemExit(f"{name}: no LoRA adapter — run `harf train-lora --condition {name}` first.")
    return build_condition_model(cfg, cdir, instruct=instruct)


def cmd_eval_ppl(cfg, args):
    from .evaluation.perplexity import evaluate_perplexity
    model, tok = _cond_model(cfg)
    evaluate_perplexity(model, tok, _eval_rows(cfg), cfg["condition"], lang_dir(cfg) / "eval",
                        cfg["eval"]["ppl_batch_size"], cfg["eval"]["ppl_max_len"])


def cmd_eval_flores(cfg, args):
    from .evaluation.flores import evaluate_flores
    use_instruct = cfg["flores"]["use_instruct"]
    model, tok = _cond_model(cfg, instruct=use_instruct)
    evaluate_flores(model, tok, cfg, cfg["condition"], lang_dir(cfg) / "eval", use_chat=use_instruct)


def cmd_eval_latency(cfg, args):
    from .evaluation.latency import measure_latency
    lc = cfg["latency"]
    model, tok = _cond_model(cfg)
    texts = [r["text"] for r in _eval_rows(cfg)][: lc["n_sentences"]]
    measure_latency(model, tok, texts, cfg["condition"], lang_dir(cfg) / "eval", lc["warmup"], lc["repeats"])


def cmd_eval_english(cfg, args):
    """English retention: BPC on FLORES English sentences (tokenization is identical across conditions)."""
    from .evaluation.flores import load_flores
    from .evaluation.perplexity import evaluate_perplexity
    sents = load_flores(cfg["paths"]["flores_dir"], "eng_Latn", "devtest")
    rows = [{"id": f"eng-{i}", "lang": "en", "text": t} for i, t in enumerate(sents)]
    model, tok = _cond_model(cfg)
    evaluate_perplexity(model, tok, rows, f"english_{cfg['condition']}", lang_dir(cfg) / "eval",
                        cfg["eval"]["ppl_batch_size"], cfg["eval"]["ppl_max_len"])


def cmd_human_eval(cfg, args):
    """--steps make:A,B[,direction][,n]  or  --steps score:<sheet.csv>"""
    from .analysis.human_eval import make_sheet, score_sheet
    spec = args.steps or "make:baseline_lora,D"
    if spec.startswith("score:"):
        print(json.dumps(score_sheet(Path(spec[6:])), indent=2))
        return
    parts = spec[5:].split(",")
    a, b = parts[0], parts[1]
    tgt = cfg["flores_codes"][cfg["lang"].split("+")[0]]
    direction = parts[2] if len(parts) > 2 and parts[2] else f"eng_Latn-{tgt}"
    n = int(parts[3]) if len(parts) > 3 and parts[3] else 100
    print("sheet:", make_sheet(lang_dir(cfg), a, b, direction, n, cfg["seed"]))


def cmd_ceiling(cfg, args):
    """Pre-tokenizer ceiling across FLORES languages (CPU). Adds achieved SCR for any extended tokenizers found."""
    from .analysis.extra import DEFAULT_CEILING_LANGS, ceiling_table
    from .evaluation.compression import tokenizers_for_budgets
    from .evaluation.flores import load_flores
    base = _base_tok(cfg)
    langs = cfg.get("ceiling_langs") or DEFAULT_CEILING_LANGS
    achieved = {}
    conds = dict(cfg["conditions"])
    try:
        conds.update(load_config(cfg.get("multilang_config", "configs/multilang.yaml"))["conditions"])
    except FileNotFoundError:
        pass
    from .tokenizer.common import load_tokenizer
    for lang, code in cfg["flores_codes"].items():
        ld = Path(cfg["paths"]["runs"]) / cfg["model"]["name"].split("/")[-1] / lang
        toks = tokenizers_for_budgets(ld, conds) if ld.exists() else {}
        for name, c in conds.items():
            if (c.get("overrides") or {}).get("tokenizer", {}).get("repair_pretokenizer") and (ld / name / "tokenizer").exists():
                toks[f"v{c['budget']}_repaired"] = load_tokenizer(str(ld / name / "tokenizer"))
        if toks:
            sents = load_flores(cfg["paths"]["flores_dir"], code, "devtest")
            nb = sum(len(base.encode(x, add_special_tokens=False)) for x in sents)
            achieved[code] = {lab: 1 - sum(len(t.encode(x, add_special_tokens=False)) for x in sents) / nb
                              for lab, t in toks.items()}
    out = Path(cfg["paths"]["runs"]) / cfg["model"]["name"].split("/")[-1] / "analysis"
    df = ceiling_table(base, cfg["paths"]["flores_dir"], langs, "devtest", out, achieved)
    print(df.to_string(index=False))
    print(f"\nwritten to {out / 'pretokenizer_ceiling.md'}")


def cmd_loops(cfg, args):
    """Looping vs fragmentation, item level (CPU; needs FLORES prediction files)."""
    from .analysis.extra import loops_vs_fertility
    tgt = cfg["flores_codes"][cfg["lang"].split("+")[0]]
    conds = [c for c in cfg["conditions"]]
    res = loops_vs_fertility(_base_tok(cfg), lang_dir(cfg) / "eval", conds, f"eng_Latn-{tgt}", lang_dir(cfg) / "reports")
    print((lang_dir(cfg) / "reports" / f"loops_vs_fertility_eng_Latn-{tgt}.md").read_text(encoding="utf-8"))


def cmd_lora_norms(cfg, args):
    """Size of the LoRA update per condition (CPU; needs adapters)."""
    from .analysis.extra import lora_update_norms
    out = {}
    for c in cfg["conditions"]:
        a = lang_dir(cfg) / c / "lora" / "adapter"
        if (a / "adapter_model.safetensors").exists():
            out[c] = lora_update_norms(a)
            print(f"{c:22s} total ||ΔW|| = {out[c]['total_norm']:.3f}   by type: "
                  + ", ".join(f"{k}={v:.2f}" for k, v in out[c]["by_module_type"].items()))
    write_json(lang_dir(cfg) / "reports" / "lora_update_norms.json", out)


def cmd_frag_decode(cfg, args):
    """Causal probe: constrain the model to write English with short tokens (GPU, inference only)."""
    from .evaluation.flores import fragmented_decoding
    model, tok = _cond_model(cfg)
    fragmented_decoding(model, tok, cfg, lang_dir(cfg) / "eval", n=cfg.get("frag_n", 200))


def cmd_rescore(cfg, args):
    """FLORES chrF++ at an equal character budget for all systems (CPU, from saved predictions)."""
    from .analysis.extra import rescore_equal_chars, write_rescore_md
    from .tokenizer.common import load_tokenizer
    ld = lang_dir(cfg)
    base = _base_tok(cfg)
    toks = {}
    for c, spec in cfg["conditions"].items():
        if not spec.get("budget"):
            toks[c] = base
        elif (ld / c / "tokenizer").exists():
            toks[c] = load_tokenizer(str(ld / c / "tokenizer"))
    res = rescore_equal_chars(ld / "eval", list(cfg["conditions"]), toks,
                              max_new_tokens=cfg["flores"]["max_new_tokens"])
    write_json(ld / "reports" / "flores_equal_chars.json", res)
    write_rescore_md(res, ld / "reports" / "flores_equal_chars.md")
    print((ld / "reports" / "flores_equal_chars.md").read_text(encoding="utf-8"))


def cmd_document(cfg, args):
    """Write docs/RESULTS.md with every saved number (CPU; rerun after each session)."""
    from .analysis.results_doc import build_results_doc
    out = build_results_doc(Path("docs/RESULTS.md"), Path(cfg["paths"]["runs"]), cfg["model"]["name"].split("/")[-1],
                            qwen_cfg=args.config, multi_cfg=cfg.get("multilang_config", "configs/multilang.yaml"))
    print(f"written {out}")


def cmd_report(cfg, args):
    from .analysis.report import failure_cases, final_table, significance
    final_table(cfg, lang_dir(cfg))
    significance(cfg, lang_dir(cfg))
    failure_cases(cfg, lang_dir(cfg))
    print(f"Reports in {lang_dir(cfg) / 'reports'}")


STEP_ORDER = ["extend", "inject", "train-lora", "eval-ppl", "eval-flores", "eval-latency"]


def cmd_run(cfg, args):
    steps = args.steps.split(",") if args.steps else STEP_ORDER
    cdir = cond_dir(cfg)
    save_resolved(cfg, cdir)
    done_marker = lambda s: cdir / f".done_{s}"  # noqa: E731
    for s in steps:
        if done_marker(s).exists() and not args.force:
            log.info("skip %s (done)", s)
            continue
        log.info("=== %s / %s / %s ===", cfg["lang"], cfg["condition"], s)
        COMMANDS[s](cfg, args)
        done_marker(s).touch()


COMMANDS = {
    "env": cmd_env, "smoke": cmd_smoke, "prepare": cmd_prepare, "audit": cmd_audit, "train-bpe": cmd_train_bpe,
    "select": cmd_select, "extend": cmd_extend, "inject": cmd_inject, "train-lora": cmd_train_lora,
    "eval-compression": cmd_eval_compression, "eval-ppl": cmd_eval_ppl, "eval-flores": cmd_eval_flores,
    "eval-latency": cmd_eval_latency, "report": cmd_report, "run": cmd_run,
    "eval-english": cmd_eval_english, "human-eval": cmd_human_eval,
    "document": cmd_document, "rescore": cmd_rescore, "ceiling": cmd_ceiling, "loops": cmd_loops, "lora-norms": cmd_lora_norms, "frag-decode": cmd_frag_decode,
}
NEEDS_CONDITION = {"extend", "inject", "train-lora", "eval-ppl", "eval-flores", "eval-latency", "run", "eval-english", "frag-decode"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="harf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", default="configs/qwen2.5-1.5b.yaml")
    ap.add_argument("--lang", default=None, help="ur | sd | ur+sd")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--steps", default=None, help="for `run`: comma-separated subset of " + ",".join(STEP_ORDER))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    setup_logging(args.log_level)
    if args.command in NEEDS_CONDITION and not args.condition:
        ap.error(f"`{args.command}` needs --condition (one of the keys under `conditions:` in the config)")
    cfg = load_config(args.config, args.set, args.lang, args.condition)
    set_seed(cfg["seed"])
    COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
