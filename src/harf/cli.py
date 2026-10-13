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


def _manifest_path(cfg):
    return shared_dir(cfg) / f"merges_{cfg['tokenizer']['strategy']}.json"


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
              cfg["seed"], freeze=True, force=args.force)


def cmd_train_bpe(cfg, args):
    from .tokenizer.audit import run_audit
    from .tokenizer.train_bpe import as_hf, train_target_bpe
    out = shared_dir(cfg) / "target_bpe"
    tok = train_target_bpe(_base_tok(cfg), _train_texts(cfg), cfg["tokenizer"]["target_bpe_vocab"],
                           cfg["tokenizer"]["min_pair_freq"], out)
    run_audit(as_hf(tok), _eval_rows(cfg, "valid"), out, "target_bpe_standalone", cfg["seed"], inspect_n=30)


def cmd_select(cfg, args):
    from .tokenizer.select import continued_bpe, target_bpe_chains
    tc = cfg["tokenizer"]
    base = _base_tok(cfg)
    texts = _train_texts(cfg)
    if tc["strategy"] == "continued":
        res = continued_bpe(base, texts, tc["max_budget"], tc["min_pair_freq"], tc["min_word_freq"], tc["max_unique_words"])
    else:
        from tokenizers import Tokenizer
        p = shared_dir(cfg) / "target_bpe" / "tokenizer.json"
        if not p.exists():
            raise SystemExit("Run `harf train-bpe` first for strategy=target_bpe.")
        res = target_bpe_chains(base, Tokenizer.from_file(str(p)), texts, tc["max_budget"])
    res["base_model"] = cfg["model"]["name"]
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
    extend_tokenizer(_base_tok(cfg), manifest, budget, cond_dir(cfg) / "tokenizer", texts)


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
}
NEEDS_CONDITION = {"extend", "inject", "train-lora", "eval-ppl", "eval-flores", "eval-latency", "run"}


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
