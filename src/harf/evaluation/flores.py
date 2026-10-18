"""Day 12: FLORES-200 translation (eng<->urd_Arab / snd_Arab), identical prompts & greedy decoding.

Data: the FLORES-200 tarball layout  <flores_dir>/{dev,devtest}/<code>.{dev,devtest}
(e.g. data/flores200_dataset/devtest/urd_Arab.devtest). FLORES+ (openlanguagedata/flores_plus on HF,
gated) also works: export each language/split to that same one-sentence-per-line layout.

eng->target is the direction that exercises the NEW tokens (the model must generate them)."""
from __future__ import annotations

from pathlib import Path

from ..analysis.stats import corpus_chrf
from ..utils import log, require_torch, write_json, write_jsonl

LANG_NAMES = {"eng_Latn": "English", "urd_Arab": "Urdu", "snd_Arab": "Sindhi", "pbt_Arab": "Pashto",
              "amh_Ethi": "Amharic", "hin_Deva": "Hindi", "tam_Taml": "Tamil", "pes_Arab": "Persian",
              "arb_Arab": "Arabic", "ben_Beng": "Bengali", "tel_Telu": "Telugu", "mal_Mlym": "Malayalam",
              "sin_Sinh": "Sinhala", "tha_Thai": "Thai", "rus_Cyrl": "Russian"}


def lang_name(code: str) -> str:
    """Prompt name for a FLORES code; never fails (falls back to the code itself)."""
    return LANG_NAMES.get(code, code)


def load_flores(flores_dir: str, code: str, split: str) -> list[str]:
    p = Path(flores_dir) / split / f"{code}.{split}"
    if not p.exists():
        raise SystemExit(f"FLORES file not found: {p}")
    return p.read_text(encoding="utf-8").rstrip("\n").split("\n")


def build_prompts(src: list[str], shots: list[tuple[str, str]], src_code: str, tgt_code: str,
                  tok=None, use_chat: bool = False) -> list[str]:
    sn, tn = lang_name(src_code), lang_name(tgt_code)
    if use_chat and tok is not None and getattr(tok, "chat_template", None):
        out = []
        for s in src:
            msgs = []
            for a, b in shots:
                msgs += [{"role": "user", "content": f"Translate from {sn} to {tn}. Output only the translation.\n{a}"},
                         {"role": "assistant", "content": b}]
            msgs.append({"role": "user", "content": f"Translate from {sn} to {tn}. Output only the translation.\n{s}"})
            out.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        return out
    head = "".join(f"{sn}: {a}\n{tn}: {b}\n\n" for a, b in shots)
    return [f"{head}{sn}: {s}\n{tn}:" for s in src]


def generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int,
             extra_suppress: list[int] | None = None) -> list[str]:
    torch = require_torch()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = model.get_input_embeddings().weight.device
    n_rows = model.get_input_embeddings().weight.shape[0]
    suppress = list(range(len(tok), n_rows)) + list(extra_suppress or [])  # padded rows are not real tokens
    outs = []
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            kw = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                      pad_token_id=tok.pad_token_id, suppress_tokens=suppress or None)
            try:  # stop each row at the end of its translation line (identical rule for all conditions)
                gen = model.generate(**batch, **kw, stop_strings=["\n"], tokenizer=tok)
            except (TypeError, ValueError):
                gen = model.generate(**batch, **kw)
            for row in gen[:, batch["input_ids"].shape[1]:]:
                text = tok.decode(row, skip_special_tokens=True)
                outs.append(text.strip().split("\n")[0].strip())  # first line = translation
    return outs


def evaluate_flores(model, tok, cfg: dict, label: str, out_dir: Path, use_chat: bool = False) -> dict:
    fc = cfg["flores"]
    tgt_code = cfg["flores_codes"][cfg["lang"].split("+")[0]]
    results = {"label": label, "split": fc["split"], "n_shots": fc["n_shots"], "prompt_mode": "chat" if use_chat else "few-shot"}
    for direction in fc["directions"]:
        s_code, t_code = [tgt_code if x == "target" else x for x in direction.split("-")]
        src = load_flores(cfg["paths"]["flores_dir"], s_code, fc["split"])
        ref = load_flores(cfg["paths"]["flores_dir"], t_code, fc["split"])
        if fc.get("max_sentences"):
            src, ref = src[: fc["max_sentences"]], ref[: fc["max_sentences"]]
        dev_s = load_flores(cfg["paths"]["flores_dir"], s_code, "dev")
        dev_t = load_flores(cfg["paths"]["flores_dir"], t_code, "dev")
        shots = list(zip(dev_s, dev_t))[: fc["n_shots"]]  # fixed shots, identical across conditions
        prompts = build_prompts(src, shots, s_code, t_code, tok, use_chat)
        hyps = generate(model, tok, prompts, fc["max_new_tokens"], fc["batch_size"])
        import sacrebleu
        chrf = corpus_chrf(hyps, ref)
        bleu = sacrebleu.corpus_bleu(hyps, [ref]).score
        key = f"{s_code}-{t_code}"
        results[key] = {"chrf++": chrf, "bleu": bleu, "n": len(ref),
                        "empty_outputs": sum(not h for h in hyps)}
        write_jsonl(out_dir / f"flores_{label}_{key}.jsonl",
                    [{"i": i, "src": s, "ref": r, "hyp": h} for i, (s, r, h) in enumerate(zip(src, ref, hyps))])
        log.info("FLORES[%s] %s chrF++=%.2f BLEU=%.2f", label, key, chrf, bleu)
    write_json(out_dir / f"flores_{label}.json", results)
    return results


def fragmented_decoding(model, tok, cfg: dict, out_dir, max_chars_list=(None, 3, 2, 1), n: int = 200,
                        direction: str | None = None) -> dict:
    """Causal probe: force the model to WRITE English with only short tokens (<= k characters), making English
    as fragmented as Urdu/Sindhi output, and measure looping and quality. Prompts are unchanged.
    Caveat: the model rarely saw English spelled this way in pretraining (it did see fragmented Urdu)."""
    from ..analysis.extra import rep4
    fc = cfg["flores"]
    tgt = cfg["flores_codes"][cfg["lang"].split("+")[0]]
    direction = direction or f"{tgt}-eng_Latn"
    s_code, t_code = direction.split("-")
    src = load_flores(cfg["paths"]["flores_dir"], s_code, fc["split"])[:n]
    ref = load_flores(cfg["paths"]["flores_dir"], t_code, fc["split"])[:n]
    shots = list(zip(load_flores(cfg["paths"]["flores_dir"], s_code, "dev"),
                     load_flores(cfg["paths"]["flores_dir"], t_code, "dev")))[: fc["n_shots"]]
    prompts = build_prompts(src, shots, s_code, t_code)
    special = set(tok.all_special_ids)
    lengths = {i: len(tok.decode([i]).strip("\n")) for i in range(len(tok))}
    results = {"direction": direction, "n": len(src), "runs": {}}
    for k in max_chars_list:
        sup = [] if k is None else [i for i, L in lengths.items() if L > k and i not in special]
        # more tokens are needed to write the same text with short pieces
        mnt = fc["max_new_tokens"] * (1 if k is None else max(2, 6 // k))
        hyps = generate(model, tok, prompts, mnt, fc["batch_size"], extra_suppress=sup)
        r = [rep4(h) for h in hyps]
        label = "unconstrained" if k is None else f"max{k}chars"
        results["runs"][label] = {
            "chrf++": corpus_chrf(hyps, ref), "mean_rep4": float(sum(r) / len(r)),
            "loop_rate": float(sum(x >= 0.5 for x in r) / len(r)),
            "median_len_ratio": float(sorted(len(h) / max(len(x), 1) for h, x in zip(hyps, ref))[len(ref) // 2]),
            "tokens_per_word_output": float(sum(len(tok.encode(" " + h, add_special_tokens=False)) for h in hyps)
                                            / max(sum(len(h.split()) for h in hyps), 1)),
            "n_suppressed": len(sup)}
        write_jsonl(out_dir / f"fragdecode_{label}_{direction}.jsonl",
                    [{"i": i, "ref": x, "hyp": h} for i, (x, h) in enumerate(zip(ref, hyps))])
        log.info("fragmented decoding %s: %s", label, results["runs"][label])
    write_json(out_dir / f"fragdecode_{direction}.json", results)
    return results
