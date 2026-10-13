"""Day 12: FLORES-200 translation (eng<->urd_Arab / snd_Arab), identical prompts & greedy decoding.

Data: the FLORES-200 tarball layout  <flores_dir>/{dev,devtest}/<code>.{dev,devtest}
(e.g. data/flores200_dataset/devtest/urd_Arab.devtest). FLORES+ (openlanguagedata/flores_plus on HF,
gated) also works: export each language/split to that same one-sentence-per-line layout.

eng->target is the direction that exercises the NEW tokens (the model must generate them)."""
from __future__ import annotations

from pathlib import Path

from ..analysis.stats import corpus_chrf
from ..utils import log, require_torch, write_json, write_jsonl

LANG_NAMES = {"eng_Latn": "English", "urd_Arab": "Urdu", "snd_Arab": "Sindhi"}


def load_flores(flores_dir: str, code: str, split: str) -> list[str]:
    p = Path(flores_dir) / split / f"{code}.{split}"
    if not p.exists():
        raise SystemExit(f"FLORES file not found: {p}")
    return p.read_text(encoding="utf-8").rstrip("\n").split("\n")


def build_prompts(src: list[str], shots: list[tuple[str, str]], src_code: str, tgt_code: str,
                  tok=None, use_chat: bool = False) -> list[str]:
    sn, tn = LANG_NAMES[src_code], LANG_NAMES[tgt_code]
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


def generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    torch = require_torch()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = model.get_input_embeddings().weight.device
    n_rows = model.get_input_embeddings().weight.shape[0]
    suppress = list(range(len(tok), n_rows))  # padded rows are not real tokens
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
