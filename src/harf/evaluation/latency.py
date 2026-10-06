"""Day 13: latency under identical hardware/settings; tokenizer CPU time reported separately.

Free-running generation is NOT a fair latency comparison between tokenizers: a fixed max_new_tokens
emits more text with a more efficient tokenizer, and outputs differ. We therefore time producing the
SAME text in every condition:
  * prefill   : one forward pass over each reference sentence
  * emit      : forced decoding with KV cache, one step per token of that condition's tokenization
                (= wall-clock to generate exactly this text, token by token)
Reported: median over repeats of per-sentence time, plus chars/second."""
from __future__ import annotations

import statistics
import time
from pathlib import Path

from ..utils import env_info, log, require_torch, write_json


def _sync():
    torch = require_torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tokenizer_time(tok, texts: list[str], repeats: int) -> dict:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for t in texts:
            tok.encode(t, add_special_tokens=False)
        ts.append(time.perf_counter() - t0)
    return {"encode_total_s_median": statistics.median(ts), "encode_per_sentence_ms": 1000 * statistics.median(ts) / len(texts)}


def forced_emit(model, ids):
    torch = require_torch()
    past = None
    x = ids[:, :1]
    for i in range(ids.shape[1]):
        out = model(input_ids=x, past_key_values=past, use_cache=True)
        past = out.past_key_values
        if i + 1 < ids.shape[1]:
            x = ids[:, i + 1:i + 2]
    return out


def measure_latency(model, tok, texts: list[str], label: str, out_dir: Path, warmup: int, repeats: int) -> dict:
    torch = require_torch()
    device = model.get_input_embeddings().weight.device
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    seqs = [torch.tensor([[start] + tok.encode(t, add_special_tokens=False)], device=device) for t in texts]
    with torch.inference_mode():
        for s in seqs[:warmup]:
            model(input_ids=s)
            forced_emit(model, s)
        _sync()
        prefill, emit = [], []
        for s in seqs:
            p_runs, e_runs = [], []
            for _ in range(repeats):
                _sync(); t0 = time.perf_counter(); model(input_ids=s); _sync()
                p_runs.append(time.perf_counter() - t0)
                _sync(); t0 = time.perf_counter(); forced_emit(model, s); _sync()
                e_runs.append(time.perf_counter() - t0)
            prefill.append(statistics.median(p_runs))
            emit.append(statistics.median(e_runs))
    chars = sum(len(t) for t in texts)
    res = {"label": label, "n_sentences": len(texts), "repeats": repeats, "warmup": warmup, "batch_size": 1,
           "prefill_ms_median": 1000 * statistics.median(prefill),
           "emit_ms_median": 1000 * statistics.median(emit),
           "emit_total_s": sum(emit), "emit_chars_per_s": chars / sum(emit),
           "prefill_total_s": sum(prefill),
           "tokens_total": int(sum(s.shape[1] - 1 for s in seqs)),
           **tokenizer_time(tok, texts, repeats),
           "hardware": env_info().get("gpus"), "dtype": str(next(model.parameters()).dtype)}
    write_json(out_dir / f"latency_{label}.json", res)
    log.info("latency[%s]: emit median %.1f ms, %.1f chars/s", label, res["emit_ms_median"], res["emit_chars_per_s"])
    return res
