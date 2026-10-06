"""Day 6: vocabulary injection — resize embeddings without touching existing rows, save/reload.

A condition is stored compactly as: tokenizer/ + new_rows.pt (+ lora/adapter). The base model is
never duplicated on disk; `build_condition_model` reconstructs any condition exactly."""
from __future__ import annotations

from pathlib import Path

from ..tokenizer.common import load_tokenizer
from ..utils import log, require_torch


def _dtype(name: str):
    torch = require_torch()
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_base_model(cfg: dict, dtype: str | None = None, device: str | None = None, instruct: bool = False):
    torch = require_torch()
    from transformers import AutoModelForCausalLM
    name = cfg["model"]["instruct_name"] if instruct else cfg["model"]["name"]
    kw = dict(revision=cfg["model"].get("revision"), attn_implementation=cfg["model"].get("attn_implementation"))
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=_dtype(dtype or cfg["model"]["dtype"]), **kw)
    except TypeError:  # transformers < 4.56 uses torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=_dtype(dtype or cfg["model"]["dtype"]), **kw)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval()


def is_tied(model) -> bool:
    out = model.get_output_embeddings()
    return out is not None and out.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()


def inject_vocab(model, n_total_tokens: int, pad_to_multiple_of: int = 64) -> dict:
    """Ensure the embedding matrix has >= n_total_tokens rows. Never shrinks.

    NB: Qwen2.5 ships an embedding matrix larger than its tokenizer (151936 rows vs 151665 ids). A naive
    `resize_token_embeddings(len(tokenizer))` would SHRINK it; we only grow when needed."""
    emb = model.get_input_embeddings()
    old_rows, tied_before = emb.weight.shape[0], is_tied(model)
    if n_total_tokens > old_rows:
        try:
            model.resize_token_embeddings(n_total_tokens, pad_to_multiple_of=pad_to_multiple_of, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(n_total_tokens, pad_to_multiple_of=pad_to_multiple_of)
    new_rows = model.get_input_embeddings().weight.shape[0]
    if tied_before and not is_tied(model):
        raise RuntimeError("Embedding tying was broken by resize.")
    info = {"old_rows": old_rows, "new_rows": new_rows, "tied": is_tied(model), "n_total_tokens": n_total_tokens}
    log.info("inject_vocab: %s", info)
    return info


def set_new_rows(model, ids, input_rows, output_rows=None) -> None:
    torch = require_torch()
    ids_t = torch.as_tensor(ids, dtype=torch.long, device=model.device)
    with torch.no_grad():
        w = model.get_input_embeddings().weight
        w[ids_t] = input_rows.to(w.device, w.dtype)
        if not is_tied(model) and output_rows is not None:
            o = model.get_output_embeddings().weight
            o[ids_t] = output_rows.to(o.device, o.dtype)


def save_rows(path: Path, ids, input_rows, output_rows, meta: dict) -> None:
    torch = require_torch()
    torch.save({"ids": list(map(int, ids)), "input": input_rows.detach().float().cpu(),
                "output": None if output_rows is None else output_rows.detach().float().cpu(), "meta": meta}, path)


def load_rows(path: Path) -> dict:
    torch = require_torch()
    return torch.load(path, map_location="cpu", weights_only=False)


def build_condition_model(cfg: dict, cdir: Path | None, dtype: str | None = None, instruct: bool = False,
                          merge_lora: bool = True, with_lora: bool = True):
    """Rebuild (model, tokenizer) for a condition directory. cdir=None -> baseline."""
    base_name = cfg["model"]["instruct_name"] if instruct else cfg["model"]["name"]
    if cdir is None or not (Path(cdir) / "tokenizer").exists():
        tok = load_tokenizer(base_name, cfg["model"].get("revision"))
        model = load_base_model(cfg, dtype, instruct=instruct)
    else:
        cdir = Path(cdir)
        tok = load_tokenizer(str(cdir / "tokenizer"))
        model = load_base_model(cfg, dtype, instruct=instruct)
        inject_vocab(model, len(tok))
        trained = cdir / "lora" / "new_rows.pt"
        rows_path = trained if (with_lora and trained.exists()) else cdir / "new_rows.pt"
        if rows_path.exists():
            r = load_rows(rows_path)
            set_new_rows(model, r["ids"], r["input"], r["output"])
    if with_lora and cdir is not None and (Path(cdir) / "lora" / "adapter").exists():
        if instruct:
            log.warning("LoRA adapter was trained on the base model; applying it to the instruct model is approximate.")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(Path(cdir) / "lora" / "adapter"))
        if merge_lora:
            model = model.merge_and_unload()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model.eval(), tok


def roundtrip_check(cfg: dict, cdir: Path, model) -> bool:
    """Save/reload test: rebuilt condition must match the in-memory model on new rows."""
    torch = require_torch()
    r = load_rows(cdir / "new_rows.pt")
    ids = torch.as_tensor(r["ids"], device=model.device)
    live = model.get_input_embeddings().weight[ids].float().cpu()
    ok = torch.allclose(live, r["input"].to(live.dtype), atol=1e-2, rtol=1e-2)
    log.info("roundtrip_check: %s", ok)
    return bool(ok)
