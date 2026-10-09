"""Day 9: LoRA adapter + training of ONLY the new embedding rows.

Why not just LoRA? LoRA alone cannot change embedding rows, so new tokens would stay at their
initialisation. Why not modules_to_save=[embed_tokens]? That trains (and stores) the whole
151k x d matrix. Instead we unfreeze the embedding and mask gradients of all pre-existing rows
(weight decay disabled on that group so old rows are bit-for-bit unchanged; this is asserted at the end).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from ..data.prepare import load_split_multi
from ..model.inject import build_condition_model, is_tied, save_rows
from ..utils import log, require_torch, write_json


def packed_blocks(tok, texts: list[str], seq_len: int, max_tokens: int, seed: int):
    sep = tok.eos_token_id
    buf, blocks, total = [], [], 0
    rng = random.Random(seed)
    texts = texts[:]
    rng.shuffle(texts)
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False) + [sep]
        buf.extend(ids)
        total += len(ids)
        while len(buf) >= seq_len:
            blocks.append(buf[:seq_len])
            buf = buf[seq_len:]
        if total >= max_tokens:
            break
    rng.shuffle(blocks)
    return blocks


def train_lora(cfg: dict, cdir: Path) -> dict:
    torch = require_torch()
    from peft import LoraConfig, get_peft_model

    lcfg = cfg["lora"]
    out = cdir / "lora"
    out.mkdir(parents=True, exist_ok=True)
    has_new = (cdir / "tokenizer").exists()
    model, tok = build_condition_model(cfg, cdir if has_new else None, dtype=lcfg["master_dtype"],
                                       with_lora=False)
    base_len = json.loads((cdir / "tokenizer" / "harf_meta.json").read_text())["first_new_id"] if has_new else None
    new_ids = list(range(base_len, len(tok))) if has_new else []

    if lcfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    peft_cfg = LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
                          target_modules=lcfg["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)

    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    groups = [{"params": lora_params, "lr": lcfg["lr"], "weight_decay": 0.0}]
    emb_params, snapshots = [], {}
    if new_ids:
        emb = model.get_input_embeddings()
        mats = [emb.weight] + ([] if is_tied(model) else [model.get_output_embeddings().weight])
        mask = torch.zeros(emb.weight.shape[0], 1, device=emb.weight.device, dtype=emb.weight.dtype)
        mask[new_ids] = 1
        for w in mats:
            w.requires_grad_(True)
            w.register_hook(lambda g, m=mask: g * m.to(g.dtype))
            emb_params.append(w)
            snapshots[id(w)] = w.detach()[: base_len].float().sum().item()
        groups.append({"params": emb_params, "lr": lcfg["embed_lr"], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))

    texts = [r["text"] for r in load_split_multi(cfg, "train")]
    blocks = packed_blocks(tok, texts, lcfg["seq_len"], lcfg["max_train_tokens"], cfg["seed"])
    if not blocks:
        raise SystemExit("No training blocks — corpus too small for seq_len.")
    steps, accum, bs = lcfg["steps"], lcfg["grad_accum"], lcfg["batch_size"]
    warm = max(1, lcfg["warmup_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    n_trainable = sum(p.numel() for p in lora_params) + len(new_ids) * model.get_input_embeddings().weight.shape[1]
    log.info("LoRA: %d blocks of %d tokens, %d steps, trainable≈%s params", len(blocks), lcfg["seq_len"], steps,
             f"{n_trainable:,}")

    device = model.get_input_embeddings().weight.device
    # bfloat16 needs Ampere+ (A100, RTX 30xx/40xx). T4/P100 (Kaggle, Colab free): use float16 + GradScaler.
    ac = lcfg.get("autocast_dtype", "bfloat16")
    use_ac = device.type == "cuda" and ac != "none"
    ac_dtype = torch.float16 if ac == "float16" else torch.bfloat16
    autocast = torch.autocast("cuda", dtype=ac_dtype, enabled=use_ac)
    scaler = torch.amp.GradScaler("cuda", enabled=use_ac and ac == "float16")
    log_f = open(out / "train_log.jsonl", "w")
    model.train()
    bi, t0 = 0, time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            batch = [blocks[(bi + j) % len(blocks)] for j in range(bs)]
            bi += bs
            x = torch.tensor(batch, device=device)
            with autocast:
                loss = model(input_ids=x, labels=x).loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % lcfg["log_every"] == 0 or step == steps - 1:
            rec = {"step": step, "loss": tot, "lr": sched.get_last_lr()[0], "tokens_seen": bi * lcfg["seq_len"],
                   "elapsed_s": round(time.time() - t0, 1)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            log.info("%s", rec)
    log_f.close()

    model.eval()
    for w in emb_params:  # old rows must be untouched
        drift = abs(w.detach()[: base_len].float().sum().item() - snapshots[id(w)])
        if drift > 1e-3:
            raise RuntimeError(f"Pre-existing embedding rows changed during training (drift={drift}).")
    model.save_pretrained(str(out / "adapter"), save_embedding_layers=False)
    if new_ids:
        emb_w = model.get_input_embeddings().weight
        out_w = None if is_tied(model) else model.get_output_embeddings().weight[new_ids]
        save_rows(out / "new_rows.pt", new_ids, emb_w[new_ids], out_w, {"source": "lora"})
    summary = {"steps": steps, "final_loss": tot, "trainable_params": n_trainable, "lora_r": lcfg["r"],
               "seq_len": lcfg["seq_len"], "n_blocks": len(blocks), "wall_s": round(time.time() - t0, 1)}
    write_json(out / "train_summary.json", summary)
    return summary
