# HARF — Hybrid Alignment for Reducing Fragmentation

*حرف (harf): "letter" in Urdu, Sindhi, Arabic and Persian.*

Post-hoc vocabulary extension for Perso-Arabic scripts. HARF adds Urdu/Sindhi subwords to an open-weight
LLM's tokenizer (Qwen2.5-1.5B, Llama-3.2-1B), initialises the new embeddings, optionally adapts with LoRA,
and measures what it buys and what it costs: compression, latency, bits-per-character, and FLORES-200 chrF++.

> **Status:** research prototype. The tokenizer pipeline is tested end-to-end (7 tests, CPU).
> The GPU half (injection, initialisation, LoRA, perplexity, FLORES, latency) is written but awaits its first GPU run.
> No result in this README is a claim; all numbers come from `runs/*/reports/`.

## Why

Byte-level BPE tokenizers trained mostly on English split Urdu and especially Sindhi text into many
more tokens per word. Users of these languages pay this "tokenization tax" in cost, latency, and usable context.

## What is different about HARF

| Design choice | Why it matters |
|---|---|
| **Continued BPE** over the base model's own segmentation | Exact ID budget; 2k/4k are exact prefixes of 8k (clean ablation); simulated savings equal real savings token-for-token (tested). |
| **Strict non-interference** | Every new token contains an Arabic-block byte, so English, code and numbers tokenize *identically* to the base model. Verified on every extension. |
| **Merge-tree projection** init | Learns `E[ab] ≈ W·[E[a]; E[b]]` from the base model's *own* merges. Self-supervised; parallel data is an optional anchor. |
| **Bits-per-character**, not token perplexity | Token perplexity is not comparable across tokenizers. |
| **Same-text latency** | Times emitting the *same text* in every condition, not a fixed number of tokens. |
| **Control conditions** | `baseline_lora` separates "new vocab" from "just fine-tuned on Urdu"; `D_mean` tests whether init still matters after LoRA. |
| **Sindhi-safe normalisation** | In Sindhi, ک (kheh) ≠ ڪ (kaf). Urdu "fixes" would corrupt Sindhi; canonicalisation is per-language and off by default. |
| **Old embedding rows are frozen** | Gradient-masked; asserted unchanged after training. Only new rows + LoRA train. |

## Install

```bash
git clone https://github.com/<you>/harf && cd harf
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu,dev]"
pytest -q                                  # 7 tests, CPU only, no downloads
pip freeze > requirements.lock.txt         # commit this
```

Llama 3.2 is gated: accept its license on Hugging Face, then `huggingface-cli login`.

## Data

```
data/raw/ur/...   .txt (blank-line separated docs) | .txt.gz | .jsonl {"text", "id", "source", "license"}
data/raw/sd/...
data/flores200_dataset/{dev,devtest}/{eng_Latn,urd_Arab,snd_Arab}.{dev,devtest}   # scripts/get_flores.sh
```

Keep source and license metadata; it flows into `data/processed/<lang>/data_card.md`.

## Run

```bash
# 15-minute shakedown on a GPU (Days 1-6 + one LoRA run)
scripts/debug_day1_to_6.sh ur

# Full matrix for one language
scripts/run_matrix.sh ur configs/qwen2.5-1.5b.yaml

# Or step by step
harf prepare --lang ur
harf audit   --lang ur                     # frozen baseline
harf select  --lang ur                     # learns 8k merges once
harf extend  --lang ur --condition F
harf eval-compression --lang ur            # CPU only — check this before any GPU time
harf run     --lang ur --condition D       # extend, inject, train-lora, eval-ppl, eval-flores, eval-latency
harf report  --lang ur
```

Override anything: `--set lora.steps=300 --set projection.pairs=all`.

## Experiment matrix

| Condition | Vocab | Init | LoRA | Purpose |
|---|---|---|---|---|
| baseline | 0 | – | no | reference |
| baseline_lora | 0 | – | yes | **control**: fine-tuning without new vocab |
| A | 2k | random | no | lower bound |
| B | 2k | mean of constituents | no | init baseline |
| C | 2k | projection | no | alignment effect |
| D | 2k | projection | yes | full small-scale method |
| D_mean | 2k | mean | yes | **control**: does init matter after LoRA? |
| E | 4k | projection | yes | vocab ablation |
| F | 8k | projection | yes | source-proposed main setting |

## Outputs

```
runs/<model>/<lang>/
  _shared/baseline_audit/    baseline_metrics.json, per-sentence CSV, inspection sheet (frozen)
  _shared/merges_continued.json, candidate_tokens_{2k,4k,8k}.json
  <cond>/tokenizer/          extended tokenizer, new_tokens.json, verification.json
  <cond>/new_rows.pt         initialised embedding rows (the base model is never duplicated)
  <cond>/init/               projection diagnostics, nearest neighbours, smoke generation
  <cond>/lora/               adapter, trained rows, train_log.jsonl
  eval/                      compression, ppl_*, flores_* (raw predictions), latency_*
  reports/                   final_results.{csv,md}, significance.json, tradeoff.png, failure_cases.md
```

## Metrics (fixed definitions)

- **SCR** = 1 − Σ tokens_extended / Σ tokens_baseline, over the fixed test sentences.
- **BPC** = total NLL / (ln 2 · #characters). Word perplexity = exp(NLL / #words).
- **Latency speedup** = baseline emit time / condition emit time, for the same texts, same GPU, batch 1, median of repeats.
- **Significance:** paired t-test and Wilcoxon per sentence, bootstrap CIs, paired-bootstrap chrF++, Holm correction.

## Reporting honestly

The source proposal's targets (35–40% compression, <2% quality loss) are hypotheses. Report what you measure,
including negative results. If compression improves but BPC worsens, that is a finding.

## Citation

TBD.
