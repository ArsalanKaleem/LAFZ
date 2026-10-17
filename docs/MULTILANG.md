# Multi-language study: Pashto, Amharic, Hindi, Tamil

## Design

| Language | Code | Script | Pre-tokenizer splits words? | Conditions |
|---|---|---|---|---|
| Pashto | `ps` | arabic | no | baseline, baseline_lora, H |
| Amharic | `am` | ethiopic | no | baseline, baseline_lora, H |
| Hindi | `hi` | devanagari | **yes** (vowel signs) | baseline, baseline_lora, H, **H_repair** |
| Tamil | `ta` | tamil | **yes** (vowel signs) | baseline, baseline_lora, H, **H_repair** |

All four use `configs/multilang*.yaml`: **750 LoRA steps for every condition** (half the Urdu protocol),
so comparisons are fair within each language.

`H_repair` = HARF built on a **script-scoped repaired pre-tokenizer**: only the target script's combining
marks are added to the word class, so words are no longer cut at vowel signs. Text without that script is
pre-tokenized and tokenized exactly as before (checked on every extension against 9 control texts).

## Step 1 — laptop: data (CPU, ~1–2 h of downloading)

```powershell
python scripts/download_data.py --langs ps am hi ta --max-docs 200000
```

If a source fails ("config not found"), open the dataset on Hugging Face, find the language under
*Subsets*, and rerun for that language with `--fineweb-config` / `--wiki-config`.

## Step 2 — laptop: prepare, audit, select, extend, compression (CPU)

```powershell
foreach ($L in "ps","am","hi","ta") {
  harf prepare --config configs/multilang.yaml --lang $L
  harf audit   --config configs/multilang.yaml --lang $L
  harf select  --config configs/multilang.yaml --lang $L
  harf extend  --config configs/multilang.yaml --lang $L --condition H
}
foreach ($L in "hi","ta") {
  harf select --config configs/multilang.yaml --lang $L --condition H_repair
  harf extend --config configs/multilang.yaml --lang $L --condition H_repair
}
foreach ($L in "ps","am","hi","ta") { harf eval-compression --config configs/multilang.yaml --lang $L }
harf ceiling
```

`select` for `H_repair` writes a separate merge list (`merges_continued_repaired.json`).
Check `data/processed/<lang>/data_card.md` for each language before going further.

## Step 3 — Kaggle (GPU), one language per session

Zip and upload each language as a private dataset, e.g. `harf-hi`:

```powershell
tar -a -c -f harf-hi.zip data/processed/hi runs/Qwen2.5-1.5B/hi
```

Per language (replace `hi`; drop `H_repair` for `ps` and `am`):

```bash
!harf run --config configs/multilang-kaggle.yaml --lang hi --condition baseline --steps eval-ppl
!for C in H H_repair baseline_lora; do PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True harf run --config configs/multilang-kaggle.yaml --lang hi --condition $C --steps extend,inject,train-lora,eval-ppl; zip -qr /kaggle/working/hi_$C.zip runs/Qwen2.5-1.5B/hi; done
!for C in baseline baseline_lora H H_repair; do harf eval-flores --config configs/multilang-kaggle.yaml --lang hi --condition $C; harf eval-english --config configs/multilang-kaggle.yaml --lang hi --condition $C; harf eval-latency --config configs/multilang-kaggle.yaml --lang hi --condition $C --set latency.repeats=3; done
!harf report --config configs/multilang-kaggle.yaml --lang hi
!zip -qr /kaggle/working/hi_all.zip runs/Qwen2.5-1.5B/hi
```

Approximate GPU time: Pashto/Amharic ~5.5 h each, Hindi/Tamil ~8 h each.

## Step 4 — laptop: analysis

```powershell
foreach ($L in "ps","am","hi","ta") {
  harf report --config configs/multilang.yaml --lang $L
  harf loops  --config configs/multilang.yaml --lang $L
  harf human-eval --config configs/multilang.yaml --lang $L --steps "make:baseline_lora,H,,50"
}
```

## What to look for

1. **Prediction:** `H` gains for Pashto and Amharic should resemble Urdu/Sindhi; for Hindi and Tamil they
   should be smaller, and `H_repair` should recover much of the gap (compression first, then BPC/chrF++).
2. **The open question:** the repaired pre-tokenizer shows the PRETRAINED model segmentations it never saw.
   Whether 750 LoRA steps overcome that shift is the result — report it either way.
3. **Degeneration:** loop rates (`failure_summary.csv`, `loops`) across all six languages.
