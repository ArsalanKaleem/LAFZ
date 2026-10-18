# HARF — all results

Generated 2026-10-18 18:41 by `harf document` · git `bc388164ce` · model `Qwen/Qwen2.5-1.5B` @ `8faed761d45a263340a0528343f099c05c9a4323`

Every table lists its source file. Regenerate after each session; do not edit numbers by hand.

Contents: [Urdu](#urdu-ur), [Sindhi](#sindhi-sd), [Pashto](#pashto-ps), [Amharic](#amharic-am), [Hindi](#hindi-hi), [Tamil](#tamil-ta), [Pre-tokenizer ceiling](#pre-tokenizer-ceiling)

## Urdu (`ur`)

### Data

- train_docs: 284,276
- train_chars: 647,296,416
- train_words: 137,176,665
- valid_sentences: 2,000
- test_sentences: 2,000
- filtering: {'docs_in': 300000, 'low_arabic_ratio': 1765, 'too_short': 76, 'duplicate_doc': 1882, 'valid_leak_removed': 18745, 'test_leak_removed': 19871}

<sub>Source: `data/processed/ur/data_card.json`</sub>

### Baseline tokenization (validation split)

- n_sentences: 2000
- total_tokens: 152945
- tokens_per_word: 3.0718
- tokens_per_char: 0.6589
- chars_per_token: 1.5177
- bytes_per_token: 2.6990
- median_tokens_per_sentence: 58.0000
- target_word_fertility: 3.1064
- fragmented_target_word_rate: 0.9602

<sub>Source: `runs/Qwen2.5-1.5B/ur/_shared/baseline_audit/baseline_metrics.json`</sub>

### Compression (test split)

Definition: SCR = 1 - sum(tokens_ext)/sum(tokens_base); n = 2000 sentences.

| tokenizer | SCR | tokens/word | chars/token | median tokens/sent | per-sentence SCR median | Wilcoxon p |
|---|---|---|---|---|---|---|
| baseline |  | 3.0424 | 1.5224 | 58.0000 |  |  |
| v2000 | 0.4905 | 1.5500 | 2.9881 | 29.0000 | 0.4935 | 0.0000 |
| v4000 | 0.5373 | 1.4078 | 3.2899 | 27.0000 | 0.5426 | 0.0000 |
| v8000 | 0.5738 | 1.2967 | 3.5720 | 24.5000 | 0.5797 | 0.0000 |

<sub>Source: `runs/Qwen2.5-1.5B/ur/eval/compression_results.json`</sub>

### All conditions

| condition | budget | init | lora | scr | tokens_per_word | median_tokens_per_sentence | bpc | word_ppl | bpc_change_pct | chrf++ eng_Latn-urd_Arab | chrf++ urd_Arab-eng_Latn | english_bpc | english_bpc_change_pct | emit_ms_median | tokenizer_ms_per_sentence | latency_speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 |  | False | 0.0000 | 3.0424 | 58.0000 | 2.5888 | 4069.4377 | 0.0000 | 8.9025 | 29.7283 | 1.0010 | 0.0000 | 2361.0139 | 0.1380 | 1.0000 |
| baseline_lora | 0 |  | True | 0.0000 | 3.0424 | 58.0000 | 1.8182 | 342.7873 | -29.7687 | 13.1447 | 27.8403 | 1.0160 | 1.5029 | 2343.1156 | 0.1225 | 1.0040 |
| B | 2000 | mean | False | 0.4905 | 1.5500 | 29.0000 | 3.7034 | 145743.6397 | 43.0542 |  |  |  |  |  |  |  |
| C | 2000 | projection | False | 0.4905 | 1.5500 | 29.0000 | 3.5436 | 87249.8178 | 36.8809 |  |  |  |  |  |  |  |
| D | 2000 | projection | True | 0.4905 | 1.5500 | 29.0000 | 1.7088 | 241.3149 | -33.9919 | 17.8634 | 34.9565 | 1.0107 | 0.9770 | 1160.0821 | 0.1167 | 1.9543 |
| D_mean | 2000 | mean | True | 0.4905 | 1.5500 | 29.0000 | 1.7163 | 247.1551 | -33.7042 | 18.7567 | 34.8888 | 0.9979 | -0.3032 | 1163.5518 | 0.1296 | 1.9425 |
| D_750 | 2000 | projection | True | 0.4905 | 1.5500 | 29.0000 | 1.8176 | 342.1618 | -29.7906 |  |  |  |  |  |  |  |
| D_mean_750 | 2000 | mean | True | 0.4905 | 1.5500 | 29.0000 | 1.8278 | 353.4861 | -29.3989 |  |  |  |  |  |  |  |

<sub>Source: `runs/Qwen2.5-1.5B/ur/reports/final_results.csv`</sub>

### Pairwise comparisons


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| baseline_lora | D | 1.8182 | 1.7088 | -0.0949 | [-0.1014, -0.0877] | 1.4e-159 |
| baseline_lora | D_mean | 1.8182 | 1.7163 | -0.0892 | [-0.0956, -0.0822] | 7.6e-150 |
| D_mean | D | 1.7163 | 1.7088 | -0.0057 | [-0.0088, -0.0025] | 3.2e-07 |
| baseline_lora | D_750 | 1.8182 | 1.8176 | +0.0094 | [+0.0033, +0.0160] | 0.32 |
| baseline_lora | D_mean_750 | 1.8182 | 1.8278 | +0.0180 | [+0.0119, +0.0243] | 2e-06 |
| D_mean_750 | D_750 | 1.8278 | 1.8176 | -0.0086 | [-0.0119, -0.0051] | 9.5e-12 |

## chrF++ eng_Latn-urd_Arab

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | D | 13.14 | 17.86 | +4.72 | [+4.00, +5.44] |
| baseline_lora | D_mean | 13.14 | 18.76 | +5.61 | [+4.93, +6.28] |
| D_mean | D | 18.76 | 17.86 | -0.89 | [-1.62, -0.19] |

## chrF++ urd_Arab-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | D | 27.84 | 34.96 | +7.12 | [+6.30, +7.91] |
| baseline_lora | D_mean | 27.84 | 34.89 | +7.05 | [+6.16, +7.89] |
| D_mean | D | 34.89 | 34.96 | +0.07 | [-0.67, +0.84] |

<sub>Source: `runs/Qwen2.5-1.5B/ur/reports/comparisons.md`</sub>

### English retention: paired tests


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| english_baseline | english_baseline_lora | 1.0010 | 1.0160 | +0.0156 | [+0.0132, +0.0180] | 9.5e-39 |
| english_baseline | english_D | 1.0010 | 1.0107 | +0.0089 | [+0.0059, +0.0119] | 1.7e-13 |
| english_baseline_lora | english_D | 1.0160 | 1.0107 | -0.0067 | [-0.0087, -0.0047] | 1.7e-13 |

## chrF++ eng_Latn-urd_Arab

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|

## chrF++ urd_Arab-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|

<sub>Source: `runs/Qwen2.5-1.5B/ur/reports/comparisons_english.md`</sub>

### Human evaluation (blinded)

| file | a | b | n | preferred a / b / tie | fluency a → b | adequacy a → b | sign-test p |
|---|---|---|---|---|---|---|---|
| human_eval_baseline_lora_vs_D_eng_Latn-urd_Arab.results.json | baseline_lora | D | 100 | 12 / 51 / 37 | 1.59 → 2.22 | 1.28 → 2.06 | 0.000001 |

Raters: record who rated each language in the paper.

### Generation failure summary (FLORES outputs)

| condition | direction | empty | mean_rep4 | mean_script_mix | median_len_ratio |
|---|---|---|---|---|---|
| baseline | eng_Latn-urd_Arab | 0 | 0.728 | 0.003 | 2.575 |
| baseline_lora | eng_Latn-urd_Arab | 0 | 0.546 | 0.003 | 2.239 |
| baseline_lora | urd_Arab-eng_Latn | 0 | 0.039 | 0.000 | 1.061 |
| baseline | urd_Arab-eng_Latn | 0 | 0.018 | 0.000 | 1.024 |
| D | eng_Latn-urd_Arab | 0 | 0.311 | 0.019 | 1.217 |
| D_mean | eng_Latn-urd_Arab | 0 | 0.257 | 0.024 | 1.164 |
| D_mean | urd_Arab-eng_Latn | 0 | 0.025 | 0.000 | 0.992 |
| D | urd_Arab-eng_Latn | 0 | 0.026 | 0.000 | 1.008 |

<sub>Source: `runs/Qwen2.5-1.5B/ur/reports/failure_summary.csv`</sub>

### Looping vs per-word fragmentation (eng_Latn-urd_Arab)

| condition | Spearman ρ | p | loop rate Q1 → Q4 (least → most fragmented) |
|---|---|---|---|
| baseline | -0.038 | 0.23 | 83.8% → 86.2% → 86.3% → 81.9% |
| baseline_lora | -0.011 | 0.74 | 64.4% → 68.5% → 72.3% → 62.2% |
| D | -0.002 | 0.96 | 32.0% → 38.2% → 32.4% → 34.9% |
| D_mean | +0.013 | 0.68 | 26.1% → 27.2% → 29.7% → 29.3% |

### Looping at matched length (eng_Latn-urd_Arab)

Loop rate (n) by the number of tokens each system's own tokenizer needs to write the reference.

| condition | 0–30 | 30–40 | 40–50 | 50–60 | 60–75 | 75–∞ | overall |
|---|---|---|---|---|---|---|---|
| baseline | – (4) | 68.0% (25) | 66.7% (69) | 82.0% (100) | 85.5% (179) | 87.4% (635) | 84.6% |
| baseline_lora | – (4) | 36.0% (25) | 56.5% (69) | 58.0% (100) | 64.2% (179) | 71.5% (635) | 66.9% |
| D | 23.8% (185) | 29.1% (251) | 35.0% (263) | 42.9% (170) | 41.0% (105) | 60.5% (38) | 34.4% |
| D_mean | 20.0% (185) | 25.5% (251) | 27.0% (263) | 29.4% (170) | 38.1% (105) | 57.9% (38) | 28.1% |

Decomposition vs `baseline_lora` (reweighting to its length distribution; approximate):

| condition | ref loop rate | at ref lengths | actual | reduction | from per-step looping | from shorter sequences | share per-step | coverage |
|---|---|---|---|---|---|---|---|---|
| D | 66.9% | 52.7% | 34.4% | 32.5% | 14.2% | 18.3% | 43.8% | 100.0% |
| D_mean | 66.9% | 48.5% | 28.1% | 38.8% | 18.4% | 20.5% | 47.3% | 100.0% |

Coverage = share of the reference system's sentences in length bins both systems occupy. Below 100%, the two parts need not sum exactly to the total.

### LoRA update size ‖ΔW‖

| condition | total | down_proj | gate_proj | k_proj | o_proj | q_proj | up_proj | v_proj |
|---|---|---|---|---|---|---|---|---|
| baseline_lora | 33.92 | 10.89 | 24.64 | 2.37 | 7.10 | 6.06 | 18.07 | 2.38 |
| D | 36.95 | 14.08 | 26.01 | 2.58 | 7.17 | 6.80 | 19.49 | 2.54 |
| D_mean | 36.47 | 13.49 | 25.68 | 2.52 | 7.16 | 6.70 | 19.48 | 2.57 |
| D_750 | 24.07 | 8.31 | 17.44 | 1.79 | 4.57 | 4.61 | 12.57 | 1.66 |
| D_mean_750 | 23.76 | 7.91 | 17.22 | 1.72 | 4.61 | 4.55 | 12.56 | 1.68 |

<sub>Source: `runs/Qwen2.5-1.5B/ur/reports/lora_update_norms.json`</sub>

### Training runs

Per-token losses are not comparable across tokenizers.

| condition | steps | final loss (per token) | trainable params | wall time (h) | λ (dev BPC) |
|---|---|---|---|---|---|
| baseline_lora | 1500 | 1.727 | 18464768 | 3.947 |  |
| D | 1500 | 2.925 | 21536768 | 3.402 | 1.000 |
| D_mean | 1500 | 2.930 | 21536768 | 3.405 |  |
| D_750 | 750 | 3.234 | 21536768 | 1.753 | 1.000 |
| D_mean_750 | 750 | 3.247 | 21536768 | 1.758 |  |

## Sindhi (`sd`)

### Data

- train_docs: 204,956
- train_chars: 581,624,108
- train_words: 122,903,712
- valid_sentences: 2,000
- test_sentences: 2,000
- filtering: {'docs_in': 216928, 'low_arabic_ratio': 2833, 'other_language_markers': 11, 'duplicate_doc': 467, 'too_short': 114, 'valid_leak_removed': 16399, 'test_leak_removed': 16605}

<sub>Source: `data/processed/sd/data_card.json`</sub>

### Baseline tokenization (validation split)

- n_sentences: 2000
- total_tokens: 128164
- tokens_per_word: 3.0262
- tokens_per_char: 0.6478
- chars_per_token: 1.5436
- bytes_per_token: 2.7295
- median_tokens_per_sentence: 48.0000
- target_word_fertility: 3.1244
- fragmented_target_word_rate: 0.9471

<sub>Source: `runs/Qwen2.5-1.5B/sd/_shared/baseline_audit/baseline_metrics.json`</sub>

### Compression (test split)

Definition: SCR = 1 - sum(tokens_ext)/sum(tokens_base); n = 2000 sentences.

| tokenizer | SCR | tokens/word | chars/token | median tokens/sent | per-sentence SCR median | Wilcoxon p |
|---|---|---|---|---|---|---|
| baseline |  | 3.0032 | 1.5521 | 49.0000 |  |  |
| v2000 | 0.4750 | 1.5766 | 2.9564 | 26.0000 | 0.4737 | 0.0000 |
| v4000 | 0.5241 | 1.4293 | 3.2611 | 24.0000 | 0.5227 | 0.0000 |
| v8000 | 0.5612 | 1.3179 | 3.5369 | 22.0000 | 0.5622 | 0.0000 |

<sub>Source: `runs/Qwen2.5-1.5B/sd/eval/compression_results.json`</sub>

### All conditions

| condition | budget | init | lora | scr | tokens_per_word | median_tokens_per_sentence | bpc | word_ppl | bpc_change_pct | chrf++ eng_Latn-snd_Arab | chrf++ snd_Arab-eng_Latn | english_bpc | english_bpc_change_pct | emit_ms_median | tokenizer_ms_per_sentence | latency_speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 |  | False | 0.0000 | 3.0032 | 49.0000 | 3.9830 | 387877.7057 | 0.0000 | 3.6488 | 19.6326 | 1.0010 | 0.0000 | 1388.0271 | 0.0939 | 1.0000 |
| baseline_lora | 0 |  | True | 0.0000 | 3.0032 | 49.0000 | 2.0331 | 712.3710 | -48.9558 | 7.5194 | 24.1104 | 1.0418 | 4.0857 | 1382.3043 | 0.0970 | 1.0197 |
| D | 2000 | projection | True | 0.4750 | 1.5766 | 26.0000 | 1.9023 | 466.8754 | -52.2393 | 15.1794 | 33.2373 | 1.0198 | 1.8799 | 760.0608 | 0.0873 | 1.8438 |

<sub>Source: `runs/Qwen2.5-1.5B/sd/reports/final_results.csv`</sub>

### Pairwise comparisons


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| baseline_lora | D | 2.0331 | 1.9023 | -0.1036 | [-0.1116, -0.0957] | 1.3e-145 |

## chrF++ eng_Latn-snd_Arab

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | D | 7.52 | 15.18 | +7.66 | [+7.00, +8.34] |

## chrF++ snd_Arab-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | D | 24.11 | 33.24 | +9.13 | [+8.19, +10.05] |

<sub>Source: `runs/Qwen2.5-1.5B/sd/reports/comparisons.md`</sub>

### English retention: paired tests


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| english_baseline | english_baseline_lora | 1.0010 | 1.0418 | +0.0436 | [+0.0412, +0.0461] | 4e-134 |
| english_baseline | english_D | 1.0010 | 1.0198 | +0.0183 | [+0.0158, +0.0206] | 2e-53 |
| english_baseline_lora | english_D | 1.0418 | 1.0198 | -0.0254 | [-0.0274, -0.0235] | 4.8e-105 |

## chrF++ eng_Latn-snd_Arab

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|

## chrF++ snd_Arab-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|

<sub>Source: `runs/Qwen2.5-1.5B/sd/reports/comparisons_english.md`</sub>

### Human evaluation (blinded)

| file | a | b | n | preferred a / b / tie | fluency a → b | adequacy a → b | sign-test p |
|---|---|---|---|---|---|---|---|
| human_eval_baseline_lora_vs_D_eng_Latn-snd_Arab.results.json | baseline_lora | D | 50 | 2 / 30 / 18 | 1.10 → 1.94 | 1.10 → 1.82 | 0.000000 |

Raters: record who rated each language in the paper.

### Generation failure summary (FLORES outputs)

| condition | direction | empty | mean_rep4 | mean_script_mix | median_len_ratio |
|---|---|---|---|---|---|
| baseline | eng_Latn-snd_Arab | 0 | 0.942 | 0.001 | 2.403 |
| baseline_lora | eng_Latn-snd_Arab | 0 | 0.785 | 0.004 | 2.519 |
| baseline_lora | snd_Arab-eng_Latn | 0 | 0.059 | 0.000 | 1.087 |
| baseline | snd_Arab-eng_Latn | 0 | 0.036 | 0.000 | 1.074 |
| D | eng_Latn-snd_Arab | 0 | 0.350 | 0.060 | 1.207 |
| D | snd_Arab-eng_Latn | 0 | 0.030 | 0.000 | 1.012 |

<sub>Source: `runs/Qwen2.5-1.5B/sd/reports/failure_summary.csv`</sub>

### Looping vs per-word fragmentation (eng_Latn-snd_Arab)

| condition | Spearman ρ | p | loop rate Q1 → Q4 (least → most fragmented) |
|---|---|---|---|
| baseline | -0.062 | 0.049 | 99.6% → 100.0% → 99.2% → 99.6% |
| baseline_lora | +0.010 | 0.76 | 89.4% → 88.0% → 87.4% → 87.7% |
| D | +0.037 | 0.24 | 34.5% → 39.0% → 37.5% → 37.9% |

### Looping at matched length (eng_Latn-snd_Arab)

Loop rate (n) by the number of tokens each system's own tokenizer needs to write the reference.

| condition | 0–30 | 30–40 | 40–50 | 50–60 | 60–75 | 75–∞ | overall |
|---|---|---|---|---|---|---|---|
| baseline | – (12) | 100.0% (51) | 100.0% (89) | 99.2% (123) | 100.0% (219) | 99.4% (518) | 99.6% |
| baseline_lora | – (12) | 72.5% (51) | 87.6% (89) | 82.9% (123) | 86.3% (219) | 91.5% (518) | 88.1% |
| D | 26.3% (232) | 31.7% (290) | 40.4% (235) | 46.3% (162) | 58.6% (70) | 56.5% (23) | 37.3% |

Decomposition vs `baseline_lora` (reweighting to its length distribution; approximate):

| condition | ref loop rate | at ref lengths | actual | reduction | from per-step looping | from shorter sequences | share per-step | coverage |
|---|---|---|---|---|---|---|---|---|
| D | 88.1% | 52.7% | 37.3% | 50.9% | 35.4% | 15.4% | 69.6% | 100.0% |

Coverage = share of the reference system's sentences in length bins both systems occupy. Below 100%, the two parts need not sum exactly to the total.

### LoRA update size ‖ΔW‖

| condition | total | down_proj | gate_proj | k_proj | o_proj | q_proj | up_proj | v_proj |
|---|---|---|---|---|---|---|---|---|
| baseline_lora | 38.74 | 12.23 | 29.50 | 2.61 | 7.56 | 6.76 | 19.10 | 2.52 |
| D | 40.34 | 14.56 | 29.41 | 2.70 | 7.33 | 7.22 | 20.73 | 2.61 |

<sub>Source: `runs/Qwen2.5-1.5B/sd/reports/lora_update_norms.json`</sub>

### Training runs

Per-token losses are not comparable across tokenizers.

| condition | steps | final loss (per token) | trainable params | wall time (h) | λ (dev BPC) |
|---|---|---|---|---|---|
| baseline_lora | 1500 | 2.003 | 18464768 | 3.766 |  |
| D | 1500 | 3.508 | 21536768 | 3.807 | 10.000 |

## Pashto (`ps`)

### Data

- train_docs: 209,861
- train_chars: 509,179,435
- train_words: 109,688,029
- valid_sentences: 2,000
- test_sentences: 2,000
- filtering: {'docs_in': 220529, 'low_script_ratio': 1019, 'duplicate_doc': 696, 'too_short': 319, 'valid_leak_removed': 9831, 'test_leak_removed': 9275}

<sub>Source: `data/processed/ps/data_card.json`</sub>

### Baseline tokenization (validation split)

- n_sentences: 2000
- total_tokens: 154450
- tokens_per_word: 2.8546
- tokens_per_char: 0.6219
- chars_per_token: 1.6080
- bytes_per_token: 2.8520
- median_tokens_per_sentence: 66.0000
- target_word_fertility: 2.8616
- fragmented_target_word_rate: 0.8330

<sub>Source: `runs/Qwen2.5-1.5B/ps/_shared/baseline_audit/baseline_metrics.json`</sub>

### Compression (test split)

Definition: SCR = 1 - sum(tokens_ext)/sum(tokens_base); n = 2000 sentences.

| tokenizer | SCR | tokens/word | chars/token | median tokens/sent | per-sentence SCR median | Wilcoxon p |
|---|---|---|---|---|---|---|
| baseline |  | 2.8409 | 1.6158 | 67.5000 |  |  |
| v2000 | 0.4471 | 1.5708 | 2.9224 | 37.0000 | 0.4505 | 0.0000 |

<sub>Source: `runs/Qwen2.5-1.5B/ps/eval/compression_results.json`</sub>

### All conditions

| condition | budget | init | lora | scr | tokens_per_word | median_tokens_per_sentence |
|---|---|---|---|---|---|---|
| baseline | 0 |  | False | 0.0000 | 2.8409 | 67.5000 |
| H | 2000 | projection | True | 0.4471 | 1.5708 | 37.0000 |

<sub>Source: `runs/Qwen2.5-1.5B/ps/reports/final_results.csv`</sub>

## Amharic (`am`)

### Data

- train_docs: 194,754
- train_chars: 445,538,591
- train_words: 85,276,914
- valid_sentences: 2,000
- test_sentences: 2,000
- filtering: {'docs_in': 213906, 'low_script_ratio': 10248, 'duplicate_doc': 647, 'too_short': 260, 'valid_leak_removed': 15223, 'test_leak_removed': 15805}

<sub>Source: `data/processed/am/data_card.json`</sub>

### Baseline tokenization (validation split)

- n_sentences: 2000
- total_tokens: 251047
- tokens_per_word: 6.4902
- tokens_per_char: 1.2531
- chars_per_token: 0.7980
- bytes_per_token: 2.0464
- median_tokens_per_sentence: 89.0000
- target_word_fertility: 6.7411
- fragmented_target_word_rate: 1.0000

<sub>Source: `runs/Qwen2.5-1.5B/am/_shared/baseline_audit/baseline_metrics.json`</sub>

### Compression (test split)

Definition: SCR = 1 - sum(tokens_ext)/sum(tokens_base); n = 2000 sentences.

| tokenizer | SCR | tokens/word | chars/token | median tokens/sent | per-sentence SCR median | Wilcoxon p |
|---|---|---|---|---|---|---|
| baseline |  | 6.4778 | 0.8006 | 77.0000 |  |  |
| v2000 | 0.6168 | 2.4823 | 2.0891 | 30.0000 | 0.6111 | 0.0000 |

<sub>Source: `runs/Qwen2.5-1.5B/am/eval/compression_results.json`</sub>

### All conditions

| condition | budget | init | lora | scr | tokens_per_word | median_tokens_per_sentence |
|---|---|---|---|---|---|---|
| baseline | 0 |  | False | 0.0000 | 6.4778 | 77.0000 |
| H | 2000 | projection | True | 0.6168 | 2.4823 | 30.0000 |

<sub>Source: `runs/Qwen2.5-1.5B/am/reports/final_results.csv`</sub>

## Hindi (`hi`)

### Data

- train_docs: 276,975
- train_chars: 640,545,057
- train_words: 124,681,362
- valid_sentences: 2,000
- test_sentences: 2,000
- filtering: {'docs_in': 300000, 'low_script_ratio': 7558, 'too_short': 637, 'duplicate_doc': 3204, 'valid_leak_removed': 14764, 'test_leak_removed': 15103}

<sub>Source: `data/processed/hi/data_card.json`</sub>

### Baseline tokenization (validation split)

- n_sentences: 2000
- total_tokens: 167336
- tokens_per_word: 4.6075
- tokens_per_char: 0.9115
- chars_per_token: 1.0971
- bytes_per_token: 2.7922
- median_tokens_per_sentence: 69.0000
- target_word_fertility: 4.7325
- fragmented_target_word_rate: 1.0000

<sub>Source: `runs/Qwen2.5-1.5B/hi/_shared/baseline_audit/baseline_metrics.json`</sub>

### Compression (test split)

Definition: SCR = 1 - sum(tokens_ext)/sum(tokens_base); n = 2000 sentences.

| tokenizer | SCR | tokens/word | chars/token | median tokens/sent | per-sentence SCR median | Wilcoxon p |
|---|---|---|---|---|---|---|
| baseline |  | 4.6003 | 1.0995 | 71.0000 |  |  |
| v2000 | 0.4438 | 2.5587 | 1.9768 | 39.0000 | 0.4447 | 0.0000 |
| v2000_repaired | 0.6069 | 1.8083 | 2.7971 | 28.0000 | 0.6054 | 0.0000 |

<sub>Source: `runs/Qwen2.5-1.5B/hi/eval/compression_results.json`</sub>

### All conditions

| condition | budget | init | lora | scr | tokens_per_word | median_tokens_per_sentence |
|---|---|---|---|---|---|---|
| baseline | 0 |  | False | 0.0000 | 4.6003 | 71.0000 |
| H | 2000 | projection | True | 0.4438 | 2.5587 | 39.0000 |
| H_repair | 2000 | projection | True | 0.6069 | 1.8083 | 28.0000 |

<sub>Source: `runs/Qwen2.5-1.5B/hi/reports/final_results.csv`</sub>

## Tamil (`ta`)

### Data

- train_docs: 283,253
- train_chars: 820,576,705
- train_words: 91,085,043
- valid_sentences: 2,000
- test_sentences: 2,000
- filtering: {'docs_in': 300000, 'low_script_ratio': 4866, 'duplicate_doc': 60, 'too_short': 12, 'valid_leak_removed': 26877, 'test_leak_removed': 27104}

<sub>Source: `data/processed/ta/data_card.json`</sub>

### Baseline tokenization (validation split)

- n_sentences: 2000
- total_tokens: 192349
- tokens_per_word: 9.7679
- tokens_per_char: 1.0957
- chars_per_token: 0.9127
- bytes_per_token: 2.4917
- median_tokens_per_sentence: 80.0000
- target_word_fertility: 10.1386
- fragmented_target_word_rate: 1.0000

<sub>Source: `runs/Qwen2.5-1.5B/ta/_shared/baseline_audit/baseline_metrics.json`</sub>

### Compression (test split)

Definition: SCR = 1 - sum(tokens_ext)/sum(tokens_base); n = 2000 sentences.

| tokenizer | SCR | tokens/word | chars/token | median tokens/sent | per-sentence SCR median | Wilcoxon p |
|---|---|---|---|---|---|---|
| baseline |  | 9.7521 | 0.9184 | 81.5000 |  |  |
| v2000 | 0.5546 | 4.3435 | 2.0620 | 36.0000 | 0.5568 | 0.0000 |
| v2000_repaired | 0.7201 | 2.7300 | 3.2807 | 23.0000 | 0.7200 | 0.0000 |

<sub>Source: `runs/Qwen2.5-1.5B/ta/eval/compression_results.json`</sub>

### All conditions

| condition | budget | init | lora | scr | tokens_per_word | median_tokens_per_sentence |
|---|---|---|---|---|---|---|
| baseline | 0 |  | False | 0.0000 | 9.7521 | 81.5000 |
| H | 2000 | projection | True | 0.5546 | 4.3435 | 36.0000 |
| H_repair | 2000 | projection | True | 0.7201 | 2.7300 | 23.0000 |

<sub>Source: `runs/Qwen2.5-1.5B/ta/reports/final_results.csv`</sub>

## Pre-tokenizer ceiling

`ceiling_scr` = 1 − pre-tokens/tokens: the most any merge-based extension can compress under the inherited pre-tokenizer; `ceiling_scr_repaired` = the same after the script-scoped repair.

| lang | tokens_per_word | pretokens_per_word | tokens_per_char | ceiling_scr | n_sentences | pretokens_per_word_repaired | ceiling_scr_repaired | achieved_v2000 | share_of_ceiling_v2000 | achieved_v4000 | share_of_ceiling_v4000 | achieved_v8000 | share_of_ceiling_v8000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eng_Latn | 1.261 | 1.180 | 0.209 | 0.064 | 1012 |  |  |  |  |  |  |  |  |
| arb_Arab | 2.291 | 1.231 | 0.387 | 0.463 | 1012 | 1.171 | 0.489 |  |  |  |  |  |  |
| hin_Deva | 4.757 | 2.535 | 0.931 | 0.467 | 1012 | 1.140 | 0.760 |  |  |  |  |  |  |
| rus_Cyrl | 2.466 | 1.229 | 0.341 | 0.502 | 1012 |  |  |  |  |  |  |  |  |
| tha_Thai | 16.373 | 7.401 | 0.558 | 0.548 | 1012 | 1.416 | 0.913 |  |  |  |  |  |  |
| ben_Beng | 7.116 | 3.194 | 1.076 | 0.551 | 1012 | 1.179 | 0.834 |  |  |  |  |  |  |
| tam_Taml | 10.064 | 4.340 | 1.095 | 0.569 | 1012 | 1.231 | 0.878 |  |  |  |  |  |  |
| pes_Arab | 3.025 | 1.227 | 0.575 | 0.595 | 1012 | 1.222 | 0.596 |  |  |  |  |  |  |
| pbt_Arab | 2.927 | 1.124 | 0.631 | 0.616 | 1012 | 1.123 | 0.616 |  |  |  |  |  |  |
| snd_Arab | 3.099 | 1.153 | 0.657 | 0.628 | 1012 | 1.135 | 0.634 | 0.475 | 0.756 | 0.522 | 0.831 | 0.558 | 0.889 |
| urd_Arab | 3.145 | 1.128 | 0.669 | 0.641 | 1012 | 1.117 | 0.645 | 0.493 | 0.768 | 0.542 | 0.844 | 0.577 | 0.900 |
| tel_Telu | 11.406 | 3.936 | 1.458 | 0.655 | 1012 | 1.264 | 0.889 |  |  |  |  |  |  |
| mal_Mlym | 13.368 | 4.603 | 1.336 | 0.656 | 1012 | 1.278 | 0.904 |  |  |  |  |  |  |
| sin_Sinh | 9.173 | 2.959 | 1.449 | 0.677 | 1012 | 1.235 | 0.865 |  |  |  |  |  |  |
| amh_Ethi | 6.448 | 1.184 | 1.282 | 0.816 | 1012 | 1.184 | 0.816 |  |  |  |  |  |  |

<sub>Source: `runs/Qwen2.5-1.5B/analysis/pretokenizer_ceiling.csv`</sub>

