# HARF — all results

Generated 2026-10-22 18:52 by `harf document` · git `2ba56c1556` · model `Qwen/Qwen2.5-1.5B` @ `8faed761d45a263340a0528343f099c05c9a4323`

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
| D_750 | 2000 | projection | True | 0.4905 | 1.5500 | 29.0000 | 1.8176 | 342.1618 | -29.7906 |  |  | 1.0160 | 1.5035 |  |  |  |
| D_mean_750 | 2000 | mean | True | 0.4905 | 1.5500 | 29.0000 | 1.8278 | 353.4861 | -29.3989 |  |  |  |  |  |  |  |
| D_frozen_rows | 2000 | projection | True | 0.4905 | 1.5500 | 29.0000 | 2.2038 | 1182.3271 | -14.8716 | 6.7968 | 23.5593 | 1.0403 | 3.9350 |  |  |  |
| D_seed2 | 2000 | projection | True | 0.4905 | 1.5500 | 29.0000 | 1.7045 | 237.9826 | -34.1592 |  |  |  |  |  |  |  |
| E | 4000 | projection | True | 0.5373 | 1.4078 | 27.0000 | 1.6744 | 216.0555 | -35.3223 | 20.9776 | 37.8762 | 1.0109 | 0.9898 | 1002.5618 | 0.1133 | 2.2960 |
| F | 8000 | projection | True | 0.5738 | 1.2967 | 24.5000 | 1.6428 | 195.2016 | -36.5435 | 22.5012 | 39.1917 | 1.0134 | 1.2465 | 937.7875 | 0.1131 | 2.4506 |

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
| baseline_lora | D_seed2 | 1.8182 | 1.7045 | -0.1010 | [-0.1074, -0.0939] | 1.9e-175 |
| D | D_frozen_rows | 1.7088 | 2.2038 | +0.4647 | [+0.4563, +0.4732] | 0 |
| baseline_lora | E | 1.8182 | 1.6744 | -0.1227 | [-0.1303, -0.1150] | 3.6e-180 |
| baseline_lora | F | 1.8182 | 1.6428 | -0.1519 | [-0.1604, -0.1432] | 9.8e-195 |

## chrF++ eng_Latn-urd_Arab

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | D | 13.14 | 17.86 | +4.72 | [+4.00, +5.44] |
| baseline_lora | D_mean | 13.14 | 18.76 | +5.61 | [+4.93, +6.28] |
| D_mean | D | 18.76 | 17.86 | -0.89 | [-1.62, -0.19] |
| D | D_frozen_rows | 17.86 | 6.80 | -11.07 | [-11.78, -10.45] |
| baseline_lora | E | 13.14 | 20.98 | +7.83 | [+7.01, +8.60] |
| baseline_lora | F | 13.14 | 22.50 | +9.36 | [+8.50, +10.23] |

## chrF++ urd_Arab-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | D | 27.84 | 34.96 | +7.12 | [+6.30, +7.91] |
| baseline_lora | D_mean | 27.84 | 34.89 | +7.05 | [+6.16, +7.89] |
| D_mean | D | 34.89 | 34.96 | +0.07 | [-0.67, +0.84] |
| D | D_frozen_rows | 34.96 | 23.56 | -11.40 | [-12.18, -10.52] |
| baseline_lora | E | 27.84 | 37.88 | +10.04 | [+9.20, +10.83] |
| baseline_lora | F | 27.84 | 39.19 | +11.35 | [+10.45, +12.20] |

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

### FLORES at an equal character budget

Each system's outputs are truncated to the same number of characters (the longest output the base model could produce under the 256-token cap). `hit cap` = share of outputs that ran into the token cap, i.e. the model did not stop by itself.

#### eng_Latn-urd_Arab (character cap = 588)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 8.90 | 8.90 | 85% | 0% | 0.728 | 2.57 |
| baseline_lora | 13.14 | 13.15 | 68% | 0% | 0.546 | 2.24 |
| D | 17.86 | 19.28 | 34% | 30% | 0.298 | 1.22 |
| D_mean | 18.76 | 19.94 | 27% | 23% | 0.247 | 1.16 |
| D_frozen_rows | 6.80 | 7.03 | 65% | 36% | 0.701 | 3.89 |
| E | 20.98 | 22.52 | 20% | 19% | 0.180 | 1.08 |
| F | 22.50 | 24.15 | 14% | 14% | 0.132 | 1.04 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| D | 13.15 | 19.28 | +6.14 | [+5.45, +6.83] |
| D_mean | 13.15 | 19.94 | +6.80 | [+6.15, +7.44] |
| D_frozen_rows | 13.15 | 7.03 | -6.12 | [-6.54, -5.70] |
| E | 13.15 | 22.52 | +9.37 | [+8.61, +10.08] |
| F | 13.15 | 24.15 | +11.00 | [+10.29, +11.71] |

#### urd_Arab-eng_Latn (character cap = 1452)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 29.73 | 29.73 | 1% | 0% | 0.018 | 1.02 |
| baseline_lora | 27.84 | 27.84 | 3% | 0% | 0.039 | 1.06 |
| D | 34.96 | 34.96 | 2% | 0% | 0.026 | 1.01 |
| D_mean | 34.89 | 34.89 | 2% | 0% | 0.025 | 0.99 |
| D_frozen_rows | 23.56 | 23.57 | 4% | 0% | 0.067 | 1.10 |
| E | 37.88 | 37.88 | 1% | 0% | 0.014 | 0.97 |
| F | 39.19 | 39.19 | 1% | 0% | 0.013 | 0.97 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| D | 27.84 | 34.96 | +7.11 | [+6.29, +7.90] |
| D_mean | 27.84 | 34.89 | +7.05 | [+6.17, +7.90] |
| D_frozen_rows | 27.84 | 23.57 | -4.27 | [-5.04, -3.54] |
| E | 27.84 | 37.88 | +10.03 | [+9.20, +10.83] |
| F | 27.84 | 39.19 | +11.35 | [+10.45, +12.19] |


<sub>Source: `runs/Qwen2.5-1.5B/ur/reports/flores_equal_chars.md`</sub>

### Generation failure summary (FLORES outputs)

| condition | direction | empty | mean_rep4 | mean_script_mix | median_len_ratio |
|---|---|---|---|---|---|
| baseline | eng_Latn-urd_Arab | 0 | 0.728 | 0.003 | 2.575 |
| baseline_lora | eng_Latn-urd_Arab | 0 | 0.546 | 0.003 | 2.239 |
| baseline_lora | urd_Arab-eng_Latn | 0 | 0.039 | 0.000 | 1.061 |
| baseline | urd_Arab-eng_Latn | 0 | 0.018 | 0.000 | 1.024 |
| D | eng_Latn-urd_Arab | 0 | 0.311 | 0.019 | 1.217 |
| D_frozen_rows | eng_Latn-urd_Arab | 0 | 0.707 | 0.048 | 4.069 |
| D_frozen_rows | urd_Arab-eng_Latn | 0 | 0.067 | 0.000 | 1.101 |
| D_mean | eng_Latn-urd_Arab | 0 | 0.257 | 0.024 | 1.164 |
| D_mean | urd_Arab-eng_Latn | 0 | 0.025 | 0.000 | 0.992 |
| D | urd_Arab-eng_Latn | 0 | 0.026 | 0.000 | 1.008 |
| E | eng_Latn-urd_Arab | 0 | 0.191 | 0.040 | 1.085 |
| E | urd_Arab-eng_Latn | 0 | 0.014 | 0.000 | 0.972 |
| F | eng_Latn-urd_Arab | 0 | 0.140 | 0.081 | 1.043 |
| F | urd_Arab-eng_Latn | 0 | 0.013 | 0.000 | 0.975 |

<sub>Source: `runs/Qwen2.5-1.5B/ur/reports/failure_summary.csv`</sub>

### Looping vs per-word fragmentation (eng_Latn-urd_Arab)

| condition | Spearman ρ | p | loop rate Q1 → Q4 (least → most fragmented) |
|---|---|---|---|
| baseline | -0.038 | 0.23 | 83.8% → 86.2% → 86.3% → 81.9% |
| baseline_lora | -0.011 | 0.74 | 64.4% → 68.5% → 72.3% → 62.2% |
| D | -0.002 | 0.96 | 32.0% → 38.2% → 32.4% → 34.9% |
| D_mean | +0.013 | 0.68 | 26.1% → 27.2% → 29.7% → 29.3% |
| D_frozen_rows | -0.064 | 0.041 | 77.1% → 83.1% → 81.2% → 75.5% |
| E | -0.041 | 0.2 | 18.6% → 21.7% → 22.7% → 20.1% |
| F | +0.001 | 0.98 | 9.5% → 18.5% → 13.3% → 16.5% |

### Looping at matched length (eng_Latn-urd_Arab)

Loop rate (n) by the number of tokens each system's own tokenizer needs to write the reference.

| condition | 0–30 | 30–40 | 40–50 | 50–60 | 60–75 | 75–∞ | overall |
|---|---|---|---|---|---|---|---|
| baseline | – (4) | 68.0% (25) | 66.7% (69) | 82.0% (100) | 85.5% (179) | 87.4% (635) | 84.6% |
| baseline_lora | – (4) | 36.0% (25) | 56.5% (69) | 58.0% (100) | 64.2% (179) | 71.5% (635) | 66.9% |
| D | 23.8% (185) | 29.1% (251) | 35.0% (263) | 42.9% (170) | 41.0% (105) | 60.5% (38) | 34.4% |
| D_mean | 20.0% (185) | 25.5% (251) | 27.0% (263) | 29.4% (170) | 38.1% (105) | 57.9% (38) | 28.1% |
| D_frozen_rows | 71.9% (185) | 80.1% (251) | 78.7% (263) | 83.5% (170) | 81.9% (105) | 86.8% (38) | 79.2% |
| E | 11.0% (255) | 14.7% (299) | 20.6% (247) | 35.1% (134) | 53.4% (58) | 47.4% (19) | 20.8% |
| F | 12.0% (325) | 12.7% (331) | 16.8% (214) | 17.4% (86) | 25.0% (44) | – (12) | 14.4% |

Decomposition vs `baseline_lora` (reweighting to its length distribution; approximate):

| condition | ref loop rate | at ref lengths | actual | reduction | from per-step looping | from shorter sequences | share per-step | coverage |
|---|---|---|---|---|---|---|---|---|
| D | 66.9% | 52.7% | 34.4% | 32.5% | 14.2% | 18.3% | 43.8% | 100.0% |
| D_mean | 66.9% | 48.5% | 28.1% | 38.8% | 18.4% | 20.5% | 47.3% | 100.0% |
| D_frozen_rows | 66.9% | 84.9% | 79.2% | -12.4% | -18.0% | 5.6% | 145.4% | 100.0% |
| E | 66.9% | 44.5% | 20.8% | 46.1% | 22.4% | 23.7% | 48.6% | 100.0% |
| F | 66.9% | 23.3% | 14.4% | 52.5% | 43.6% | 8.9% | 83.0% | 100.0% |

Coverage = share of the reference system's sentences in length bins both systems occupy. Below 100%, the two parts need not sum exactly to the total.

### LoRA update size ‖ΔW‖

| condition | total | down_proj | gate_proj | k_proj | o_proj | q_proj | up_proj | v_proj |
|---|---|---|---|---|---|---|---|---|
| baseline_lora | 33.92 | 10.89 | 24.64 | 2.37 | 7.10 | 6.06 | 18.07 | 2.38 |
| D | 36.95 | 14.08 | 26.01 | 2.58 | 7.17 | 6.80 | 19.49 | 2.54 |
| D_mean | 36.47 | 13.49 | 25.68 | 2.52 | 7.16 | 6.70 | 19.48 | 2.57 |
| D_750 | 24.07 | 8.31 | 17.44 | 1.79 | 4.57 | 4.61 | 12.57 | 1.66 |
| D_mean_750 | 23.76 | 7.91 | 17.22 | 1.72 | 4.61 | 4.55 | 12.56 | 1.68 |
| D_frozen_rows | 46.13 | 16.91 | 33.43 | 3.09 | 8.92 | 8.40 | 23.60 | 2.81 |
| D_seed2 | 37.01 | 14.08 | 25.96 | 2.57 | 7.09 | 6.85 | 19.68 | 2.55 |
| E | 36.43 | 14.11 | 25.47 | 2.63 | 7.01 | 7.06 | 19.15 | 2.56 |
| F | 35.83 | 13.87 | 24.96 | 2.68 | 6.88 | 7.22 | 18.85 | 2.53 |

<sub>Source: `runs/Qwen2.5-1.5B/ur/reports/lora_update_norms.json`</sub>

### Forced fragmentation of English output (urd_Arab-eng_Latn, n=200)

| constraint | chrF++ | mean rep4 | loop rate | median len ratio | output tokens/word |
|---|---|---|---|---|---|
| unconstrained | 30.90 | 0.013 | 0.5% | 1.03 | 1.23 |
| max3chars | 3.35 | 0.715 | 76.0% | 6.08 | 3.68 |
| max2chars | 1.51 | 0.724 | 75.5% | 7.95 | 5.31 |
| max1chars | 0.70 | 0.584 | 59.5% | 11.72 | 6.01 |

<sub>Source: `runs/Qwen2.5-1.5B/ur/eval/fragdecode_urd_Arab-eng_Latn.json`</sub>

### Training runs

Per-token losses are not comparable across tokenizers.

| condition | steps | final loss (per token) | trainable params | wall time (h) | λ (dev BPC) |
|---|---|---|---|---|---|
| baseline_lora | 1500 | 1.727 | 18464768 | 3.947 |  |
| D | 1500 | 2.925 | 21536768 | 3.402 | 1.000 |
| D_mean | 1500 | 2.930 | 21536768 | 3.405 |  |
| D_750 | 750 | 3.234 | 21536768 | 1.753 | 1.000 |
| D_mean_750 | 750 | 3.247 | 21536768 | 1.758 |  |
| D_frozen_rows | 1500 | 3.927 | 18464768 | 3.611 | 1.000 |
| D_seed2 | 1500 | 2.722 | 21536768 | 3.649 | 1.000 |
| E | 1500 | 3.419 | 24608768 | 4.496 | 1.000 |
| F | 1500 | 3.369 | 30752768 | 4.511 | 1.000 |

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

### FLORES at an equal character budget

Each system's outputs are truncated to the same number of characters (the longest output the base model could produce under the 256-token cap). `hit cap` = share of outputs that ran into the token cap, i.e. the model did not stop by itself.

#### eng_Latn-snd_Arab (character cap = 409)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 3.65 | 3.65 | 99% | 0% | 0.942 | 2.40 |
| baseline_lora | 7.52 | 7.53 | 88% | 2% | 0.785 | 2.52 |
| D | 15.18 | 17.45 | 36% | 37% | 0.324 | 1.21 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| D | 7.53 | 17.45 | +9.93 | [+9.29, +10.58] |

#### snd_Arab-eng_Latn (character cap = 1489)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 19.63 | 19.63 | 2% | 0% | 0.036 | 1.07 |
| baseline_lora | 24.11 | 24.11 | 4% | 0% | 0.059 | 1.09 |
| D | 33.24 | 33.24 | 3% | 0% | 0.030 | 1.01 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| D | 24.11 | 33.24 | +9.12 | [+8.19, +10.04] |


<sub>Source: `runs/Qwen2.5-1.5B/sd/reports/flores_equal_chars.md`</sub>

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

| condition | budget | init | lora | scr | tokens_per_word | median_tokens_per_sentence | bpc | word_ppl | bpc_change_pct | chrf++ eng_Latn-pbt_Arab | chrf++ pbt_Arab-eng_Latn | english_bpc | english_bpc_change_pct | emit_ms_median | tokenizer_ms_per_sentence | latency_speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 |  | False | 0.0000 | 2.8409 | 67.5000 | 3.2165 | 27844.8852 | 0.0000 | 5.4225 | 22.5608 | 1.0010 | 0.0000 | 2128.8768 | 0.1281 | 1.0000 |
| baseline_lora | 0 |  | True | 0.0000 | 2.8409 | 67.5000 | 2.0865 | 764.2393 | -35.1317 | 8.2093 | 24.0690 | 1.0100 | 0.9003 | 2211.0992 | 0.1218 | 0.9660 |
| H | 2000 | projection | True | 0.4471 | 1.5708 | 37.0000 | 1.9340 | 470.4970 | -39.8715 | 9.7957 | 30.4040 | 1.0162 | 1.5206 | 1202.7633 | 0.1197 | 1.7490 |

<sub>Source: `runs/Qwen2.5-1.5B/ps/reports/final_results.csv`</sub>

### Pairwise comparisons


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| baseline_lora | H | 2.0865 | 1.9340 | -0.1438 | [-0.1506, -0.1370] | 2e-229 |

## chrF++ eng_Latn-pbt_Arab

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | H | 8.21 | 9.80 | +1.59 | [+1.28, +1.94] |

## chrF++ pbt_Arab-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | H | 24.07 | 30.40 | +6.33 | [+5.42, +7.24] |

<sub>Source: `runs/Qwen2.5-1.5B/ps/reports/comparisons.md`</sub>

### English retention: paired tests


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| english_baseline_lora | english_H | 1.0100 | 1.0162 | +0.0065 | [+0.0043, +0.0088] | 6.6e-06 |

## chrF++ eng_Latn-pbt_Arab

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|

## chrF++ pbt_Arab-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|

<sub>Source: `runs/Qwen2.5-1.5B/ps/reports/comparisons_english.md`</sub>

### FLORES at an equal character budget

Each system's outputs are truncated to the same number of characters (the longest output the base model could produce under the 256-token cap). `hit cap` = share of outputs that ran into the token cap, i.e. the model did not stop by itself.

#### eng_Latn-pbt_Arab (character cap = 590)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 5.42 | 5.42 | 98% | 0% | 0.819 | 2.80 |
| baseline_lora | 8.21 | 8.21 | 97% | 0% | 0.830 | 2.95 |
| H | 9.80 | 10.72 | 65% | 56% | 0.661 | 4.15 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| H | 8.21 | 10.72 | +2.51 | [+2.18, +2.88] |

#### pbt_Arab-eng_Latn (character cap = 1559)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 22.56 | 22.56 | 2% | 0% | 0.025 | 0.97 |
| baseline_lora | 24.07 | 24.07 | 6% | 0% | 0.070 | 1.12 |
| H | 30.40 | 30.41 | 4% | 0% | 0.047 | 1.06 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| H | 24.07 | 30.41 | +6.34 | [+5.43, +7.24] |


<sub>Source: `runs/Qwen2.5-1.5B/ps/reports/flores_equal_chars.md`</sub>

### Generation failure summary (FLORES outputs)

| condition | direction | empty | mean_rep4 | mean_script_mix | median_len_ratio |
|---|---|---|---|---|---|
| baseline | eng_Latn-pbt_Arab | 0 | 0.819 | 0.003 | 2.799 |
| baseline_lora | eng_Latn-pbt_Arab | 0 | 0.830 | 0.005 | 2.948 |
| baseline_lora | pbt_Arab-eng_Latn | 0 | 0.070 | 0.000 | 1.119 |
| baseline | pbt_Arab-eng_Latn | 0 | 0.025 | 0.000 | 0.972 |
| H | eng_Latn-pbt_Arab | 0 | 0.678 | 0.041 | 4.855 |
| H | pbt_Arab-eng_Latn | 0 | 0.047 | 0.000 | 1.061 |

<sub>Source: `runs/Qwen2.5-1.5B/ps/reports/failure_summary.csv`</sub>

### Looping vs per-word fragmentation (eng_Latn-pbt_Arab)

| condition | Spearman ρ | p | loop rate Q1 → Q4 (least → most fragmented) |
|---|---|---|---|
| baseline | +0.033 | 0.3 | 87.0% → 89.8% → 91.4% → 89.2% |
| baseline_lora | -0.035 | 0.27 | 95.7% → 97.6% → 96.9% → 97.2% |
| H | +0.034 | 0.28 | 67.6% → 77.2% → 79.7% → 78.3% |

### Looping at matched length (eng_Latn-pbt_Arab)

Loop rate (n) by the number of tokens each system's own tokenizer needs to write the reference.

| condition | 0–30 | 30–40 | 40–50 | 50–60 | 60–75 | 75–∞ | overall |
|---|---|---|---|---|---|---|---|
| baseline | – (8) | 79.5% (44) | 93.1% (101) | 87.6% (137) | 88.9% (234) | 90.6% (488) | 89.3% |
| baseline_lora | – (8) | 93.2% (44) | 95.0% (101) | 97.1% (137) | 95.7% (234) | 98.0% (488) | 96.8% |
| H | 60.4% (187) | 74.2% (264) | 78.7% (230) | 80.2% (182) | 87.1% (116) | 87.9% (33) | 75.7% |

Decomposition vs `baseline_lora` (reweighting to its length distribution; approximate):

| condition | ref loop rate | at ref lengths | actual | reduction | from per-step looping | from shorter sequences | share per-step | coverage |
|---|---|---|---|---|---|---|---|---|
| H | 96.8% | 84.9% | 75.7% | 21.1% | 11.9% | 9.2% | 56.3% | 100.0% |

Coverage = share of the reference system's sentences in length bins both systems occupy. Below 100%, the two parts need not sum exactly to the total.

### Training runs

Per-token losses are not comparable across tokenizers.

| condition | steps | final loss (per token) | trainable params | wall time (h) | λ (dev BPC) |
|---|---|---|---|---|---|
| baseline_lora | 750 | 2.057 | 18464768 | 1.843 |  |
| H | 750 | 3.285 | 21536768 | 1.864 | 1.000 |

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

| condition | budget | init | lora | scr | tokens_per_word | median_tokens_per_sentence | bpc | word_ppl | bpc_change_pct | chrf++ eng_Latn-amh_Ethi | chrf++ amh_Ethi-eng_Latn | english_bpc | english_bpc_change_pct | emit_ms_median | tokenizer_ms_per_sentence | latency_speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 |  | False | 0.0000 | 6.4778 | 77.0000 | 4.6397 | 17502695.1580 | 0.0000 | 1.8518 | 16.0322 | 1.0010 | 0.0000 | 2679.6050 | 0.1301 | 1.0000 |
| baseline_lora | 0 |  | True | 0.0000 | 6.4778 | 77.0000 | 3.0047 | 49057.1366 | -35.2391 | 3.4276 | 14.2002 | 1.0073 | 0.6346 | 2712.7935 | 0.1240 | 0.9897 |
| H | 2000 | projection | True | 0.6168 | 2.4823 | 30.0000 | 2.7458 | 19338.9700 | -40.8205 | 2.6409 | 19.5874 | 1.0044 | 0.3447 | 1059.6094 | 0.1096 | 2.6082 |

<sub>Source: `runs/Qwen2.5-1.5B/am/reports/final_results.csv`</sub>

### Pairwise comparisons


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| baseline_lora | H | 3.0047 | 2.7458 | -0.2508 | [-0.2615, -0.2410] | 1.9e-257 |

## chrF++ amh_Ethi-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | H | 14.20 | 19.59 | +5.39 | [+4.74, +6.05] |

## chrF++ eng_Latn-amh_Ethi

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | H | 3.43 | 2.64 | -0.79 | [-1.00, -0.58] |

<sub>Source: `runs/Qwen2.5-1.5B/am/reports/comparisons.md`</sub>

### FLORES at an equal character budget

Each system's outputs are truncated to the same number of characters (the longest output the base model could produce under the 256-token cap). `hit cap` = share of outputs that ran into the token cap, i.e. the model did not stop by itself.

#### amh_Ethi-eng_Latn (character cap = 1286)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 16.03 | 16.03 | 2% | 0% | 0.028 | 0.94 |
| baseline_lora | 14.20 | 14.28 | 17% | 5% | 0.141 | 1.31 |
| H | 19.59 | 19.66 | 7% | 2% | 0.078 | 1.13 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| H | 14.28 | 19.66 | +5.37 | [+4.72, +6.04] |

#### eng_Latn-amh_Ethi (character cap = 758)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 1.85 | 1.85 | 86% | 0% | 0.856 | 2.52 |
| baseline_lora | 3.43 | 3.43 | 81% | 0% | 0.725 | 2.41 |
| H | 2.64 | 2.66 | 69% | 4% | 0.615 | 4.62 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| H | 3.43 | 2.66 | -0.76 | [-0.97, -0.56] |


<sub>Source: `runs/Qwen2.5-1.5B/am/reports/flores_equal_chars.md`</sub>

### Generation failure summary (FLORES outputs)

| condition | direction | empty | mean_rep4 | mean_script_mix | median_len_ratio |
|---|---|---|---|---|---|
| baseline | amh_Ethi-eng_Latn | 0 | 0.028 | 0.000 | 0.941 |
| baseline | eng_Latn-amh_Ethi | 0 | 0.856 | 0.007 | 2.523 |
| baseline_lora | amh_Ethi-eng_Latn | 0 | 0.141 | 0.003 | 1.311 |
| baseline_lora | eng_Latn-amh_Ethi | 0 | 0.725 | 0.001 | 2.414 |
| H | amh_Ethi-eng_Latn | 0 | 0.079 | 0.000 | 1.128 |
| H | eng_Latn-amh_Ethi | 0 | 0.616 | 0.197 | 4.628 |

<sub>Source: `runs/Qwen2.5-1.5B/am/reports/failure_summary.csv`</sub>

### Looping vs per-word fragmentation (eng_Latn-amh_Ethi)

| condition | Spearman ρ | p | loop rate Q1 → Q4 (least → most fragmented) |
|---|---|---|---|
| baseline | -0.008 | 0.81 | 94.5% → 96.9% → 94.5% → 96.0% |
| baseline_lora | -0.024 | 0.45 | 88.9% → 93.3% → 92.9% → 92.0% |
| H | +0.046 | 0.14 | 63.6% → 67.7% → 67.1% → 69.2% |

### Looping at matched length (eng_Latn-amh_Ethi)

Loop rate (n) by the number of tokens each system's own tokenizer needs to write the reference.

| condition | 0–30 | 30–40 | 40–50 | 50–60 | 60–75 | 75–∞ | overall |
|---|---|---|---|---|---|---|---|
| baseline | – (0) | – (4) | 100.0% (23) | 92.7% (41) | 95.3% (107) | 95.6% (837) | 95.5% |
| baseline_lora | – (0) | – (4) | 87.0% (23) | 90.2% (41) | 89.7% (107) | 92.2% (837) | 91.8% |
| H | 59.8% (189) | 69.3% (280) | 68.2% (242) | 68.5% (178) | 69.1% (94) | 62.1% (29) | 66.9% |

Decomposition vs `baseline_lora` (reweighting to its length distribution; approximate):

| condition | ref loop rate | at ref lengths | actual | reduction | from per-step looping | from shorter sequences | share per-step | coverage |
|---|---|---|---|---|---|---|---|---|
| H | 91.8% | 63.2% | 66.9% | 24.9% | 28.6% | -3.7% | 114.7% | 100.0% |

Coverage = share of the reference system's sentences in length bins both systems occupy. Below 100%, the two parts need not sum exactly to the total.

### Training runs

Per-token losses are not comparable across tokenizers.

| condition | steps | final loss (per token) | trainable params | wall time (h) | λ (dev BPC) |
|---|---|---|---|---|---|
| baseline_lora | 750 | 1.554 | 18464768 | 1.815 |  |
| H | 750 | 3.608 | 21536768 | 1.837 | 1000.000 |

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

| condition | budget | init | lora | scr | tokens_per_word | median_tokens_per_sentence | bpc | word_ppl | bpc_change_pct | chrf++ eng_Latn-hin_Deva | chrf++ hin_Deva-eng_Latn | english_bpc | english_bpc_change_pct | emit_ms_median | tokenizer_ms_per_sentence | latency_speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 |  | False | 0.0000 | 4.6003 | 71.0000 | 2.2080 | 2301.2316 | 0.0000 | 15.6840 | 36.6208 | 1.0010 | 0.0000 | 2266.5546 | 0.1648 | 1.0000 |
| baseline_lora | 0 |  | True | 0.0000 | 4.6003 | 71.0000 | 1.8927 | 762.0580 | -14.2766 | 13.5773 | 32.9692 | 0.9934 | -0.7519 | 2252.8615 | 0.1565 | 1.0143 |
| H | 2000 | projection | True | 0.4438 | 2.5587 | 39.0000 | 1.9032 | 790.4885 | -13.8034 | 12.1985 | 31.7249 | 0.9948 | -0.6180 | 1144.8966 | 0.1557 | 1.8496 |
| H_repair | 2000 | projection | True | 0.6069 | 1.8083 | 28.0000 | 1.8361 | 624.8251 | -16.8414 | 15.6392 | 34.8535 | 0.9978 | -0.3197 | 895.3874 | 0.1164 | 2.5637 |

<sub>Source: `runs/Qwen2.5-1.5B/hi/reports/final_results.csv`</sub>

### Pairwise comparisons


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| baseline_lora | H | 1.8927 | 1.9032 | +0.0311 | [+0.0234, +0.0389] | 1.8e-10 |
| baseline_lora | H_repair | 1.8927 | 1.8361 | -0.0468 | [-0.0543, -0.0394] | 2.7e-49 |
| H | H_repair | 1.9032 | 1.8361 | -0.0779 | [-0.0848, -0.0715] | 7.4e-121 |

## chrF++ eng_Latn-hin_Deva

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | H | 13.58 | 12.20 | -1.38 | [-1.93, -0.86] |
| baseline_lora | H_repair | 13.58 | 15.64 | +2.06 | [+1.36, +2.74] |
| H | H_repair | 12.20 | 15.64 | +3.44 | [+2.82, +4.07] |

## chrF++ hin_Deva-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | H | 32.97 | 31.72 | -1.24 | [-2.21, -0.34] |
| baseline_lora | H_repair | 32.97 | 34.85 | +1.88 | [+1.09, +2.75] |
| H | H_repair | 31.72 | 34.85 | +3.13 | [+2.22, +4.12] |

<sub>Source: `runs/Qwen2.5-1.5B/hi/reports/comparisons.md`</sub>

### English retention: paired tests


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| baseline_lora | H | 1.8927 | 1.9032 | +0.0311 | [+0.0234, +0.0389] | 1.8e-10 |
| baseline_lora | H_repair | 1.8927 | 1.8361 | -0.0468 | [-0.0543, -0.0394] | 2.7e-49 |
| H | H_repair | 1.9032 | 1.8361 | -0.0779 | [-0.0848, -0.0715] | 7.4e-121 |

## chrF++ eng_Latn-hin_Deva

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | H | 13.58 | 12.20 | -1.38 | [-1.93, -0.86] |
| baseline_lora | H_repair | 13.58 | 15.64 | +2.06 | [+1.36, +2.74] |
| H | H_repair | 12.20 | 15.64 | +3.44 | [+2.82, +4.07] |

## chrF++ hin_Deva-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | H | 32.97 | 31.72 | -1.24 | [-2.21, -0.34] |
| baseline_lora | H_repair | 32.97 | 34.85 | +1.88 | [+1.09, +2.75] |
| H | H_repair | 31.72 | 34.85 | +3.13 | [+2.22, +4.12] |

<sub>Source: `runs/Qwen2.5-1.5B/hi/reports/comparisons_english.md`</sub>

### FLORES at an equal character budget

Each system's outputs are truncated to the same number of characters (the longest output the base model could produce under the 256-token cap). `hit cap` = share of outputs that ran into the token cap, i.e. the model did not stop by itself.

#### eng_Latn-hin_Deva (character cap = 311)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 15.68 | 15.68 | 54% | 0% | 0.382 | 1.47 |
| baseline_lora | 13.58 | 13.58 | 66% | 0% | 0.484 | 1.71 |
| H | 12.20 | 14.24 | 79% | 82% | 0.570 | 2.29 |
| H_repair | 15.64 | 19.15 | 38% | 41% | 0.303 | 1.26 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| H | 13.58 | 14.24 | +0.67 | [+0.09, +1.20] |
| H_repair | 13.58 | 19.15 | +5.57 | [+4.94, +6.20] |

#### hin_Deva-eng_Latn (character cap = 1419)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 36.62 | 36.62 | 2% | 0% | 0.018 | 0.98 |
| baseline_lora | 32.97 | 32.97 | 2% | 0% | 0.031 | 1.04 |
| H | 31.72 | 31.74 | 3% | 0% | 0.034 | 1.03 |
| H_repair | 34.85 | 34.85 | 1% | 0% | 0.018 | 1.02 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| H | 32.97 | 31.74 | -1.23 | [-2.18, -0.32] |
| H_repair | 32.97 | 34.85 | +1.88 | [+1.09, +2.75] |


<sub>Source: `runs/Qwen2.5-1.5B/hi/reports/flores_equal_chars.md`</sub>

### Generation failure summary (FLORES outputs)

| condition | direction | empty | mean_rep4 | mean_script_mix | median_len_ratio |
|---|---|---|---|---|---|
| baseline | eng_Latn-hin_Deva | 0 | 0.382 | 0.004 | 1.469 |
| baseline | hin_Deva-eng_Latn | 0 | 0.018 | 0.000 | 0.981 |
| baseline_lora | eng_Latn-hin_Deva | 0 | 0.484 | 0.003 | 1.712 |
| baseline_lora | hin_Deva-eng_Latn | 0 | 0.031 | 0.000 | 1.038 |
| H | eng_Latn-hin_Deva | 0 | 0.662 | 0.029 | 3.469 |
| H | hin_Deva-eng_Latn | 0 | 0.034 | 0.000 | 1.034 |
| H_repair | eng_Latn-hin_Deva | 0 | 0.370 | 0.042 | 1.290 |
| H_repair | hin_Deva-eng_Latn | 0 | 0.018 | 0.000 | 1.021 |

<sub>Source: `runs/Qwen2.5-1.5B/hi/reports/failure_summary.csv`</sub>

### Looping vs per-word fragmentation (eng_Latn-hin_Deva)

| condition | Spearman ρ | p | loop rate Q1 → Q4 (least → most fragmented) |
|---|---|---|---|
| baseline | -0.016 | 0.62 | 43.5% → 49.0% → 45.7% → 43.3% |
| baseline_lora | +0.021 | 0.5 | 50.6% → 57.7% → 63.8% → 55.6% |
| H | -0.032 | 0.31 | 79.4% → 83.4% → 78.3% → 82.5% |
| H_repair | +0.007 | 0.83 | 37.5% → 41.1% → 44.9% → 39.7% |

### Looping at matched length (eng_Latn-hin_Deva)

Loop rate (n) by the number of tokens each system's own tokenizer needs to write the reference.

| condition | 0–30 | 30–40 | 40–50 | 50–60 | 60–75 | 75–∞ | overall |
|---|---|---|---|---|---|---|---|
| baseline | – (0) | – (2) | 46.7% (15) | 38.1% (21) | 36.7% (98) | 46.5% (876) | 45.4% |
| baseline_lora | – (0) | – (2) | 66.7% (15) | 28.6% (21) | 54.1% (98) | 57.8% (876) | 56.9% |
| H | 77.3% (22) | 77.8% (99) | 79.2% (144) | 80.8% (167) | 78.9% (261) | 84.6% (319) | 80.9% |
| H_repair | 36.1% (147) | 32.6% (233) | 40.2% (254) | 41.1% (190) | 49.6% (141) | 72.3% (47) | 40.8% |

Decomposition vs `baseline_lora` (reweighting to its length distribution; approximate):

| condition | ref loop rate | at ref lengths | actual | reduction | from per-step looping | from shorter sequences | share per-step | coverage |
|---|---|---|---|---|---|---|---|---|
| H | 56.9% | 83.9% | 80.9% | -24.0% | -27.0% | 3.0% | 112.4% | 100.0% |
| H_repair | 56.9% | 68.9% | 40.8% | 16.1% | -12.0% | 28.1% | -74.6% | 100.0% |

Coverage = share of the reference system's sentences in length bins both systems occupy. Below 100%, the two parts need not sum exactly to the total.

### Training runs

Per-token losses are not comparable across tokenizers.

| condition | steps | final loss (per token) | trainable params | wall time (h) | λ (dev BPC) |
|---|---|---|---|---|---|
| baseline_lora | 750 | 1.193 | 18464768 | 2.072 |  |
| H | 750 | 2.213 | 21536768 | 2.096 | 0.100 |
| H_repair | 750 | 3.237 | 21536768 | 2.091 | 1.000 |

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

| condition | budget | init | lora | scr | tokens_per_word | median_tokens_per_sentence | bpc | word_ppl | bpc_change_pct | chrf++ eng_Latn-tam_Taml | chrf++ tam_Taml-eng_Latn | english_bpc | english_bpc_change_pct | emit_ms_median | tokenizer_ms_per_sentence | latency_speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0 |  | False | 0.0000 | 9.7521 | 81.5000 | 2.4630 | 4370064.9542 | 0.0000 | 12.0449 | 18.6741 | 1.0010 | 0.0000 | 2644.8170 | 0.1485 | 1.0000 |
| baseline_lora | 0 |  | True | 0.0000 | 9.7521 | 81.5000 | 1.9142 | 144846.3576 | -22.2812 | 12.5222 | 17.7888 | 1.0060 | 0.5034 | 2649.9252 | 0.1423 | 1.0179 |
| H | 2000 | projection | True | 0.5546 | 4.3435 | 36.0000 | 1.8318 | 86877.6474 | -25.6243 | 11.2061 | 19.3838 | 1.0037 | 0.2747 | 1159.9886 | 0.1269 | 2.3093 |
| H_repair | 2000 | projection | True | 0.7201 | 2.7300 | 23.0000 | 1.7645 | 57202.2268 | -28.3575 | 10.3590 | 23.4653 | 1.0085 | 0.7558 | 722.0641 | 0.0883 | 3.6355 |

<sub>Source: `runs/Qwen2.5-1.5B/ta/reports/final_results.csv`</sub>

### Pairwise comparisons


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| baseline_lora | H | 1.9142 | 1.8318 | -0.0646 | [-0.0720, -0.0571] | 1e-88 |
| baseline_lora | H_repair | 1.9142 | 1.7645 | -0.1410 | [-0.1490, -0.1332] | 1.1e-204 |
| H | H_repair | 1.8318 | 1.7645 | -0.0764 | [-0.0837, -0.0691] | 2.7e-105 |

## chrF++ eng_Latn-tam_Taml

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | H | 12.52 | 11.21 | -1.32 | [-1.71, -0.93] |
| baseline_lora | H_repair | 12.52 | 10.36 | -2.16 | [-2.55, -1.77] |
| H | H_repair | 11.21 | 10.36 | -0.85 | [-1.26, -0.45] |

## chrF++ tam_Taml-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|
| baseline_lora | H | 17.79 | 19.38 | +1.59 | [+0.90, +2.25] |
| baseline_lora | H_repair | 17.79 | 23.47 | +5.68 | [+5.00, +6.43] |
| H | H_repair | 19.38 | 23.47 | +4.08 | [+3.38, +4.76] |

<sub>Source: `runs/Qwen2.5-1.5B/ta/reports/comparisons.md`</sub>

### English retention: paired tests


Paired over identical test sentences. BPC: per-sentence bits/char, diff = b - a (negative = b better), Wilcoxon p Holm-corrected across pairs. chrF++: paired bootstrap, diff = b - a.

## BPC

| a | b | BPC a | BPC b | diff | 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| english_baseline_lora | english_H | 1.0060 | 1.0037 | -0.0032 | [-0.0048, -0.0016] | 3.1e-06 |
| english_baseline_lora | english_H_repair | 1.0060 | 1.0085 | +0.0021 | [+0.0003, +0.0040] | 0.087 |
| english_H | english_H_repair | 1.0037 | 1.0085 | +0.0053 | [+0.0038, +0.0068] | 4.7e-15 |

## chrF++ eng_Latn-tam_Taml

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|

## chrF++ tam_Taml-eng_Latn

| a | b | chrF a | chrF b | diff | 95% CI |
|---|---|---|---|---|---|

<sub>Source: `runs/Qwen2.5-1.5B/ta/reports/comparisons_english.md`</sub>

### FLORES at an equal character budget

Each system's outputs are truncated to the same number of characters (the longest output the base model could produce under the 256-token cap). `hit cap` = share of outputs that ran into the token cap, i.e. the model did not stop by itself.

#### eng_Latn-tam_Taml (character cap = 256)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 12.04 | 12.04 | 77% | 0% | 0.533 | 1.35 |
| baseline_lora | 12.52 | 12.52 | 88% | 0% | 0.590 | 1.50 |
| H | 11.21 | 13.43 | 85% | 88% | 0.581 | 1.65 |
| H_repair | 10.36 | 14.75 | 74% | 76% | 0.489 | 1.48 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| H | 12.52 | 13.43 | +0.91 | [+0.50, +1.32] |
| H_repair | 12.52 | 14.75 | +2.22 | [+1.75, +2.68] |

#### tam_Taml-eng_Latn (character cap = 1618)

| condition | chrF++ original | chrF++ equal chars | hit token cap | truncated | rep4 (equal chars) | len ratio (equal chars) |
|---|---|---|---|---|---|---|
| baseline | 18.67 | 18.67 | 5% | 0% | 0.063 | 1.09 |
| baseline_lora | 17.79 | 17.80 | 10% | 0% | 0.109 | 1.32 |
| H | 19.38 | 19.38 | 7% | 0% | 0.087 | 1.21 |
| H_repair | 23.47 | 23.47 | 5% | 0% | 0.059 | 1.14 |

Paired bootstrap at equal characters, vs `baseline_lora`:

| condition | chrF++ ref | chrF++ cond | diff | 95% CI |
|---|---|---|---|---|
| H | 17.80 | 19.38 | +1.59 | [+0.90, +2.25] |
| H_repair | 17.80 | 23.47 | +5.67 | [+5.00, +6.42] |


<sub>Source: `runs/Qwen2.5-1.5B/ta/reports/flores_equal_chars.md`</sub>

### Generation failure summary (FLORES outputs)

| condition | direction | empty | mean_rep4 | mean_script_mix | median_len_ratio |
|---|---|---|---|---|---|
| baseline | eng_Latn-tam_Taml | 0 | 0.533 | 0.002 | 1.349 |
| baseline_lora | eng_Latn-tam_Taml | 0 | 0.590 | 0.001 | 1.498 |
| baseline_lora | tam_Taml-eng_Latn | 0 | 0.109 | 0.000 | 1.317 |
| baseline | tam_Taml-eng_Latn | 0 | 0.063 | 0.000 | 1.091 |
| H | eng_Latn-tam_Taml | 0 | 0.721 | 0.024 | 3.114 |
| H_repair | eng_Latn-tam_Taml | 0 | 0.657 | 0.090 | 3.990 |
| H_repair | tam_Taml-eng_Latn | 0 | 0.059 | 0.000 | 1.135 |
| H | tam_Taml-eng_Latn | 0 | 0.087 | 0.000 | 1.211 |

<sub>Source: `runs/Qwen2.5-1.5B/ta/reports/failure_summary.csv`</sub>

### Looping vs per-word fragmentation (eng_Latn-tam_Taml)

| condition | Spearman ρ | p | loop rate Q1 → Q4 (least → most fragmented) |
|---|---|---|---|
| baseline | -0.036 | 0.25 | 67.6% → 64.4% → 63.2% → 67.6% |
| baseline_lora | -0.030 | 0.34 | 76.3% → 75.9% → 72.3% → 75.5% |
| H | +0.038 | 0.23 | 82.6% → 84.2% → 85.0% → 87.4% |
| H_repair | +0.082 | 0.0087 | 65.6% → 72.3% → 77.1% → 77.1% |

### Looping at matched length (eng_Latn-tam_Taml)

Loop rate (n) by the number of tokens each system's own tokenizer needs to write the reference.

| condition | 0–30 | 30–40 | 40–50 | 50–60 | 60–75 | 75–∞ | overall |
|---|---|---|---|---|---|---|---|
| baseline | – (0) | – (0) | – (0) | – (4) | 59.1% (22) | 66.1% (986) | 65.7% |
| baseline_lora | – (0) | – (0) | – (0) | – (4) | 81.8% (22) | 74.8% (986) | 75.0% |
| H | – (10) | 71.4% (49) | 74.6% (114) | 83.7% (153) | 84.7% (268) | 89.5% (418) | 84.8% |
| H_repair | 63.7% (157) | 68.7% (278) | 75.5% (241) | 78.2% (170) | 78.4% (125) | 85.4% (41) | 73.0% |

Decomposition vs `baseline_lora` (reweighting to its length distribution; approximate):

| condition | ref loop rate | at ref lengths | actual | reduction | from per-step looping | from shorter sequences | share per-step | coverage |
|---|---|---|---|---|---|---|---|---|
| H | 75.0% | 89.3% | 84.8% | -9.8% | -14.3% | 4.6% | 146.7% | 100.0% |
| H_repair | 75.0% | 85.2% | 73.0% | 2.0% | -10.2% | 12.2% | -515.4% | 100.0% |

Coverage = share of the reference system's sentences in length bins both systems occupy. Below 100%, the two parts need not sum exactly to the total.

### Training runs

Per-token losses are not comparable across tokenizers.

| condition | steps | final loss (per token) | trainable params | wall time (h) | λ (dev BPC) |
|---|---|---|---|---|---|
| baseline_lora | 750 | 1.020 | 18464768 | 2.068 |  |
| H | 750 | 2.356 | 21536768 | 2.083 | 1.000 |
| H_repair | 750 | 3.343 | 21536768 | 2.087 | 1.000 |

## Pre-tokenizer ceiling

`ceiling_scr` = 1 − pre-tokens/tokens: the most any merge-based extension can compress under the inherited pre-tokenizer; `ceiling_scr_repaired` = the same after the script-scoped repair.

| lang | tokens_per_word | pretokens_per_word | tokens_per_char | ceiling_scr | n_sentences | pretokens_per_word_repaired | ceiling_scr_repaired | achieved_v2000 | share_of_ceiling_v2000 | achieved_v4000 | share_of_ceiling_v4000 | achieved_v8000 | share_of_ceiling_v8000 | achieved_v2000_repaired | share_of_ceiling_v2000_repaired |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eng_Latn | 1.261 | 1.180 | 0.209 | 0.064 | 1012 |  |  |  |  |  |  |  |  |  |  |
| arb_Arab | 2.291 | 1.231 | 0.387 | 0.463 | 1012 | 1.171 | 0.489 |  |  |  |  |  |  |  |  |
| hin_Deva | 4.757 | 2.535 | 0.931 | 0.467 | 1012 | 1.140 | 0.760 | 0.455 | 0.974 |  |  |  |  | 0.617 | 0.812 |
| rus_Cyrl | 2.466 | 1.229 | 0.341 | 0.502 | 1012 |  |  |  |  |  |  |  |  |  |  |
| tha_Thai | 16.373 | 7.401 | 0.558 | 0.548 | 1012 | 1.416 | 0.913 |  |  |  |  |  |  |  |  |
| ben_Beng | 7.116 | 3.194 | 1.076 | 0.551 | 1012 | 1.179 | 0.834 |  |  |  |  |  |  |  |  |
| tam_Taml | 10.064 | 4.340 | 1.095 | 0.569 | 1012 | 1.231 | 0.878 | 0.562 | 0.988 |  |  |  |  | 0.732 | 0.833 |
| pes_Arab | 3.025 | 1.227 | 0.575 | 0.595 | 1012 | 1.222 | 0.596 |  |  |  |  |  |  |  |  |
| pbt_Arab | 2.927 | 1.124 | 0.631 | 0.616 | 1012 | 1.123 | 0.616 | 0.428 | 0.695 |  |  |  |  |  |  |
| snd_Arab | 3.099 | 1.153 | 0.657 | 0.628 | 1012 | 1.135 | 0.634 | 0.475 | 0.756 | 0.522 | 0.831 | 0.558 | 0.889 |  |  |
| urd_Arab | 3.145 | 1.128 | 0.669 | 0.641 | 1012 | 1.117 | 0.645 | 0.493 | 0.768 | 0.542 | 0.844 | 0.577 | 0.900 |  |  |
| tel_Telu | 11.406 | 3.936 | 1.458 | 0.655 | 1012 | 1.264 | 0.889 |  |  |  |  |  |  |  |  |
| mal_Mlym | 13.368 | 4.603 | 1.336 | 0.656 | 1012 | 1.278 | 0.904 |  |  |  |  |  |  |  |  |
| sin_Sinh | 9.173 | 2.959 | 1.449 | 0.677 | 1012 | 1.235 | 0.865 |  |  |  |  |  |  |  |  |
| amh_Ethi | 6.448 | 1.184 | 1.282 | 0.816 | 1012 | 1.184 | 0.816 | 0.611 | 0.749 |  |  |  |  |  |  |

<sub>Source: `runs/Qwen2.5-1.5B/analysis/pretokenizer_ceiling.csv`</sub>

