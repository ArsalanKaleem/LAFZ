# HARF — Halving the Tokenization Tax for Urdu and Sindhi

*حرف (harf): "letter" in Urdu, Sindhi, Arabic and Persian.*

**HARF (Hybrid Alignment for Reducing Fragmentation)** is a post-hoc vocabulary-extension pipeline for open-weight language models. It adds Urdu or Sindhi subwords to a model's tokenizer, initialises their embeddings, adapts the model with LoRA, and evaluates what that buys and what it costs.

Open models such as Qwen2.5 spend about **three tokens per Urdu or Sindhi word**, roughly three times English. That makes these languages slower and more expensive to process and shrinks the usable context. HARF halves it.

## Results (Qwen2.5-1.5B, 2,000 new tokens, 1,500 LoRA steps)


|                                                                   | Urdu                      | Sindhi                    |
| ----------------------------------------------------------------- | ------------------------- | ------------------------- |
| Tokens per word (baseline → HARF)                                | 3.04 →**1.55**(−49%)    | 3.00 →**1.58**(−47.5%)  |
| Time to generate the same text                                    | **1.95× faster**         | **1.84× faster**         |
| Bits per character vs. fine-tuning alone                          | 1.818 →**1.709**(−6.0%) | 2.033 →**1.902**(−6.4%) |
| chrF++ English → target vs. fine-tuning alone                    | 13.1 →**17.9**           | 7.5 →**15.2**            |
| chrF++ target → English vs. fine-tuning alone                    | 27.8 →**35.0**           | 24.1 →**33.2**           |
| Blinded native-speaker preference (decided items)                 | **81%**HARF               | **94%**HARF               |
| Repeated 4-grams in English → target output (baseline → HARF)   | 73% →**31%**             | 94% →**35%**             |
| English BPC change vs. original model (fine-tuning alone → HARF) | +1.5% →**+1.0%**         | +4.1% →**+1.9%**         |

All comparisons against the fine-tuned baseline (same LoRA budget, no new vocabulary) are statistically significant (paired Wilcoxon / paired bootstrap, Holm-corrected). With **equal training data**, the extended Urdu model matches the fine-tuned baseline within 0.5% BPC using **half the compute**.

Absolute translation quality is low for every system (a 1.5B base model translating from three examples), so read the chrF++ and human scores as relative comparisons. Human ratings come from one blinded native speaker per language (100 Urdu items, 50 Sindhi items). Each condition was trained once (one seed).

## How it works

1. **Continued BPE.** Pre-tokenize the target-language corpus with the base model's own pipeline, segment every word with the base BPE, then keep merging the most frequent adjacent pair.
   * **Exact budget:** every merge adds exactly one token.
   * **Nested vocabularies:** the 2k set is a prefix of the 4k and 8k sets.
   * **Faithful:** savings measured on the training word table equal the real tokenizer's savings token-for-token (tested).
   * **Non-interference:** every new token contains an Arabic-block byte, so English, code and numbers are tokenized **exactly** as before. Checked on every extension.
2. **Extension.** New vocabulary entries and merges are appended; no existing ID changes.
3. **Initialisation.** Each new row starts from its merge parents: mean of parts, or a ridge projection learned on the base model's own merges with the regularisation chosen by validation BPC.
4. **Adaptation.** LoRA on all attention and MLP projections, plus a small trainable table for the new rows. The original embedding matrix stays frozen, so old rows cannot change.

## Install

```bash
git clone https://github.com/ArsalanKaleem/LAFZ.git && cd LAFZ
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[gpu,dev]"
pytest -q                                              # 7 tests, CPU only, no downloads
```

Python 3.10–3.12 is recommended. On Kaggle or Colab, use `pip install -e . peft` (keep the preinstalled PyTorch) and `pip uninstall -y torchao` if PEFT complains about its version.

## Quick start

**CPU steps** (tokenizer only, no GPU):

```bash
harf prepare --lang ur                       # clean, deduplicate, split → data/processed/ur
harf audit   --lang ur                       # frozen baseline tokenization audit
harf select  --lang ur                       # learn up to 8k merges once
harf extend  --lang ur --condition D         # 2k-token extended tokenizer (+ verification)
harf eval-compression --lang ur              # compression for every budget
```

**GPU steps** (a 16 GB T4 works with `configs/kaggle-t4.yaml`, about 3.5 h per LoRA run):

```bash
harf run --config configs/kaggle-t4.yaml --lang ur --condition baseline --steps eval-ppl
harf run --config configs/kaggle-t4.yaml --lang ur --condition D             # extend, inject, LoRA, evaluate
harf run --config configs/kaggle-t4.yaml --lang ur --condition baseline_lora
harf eval-flores  --config configs/kaggle-t4.yaml --lang ur --condition D
harf eval-latency --config configs/kaggle-t4.yaml --lang ur --condition D
harf eval-english --config configs/kaggle-t4.yaml --lang ur --condition D
harf report       --config configs/kaggle-t4.yaml --lang ur
```

**Human evaluation** (blinded A/B sheet; scoring unblinds it):

```bash
harf human-eval --lang ur --steps "make:baseline_lora,D"
harf human-eval --lang ur --steps "score:runs/Qwen2.5-1.5B/ur/reports/human_eval_baseline_lora_vs_D_eng_Latn-urd_Arab.csv"
```

## Data

Raw text goes in `data/raw/<lang>/` as `.txt` (blank-line separated documents), `.txt.gz`, or `.jsonl` with a `text` field. Our runs used FineWeb-2 (`urd_Arab`, `snd_Arab`; ODC-By) and Wikipedia (CC BY-SA). Evaluation uses FLORES-200 devtest (CC BY-SA 4.0).

Normalisation is language-specific and conservative by default. In Sindhi, ک (U+06A9, kheh) and ڪ (U+06AA, kaf) are different letters, so Urdu spelling rules must never be applied to Sindhi text.

## Experimental conditions


| Condition            | New tokens | Init              | LoRA      | Purpose                                         |
| -------------------- | ---------- | ----------------- | --------- | ----------------------------------------------- |
| `baseline`           | 0          | –                | no        | Reference                                       |
| `baseline_lora`      | 0          | –                | yes       | Control: fine-tuning without new vocabulary     |
| `B`/`C`              | 2k         | mean / projection | no        | Initialisation before training                  |
| `D`                  | 2k         | projection        | yes       | Main method                                     |
| `D_mean`             | 2k         | mean              | yes       | Control: does initialisation matter after LoRA? |
| `D_750`,`D_mean_750` | 2k         | projection / mean | 750 steps | Control: equal training data                    |
| `E`/`F`              | 4k / 8k    | projection        | yes       | Vocabulary-size ablation                        |

## Metrics

* **SCR** = 1 − Σ tokens<sub>extended</sub> / Σ tokens<sub>baseline</sub> on fixed test sentences.
* **BPC** (bits per character) = total NLL / (ln 2 · characters). Token-level perplexity is not comparable across tokenizers, so it is reported but never used for comparisons.
* **Latency** is the time to emit the *same* reference text by forced decoding (batch 1, median of repeats). Free-running generation with a fixed token cap would favour longer tokens unfairly.
* **Statistics:** paired t-test and Wilcoxon per sentence, bootstrap CIs, paired-bootstrap chrF++, Holm correction.

## Outputs

```
runs/<model>/<lang>/
  _shared/              baseline audit (frozen), merge list, candidate_tokens_{2k,4k,8k}.json
  <condition>/          tokenizer/, new_rows.pt, init/ diagnostics, lora/ adapter + training log
  eval/                 compression, perplexity, FLORES predictions, latency, English check
  reports/              final_results.md, comparisons.md, significance.json, failure_cases.md, human_eval_*
```

The base model is never duplicated on disk: each condition is stored as a tokenizer plus its new embedding rows and LoRA adapter.

## Limitations

* One model (Qwen2.5-1.5B) and one seed per condition.
* Web-crawled training data; FineWeb-2 Sindhi is narrower in domain than Urdu.
* New tokens also apply to other Arabic-script text (Persian, Arabic, Pashto); the non-interference guarantee covers text *without* Arabic-script characters.
* Qwen's pre-tokenizer splits words at Arabic diacritics and ZWNJ, so new merges cannot cross those points.
* Only byte-level BPE tokenizers are supported (Qwen2/2.5, Llama 3.x, GPT-2 style), not SentencePiece models.

## Other languages

The pipeline is language-agnostic in design, but the current release targets Arabic-script languages. See [`docs/RESEARCH_NOTES.md`](https://claude.ai/chat/docs/RESEARCH_NOTES.md) for what changes for other scripts.

## Citation

A paper is in preparation. Citation details will be added here.

## License

Apache-2.0. Data and model licenses follow their sources (FineWeb-2: ODC-By; Wikipedia: CC BY-SA; FLORES-200: CC BY-SA 4.0; Qwen2.5: see its model card).
