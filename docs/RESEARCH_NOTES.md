# Research notes

## Positioning (verify every citation with a literature search before writing)

Related work to read and compare against:
- Embedding initialisation for new vocabularies: WECHSEL (Minixhofer et al.), FOCUS (Dobler & de Melo), OFA.
- Zero-shot tokenizer transfer (Minixhofer et al.).
- Tokenization disparity across languages: Petrov et al., "Language Model Tokenizers Introduce Unfairness Between Languages".
- Vocabulary expansion for specific languages (Chinese LLaMA, various Indic and Arabic efforts).

Candidate contributions HARF can defend:
1. The first controlled study of the tokenization tax for **Sindhi**, alongside Urdu, with a released benchmark.
2. Continued-BPE extension with **provable non-interference** for out-of-script text and an exact, prefix-nested budget.
3. **Merge-tree projection**: self-supervised initialisation from the base model's own merges, with no parallel data.
4. Evaluation that is comparable across tokenizers (BPC, same-text latency) and includes control conditions.
5. Released artefacts: tokenizers, embedding rows, adapters, and the evaluation harness.

Strong baselines to add before submission: FOCUS-style initialisation and a WECHSEL-style bilingual initialisation.
Without these, reviewers will ask why the projection is better than existing methods.

## Known risks and what to watch

- **Qwen's padded embedding matrix** (151936 rows vs 151665 ids). `inject_vocab` never shrinks it.
  Padded rows are excluded from NLL and suppressed in generation for every condition.
- **Tied embeddings** (Qwen2.5-1.5B, Llama-3.2-1B): one matrix serves input and output. A new row's init
  affects both reading and predicting that token.
- **Random init (A)** will likely make generation degenerate. That is the expected lower bound, not a bug.
- **ZWNJ and diacritics** split pre-tokens in the Qwen regex (they are not `\p{L}`). Merges never cross
  pre-token boundaries, so words with ZWNJ stay fragmented. Quantify this in the audit; it may be a finding.
- **Persian and Arabic** share the script: they will be affected by the new tokens (non-interference
  covers only text without Arabic-block characters). Measure on a small Persian/Arabic sample.
- **Latency:** fewer tokens does not guarantee wall-clock speedup at batch 1 for a 1.5B model.
  Report token savings and runtime separately.

## Day-to-command map

| Day | Command | Deliverable |
|---|---|---|
| 1 | `harf env`, `harf smoke` | environment.json, smoke_test.json |
| 2 | `harf prepare` | data/processed/<lang>/, data_card.md |
| 3 | `harf audit` | _shared/baseline_audit/ (frozen) |
| 4 | `harf train-bpe` | _shared/target_bpe/ |
| 5 | `harf select` | candidate_tokens_{2k,4k,8k}.json |
| 6 | `harf extend`, `harf inject` | tokenizer/, new_rows.pt, verification.json |
| 7–8 | `harf inject` (init=random/mean/projection) | init/, projection diagnostics, nearest_neighbors.md |
| 9 | `harf train-lora` | lora/adapter, train_log.jsonl |
| 10 | `harf eval-compression` | eval/compression_* |
| 11 | `harf eval-ppl` | eval/ppl_* |
| 12 | `harf eval-flores` | eval/flores_* (raw predictions) |
| 13 | `harf eval-latency` | eval/latency_* |
| 14 | `harf report` | final_results, significance.json, failure_cases.md |
| 15 | README + archive | reproducible repository |
