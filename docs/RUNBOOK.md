# HARF runbook — everything from here to the paper

Budget: about **44 of 48 GPU hours** in 6 Kaggle sessions, plus laptop (CPU) work.
Commands are PowerShell on the laptop (`C:\Projects\harf`) and notebook cells on Kaggle.

| Part | Where | Time |
|---|---|---|
| A. One-time setup | Laptop | 10 min |
| B. Analyses on results you already have | Laptop | 30 min |
| C. Data + tokenizer work for Pashto, Amharic, Hindi, Tamil | Laptop | 4–7 h (mostly waiting) |
| D. Decision gate | Laptop | 10 min |
| E. Six Kaggle sessions | Kaggle | ~44 GPU h |
| F. Merge results | Laptop | 15 min |
| G. Final analysis + human evaluation | Laptop | 1–2 days |

---

## Do I need to re-upload `harf-ur` / `harf-sd`?

- **`harf-ur`: no.** Its processed data and merge list are still valid; the new code keeps the Arabic
  checks byte-for-byte identical. The Urdu sessions (U1, U2) only need it plus the code from GitHub.
- **`harf-sd`: no.** No new Sindhi GPU runs are planned.
- **New:** one dataset each for `harf-ps`, `harf-am`, `harf-hi`, `harf-ta` (Part C, step C5).

---

## A. One-time setup (laptop)

```powershell
cd C:\Projects\harf
.venv\Scripts\Activate.ps1            # or: conda activate harf
git am 0010-novelty-experiments.patch # skip any patch already applied
git am 0011-multiscript-and-repair.patch
git am 0012-runbook.patch
git push
pytest -q                             # expect: 15 passed
```

Download FLORES-200 once on the laptop:

```powershell
curl.exe -L -o f.tgz https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
tar -xf f.tgz -C data
del f.tgz
```

Make sure every earlier result zip is extracted, oldest first, newest last:

```powershell
cd C:\Projects\harf
tar -xf kaggle_results\results_session5.zip     # Urdu: D, D_mean, baseline_lora, FLORES, latency
tar -xf kaggle_results\english_check.zip
tar -xf kaggle_results\sd_all.zip
```

(Use whatever your zips are actually named; the newest Urdu and Sindhi zips are what matter.)

---

## B. Analyses on existing results (laptop, ~30 min, free)

```powershell
harf ceiling                    # pre-tokenizer ceiling for 16 languages (with repaired column)
harf loops --lang ur            # looping vs fragmentation, item level
harf loops --lang sd
harf lora-norms --lang ur       # size of the LoRA update per condition
harf lora-norms --lang sd
```

Outputs: `runs\Qwen2.5-1.5B\analysis\pretokenizer_ceiling.md`,
`runs\Qwen2.5-1.5B\<lang>\reports\loops_vs_fertility_*.md`, `...\reports\lora_update_norms.json`.

---

## C. New languages: data and tokenizers (laptop)

**C1. Download** (~1–2 h):

```powershell
python scripts/download_data.py --langs ps am hi ta --max-docs 200000
```

If a line says `FAILED`, find the right subset name on the dataset's Hugging Face page and rerun for
that language, e.g. `python scripts/download_data.py --langs am --wiki-config 20231101.am`.

**C2. Prepare and audit** (~30 min):

```powershell
foreach ($L in "ps","am","hi","ta") {
  harf prepare --config configs/multilang.yaml --lang $L
  harf audit   --config configs/multilang.yaml --lang $L
}
```

Open each `data\processed\<lang>\data_card.md`: the top characters must be in the right script.

**C3. Learn merges** (slowest: up to ~1 h per run; 6 runs):

```powershell
foreach ($L in "ps","am","hi","ta") {
  harf select --config configs/multilang.yaml --lang $L --set tokenizer.max_unique_words=500000
}
foreach ($L in "hi","ta") {
  harf select --config configs/multilang.yaml --lang $L --condition H_repair --set tokenizer.max_unique_words=500000
}
```

**C4. Extend and measure compression** (~10 min):

```powershell
foreach ($L in "ps","am","hi","ta") { harf extend --config configs/multilang.yaml --lang $L --condition H }
foreach ($L in "hi","ta")            { harf extend --config configs/multilang.yaml --lang $L --condition H_repair }
foreach ($L in "ps","am","hi","ta") { harf eval-compression --config configs/multilang.yaml --lang $L }
harf ceiling
```

**C5. Zip each language for Kaggle**:

```powershell
foreach ($L in "ps","am","hi","ta") {
  tar -a -c -f "harf-$L.zip" "data/processed/$L" "runs/Qwen2.5-1.5B/$L"
}
```

Upload each zip as a **private** Kaggle dataset named exactly `harf-ps`, `harf-am`, `harf-hi`, `harf-ta`.
Then move the zips into `kaggle_results\`.

---

## D. Decision gate (before any new-language GPU time)

Look at `runs\Qwen2.5-1.5B\hi\eval\compression_results.csv` and the same for `ta`:

- `v2000_repaired` clearly better than `v2000` → run `H_repair` as planned.
- About the same → drop `H_repair` from sessions TA and HI (saves ~4 GPU h).

---

## E. Kaggle sessions

**Every session:** new notebook, Accelerator **GPU T4 x2**, Internet **On**, attach the datasets listed,
then **Save Version → Save & Run All (Commit)**. Download the session's last zip from the Output tab
into `kaggle_results\` when it finishes.

**Cell 1 of every session** (change `--langs`, and `--restore` where listed):

```bash
%cd /kaggle/working
!rm -rf LAFZ && git clone -q https://github.com/ArsalanKaleem/LAFZ.git
%cd /kaggle/working/LAFZ
!git log --oneline | head -1
!pip install -q -e . peft && pip uninstall -y -q torchao
!python scripts/kaggle_setup.py --langs ta
```

The last line must print `SETUP OK`. `git log` must show the runbook commit or later.

Suggested order (most novel first): **TA → U2 → HI → U1 → PS → AM**.

### Session TA — Tamil (~8 h) · attach `harf-ta`

```bash
!harf run --config configs/multilang-kaggle.yaml --lang ta --condition baseline --steps eval-ppl
```
```bash
!for C in H H_repair baseline_lora; do PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True harf run --config configs/multilang-kaggle.yaml --lang ta --condition $C --steps extend,inject,train-lora,eval-ppl; zip -qr /kaggle/working/ta_$C.zip runs/Qwen2.5-1.5B/ta; done
```
```bash
!for C in baseline baseline_lora H H_repair; do harf eval-flores --config configs/multilang-kaggle.yaml --lang ta --condition $C; harf eval-english --config configs/multilang-kaggle.yaml --lang ta --condition $C; harf eval-latency --config configs/multilang-kaggle.yaml --lang ta --condition $C --set latency.repeats=3; done
!harf report --config configs/multilang-kaggle.yaml --lang ta
!zip -qr /kaggle/working/ta_all.zip runs/Qwen2.5-1.5B/ta
```

`baseline_lora` is the cheapest condition to lose, so it runs last of the three trainings.

### Session U2 — Urdu mechanism + seed (~8.5 h) · attach `harf-ur`

Cell 1 with `--langs ur`. Then:

```bash
!harf frag-decode --config configs/kaggle-t4.yaml --lang ur --condition baseline
!zip -qr /kaggle/working/u2_frag.zip runs/Qwen2.5-1.5B/ur/eval
```
```bash
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True harf run --config configs/kaggle-t4.yaml --lang ur --condition D_frozen_rows --steps extend,inject,train-lora,eval-ppl
!harf eval-english --config configs/kaggle-t4.yaml --lang ur --condition D_frozen_rows
!harf eval-flores  --config configs/kaggle-t4.yaml --lang ur --condition D_frozen_rows
!zip -qr /kaggle/working/u2_frozen.zip runs/Qwen2.5-1.5B/ur
```
```bash
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True harf run --config configs/kaggle-t4.yaml --lang ur --condition D_seed2 --steps extend,inject,train-lora,eval-ppl
!zip -qr /kaggle/working/u2_all.zip runs/Qwen2.5-1.5B/ur
```

### Session HI — Hindi (~8 h) · attach `harf-hi`

Identical to Session TA with `ta` replaced by `hi` everywhere (Cell 1: `--langs hi`).

### Session U1 — Urdu vocabulary size 4k / 8k (~8.5 h) · attach `harf-ur`

Cell 1 with `--langs ur`. Then:

```bash
!for C in E F; do PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True harf run --config configs/kaggle-t4.yaml --lang ur --condition $C --steps extend,inject,train-lora,eval-ppl; zip -qr /kaggle/working/u1_$C.zip runs/Qwen2.5-1.5B/ur; done
```
```bash
!for C in E F; do harf eval-flores --config configs/kaggle-t4.yaml --lang ur --condition $C; harf eval-latency --config configs/kaggle-t4.yaml --lang ur --condition $C --set latency.repeats=3; harf eval-english --config configs/kaggle-t4.yaml --lang ur --condition $C; done
!zip -qr /kaggle/working/u1_all.zip runs/Qwen2.5-1.5B/ur
```

### Session PS — Pashto (~5.5 h) · attach `harf-ps`

```bash
!harf run --config configs/multilang-kaggle.yaml --lang ps --condition baseline --steps eval-ppl
!for C in H baseline_lora; do PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True harf run --config configs/multilang-kaggle.yaml --lang ps --condition $C --steps extend,inject,train-lora,eval-ppl; zip -qr /kaggle/working/ps_$C.zip runs/Qwen2.5-1.5B/ps; done
!for C in baseline baseline_lora H; do harf eval-flores --config configs/multilang-kaggle.yaml --lang ps --condition $C; harf eval-english --config configs/multilang-kaggle.yaml --lang ps --condition $C; harf eval-latency --config configs/multilang-kaggle.yaml --lang ps --condition $C --set latency.repeats=3; done
!harf report --config configs/multilang-kaggle.yaml --lang ps
!zip -qr /kaggle/working/ps_all.zip runs/Qwen2.5-1.5B/ps
```

### Session AM — Amharic (~5.5 h) · attach `harf-am`

Identical to Session PS with `ps` replaced by `am`.

### Budget

| Session | GPU h | Cumulative |
|---|---|---|
| TA | 8 | 8 |
| U2 | 8.5 | 16.5 |
| HI | 8 | 24.5 |
| U1 | 8.5 | 33 |
| PS | 5.5 | 38.5 |
| AM | 5.5 | 44 |

Kaggle quota resets weekly; check the remaining hours before each session.

### If something fails

- A `!harf ...` failure does **not** stop a committed notebook. After each session, search the log for
  `Traceback` and paste it to me.
- Out of memory: rerun that condition alone; the other conditions' zips are already saved.
- Session hit 12 h: everything zipped before the cut-off is in the Output tab.

---

## F. Merge results on the laptop

Download each session's **last** zip (`ta_all.zip`, `u2_all.zip`, `hi_all.zip`, `u1_all.zip`,
`ps_all.zip`, `am_all.zip`) into `kaggle_results\`, then extract in the order the sessions ran:

```powershell
cd C:\Projects\harf
foreach ($z in "ta_all","u2_all","hi_all","u1_all","ps_all","am_all") {
  if (Test-Path "kaggle_results\$z.zip") { tar -xf "kaggle_results\$z.zip" }
}
```

U1 and U2 each contain the full Urdu folder; whichever ran later must be extracted later. Because the
sessions only *add* conditions, extracting both keeps everything.

---

## G. Final analysis (laptop)

```powershell
harf report --lang ur
harf report --lang sd
foreach ($L in "ps","am","hi","ta") { harf report --config configs/multilang.yaml --lang $L }
foreach ($L in "ur","sd") { harf loops --lang $L; harf lora-norms --lang $L }
foreach ($L in "ps","am","hi","ta") { harf loops --config configs/multilang.yaml --lang $L }
harf ceiling
```

Send to me: every `reports\comparisons.md`, `reports\final_results.md`, the `loops_vs_fertility_*.md`
files, `lora_update_norms.json` (ur), `analysis\pretokenizer_ceiling.md`, and
`runs\Qwen2.5-1.5B\ur\eval\fragdecode_urd_Arab-eng_Latn.json`.

**Human evaluation** (native speakers, blinded, 50 items each):

```powershell
foreach ($L in "ps","am","hi","ta") {
  harf human-eval --config configs/multilang.yaml --lang $L --steps "make:baseline_lora,H,,50"
}
```

Give each rater the CSV and the rating guide (fluency, adequacy, preferred). Never share the
`.key.json`. Record each rater's language background for the paper. Score with:

```powershell
harf human-eval --config configs/multilang.yaml --lang hi --steps "score:runs\Qwen2.5-1.5B\hi\reports\human_eval_baseline_lora_vs_H_eng_Latn-hin_Deva.csv"
```

---

## Checklist

- [ ] A: patches applied, 15 tests pass, FLORES on laptop, old zips extracted
- [ ] B: ceiling, loops, lora-norms on Urdu/Sindhi
- [ ] C: 4 languages downloaded, prepared, selected (+ repaired for hi/ta), extended, zipped, uploaded
- [ ] D: decision on `H_repair`
- [ ] E: TA, U2, HI, U1, PS, AM — last zip of each downloaded
- [ ] F: results merged
- [ ] G: reports, analyses, human evaluation
