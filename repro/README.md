# Reproduction: Table 2 (columns b, c, F1 diff) for Qwen 2.5 7B and BioMistral 7B

Paper: *Facts Fade Fast: Evaluating Memorization of Outdated Medical Knowledge in LLMs* (arXiv 2509.04304).
Target: macro precision / recall / F1 on the 512 changed-knowledge questions, scored against
**(b) outdated** and **(c) latest** labels, plus **F1 diff = F1(c) − F1(b)**.
Inference ran on the ITU HPC instead of the Together AI API used in the original code.

## Summary

| Model | Outcome |
|---|---|
| **Qwen 2.5 7B** | **Reproduced.** All P/R/F1 within 1.2 points, F1 diff −4.9 vs. −4.8; 97% of predicted labels identical to the authors' answers. |
| **BioMistral 7B** | **Not reproducible from the published information.** With the documented prompt and greedy decoding the model answers NOT ENOUGH INFORMATION for ~80% of questions; F1 is 7–10 points below the paper. The inference code for this model is not in the repository and four plausible variants of the setup did not recover the authors' outputs. |

Separately, we found several problems in the released data and code (section 5) — most importantly,
the released `Datasets/MedChangeQA.csv` pairs every question with another topic's labels.

## 1. Results

Scores in %, P / R / F1. "pred = outdated" is the share of answers that predicted the outdated label;
it equals the paper's unlabeled "Answers" column (35.4% / 37.1%), which settles what that column means.

| Run | (b) outdated | (c) latest | F1 diff | pred = outdated |
|---|---|---|---|---|
| **Qwen** — paper Table 2 | 42.6 / 37.1 / 30.8 | 27.1 / 30.8 / 26.0 | −4.8 | 35.4% |
| Qwen — authors' answer file, re-scored | 42.6 / 37.1 / 30.8 | 27.1 / 30.8 / 26.0 | −4.7¹ | 35.4% |
| **Qwen — ours** (bf16, notebook prompt) | **43.1 / 37.2 / 30.9** | **28.3 / 30.7 / 26.0** | **−4.9** | 35.9% |
| **BioMistral** — paper Table 2 | 36.8 / 37.2 / 36.3 | 35.4 / 35.5 / 35.3 | −1.5² | 37.1% |
| BioMistral — authors' answer file, re-scored | 36.8 / 37.2 / 36.3 | 35.4 / 35.5 / 35.3 | −1.0 | 37.1% |
| **BioMistral — ours** (8-bit, Table 6 prompt) | 27.6 / 35.3 / 29.7 | 34.7 / 33.5 / 25.6 | −4.1 | 44.9% |
| BioMistral — ours (8-bit, notebook prompt) | 59.6 / 34.2 / 28.1 | 28.7 / 35.2 / 25.2 | −2.9 | 44.5% |

¹ The paper subtracts rounded F1 values (26.0 − 30.8); we subtract unrounded ones.
² Arithmetic error in the paper: 35.3 − 36.3 = −1.0.

Re-scoring the authors' own answer files with their notebook logic reproduces every Table 2 value,
which validates our evaluation code and subset (`evaluate.py`, `data/medchange_subset.csv`).

Predicted label distribution (SUPPORTED / NOT ENOUGH INFO / REFUTED) over the 512 questions:

| | Authors' answers | Ours |
|---|---|---|
| Qwen | 389 / 65 / 58 | 394 / 72 / 46 |
| BioMistral | 198 / 171 / 143 | 107 / 401 / 4 (Table 6 prompt) |

## 2. Setup and deviations from the paper

| | Paper / original code | Ours |
|---|---|---|
| Inference | Together AI API (Qwen); local, code not published (BioMistral) | Hugging Face transformers 5.22.2, torch 2.14.1+cu126, ITU HPC (SLURM) |
| Questions | All 16,501 answered, 512 scored | Only the 512 scored questions answered (identical answers; 30× less compute) |
| Which question is asked | Question of the *outdated* review version (`outdated_keys`) | Same (`data/medchange_subset.csv`, rebuilt from the notebook logic) |
| Qwen prompt | System + user message, `answer_questions.ipynb` cell 3 | Verbatim, through Qwen's chat template |
| BioMistral prompt | Paper Table 6 (`<s>[INST] Based on your knowledge, … QUESTION: {q} [/INST]`) | Verbatim Table 6 (first run used the notebook prompt) |
| Decoding | Temperature 0, 512 tokens | Greedy, `max_new_tokens=512` |
| Qwen precision | Not stated; Together "Turbo" endpoints are likely FP8 | bf16 (native release precision) |
| BioMistral precision | "8-bit quantized", library not stated | bitsandbytes 8-bit |
| Batching | One request at a time | 16 per batch, left padding (only rounding-level differences) |
| Label parsing | Substring order SUPPORTED → REFUTED → NEI | Same, plus NEI fallback for answers without a label (see 5.4); a stricter `LABEL:`-tag parser gives identical Qwen scores |

GPUs: Qwen on an NVIDIA A30; BioMistral on an H100 NVL (notebook prompt) and an RTX 6000 Ada (Table 6 prompt).

## 3. Format conformance (TA question)

The prompt asks for `LABEL: <label>`. Share of the 512 answers that comply:

| | `LABEL:` tag | No label at all | >1 different label |
|---|---|---|---|
| Qwen — authors' answers | 52.7% | 0% | 0% |
| Qwen — ours | 73.4% (always at the start) | 0% | 0% |
| BioMistral — authors' answers | 3.7% | 5.1% | 1.2% |
| BioMistral — ours, notebook prompt | 3.3% | 0.2% | 0% |
| BioMistral — ours, Table 6 prompt | 0% | 2.1% | 0% |

Qwen mostly follows the template; BioMistral essentially never does (it writes e.g.
`NOT ENOUGH INFORMATION. …` or `SUPPORTED: …`). Because the label words themselves are almost always
present, the lenient substring parser still extracts a label, and a stricter tag-first parser changes
the scores by at most 0.5 points. Non-conformance therefore does not explain the BioMistral gap.

## 4. BioMistral: why it is not reproducible

Diagnostic job (`diagnostics/biomistral_diagnostics.sh`, first 64 questions, compared with the authors' answers
by `diagnostics/compare_authors.py`; answers in `diagnostics/outputs/`). With greedy decoding the right setup should give near word-for-word identical text.

| Variant | Same label as authors | Starts with `ANSWER:` | Identical text | NEI / SUP / REF |
|---|---|---|---|---|
| Table 6 prompt | 44% | 0% | 0% | 56 / 8 / 0 |
| + indented prompt | 42% | 0% | 0% | 58 / 6 / 0 |
| + double `<s>` | 50% | 0% | 0% | 52 / 11 / 1 |
| + both | 55% | 0% | 0% | 50 / 14 / 0 |
| + sampling (T = 1.0) | 42% | 5% | 0% | 41 / 16 / 7 |
| **Authors' answers** | — | **91%** | — | **21 / 25 / 18** |

Clues about the authors' actual setup:
- 91% of their BioMistral answers start with `ANSWER:`; their OLMo and PMC-LLaMA answers never do.
  Something specific to their BioMistral prompt triggered this, and the Table 6 prompt does not.
- The newline + 12 spaces inside their answers also appears for OLMo and PMC-LLaMA (different prompts),
  so it is an artifact of their logging script, not of the prompt.
- The OLMo file is named `olmo_gguf_answers_13b.txt` and all three local models share one output format,
  suggesting the local models ran through llama.cpp with GGUF files (different quantization and default
  sampling settings). Not tested: it would still require guessing the prompt.

Conclusion: the BioMistral numbers depend on unpublished inference details. With the documented prompt,
decoding and quantization, BioMistral 7B is dominated by NOT ENOUGH INFORMATION answers.

## 5. Problems found in the original data and code

1. **`Datasets/MedChangeQA.csv` pairs questions with the wrong labels** (verified). Its row numbers and
   both label columns match the notebook exactly (512/512), but the Question column was taken from the
   topic list sorted in a different order (by line number instead of the notebook's `set()`-based order).
   **0 of 512** rows pair a question with its own review history. Example, row 1: the question
   *"Is cranberry juice effective for treating urinary tract infections?"* (review CD001322, REFUTED in
   both versions) carries the labels REFUTED → SUPPORTED of review CD001321 (cranberry for *preventing* UTI).
   Most pairs are unrelated topics (row 1495: clotiapine for psychosis, labeled with hallux-valgus surgery).
   Scoring the authors' own answers against the CSV changes the results and flips Qwen's F1 diff from
   −4.8 to +4.1. No code in the repository reads or writes this file.
2. **Table 2 uses the automatically generated (silver) labels**, not the manually corrected gold labels
   the paper describes; no corrected labels are in the repository.
3. **Version grouping misses most review histories** (quick check, needs spot-checking). Cell 4 of
   `experiments_results.ipynb` joins versions by DOI prefix, which only works when the original DOI
   without `.pubN` is in the data. Example: lamotrigine for generalised tonic-clonic seizures (CD007783)
   changed from SUPPORTED (pub2) to NOT ENOUGH INFORMATION (pub3) and is missed. Grouping by Cochrane ID
   gives ~4,161 multi-version reviews and ~1,323 verdict changes instead of 1,535 and 512.
4. **Answers without a label shift all later answers** (verified). The MedChange scoring cell (cell 13)
   skips such answers instead of counting them as NEI, so every later answer is scored against the wrong
   question; with BioMistral (888 such answers among 16,501) the notebook crashes. The published notebook
   does not include BioMistral, and the paper's BioMistral numbers are reproduced only with the NEI fallback.
5. **24 of the 512 questions are generic**: *"Is there a beneficial effect of the intervention on patient
   outcomes?"* — question generation failed for these reviews, so they are unanswerable.
6. **Naive label parsing**: the first of SUPPORTED / REFUTED / NEI found anywhere in the text, in that
   fixed order ("NOT SUPPORTED" counts as SUPPORTED).
7. **Missing or broken pieces**: inference code for the local models (BioMistral, PMC-LLaMA, OLMo);
   the code that created `MedChangeQA.csv`; the notebook opens `all_questions_labels.txt` but the file is
   `generated_questions_labels.txt`; the n-gram cells need `FullQA.csv`, which is not included;
   one review version (CD001909) appears twice in the data.
8. **Paper errata**: F1 diff for BioMistral (−1.5 → −1.0) and OLMo (+2.9 → +3.9); Table 2 caption and
   Section 5 disagree on which experiment the "Answers" column refers to; "four general-purpose models"
   but five are listed.

## 6. Files and how to rerun

| File | Purpose |
|---|---|
| `build_subset.py` | Rebuilds the 512 questions + outdated/latest labels from the notebook logic → `data/medchange_subset.csv` |
| `run_inference.py` | Answers the questions on a GPU (fixed precision per model, resumable, `--limit` for tests) |
| `slurm/run_model.sh` | SLURM job wrapper for `run_inference.py` |
| `evaluate.py` | Format conformance + Table 2 scores, for our `.jsonl` and the authors' `.txt` answer files |
| `outputs/` | Answers of the three full runs (`.jsonl`) with run metadata (`.meta.json`): Qwen, BioMistral (Table 6 prompt), BioMistral (first attempt, notebook prompt) |
| `diagnostics/` | BioMistral investigation of section 4: the job script, `compare_authors.py` (question-by-question comparison with the authors' answers) and the four 64-question runs in `diagnostics/outputs/` |

On the HPC (from `~/medchange`, environment `~/medchange-env`):

```bash
sbatch --job-name=qwen --mem=4G --time=01:00:00 --exclude=desktop[1,6-8],cn[3-6,12,16-17] repro/slurm/run_model.sh --model qwen
sbatch --job-name=biomistral --mem=24G --time=02:00:00 repro/slurm/run_model.sh --model biomistral
```

Scoring (anywhere, needs pandas + scikit-learn):

```bash
python repro/evaluate.py repro/outputs/qwen_bf16_notebook.jsonl repro/outputs/biomistral_8bit_table6.jsonl
python repro/evaluate.py Code/GeneratedAnswers/qwen25-7b_answers.txt Code/GeneratedAnswers/biomistral_answers.txt
```
