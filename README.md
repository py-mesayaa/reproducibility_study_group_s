# reproducibility_study_group_s

Code for Group S's reproducibility study in Seminars in Data Science (ITU, 2026).
We reproduce Table 2 (changed-knowledge experiment) of *Facts Fade Fast: Evaluating Memorization of
Outdated Medical Knowledge in LLMs* (arXiv 2509.04304) for Qwen 2.5 7B and BioMistral 7B.

## Contents

| Path | What it is |
|---|---|
| `repro/` | Main reproduction pipeline: building the 512-question subset, GPU inference, scoring and the BioMistral diagnostics. **Results, setup and how to rerun are in [`repro/README.md`](repro/README.md).** |
| `Code/generate_answers.ipynb` | First answer-generation run (Qwen through Ollama, BioMistral). `jupytersbatchfile.sh` starts a Jupyter server on the HPC for it. |
| `Code/parse_answers.ipynb` | Extracts labels from the model answers → `Code/output/parsed_labels*.csv` |
| `Code/prompt_compliance.ipynb` | Checks whether answers follow the `LABEL:` format → `Code/output/prompt_compliance.csv` |
| `Code/experiments_results.ipynb` | Table 2 scores → `Code/output/table2_changed_knowledge.csv` |
| `Code/output/` | Model answers and the CSV outputs of the notebooks |
| `Datasets/MedChangeQA.csv` | The 512-question dataset as released by the original authors |

## Data

All data comes from public sources. The original data is the authors' public GitHub repository
[jvladika/MedChange](https://github.com/jvladika/MedChange) (we used commit `474f969`). The models are
public on Hugging Face.

**Included in this archive** (enough to run all scoring and notebooks without downloading anything):

- `Datasets/MedChangeQA.csv`: unchanged copy from the authors' repository (64 KB)
- `repro/data/medchange_subset.csv`: the 512 questions with their outdated and latest labels, built by `repro/build_subset.py`
- All our model answers: `repro/outputs/`, `repro/diagnostics/outputs/` and `Code/output/`

**Download on demand** (too large to include, only needed to rebuild the subset or re-score the authors' answers).
Run from the archive's root folder:

```bash
BASE=https://raw.githubusercontent.com/jvladika/MedChange/474f969ade22b8f96a24c3970883e68a6f2799d1
curl -L -o Code/FullScraped.csv "$BASE/Code/FullScraped.csv"
curl -L -o Code/generated_questions_labels.txt "$BASE/Code/generated_questions_labels.txt"
mkdir -p Code/GeneratedAnswers
curl -L -o Code/GeneratedAnswers/qwen25-7b_answers.txt "$BASE/Code/GeneratedAnswers/qwen25-7b_answers.txt"
curl -L -o Code/GeneratedAnswers/biomistral_answers.txt "$BASE/Code/GeneratedAnswers/biomistral_answers.txt"
```

- `Code/FullScraped.csv` (63 MB) and `Code/generated_questions_labels.txt` (2.4 MB): all review versions and
  their generated questions; inputs of `repro/build_subset.py`
- `Code/GeneratedAnswers/*.txt`: the authors' model answers; used by `repro/evaluate.py` and `repro/diagnostics/compare_authors.py`

**Models** (downloaded automatically by `repro/run_inference.py` through Hugging Face):
[`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and
[`BioMistral/BioMistral-7B`](https://huggingface.co/BioMistral/BioMistral-7B).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Scoring and the notebooks need only pandas, numpy and scikit-learn. GPU inference
(`repro/run_inference.py`) needs torch, transformers, accelerate and bitsandbytes on a CUDA machine.
Run the notebooks from inside `Code/`; they use paths relative to it.
