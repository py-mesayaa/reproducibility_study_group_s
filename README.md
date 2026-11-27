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
| `Code/compare_medchange_datasets.ipynb` | Shows that the released `MedChangeQA.csv` pairs questions with another topic's labels |
| `Code/output/` | Model answers and the CSV outputs of the notebooks |
| `Datasets/MedChangeQA.csv` | The 512-question dataset as released by the original authors |

## Files not included

These come from the original authors' repository and are needed only for some steps:

- `Datasets/MedRevQA.csv` (22 MB): used by `compare_medchange_datasets.ipynb`
- `Code/FullScraped.csv`: used by `repro/build_subset.py` (its output, `repro/data/medchange_subset.csv`, is included)
- The authors' answer files (`Code/GeneratedAnswers/*.txt`): used when re-scoring their answers with `repro/evaluate.py`

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Scoring and the notebooks need only pandas, numpy and scikit-learn. GPU inference
(`repro/run_inference.py`) needs torch, transformers, accelerate and bitsandbytes on a CUDA machine.
Run the notebooks from inside `Code/`; they use paths relative to it.
