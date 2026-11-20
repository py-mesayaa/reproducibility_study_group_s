"""Score model answers on the 512 changed-knowledge questions (paper Table 2, columns b and c).

Works on both
  * our outputs:            repro/outputs/<model>_<precision>.jsonl   (one {"key", "output"} per line)
  * the authors' answers:   Code/GeneratedAnswers/*.txt               (one answer per MedRevQA line)

For every file it reports
  1. format conformance  - does the answer follow the requested "LABEL: <label>" format?
  2. Table 2 scores       - macro precision / recall / F1 against (b) outdated and (c) latest labels,
                            and F1 diff = F1(c) - F1(b), next to the paper's numbers
under two label parsers:
  paper   - the authors' rule (experiments_results.ipynb, cell 13): first of SUPPORTED, REFUTED,
            NOT ENOUGH INFORMATION that occurs *anywhere* in the text, in that fixed order.
            Answers with no label count as NOT ENOUGH INFORMATION (cell 9's fallback; cell 13
            skips them instead, which shifts every later answer onto the wrong question).
  strict  - the "LABEL: <x>" tag if present, otherwise the label mentioned *first* in the text,
            otherwise NOT ENOUGH INFORMATION.

Examples (from the repo root):
  python repro/evaluate.py Code/GeneratedAnswers/qwen25-7b_answers.txt Code/GeneratedAnswers/biomistral_answers.txt
  python repro/evaluate.py repro/outputs/qwen_bf16.jsonl repro/outputs/biomistral_8bit.jsonl
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

HERE = Path(__file__).resolve().parent

S, R, N = "SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"
LABELS = [S, N, R]
TAG = re.compile(r"LABEL\s*:\s*\**\s*(SUPPORTED|REFUTED|NOT ENOUGH INFORMATION)", re.IGNORECASE)

# Paper Table 2: (b) outdated P/R/F1, (c) latest P/R/F1, F1 diff as printed
PAPER = {
    "qwen":       {"b": (42.6, 37.1, 30.8), "c": (27.1, 30.8, 26.0), "diff": -4.8},
    "biomistral": {"b": (36.8, 37.2, 36.3), "c": (35.4, 35.5, 35.3), "diff": -1.5},  # -1.0 by arithmetic
}


# ---------------------------------------------------------------- parsers
def parse_paper(text):
    for label in (S, R, N):
        if label in text:
            return label
    return N


def parse_strict(text):
    m = TAG.search(text)
    if m:
        return m.group(1).upper()
    found = [(text.find(label), label) for label in (S, R, N) if label in text]
    return min(found)[1] if found else N


def mentioned(text):
    """Distinct labels mentioned in the text (NOT ENOUGH INFORMATION does not contain the others)."""
    return {label for label in (S, R, N) if label in text}


# ---------------------------------------------------------------- loading
def load_answers(path, keys):
    """Return {key: answer text} for the requested keys."""
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        answers = {r["key"]: r["output"] for r in rows}
    else:  # authors' txt: line i = answer to MedRevQA question i
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].startswith("Question:"):  # BioMistral / PMC-LLaMA format: Question / Answer / ---
            lines = lines[1::3]
        answers = {k: lines[k] for k in keys}
    missing = [k for k in keys if k not in answers]
    if missing:
        print(f"  WARNING: {len(missing)} of {len(keys)} questions have no answer yet - scoring the rest")
    return {k: answers[k] for k in keys if k in answers}


# ---------------------------------------------------------------- metrics
def prf(gold, pred):
    p, r, f, _ = precision_recall_fscore_support(gold, pred, labels=LABELS, average="macro", zero_division=0)
    return 100 * p, 100 * r, 100 * f


def conformance(texts):
    n = len(texts)
    tagged = sum(bool(TAG.search(t)) for t in texts)
    starts = sum(bool(TAG.match(t.strip())) for t in texts)
    none = sum(not mentioned(t) for t in texts)
    multi = sum(len(mentioned(t)) > 1 for t in texts)
    pct = lambda x: f"{x:4d} ({100 * x / n:5.1f}%)"
    print(f"  answers scored            : {n}")
    print(f"  has 'LABEL: <label>' tag   : {pct(tagged)}")
    print(f"  ...right at the start      : {pct(starts)}")
    print(f"  no label mentioned at all  : {pct(none)}")
    print(f"  mentions >1 different label: {pct(multi)}")


def report(path, subset):
    path = Path(path)
    name = "qwen" if "qwen" in path.name.lower() else "biomistral" if "biomistral" in path.name.lower() else None
    print(f"\n=== {path}")
    answers = load_answers(path, subset.key.tolist())
    sub = subset[subset.key.isin(answers)]
    texts = [answers[k] for k in sub.key]
    conformance(texts)

    preds = {"paper": [parse_paper(t) for t in texts], "strict": [parse_strict(t) for t in texts]}
    changed = sum(a != b for a, b in zip(preds["paper"], preds["strict"]))
    print(f"  parsers disagree on        : {changed} answers")

    print(f"\n  {'':22s}{'(b) outdated P / R / F1':>26s}{'(c) latest P / R / F1':>26s}{'F1 diff':>9s}"
          f"{'pred=outdated':>15s}")
    fmt = lambda t: f"{t[0]:5.1f} /{t[1]:5.1f} /{t[2]:5.1f}"
    if name:
        ref = PAPER[name]
        print(f"  {'paper Table 2':22s}{fmt(ref['b']):>26s}{fmt(ref['c']):>26s}{ref['diff']:+9.1f}")
    for parser, pred in preds.items():
        b, c = prf(sub.outdated_label, pred), prf(sub.latest_label, pred)
        old = 100 * sum(p == o for p, o in zip(pred, sub.outdated_label)) / len(pred)
        print(f"  {'ours, ' + parser + ' parser':22s}{fmt(b):>26s}{fmt(c):>26s}{c[2] - b[2]:+9.1f}{old:14.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", help=".jsonl from run_inference.py or an authors' .txt answer file")
    ap.add_argument("--data", default=str(HERE / "data/medchange_subset.csv"))
    args = ap.parse_args()
    subset = pd.read_csv(args.data)
    for path in args.predictions:
        report(path, subset)


if __name__ == "__main__":
    main()
