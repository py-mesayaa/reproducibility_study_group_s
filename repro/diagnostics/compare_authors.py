"""How close are our BioMistral answers to the authors' own answers, question by question?

With greedy decoding the model is deterministic, so if we hit the authors' exact setup our
answers should be (nearly) word-for-word identical to theirs. This is a much sharper test
than comparing aggregate scores.

  python repro/diagnostics/compare_authors.py repro/outputs/biomistral_8bit_table6.jsonl repro/diagnostics/outputs/*.jsonl
"""
import argparse
import collections
import difflib
import json
import re
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parents[1]  # repro/
sys.path.insert(0, str(REPRO))
from evaluate import parse_paper  # noqa: E402

AUTHORS = REPRO.parent / "Code/GeneratedAnswers/biomistral_answers.txt"


def normalise(text):
    return re.sub(r"\s+", " ", text).strip().lower()


def load_authors():
    lines = AUTHORS.read_text(encoding="utf-8").splitlines()[1::3]  # Question / Answer / --- blocks
    # Their script logged "Answer: <output>" and replaced newlines with " ||| ".
    return [line[len("Answer: "):].replace(" ||| ", "\n") if line.startswith("Answer: ") else line for line in lines]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    authors = load_authors()

    print(f"{'file':48s} {'n':>4s} {'same label':>11s} {'starts ANSWER':>14s} {'text sim.':>10s}"
          f" {'identical':>10s}  predicted labels")
    for path in args.files:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        same_label = starts = identical = 0
        sims, dist = [], collections.Counter()
        for r in rows:
            ours, theirs = r["output"], authors[r["key"]]
            pred = parse_paper(ours)
            dist[pred[:3]] += 1
            same_label += pred == parse_paper(theirs)
            starts += ours.lstrip().upper().startswith("ANSWER")
            a, b = normalise(ours)[:300], normalise(theirs)[:300]
            sims.append(difflib.SequenceMatcher(None, a, b).ratio())
            identical += a[:100] == b[:100]
        n = len(rows)
        sims.sort()
        print(f"{Path(path).name:48s} {n:4d} {100 * same_label / n:10.0f}% {100 * starts / n:13.0f}%"
              f" {sims[n // 2]:10.2f} {100 * identical / n:9.0f}%  {dict(dist)}")
    print("\nauthors' answers start with ANSWER: "
          f"{100 * sum(a.lstrip().upper().startswith('ANSWER') for a in authors) / len(authors):.0f}% (all 16,501)")
    print("text sim. = median similarity of the first 300 characters (1.0 = identical); "
          "identical = first 100 characters match exactly")


if __name__ == "__main__":
    main()
