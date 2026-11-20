"""Rebuild the 512-question changed-knowledge subset (MedChangeQA) exactly as
Code/experiments_results.ipynb does, and save it as one self-contained CSV.

Output columns:
  key              row index into the 16,501 MedRevQA questions (the notebook's `outdated_keys`);
                   this is the question the model is asked and the line of the answer file that is scored
  question         question text at that index
  outdated_label   silver label of that (older) review  -> Table 2 column (b)
  latest_label     silver label of the newest review    -> Table 2 column (c)
  gold_outdated / gold_latest   manually corrected labels from Datasets/MedChangeQA.csv (if the question matches)

Run from the repo root:  python repro/build_subset.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_questions_labels():
    questions, labels = [], []
    for line in (ROOT / "Code/generated_questions_labels.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split(" ||| ")
        questions.append(parts[0].split("QUESTION: ")[1].strip())
        labels.append(parts[1].split("LABEL: ")[1].strip())
    return questions, labels


def doi_groups():
    """Group review versions by DOI prefix (same logic as notebook cell 4)."""
    df = pd.read_csv(ROOT / "Code/FullScraped.csv", index_col=0)
    dois = [d.split("doi: ")[1].strip() if "doi: " in d else "" for d in df.date.tolist()]
    first_index = {}
    for idx, d in enumerate(dois):
        first_index.setdefault(d, idx)
    doisorted = sorted(dois)
    groups, i = [], 0
    while i < len(doisorted):
        pivot = doisorted[i]
        if pivot == "":
            i += 1
            continue
        group = []
        while i < len(doisorted) and doisorted[i].startswith(pivot):
            group.append(first_index[doisorted[i]])
            i += 1
        groups.append(list(dict.fromkeys(group)))
    return sorted(groups)


def main():
    questions, labels = load_questions_labels()
    rows = []
    for group in doi_groups():
        if len(group) < 2:
            continue
        keys = sorted(group)
        latest = labels[keys[0]]  # the notebook treats the lowest index as the newest version
        for k in keys:
            if labels[k] != latest:  # first (most recent) version whose verdict differs
                rows.append({"key": k, "question": questions[k],
                             "outdated_label": labels[k], "latest_label": latest,
                             "latest_key": keys[0]})
                break

    sub = pd.DataFrame(rows)
    gold = pd.read_csv(ROOT / "Datasets/MedChangeQA.csv", index_col=0)
    gold = gold.drop_duplicates("Question").set_index("Question")
    sub["gold_latest"] = sub.question.map(gold["Newest Label"])
    sub["gold_outdated"] = sub.question.map(gold["Outdated Label"])

    out = ROOT / "repro/data/medchange_subset.csv"
    sub.to_csv(out, index=False)
    print(f"{len(sub)} questions -> {out}")
    print(f"matched to MedChangeQA.csv gold labels: {sub.gold_latest.notna().sum()}")
    both = sub.dropna(subset=["gold_latest"])
    print(f"silver==gold  outdated: {(both.outdated_label == both.gold_outdated).mean():.1%}  "
          f"latest: {(both.latest_label == both.gold_latest).mean():.1%}")


if __name__ == "__main__":
    main()
