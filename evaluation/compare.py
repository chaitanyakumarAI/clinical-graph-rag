"""
evaluation/compare.py
Print RAGAS scores per configuration, overall and per question category.

    python evaluation/compare.py [results_dir] [--configs rewrite completeness]

Reads scores_<config>.json (and answers_<config>.json for completeness info)
from results_dir (default evaluation/results). Only questions scored in EVERY
listed config are compared, so the means are over the same questions.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from collections import defaultdict

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS  = ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]


def _load(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")


def compare(results_dir, configs, testset):
    category = {item["id"]: item["category"] for item in testset}
    scores   = {c: _load(os.path.join(results_dir, f"scores_{c}.json")) for c in configs}
    common   = sorted(set.intersection(*(set(s) for s in scores.values())) if scores else set(),
                      key=lambda q: int(q.lstrip("Q")) if q.lstrip("Q").isdigit() else q)
    if not common:
        print("No question is scored in every config yet.")
        return {}

    groups = defaultdict(list)
    for qid in common:
        if category.get(qid) != "not-in-documents":
            groups["all answerable"].append(qid)
        groups[category.get(qid, "?")].append(qid)

    table = {}
    for group, qids in groups.items():
        table[group] = {c: {m: mean(scores[c][q].get(m) for q in qids) for m in METRICS}
                        for c in configs}
        print(f"\n## {group} ({len(qids)} question{'s' if len(qids) != 1 else ''}: {', '.join(qids)})")
        print(f"{'metric':<20}" + "".join(f"{c:>15}" for c in configs)
              + (f"{'Δ':>9}" if len(configs) == 2 else ""))
        for m in METRICS:
            vals = [table[group][c][m] for c in configs]
            delta = f"{vals[1] - vals[0]:+9.3f}" if len(configs) == 2 else ""
            print(f"{m:<20}" + "".join(f"{v:>15.3f}" for v in vals) + delta)

    print("\n## per question — answer_correctness")
    for qid in common:
        row = "".join(f"{scores[c][qid].get('answer_correctness', float('nan')):>15.3f}" for c in configs)
        info = ""
        for c in configs:
            comp = _load(os.path.join(results_dir, f"answers_{c}.json")).get(qid, {}).get("completeness")
            if comp:
                info = f"   missing={comp.get('missing')} repaired={comp.get('repaired')}"
        print(f"{qid:<5}{category.get(qid, '?'):<18}{row}{info}")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", default=os.path.join(EVAL_DIR, "results"))
    parser.add_argument("--configs", nargs="+", default=None)
    args = parser.parse_args()

    configs = args.configs or [c for c in ("baseline", "rewrite", "completeness")
                               if os.path.exists(os.path.join(args.results_dir, f"scores_{c}.json"))]
    compare(args.results_dir, configs, _load(os.path.join(EVAL_DIR, "testset.json")))
