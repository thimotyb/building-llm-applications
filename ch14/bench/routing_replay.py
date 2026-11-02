"""Replay a cascade over stored benchmark results (no model calls).

    python bench/routing_replay.py --tiers jev-1.14.0 deepseek-v4-flash [--md out.md]

For every (ticket, task): tier 1 answers if its confidence >= threshold, else tier 2, ...; still unsure after
the last tier -> human. Reported per threshold, on the tickets all tiers attempted:
  auto-acc   accuracy over the questions resolved without a human
  human%     share of questions sent to a human
  eff-acc    accuracy if the human is always right (upper bound; humans make mistakes too)
  escal%     share of tickets where at least one question reached tier 2 or later
  $/1k, ms   cost per 1000 tickets and mean latency, charging each tier a ticket asks it (approximation:
             the live CascadeJudge re-asks only the unsure questions, so it is slightly cheaper)
Baselines: each single tier alone, at the same thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
from report import RESULTS, TASKS, _truth_index, _vector  # noqa: E402

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def load(label: str) -> dict[int, dict]:
    data = json.loads((RESULTS / f"{label}.json").read_text())
    return {r["id"]: r for r in data["rows"]}


def replay(tiers: list[dict[int, dict]], items: dict[int, dict], ids: list[int], threshold: float) -> dict:
    correct = auto = human = escalated = 0
    cost = latency = 0.0
    for tid in ids:
        deepest = 0
        for task in TASKS:
            truth = _truth_index(task, items[tid])
            for level, rows in enumerate(tiers):
                row = rows[tid]
                last = level == len(tiers) - 1
                if row.get("error") or task not in row:
                    if last:
                        human += 1
                    continue
                p = _vector(task, row[task])
                pred = max(range(len(p)), key=p.__getitem__)
                if p[pred] >= threshold or last:
                    deepest = max(deepest, level)
                    if p[pred] >= threshold:
                        auto += 1
                        correct += pred == truth
                    else:
                        human += 1
                    break
                deepest = max(deepest, level + 1)
        escalated += deepest > 0
        for level in range(deepest + 1):
            cost += tiers[level][tid].get("cost", 0.0)
            latency += tiers[level][tid].get("latency", 0.0)
    n = len(ids) * len(TASKS)
    return {
        "auto_acc": correct / auto if auto else float("nan"), "human": human / n,
        "eff_acc": (correct + human) / n, "escal": escalated / len(ids),
        "cost_1k": cost / len(ids) * 1000, "ms": latency / len(ids) * 1000,
    }


def table(labels: list[str]) -> str:
    items = {i["id"]: i for i in dataset.load()}
    loaded = [load(label) for label in labels]
    ids = sorted(set.intersection(*[set(rows) for rows in loaded]))
    out = [f"\n#### Cascade: {' -> '.join(labels)} -> human  (n={len(ids)} tickets x {len(TASKS)} questions)\n",
           "| setup | threshold | auto-acc | human% | eff-acc | escal% | $/1k tickets | ms/ticket |", "|---|---|---|---|---|---|---|---|"]
    setups = [(" -> ".join(labels), loaded)] + [(f"{label} alone", [rows]) for label, rows in zip(labels, loaded)]
    for name, tiers in setups:
        for thr in THRESHOLDS:
            m = replay(tiers, items, ids, thr)
            out.append(f"| {name} | {thr:.1f} | {m['auto_acc']:.3f} | {m['human'] * 100:.1f} | {m['eff_acc']:.3f} | "
                       f"{m['escal'] * 100:.0f} | {m['cost_1k']:.2f} | {m['ms']:.0f} |")
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="+", required=True)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args()
    text = table(args.tiers)
    print(text)
    if args.md:
        args.md.write_text(args.md.read_text() + "\n" + text + "\n" if args.md.exists() else text + "\n")
