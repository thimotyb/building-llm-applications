"""Merge bench/results/*.json into one comparison table (accuracy, calibration, speed, cost).

    python bench/report.py [--md out.md]

Metrics per task, from the probability of the PREDICTED class:
  acc    accuracy
  ECE    expected calibration error (10 bins; 0 = perfectly calibrated)
  Brier  mean squared error of the probability vector (lower is better)
  conf   mean confidence; conf >> acc means over-confident
  acc@c  accuracy on the items the judge is confident about (p >= 0.7): what a router can trust
  cov@c  share of items that are confident (the rest would go to a bigger model or a human)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
TASKS = ("team", "urgent", "refund")
CONF = 0.7


def _vector(task: str, value) -> list[float]:
    if task == "team":
        v = [float(value.get(t, 0.0)) for t in dataset.TEAMS]
        s = sum(v) or 1.0
        return [x / s for x in v]
    return [1 - float(value), float(value)]  # [P(false), P(true)]


def _truth_index(task: str, item: dict) -> int:
    return dataset.TEAMS.index(item[task]) if task == "team" else int(item[task])


def task_metrics(task: str, rows: list[dict], items: dict[int, dict]) -> dict:
    recs = []
    for r in rows:
        if r.get("error") or task not in r:
            continue
        p, t = _vector(task, r[task]), _truth_index(task, items[r["id"]])
        pred = max(range(len(p)), key=p.__getitem__)
        brier = sum((p[k] - (1.0 if k == t else 0.0)) ** 2 for k in range(len(p)))
        recs.append((p[pred], pred == t, brier))
    if not recs:
        return {}
    n = len(recs)
    ece = 0.0
    for b in range(10):
        bin_ = [r for r in recs if (min(int(r[0] * 10), 9)) == b]
        if bin_:
            ece += len(bin_) / n * abs(sum(r[1] for r in bin_) / len(bin_) - sum(r[0] for r in bin_) / len(bin_))
    sure = [r for r in recs if r[0] >= CONF]
    return {
        "acc": sum(r[1] for r in recs) / n, "ece": ece, "brier": sum(r[2] for r in recs) / n,
        "conf": sum(r[0] for r in recs) / n,
        "acc_c": sum(r[1] for r in sure) / len(sure) if sure else float("nan"), "cov_c": len(sure) / n,
    }


def build_table() -> str:
    items = {i["id"]: i for i in dataset.load()}
    out = []
    files = sorted(RESULTS.glob("*.json"))
    if not files:
        return "No results yet: run bench/run_bench.py first."
    loaded = [json.loads(f.read_text()) for f in files]
    for task in TASKS:
        out += [f"\n### {task}\n", "| backend | n | acc | ECE | Brier | conf | acc@c | cov@c |", "|---|---|---|---|---|---|---|---|"]
        for res in loaded:
            m = task_metrics(task, res["rows"], items)
            if m:
                out.append(f"| {res['meta']['label']} | {res['meta']['n']} | {m['acc']:.2f} | {m['ece']:.3f} | "
                           f"{m['brier']:.3f} | {m['conf']:.2f} | {m['acc_c']:.2f} | {m['cov_c']:.2f} |")
    out += ["\n### Speed and cost (all three questions per ticket in one call)\n",
            "| backend | errors | mean latency / ticket | total cost | note |", "|---|---|---|---|---|"]
    for res in loaded:
        rows = res["rows"]
        ok = [r for r in rows if not r.get("error")]
        lat = sum(r["latency"] for r in ok) / len(ok) if ok else float("nan")
        note = res["meta"].get("note") or res["meta"].get("model") or ""
        out.append(f"| {res['meta']['label']} | {len(rows) - len(ok)} | {lat * 1000:.0f} ms | "
                   f"${sum(r['cost'] for r in rows):.4f} | {note} |")
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=Path)
    parser.add_argument("--notes", type=Path, help="markdown file with <!-- BENCH:START/END --> markers to refresh")
    args = parser.parse_args()
    table = build_table()
    print(table)
    if args.md:
        args.md.write_text("# Judge benchmark\n" + table + "\n")
    if args.notes:
        import re

        text = args.notes.read_text()
        block = f"<!-- BENCH:START -->\n{table}\n<!-- BENCH:END -->"
        new, count = re.subn(r"<!-- BENCH:START -->.*?<!-- BENCH:END -->", lambda _: block, text, flags=re.S)
        if count != 1:
            raise SystemExit("markers <!-- BENCH:START --> / <!-- BENCH:END --> not found exactly once")
        args.notes.write_text(new)
        print(f"\nupdated {args.notes}")
