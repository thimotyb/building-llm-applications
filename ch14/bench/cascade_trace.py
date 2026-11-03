"""Run a REAL cascade (Jev first, an LLM judge on the unsure questions) and save a trace.

    JUDGE_PROVIDER=deepseek python bench/cascade_trace.py --out traces/cascade-live.jsonl [--threshold 0.7] [--tickets 12]

Unlike routing_replay.py (which recomputes a cascade from stored answers), this calls the judges live.
For each ticket it records every call to every tier (which questions were asked, what came back, latency,
cost) and the final answer of each question with the tier that produced it and whether it was correct.
Output is JSON Lines: one header record, then one record per ticket.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset  # noqa: E402
from jev_judge import JevJudge  # noqa: E402
from judge import LLMJudge  # noqa: E402
from llm_factory import resolve_model  # noqa: E402
from router import CascadeJudge  # noqa: E402
from run_bench import QUESTIONS  # noqa: E402


class Recorder:
    """Wraps a judge and remembers every call, so the trace shows what each tier was asked and answered."""

    def __init__(self, name: str, inner):
        self.name, self.inner, self.calls = name, inner, []

    def judge(self, state, questions):
        verdict = self.inner.judge(state, questions)
        self.calls.append({
            "tier": self.name,
            "asked": [q.id for q in questions],
            "answers": {qid: {"value": a.value, "confidence": round(a.confidence, 3)}
                        for qid, a in verdict.answers.items()},
            "latency_s": round(verdict.latency_s, 3),
            "cost_usd": round(verdict.cost_usd, 6),
        })
        return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="JSON Lines file to write")
    # A question whose confidence is below this goes to the next tier; below it after the last tier -> person
    parser.add_argument("--threshold", type=float, default=0.7)
    # How many tickets: the hand-written tricky ones come first (they are the ones that escalate),
    # then template-built ones, so the trace shows both a fast path and an escalation path.
    parser.add_argument("--tickets", type=int, default=12)
    args = parser.parse_args()

    items = dataset.load()
    hard = [i for i in items if i.get("hard")]
    easy = [i for i in items if not i.get("hard")]
    chosen = hard[: args.tickets - 3] + easy[:3]

    fast = Recorder("jev", JevJudge())
    strong_model = resolve_model(role="judge")
    strong = Recorder(strong_model.label, LLMJudge(strong_model))
    cascade = CascadeJudge([fast, strong], args.threshold)

    rows, total_cost = [], 0.0
    for item in chosen:
        fast.calls.clear(), strong.calls.clear()
        verdict = cascade.judge(item["text"], QUESTIONS)
        total_cost += verdict.cost_usd
        final = {}
        for qid, answer in verdict.answers.items():
            correct = answer.value == item[qid]
            final[qid] = {"value": answer.value, "confidence": round(answer.confidence, 3),
                          "answered_by": verdict.routes[qid], "truth": item[qid], "correct": correct}
        rows.append({
            "ticket": item["id"], "hard": bool(item.get("hard")), "text": item["text"],
            "calls": [*fast.calls, *strong.calls], "final": final,
            "needs_person": list(verdict.needs_human),
            "latency_s": round(verdict.latency_s, 3), "cost_usd": round(verdict.cost_usd, 6),
        })
        escalated = [q for q, f in final.items() if not f["answered_by"].startswith("tier0")]
        print(f"ticket {item['id']:3d}  escalated={escalated or '-'}  person={list(verdict.needs_human) or '-'}  "
              f"${verdict.cost_usd:.5f}")

    answered = [f for r in rows for f in r["final"].values()]
    header = {"node": "cascade-run", "tiers": ["jev", strong_model.label], "threshold": args.threshold,
              "tickets": len(rows), "questions": len(answered),
              "escalated_questions": sum(not f["answered_by"].startswith("tier0") for f in answered),
              "questions_to_person": sum(len(r["needs_person"]) for r in rows),
              "correct": sum(f["correct"] for f in answered), "cost_usd": round(total_cost, 6)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [header, *rows]) + "\n", encoding="utf-8")
    print(json.dumps(header))


if __name__ == "__main__":
    main()
