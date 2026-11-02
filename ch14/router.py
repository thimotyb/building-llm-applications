"""Routing for chapter 14: confidence-based cascades.

1. CascadeJudge  - a `Judge` made of tiers (cheap/fast first). Each question is answered by the first tier
   whose confidence reaches the threshold; only the unsure questions go to the next tier; what is still
   unsure after the last tier is flagged `needs_human`. It is itself a Judge, so it plugs into the loop.
2. escalating_generate - the worker starts on the small model and moves to the large one as soon as the
   previous attempt failed (deterministic signal from the harness, no probability involved).

Thresholds are code, not model output: calibrate them with bench/routing_replay.py before trusting them.
"""

from __future__ import annotations

from typing import Callable, Sequence

from judge import Judge, Question, Verdict

GenerateFn = Callable[[str, str, str], tuple[str, float]]


class CascadeJudge:
    def __init__(self, tiers: Sequence[Judge], threshold: float = 0.7):
        if not tiers:
            raise ValueError("a cascade needs at least one tier")
        self.tiers, self.threshold = list(tiers), threshold

    def judge(self, state: str, questions: Sequence[Question]) -> Verdict:
        pending, answers, routes, human = list(questions), {}, {}, []
        latency = cost = 0.0
        for index, tier in enumerate(self.tiers):
            if not pending:
                break
            verdict = tier.judge(state, pending)
            latency, cost = latency + verdict.latency_s, cost + verdict.cost_usd
            last = index == len(self.tiers) - 1
            unsure = []
            for q in pending:
                answer = verdict[q.id]
                if answer.confidence >= self.threshold or last:
                    answers[q.id], routes[q.id] = answer, f"tier{index}:{verdict.backend}"
                    if answer.confidence < self.threshold:
                        human.append(q.id)
                else:
                    unsure.append(q)
            pending = unsure
        return Verdict(answers, latency, cost, "cascade", routes, tuple(human))


def escalating_generate(small: GenerateFn, large: GenerateFn) -> GenerateFn:
    """Small model for the first attempt; large model whenever there is feedback on a failed attempt."""

    def generate(spec: str, code: str, feedback: str) -> tuple[str, float]:
        return (large if feedback else small)(spec, code, feedback)

    return generate
