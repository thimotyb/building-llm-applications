import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge import Answer, FakeJudge, Noul, Verdict  # noqa: E402
from router import CascadeJudge, escalating_generate  # noqa: E402

Q = [Noul(id="a", prompt="a?"), Noul(id="b", prompt="b?")]


def noul(qid, p):
    return Answer(qid, p >= 0.5, {"true": p, "false": 1 - p})


class Tier:
    def __init__(self, name, probs, cost=1.0, latency=1.0):
        self.name, self.probs, self.cost, self.latency, self.asked = name, probs, cost, latency, []

    def judge(self, state, questions):
        self.asked.append([q.id for q in questions])
        return Verdict({q.id: noul(q.id, self.probs[q.id]) for q in questions},
                       self.latency, self.cost, self.name)


# In a cascade, questions the cheap judge is sure about stop there; only the unsure ones go to the
# stronger judge, and cost, latency and routes add up.
def test_only_unsure_questions_go_to_the_next_tier():
    cheap, strong = Tier("cheap", {"a": 0.95, "b": 0.55}, 0.1), Tier("strong", {"a": 0.5, "b": 0.9}, 1.0)
    verdict = CascadeJudge([cheap, strong], 0.7).judge("s", Q)
    assert cheap.asked == [["a", "b"]] and strong.asked == [["b"]]
    assert verdict.routes == {"a": "tier0:cheap", "b": "tier1:strong"}
    assert verdict.cost_usd == pytest.approx(1.1) and verdict.latency_s == 2.0
    assert verdict.needs_human == ()


# If the first tier is confident on every question, the second (more expensive) tier is never
# called.
def test_confident_first_tier_never_calls_the_second():
    strong = Tier("strong", {"a": 0.5, "b": 0.5})
    CascadeJudge([Tier("cheap", {"a": 0.9, "b": 0.05}), strong]).judge("s", Q)
    assert strong.asked == []


# A question still below the threshold after the last tier is flagged needs_human instead of being
# answered.
def test_still_unsure_after_last_tier_is_flagged_for_a_human():
    verdict = CascadeJudge([Tier("c", {"a": 0.6, "b": 0.9}), Tier("s", {"a": 0.55, "b": 0.9})]).judge("s", Q)
    assert verdict.needs_human == ("a",) and verdict["a"].confidence == pytest.approx(0.55)


# The worker starts on the small model and moves to the large one once there is feedback from a
# failed attempt.
def test_escalating_generate_switches_model_on_feedback():
    log = []
    gen = escalating_generate(lambda *a: (log.append("small") or "x", 0.0), lambda *a: (log.append("large") or "y", 0.0))
    gen("spec", "code", "")
    gen("spec", "code", "The harness failed")
    assert log == ["small", "large"]
