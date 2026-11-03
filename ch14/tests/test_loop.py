import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_loop import build_graph, run  # noqa: E402
from harness import CASE_DIR, run_harness  # noqa: E402
from judge import Answer, Verdict  # noqa: E402

STARTER = (CASE_DIR / "payment_retry.py").read_text()
GOOD = (CASE_DIR / "expected" / "payment_retry.py").read_text()
SPEC = (CASE_DIR / "spec.md").read_text()
CREEP = GOOD.replace("max_attempts = 3", "max_attempts = 3\n    import time  # noqa\n    time.sleep(0)")


def noul(qid, p):
    return Answer(qid, p >= 0.5, {"true": p, "false": 1 - p})


class ScriptedJudge:
    """Returns one (p_spec_ok, p_scope_creep) pair per call."""

    def __init__(self, *pairs):
        self.pairs, self.calls = list(pairs), 0

    def judge(self, state, questions):
        p_ok, p_creep = self.pairs[min(self.calls, len(self.pairs) - 1)]
        self.calls += 1
        return Verdict({"spec_ok": noul("spec_ok", p_ok), "scope_creep": noul("scope_creep", p_creep)},
                       backend="scripted")


def scripted_generator(*codes):
    calls = {"n": 0, "feedback": []}

    def generate(spec, code, feedback):
        calls["feedback"].append(feedback)
        out = codes[min(calls["n"], len(codes) - 1)]
        calls["n"] += 1
        return out, 0.0

    generate.calls = calls
    return generate


def test_harness_fails_the_starter_and_passes_the_reference():
    assert not run_harness(STARTER).passed
    report = run_harness(GOOD)
    assert report.passed and report.tests_run == 3


def test_harness_rejects_out_of_scope_code_even_if_tests_pass():
    report = run_harness(CREEP)
    assert not report.passed and any("sleep" in v for v in report.static_violations)


def test_loop_repairs_then_accepts_without_human():
    generator = scripted_generator(STARTER, GOOD)
    graph = build_graph(generator, ScriptedJudge((0.95, 0.05)))
    final = run(graph, SPEC, STARTER, max_iters=4, human="stop", thread="t1")
    assert final["status"] == "accepted" and final["iteration"] == 2
    assert "The harness failed" in generator.calls["feedback"][1]  # failure fed back to the generator
    assert [r["node"] for r in final["trace"]] == ["generate", "harness", "generate", "harness", "judge"]


def test_uncertain_judge_goes_to_human_who_can_approve():
    graph = build_graph(scripted_generator(GOOD), ScriptedJudge((0.55, 0.5)))
    final = run(graph, SPEC, STARTER, max_iters=4, human="approve", thread="t2")
    assert final["status"] == "accepted"
    assert final["trace"][-1] == {"iter": 1, "node": "human", "decision": "approve"}


def test_human_can_stop_the_loop():
    graph = build_graph(scripted_generator(GOOD), ScriptedJudge((0.55, 0.5)))
    assert run(graph, SPEC, STARTER, 4, "stop", thread="t3")["status"] == "stopped"


def test_budget_exhaustion_escalates_to_human():
    graph = build_graph(scripted_generator(STARTER), ScriptedJudge((0.9, 0.1)))
    final = run(graph, SPEC, STARTER, max_iters=2, human="stop", thread="t4")
    assert final["iteration"] == 2 and final["status"] == "stopped"
    assert final["trace"][-1]["node"] == "human"


def test_judge_revise_loops_back_with_feedback():
    generator = scripted_generator(GOOD, GOOD)
    graph = build_graph(generator, ScriptedJudge((0.1, 0.9), (0.95, 0.05)))
    final = run(graph, SPEC, STARTER, max_iters=4, human="stop", thread="t5")
    assert final["status"] == "accepted" and final["iteration"] == 2
    assert "outside the spec" in generator.calls["feedback"][1]


def test_static_violation_reaches_the_generator_as_feedback():
    generator = scripted_generator(CREEP, GOOD)
    graph = build_graph(generator, ScriptedJudge((0.95, 0.05)))
    final = run(graph, SPEC, STARTER, max_iters=4, human="stop", thread="t6")
    assert final["status"] == "accepted" and final["iteration"] == 2
    assert "no sleeping/backoff" in generator.calls["feedback"][1]  # the reason, not only an all-green test log
