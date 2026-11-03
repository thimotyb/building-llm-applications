"""Live demo: an agentic loop with harness and judges, narrated in the log.

Starting situation: a payment-retry function that makes a single attempt, a specification with three
acceptance scenarios, and a test suite the agent cannot edit. The loop generates a candidate, the harness
runs tests and static rules, the judge answers typed questions with probabilities, and the code decides
whether to repeat, ask a person, or stop.

    python demo_live.py --mode replay                    # offline, scripted, no API key needed
    python demo_live.py --mode replay --story short      # shortest path
    python demo_live.py --mode live --inject off-by-one  # real models; the first attempt is a planted flaw
    JUDGE_BACKEND=cascade python demo_live.py --mode live --human ask
Options: --human ask|approve|stop  --max-iters N  --delay SECONDS  --log FILE  --out FILE  --no-color
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import time
from pathlib import Path

from agentic_loop import ACCEPT, QUESTIONS, REJECT, build_graph, llm_generate, run
from harness import CASE_DIR, HarnessReport, run_harness
from judge import Answer, Verdict
from judge_factory import make_judge
from llm_factory import resolve_model
from router import escalating_generate

WIDTH = 78
TESTS = {
    "test_approval_on_first_attempt_stops": "AC-1  approval on the first attempt stops the loop",
    "test_transient_then_approval_retries_and_stops": "AC-2  transient error, then approval",
    "test_three_transient_errors_suggest_alternative": "AC-3  three transient errors, suggest an alternative",
}
ANSI = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "dim": "90", "bold": "1", "magenta": "35"}

REFERENCE = (CASE_DIR / "expected" / "payment_retry.py").read_text()
STARTER = (CASE_DIR / "payment_retry.py").read_text()
FLAWS = {  # planted candidates: each fails a different layer of the harness
    "off-by-one": REFERENCE.replace("max_attempts = 3", "max_attempts = 2"),
    "backoff": REFERENCE.replace("                continue", "                time.sleep(0.2)\n                continue")
    .replace("from dataclasses import", "import time\nfrom dataclasses import", 1),
    "scope-creep": REFERENCE + "\n\ndef describe(result: PaymentResult) -> str:\n    return f\"{result.status} after {result.attempts} attempt(s)\"\n",
}


class Log:
    def __init__(self, color: bool, logfile: Path | None, delay: float):
        self.color, self.delay = color, delay
        self.fh = logfile.open("w", encoding="utf-8") if logfile else None

    def c(self, text: str, *styles: str) -> str:
        if not self.color:
            return text
        return f"\033[{';'.join(ANSI[s] for s in styles)}m{text}\033[0m"

    def out(self, text: str = "") -> None:
        print(text, flush=True)
        if self.fh:
            self.fh.write(re.sub(r"\033\[[0-9;]*m", "", text) + "\n")
            self.fh.flush()

    def banner(self, icon: str, title: str, style: str = "cyan") -> None:
        time.sleep(self.delay)
        self.out()
        self.out(self.c("=" * WIDTH, "dim"))
        self.out(self.c(f"{icon} {title}", style, "bold"))

    def step(self, icon: str, text: str, style: str | None = None, indent: int = 0) -> None:
        time.sleep(self.delay / 3)
        self.out("  " * indent + f"{icon} " + (self.c(text, style) if style else text))

    def quote(self, text: str, limit: int = 6, indent: int = 2) -> None:
        for line in text.strip().splitlines()[:limit]:
            self.out(self.c("  " * indent + "│ " + line[: WIDTH - 8], "dim"))

    def diff(self, old: str, new: str, limit: int = 14) -> None:
        lines = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
                 if not l.startswith(("---", "+++", "@@"))]
        for line in lines[:limit]:
            self.out("    " + self.c(line[: WIDTH - 6], "green" if line.startswith("+") else "red"))
        if len(lines) > limit:
            self.out(self.c(f"    … {len(lines) - limit} more changed lines", "dim"))

    def bar(self, p: float, width: int = 20) -> str:
        filled = round(p * width)
        return "█" * filled + "░" * (width - filled)


def parse_tests(output: str) -> list[tuple[str, str]]:
    found = dict(re.findall(r"^(test_\w+) \(.*?\) \.\.\. (ok|FAIL|ERROR)", output, flags=re.M))
    order = {name: i for i, name in enumerate(TESTS)}
    return sorted(found.items(), key=lambda kv: order.get(kv[0], 99))


def summarize_feedback(feedback: str) -> list[str]:
    """Distil the harness feedback to what the model actually needs: which scenario failed and why."""
    if "The harness failed" not in feedback:
        return feedback.strip().splitlines()[:5]
    lines = []
    for block in re.split(r"^={10,}$", feedback, flags=re.M):
        name = re.search(r"^(?:FAIL|ERROR): (test_\w+)", block, flags=re.M)
        reason = re.findall(r"^(?:AssertionError|\w*Error).*$", block, flags=re.M)
        if name:
            lines.append(f"{TESTS.get(name.group(1), name.group(1)).split('  ')[0]} failed: "
                         f"{reason[-1] if reason else 'see test output'}")
    rules = re.search(r"Static rule violations \(fix these\):\n((?:- .*\n?)+)", feedback)
    if rules:
        lines += [f"static rule: {r[2:]}" for r in rules.group(1).strip().splitlines()]
    return lines or feedback.strip().splitlines()[:5]


def show_harness(log: Log, report: HarnessReport, seconds: float) -> None:
    for name, status in parse_tests(report.output):
        ok = status == "ok"
        log.step("✅" if ok else "❌", TESTS.get(name, name), "green" if ok else "red", indent=2)
    for violation in report.static_violations:
        log.step("⛔", f"static rule: {violation}", "red", indent=2)
    icon, style = ("✅", "green") if report.passed else ("❌", "red")
    log.step(icon, f"Harness result: {report.summary}  ({seconds:.1f}s)", style, indent=1)


class Story:
    """Collects one line per iteration for the closing table."""

    def __init__(self) -> None:
        self.rows: dict[int, dict] = {}
        self.cost = 0.0

    def row(self, i: int) -> dict:
        return self.rows.setdefault(i, {"gen": "-", "harness": "-", "judge": "-", "next": "-"})


class LoggedGenerate:
    def __init__(self, inner, log: Log, story: Story, label: str, max_iters: int, plant: list[str]):
        self.inner, self.log, self.story, self.label = inner, log, story, label
        self.max_iters, self.plant, self.n, self.prev = max_iters, list(plant), 0, STARTER

    def __call__(self, spec: str, code: str, feedback: str):
        self.n += 1
        log = self.log
        log.banner("🔁", f"ITERATION {self.n} / {self.max_iters}")
        log.step("🤖", f"Generate  [{self.label}]", "bold")
        if feedback:
            log.step("📨", "Feedback passed to the model:", indent=1)
            log.quote("\n".join(summarize_feedback(feedback)), limit=5, indent=2)
        start = time.perf_counter()
        if self.plant:
            flaw = self.plant.pop(0)
            candidate, cost = FLAWS[flaw], 0.0
            log.step("🎭", f"DEMO: planted candidate '{flaw}' instead of the model output", "yellow", indent=1)
        else:
            candidate, cost = self.inner(spec, code, feedback)
        seconds = time.perf_counter() - start
        self.story.cost += cost
        log.step("📝", f"Candidate: {len(candidate.splitlines())} lines, {len(candidate)} chars, "
                       f"{seconds:.1f}s, ${cost:.4f}", indent=1)
        log.diff(self.prev, candidate)
        self.prev = candidate
        self.story.row(self.n)["gen"] = f"{seconds:.1f}s"
        return candidate, cost


class LoggedHarness:
    def __init__(self, log: Log, story: Story, counter):
        self.log, self.story, self.counter = log, story, counter

    def __call__(self, code: str) -> HarnessReport:
        self.log.step("🧪", "Harness: read-only tests in a clean directory + static scope rules", "bold")
        start = time.perf_counter()
        report = run_harness(code)
        show_harness(self.log, report, time.perf_counter() - start)
        self.story.row(self.counter())["harness"] = ("✅ " if report.passed else "❌ ") + (
            f"{report.tests_run - report.failures}/{report.tests_run} ok"
            + (", static" if report.static_violations else ""))
        return report


class ReplayJudge:
    """Scripted judge (offline demo): one (P(spec ok), P(scope creep)) pair per call."""

    def __init__(self, *pairs: tuple[float, float]):
        self.pairs, self.calls = list(pairs), 0

    def judge(self, state, questions):
        p_ok, p_creep = self.pairs[min(self.calls, len(self.pairs) - 1)]
        self.calls += 1
        noul = lambda q, p: Answer(q, p >= 0.5, {"true": p, "false": 1 - p})  # noqa: E731
        return Verdict({"spec_ok": noul("spec_ok", p_ok), "scope_creep": noul("scope_creep", p_creep)},
                       latency_s=0.2, backend="replay")


class LoggedJudge:
    def __init__(self, inner, log: Log, story: Story, counter):
        self.inner, self.log, self.story, self.counter = inner, log, story, counter
        self.last: dict[str, float] = {}

    def judge(self, state, questions):
        log = self.log
        log.step("⚖️ ", f"Judge: typed questions answered with probabilities", "bold")
        verdict = self.inner.judge(state, questions)
        wants = {"spec_ok": ("spec_ok     ", f"want ≥ {ACCEPT:.2f}", lambda p: p >= ACCEPT),
                 "scope_creep": ("scope_creep ", f"want ≤ {REJECT:.2f}", lambda p: p <= REJECT)}
        for q in questions:
            p = verdict[q.id].probabilities["true"]
            label, rule, good = wants[q.id]
            style = "green" if good(p) else "red" if (p <= REJECT if q.id == "spec_ok" else p >= ACCEPT) else "yellow"
            via = f"   ← {verdict.routes[q.id]}" if verdict.routes else ""
            log.step("📊", f"{label} {log.bar(p)} {p:.2f}  ({rule}){via}", style, indent=2)
        meta = f"judge backend: {verdict.backend}, {verdict.latency_s:.1f}s, ${verdict.cost_usd:.5f}"
        if verdict.needs_human:
            meta += f", still unsure: {', '.join(verdict.needs_human)}"
        log.step("🏷️ ", meta, "dim", indent=2)
        self.last = {q.id: verdict[q.id].probabilities["true"] for q in questions}
        self.story.cost += verdict.cost_usd
        return verdict


def build_story(kind: str):
    """Scripted candidates and judge scores for the offline replay."""
    if kind == "short":
        return [FLAWS["off-by-one"], REFERENCE], [(0.96, 0.07)]
    return [FLAWS["off-by-one"], FLAWS["backoff"], FLAWS["scope-creep"], REFERENCE], [(0.92, 0.84), (0.74, 0.45)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["replay", "live"], default="replay")
    parser.add_argument("--story", choices=["full", "short"], default="full", help="replay storyline")
    parser.add_argument("--inject", default="off-by-one", help="live: comma list of planted first attempts "
                        f"({', '.join(FLAWS)}) or 'none'")
    parser.add_argument("--human", choices=["ask", "approve", "stop"], default="approve")
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--escalate", action="store_true", help="live: large model after a failed attempt")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds of pause between sections (projector pacing)")
    parser.add_argument("--log", type=Path, help="also write a plain-text log file")
    parser.add_argument("--out", type=Path, help="write the accepted payment_retry.py here")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    log = Log(color=sys.stdout.isatty() and not args.no_color, logfile=args.log, delay=args.delay)
    story = Story()
    state = {"iter": 0}
    counter = lambda: state["iter"] or 1  # noqa: E731

    # ---- initial situation ------------------------------------------------------------------
    log.banner("🎬", "AGENTIC LOOP DEMO  ·  harness + judges", "magenta")
    spec = (CASE_DIR / "spec.md").read_text()
    log.step("📋", "Goal (from the specification)", "bold")
    for line in ("FR-1  retry transient gateway errors, at most 3 attempts",
                 "FR-2  stop as soon as a payment is approved",
                 "FR-3  after 3 transient errors: fail and suggest an alternative payment method"):
        log.step("•", line, indent=1)
    log.step("🚫", "Out of scope: backoff, 4th attempt, real gateways, idempotency of the whole checkout", indent=1)

    log.banner("🧪", "STARTING SITUATION  ·  the starter code against the tests")
    start = time.perf_counter()
    baseline = run_harness(STARTER)
    show_harness(log, baseline, time.perf_counter() - start)
    log.step("⚠️ ", "The starter makes a single attempt: the loop has real work to do", "yellow")

    # ---- configuration ----------------------------------------------------------------------
    if args.mode == "replay":
        candidates, pairs = build_story(args.story)
        generator = lambda spec_, code, fb: (candidates.pop(0), 0.0)  # noqa: E731
        judge_inner, label, plant = ReplayJudge(*pairs), "replay (scripted, no network)", []
    else:
        worker = resolve_model()
        generator = llm_generate(worker)
        label = worker.label
        if args.escalate:
            large = resolve_model(role="large")
            generator = escalating_generate(generator, llm_generate(large))
            label += f" → {large.label} after a failure"
        judge_inner = make_judge()
        plant = [] if args.inject == "none" else [f for f in args.inject.split(",") if f]
        unknown = [f for f in plant if f not in FLAWS]
        if unknown:
            sys.exit(f"Unknown flaw(s): {', '.join(unknown)}. Choose from: {', '.join(FLAWS)}")

    log.banner("⚙️ ", "CONFIGURATION")
    log.step("🤖", f"Worker : {label}", indent=1)
    log.step("⚖️ ", f"Judge  : {type(judge_inner).__name__}", indent=1)
    log.step("🎚️ ", f"Thresholds in code: accept spec_ok ≥ {ACCEPT}, scope_creep ≤ {REJECT}; between = ask a person", indent=1)
    log.step("🔢", f"Budget : {args.max_iters} iterations; person: --human {args.human}", indent=1)

    judge = LoggedJudge(judge_inner, log, story, counter)
    harness = LoggedHarness(log, story, counter)
    gen = LoggedGenerate(generator, log, story, label, args.max_iters, plant)

    def on_row(row: dict) -> None:
        node, i = row["node"], row["iter"]
        if node == "judge":
            decision = row["decision"]
            story.row(i)["judge"] = f"{row['p_spec_ok']:.2f} / {row['p_scope_creep']:.2f}"
            icon, style, msg = {
                "accept": ("🧭", "green", "Decide: ACCEPT  (spec_ok high, scope_creep low)"),
                "revise": ("🧭", "yellow", "Decide: REVISE  → findings go back to the generator"),
                "unsure": ("🧭", "yellow", "Decide: UNSURE → neither accept nor reject, ask a person"),
            }[decision]
            story.row(i)["next"] = {"accept": "accepted", "revise": "revise", "unsure": "person"}[decision]
            log.step(icon, msg, style, indent=1)
        elif node == "harness":
            if not row["result"].startswith("PASS"):
                story.row(i)["next"] = "repair"
                log.step("🧭", "Decide: harness failed → failure text goes back as feedback (judge is skipped)",
                         "yellow", indent=1)
        elif node == "human":
            story.row(i)["next"] = {"approve": "approved", "stop": "stopped", "feedback": "feedback"}[row["decision"]]
            log.step("🙋", f"Person decided: {row['decision']}", "green" if row["decision"] == "approve" else "yellow",
                     indent=1)

    def on_interrupt(info: dict) -> None:
        log.banner("🙋", "HUMAN IN THE LOOP", "yellow")
        reasons = []
        if info.get("tests_passed"):
            reasons.append("tests pass, but the judge is not sure enough")
        else:
            reasons.append("iteration budget exhausted with failing tests")
        log.step("❓", reasons[0], indent=1)
        if judge.last:
            log.step("📊", "judge: " + ", ".join(f"{k}={v:.2f}" for k, v in judge.last.items()), indent=1)
        log.step("📄", "Code awaiting approval:", indent=1)
        log.quote(info["code"], limit=6, indent=2)
        if args.human != "ask":
            log.step("⌨️ ", f"(demo) automatic answer: {args.human}", "dim", indent=1)

    # ---- the loop ---------------------------------------------------------------------------
    class Wired:
        """Generate wrapper that also advances the iteration counter used by the other wrappers."""

        def __call__(self, spec_, code, fb):
            state["iter"] += 1
            return gen(spec_, code, fb)

    graph = build_graph(Wired(), judge, harness=harness)
    started = time.perf_counter()
    final = run(graph, spec, STARTER, args.max_iters, args.human, thread="demo",
                on_row=on_row, on_interrupt=on_interrupt)
    elapsed = time.perf_counter() - started

    # ---- outcome ----------------------------------------------------------------------------
    log.banner("🏁", "OUTCOME", "magenta")
    status = final["status"]
    log.step("✅" if status == "accepted" else "🛑", f"Status: {status.upper()}  after {final['iteration']} iteration(s)",
             "green" if status == "accepted" else "red", indent=1)
    header = f"  {'iter':<5}{'generate':<10}{'harness':<18}{'judge ok/creep':<16}{'then'}"
    log.out(log.c(header, "bold"))
    for i, r in sorted(story.rows.items()):
        log.out(f"  {i:<5}{r['gen']:<10}{r['harness']:<18}{r['judge']:<16}{r['next']}")
    log.step("💰", f"Cost: ${story.cost:.4f}   ⏱️  Wall time: {elapsed:.1f}s", indent=1)

    if status == "accepted":
        log.step("🧪", "Final re-run of the harness on the accepted code", "bold", indent=1)
        start = time.perf_counter()
        show_harness(log, run_harness(final["code"]), time.perf_counter() - start)
        log.step("🧬", "Change from the starter to the accepted code:", indent=1)
        log.diff(STARTER, final["code"], limit=24)
    if args.out and status == "accepted":
        args.out.write_text(final["code"], encoding="utf-8")
        log.step("💾", f"Saved to {args.out}", indent=1)
    log.out()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
