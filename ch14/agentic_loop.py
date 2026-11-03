"""Agentic loop: generate -> harness -> judge -> decide (repeat / human / stop).

    START -> generate -> harness --fail--> (budget left? generate : human)
                            |
                          pass
                            v
                          judge --accept--> finish
                            |--revise--> (budget left? generate : human)
                            `--unsure--> human --approve--> finish
                                                 `-feedback--> generate

Design rules shown by this file:
  * the hard stop criterion is the harness (tests), not the model's opinion;
  * the judge answers typed questions with probabilities, the thresholds are code;
  * state is explicit and typed; iterations are budgeted; every step is traced;
  * a human is asked only in the uncertain band or when the budget is exhausted.

Run:  python agentic_loop.py [--human ask|approve|stop] [--max-iters 4]
"""

from __future__ import annotations

import argparse
import json
import operator
import time
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from harness import CASE_DIR, run_harness
from judge import Judge, Noul
from judge_factory import make_judge
from llm_factory import ModelSpec, complete, resolve_model
from router import escalating_generate

# A generator is any function (spec, current_code, feedback) -> (new_code, cost_usd).
# The loop does not care whether it wraps an LLM, a stub, or an escalating pair of models.
GenerateFn = Callable[[str, str, str], tuple[str, float]]  # (spec, current_code, feedback) -> (code, cost)

# Decision thresholds live in code, not in the model. A probability >= ACCEPT counts as "yes",
# <= REJECT as "no"; everything in between is the uncertain band that goes to a person.
# Calibrate both values on labelled examples (see bench/routing_replay.py).
ACCEPT, REJECT = 0.7, 0.3


# The whole run is one explicit, typed dictionary. Each node returns only the fields it changes.
# LangGraph checkpoints this state after every node, which is what makes pause/resume possible.
class LoopState(TypedDict, total=False):
    spec: str                # the requirements; never modified by the loop
    code: str                # the current artifact (latest attempt)
    feedback: str            # why the previous attempt was rejected; empty on the first attempt
    iteration: int           # attempts made so far, incremented by generate
    max_iters: int           # iteration budget: the guaranteed end of the loop
    tests_passed: bool       # result of the last harness run
    status: str              # running | accepted | stopped
    cost_usd: float          # accumulated cost of generator and judge calls
    # `operator.add` is a reducer: nodes append rows instead of overwriting the list
    trace: Annotated[list[dict], operator.add]


# Typed questions the judge answers about the code. The judge returns a probability for each,
# and _judge_decision() turns those probabilities into accept / revise / unsure.
QUESTIONS = [
    Noul(id="spec_ok", prompt="Does the code implement FR-1, FR-2 and FR-3 of the spec exactly?"),
    Noul(
        id="scope_creep",
        prompt="Does the code add behaviour NOT in the spec (more than three attempts, backoff, "
        "handling of non-transient errors, logging, extra classes)?",
    ),
]

# System prompt of the worker: it must return the whole file in one code block so that
# _extract_code() can pull it out, and it is told to stay inside the spec (the judge checks this).
_GEN_SYSTEM = (
    "You are a careful Python developer. Return the COMPLETE new content of payment_retry.py in a single "
    "```python block and nothing else. Keep FakeGateway, TemporaryGatewayError and PaymentResult unchanged. "
    "Implement only what the spec requires. Messages to the customer stay in Italian."
)


def llm_generate(spec_model: ModelSpec) -> GenerateFn:
    """Build a generator on top of any provider selected by the LLM factory (Ollama, DeepSeek, Gemini...)."""

    def generate(spec: str, code: str, feedback: str) -> tuple[str, float]:
        user = f"SPEC:\n{spec}\n\nCURRENT payment_retry.py:\n```python\n{code}\n```\n"
        # On a retry the previous failure is appended to the prompt: this is the loop's feedback channel
        if feedback:
            user += f"\nFEEDBACK ON THE PREVIOUS ATTEMPT (fix this):\n{feedback}\n"
        response = complete(
            [{"role": "system", "content": _GEN_SYSTEM}, {"role": "user", "content": user}], spec_model
        )
        text = response.choices[0].message.content or ""
        # Cost is best effort: unknown models (e.g. local ones) simply count as zero
        try:
            import litellm

            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            cost = 0.0
        return _extract_code(text), cost

    return generate


def _extract_code(text: str) -> str:
    """Take the content of the first fenced code block; fall back to the whole reply."""
    import re

    match = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    return (match.group(1) if match else text).strip() + "\n"


def build_graph(generate: GenerateFn, judge: Judge, checkpointer=None, harness=run_harness):
    """Wire the nodes and the routing functions into a LangGraph state graph.

    generate, judge and harness are injected, so tests and the replay demo can plug in scripted
    versions and the real run can plug in LLMs, Jev, or a cascade without changing the graph.
    """

    # --- nodes: each one receives the state and returns only the fields it changes ---

    def n_generate(state: LoopState) -> dict:
        # Produce (or repair) the artifact, passing along the feedback of the previous attempt
        start = time.perf_counter()
        code, cost = generate(state["spec"], state["code"], state.get("feedback", ""))
        i = state.get("iteration", 0) + 1  # the iteration counter is incremented here and only here
        return {
            "code": code, "iteration": i, "cost_usd": state.get("cost_usd", 0.0) + cost,
            "trace": [{"iter": i, "node": "generate", "chars": len(code), "cost": cost,
                       "seconds": round(time.perf_counter() - start, 2)}],
        }

    def n_harness(state: LoopState) -> dict:
        # Executable checks (tests + static rules): the hard stop criterion. No model opinion involved.
        report = harness(state["code"])
        return {
            "tests_passed": report.passed,
            # on failure, the harness output (static violations first) becomes the next feedback
            "feedback": "" if report.passed else f"The harness failed:\n{report.feedback}",
            "trace": [{"iter": state["iteration"], "node": "harness", "result": report.summary}],
        }

    def n_judge(state: LoopState) -> dict:
        # Only reached when the harness passed: the judge covers what tests cannot see (spec fit, scope creep)
        verdict = judge.judge(f"SPEC:\n{state['spec']}\n\nCODE:\n{state['code']}", QUESTIONS)
        p_ok = verdict["spec_ok"].probabilities["true"]
        p_creep = verdict["scope_creep"].probabilities["true"]
        decision = _judge_decision(p_ok, p_creep)
        # Turn the probabilities into feedback sentences the generator can act on
        notes = []
        if p_ok < ACCEPT:
            notes.append("the code may not implement the spec exactly")
        if p_creep > REJECT:
            notes.append("the code may add behaviour outside the spec: remove it")
        return {
            "feedback": "Judge feedback: " + "; ".join(notes) if decision != "accept" else "",
            "cost_usd": state.get("cost_usd", 0.0) + verdict.cost_usd,
            "trace": [{"iter": state["iteration"], "node": "judge", "backend": verdict.backend,
                       "p_spec_ok": round(p_ok, 2), "p_scope_creep": round(p_creep, 2),
                       "decision": decision, "cost": verdict.cost_usd}],
        }

    def n_human(state: LoopState) -> dict:
        # interrupt() pauses the run and saves the state; graph.stream(Command(resume=...)) continues
        # from here and `answer` receives the resume value. Reached only when the judge is unsure
        # or the iteration budget is exhausted.
        answer = interrupt({
            "question": "Approve this code? Reply 'approve', 'stop', or write feedback for another round.",
            "iteration": state["iteration"], "tests_passed": state.get("tests_passed"),
            "feedback": state.get("feedback", ""), "code": state["code"],
        })
        answer = str(answer).strip()
        # three possible replies: approve (accept as is), stop (give up), anything else = free-text feedback
        if answer.lower() == "approve":
            return {"status": "accepted", "trace": [{"iter": state["iteration"], "node": "human", "decision": "approve"}]}
        if answer.lower() == "stop":
            return {"status": "stopped", "trace": [{"iter": state["iteration"], "node": "human", "decision": "stop"}]}
        return {
            "feedback": f"Human feedback: {answer}", "status": "running",
            "max_iters": state["max_iters"] + 2,  # a human note buys two more iterations
            "trace": [{"iter": state["iteration"], "node": "human", "decision": "feedback"}],
        }

    def n_finish(state: LoopState) -> dict:
        # "running" here means the judge accepted; human decisions already set the status
        return {"status": "accepted" if state.get("status") in (None, "running") else state["status"]}

    # --- routing functions: all stop/repeat logic is ordinary, testable code that returns a node name ---

    def after_harness(state: LoopState) -> str:
        # tests green -> judge; tests red -> retry while budget remains, otherwise ask a person
        if state["tests_passed"]:
            return "judge"
        return "generate" if state["iteration"] < state["max_iters"] else "human"

    def after_judge(state: LoopState) -> str:
        # accept -> finish; unsure -> person; revise -> retry while budget remains, otherwise a person
        decision = state["trace"][-1]["decision"]
        if decision == "accept":
            return "finish"
        if decision == "unsure":
            return "human"
        return "generate" if state["iteration"] < state["max_iters"] else "human"

    def after_human(state: LoopState) -> str:
        # free-text feedback keeps status "running" and loops back; approve/stop already set the final status
        return "generate" if state["status"] == "running" else "finish"

    # --- graph wiring: the loop is the conditional edge that returns to "generate" ---
    g = StateGraph(LoopState)
    for name, fn in [("generate", n_generate), ("harness", n_harness), ("judge", n_judge),
                     ("human", n_human), ("finish", n_finish)]:
        g.add_node(name, fn)
    g.add_edge(START, "generate")
    g.add_edge("generate", "harness")
    g.add_conditional_edges("harness", after_harness, ["judge", "generate", "human"])
    g.add_conditional_edges("judge", after_judge, ["finish", "human", "generate"])
    g.add_conditional_edges("human", after_human, ["generate", "finish"])
    g.add_edge("finish", END)
    # The checkpointer saves the state after every node; it is required for interrupt()/resume
    return g.compile(checkpointer=checkpointer or MemorySaver())


def _judge_decision(p_ok: float, p_creep: float) -> str:
    """Thresholds live in code. The uncertain band goes to a human.

    accept: confident the spec is met and confident there is no extra behaviour.
    revise: confident something is wrong (spec not met, or scope creep): go back with feedback.
    unsure: anything in between: ask a person instead of guessing.
    """
    if p_ok >= ACCEPT and p_creep <= REJECT:
        return "accept"
    if p_ok <= REJECT or p_creep >= ACCEPT:
        return "revise"
    return "unsure"


def run(graph, spec: str, code: str, max_iters: int, human: str, thread: str = "run",
        on_row=None, on_interrupt=None) -> LoopState:
    """Drive the graph until it ends, printing a live trace; `human` is ask | approve | stop.

    on_row / on_interrupt let a caller (the narrated demo) replace the default printing.
    """
    config = {"configurable": {"thread_id": thread}}
    payload: object = {"spec": spec, "code": code, "iteration": 0, "max_iters": max_iters,
                       "status": "running", "cost_usd": 0.0, "feedback": "", "trace": []}
    # Outer loop: each pass runs the graph until it ends or hits interrupt(); after an interrupt we
    # resume the same thread with the human's reply.
    while True:
        interrupted = None
        for event in graph.stream(payload, config, stream_mode="updates"):
            for node, update in event.items():
                # LangGraph reports a pause as a pseudo-node named __interrupt__
                if node == "__interrupt__":
                    interrupted = update[0].value
                    continue
                for row in update.get("trace", []):
                    if on_row:
                        on_row(row)
                    else:
                        print("  " + json.dumps(row, ensure_ascii=False))
        if interrupted is None:  # no pause: the graph reached END, return the final state
            return graph.get_state(config).values
        if on_interrupt:
            on_interrupt(interrupted)
        else:
            print(f"\n[HUMAN] {interrupted['question']}  (iter {interrupted['iteration']}, "
                  f"tests_passed={interrupted['tests_passed']})")
        # --human approve|stop answers automatically (for scripts); "ask" reads from the keyboard
        reply = {"approve": "approve", "stop": "stop"}.get(human) or input("> ")
        payload = Command(resume=reply)


def main() -> None:
    """CLI entry point: pick worker, judge and options from the environment and flags, then run the loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", choices=["ask", "approve", "stop"], default="ask")
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--judge-samples", type=int, default=1)
    parser.add_argument("--escalate", action="store_true",
                        help="after a failed attempt, regenerate with the LARGE_PROVIDER/LARGE_MODEL model")
    parser.add_argument("--out", type=Path, help="write the final payment_retry.py here")
    args = parser.parse_args()

    worker = resolve_model()
    judge = make_judge(samples=args.judge_samples)
    print(f"worker={worker.label}  judge_backend={type(judge).__name__}  max_iters={args.max_iters}")
    generate = llm_generate(worker)
    if args.escalate:
        large = resolve_model(role="large")
        print(f"escalation model={large.label}")
        generate = escalating_generate(generate, llm_generate(large))
    graph = build_graph(generate, judge)
    final = run(graph, (CASE_DIR / "spec.md").read_text(), (CASE_DIR / "payment_retry.py").read_text(),
                args.max_iters, args.human)
    print(f"\nstatus={final['status']}  iterations={final['iteration']}  cost=${final['cost_usd']:.4f}")
    if args.out:
        args.out.write_text(final["code"], encoding="utf-8")


if __name__ == "__main__":
    main()
