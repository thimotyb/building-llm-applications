"""Run the same typed questions against any provider: `LLM_PROVIDER=deepseek python demo_judge.py`.

Usage:
    python demo_judge.py                 # provider from .env (JUDGE_PROVIDER or LLM_PROVIDER)
    python demo_judge.py --samples 5     # empirical probabilities by repeated sampling
"""

import argparse

from judge import Choice, LLMJudge, Noul, decide
from llm_factory import resolve_model

TICKET = (
    "Customer writes: 'My card was charged twice for order 4471 and the checkout page now "
    "returns a 500 error. I need this fixed today, I am travelling tomorrow.'"
)
QUESTIONS = [
    Choice(id="team", prompt="Which team should handle this ticket?", options=["billing", "engineering", "sales"]),
    Noul(id="urgent", prompt="Does the customer need a resolution within 24 hours?"),
    Noul(id="refund", prompt="Is a refund explicitly requested?"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    spec = resolve_model(role="judge")
    verdict = LLMJudge(spec, samples=args.samples).judge(TICKET, QUESTIONS)
    print(f"backend={verdict.backend}  latency={verdict.latency_s:.2f}s  cost=${verdict.cost_usd:.5f}")
    for qid, answer in verdict.answers.items():
        probs = {k: round(v, 2) for k, v in answer.probabilities.items()}
        extra = f"  -> {decide(answer)}" if qid != "team" else ""
        print(f"  {qid:8s} value={answer.value!s:12s} p={probs}{extra}")


if __name__ == "__main__":
    main()
