"""Run the same typed questions against any provider: `LLM_PROVIDER=deepseek python demo_judge.py`.

Usage:
    python demo_judge.py                 # provider from .env (JUDGE_PROVIDER or LLM_PROVIDER)
    python demo_judge.py --samples 5     # empirical probabilities by repeated sampling
"""

import argparse

from judge import Choice, Noul, decide
from judge_factory import make_judge

TICKET = (
    "Customer writes: 'My card was charged twice for order 4471 and the checkout page now "
    "returns a 500 error. I need this fixed today, I am travelling tomorrow.'"
)
QUESTIONS = [
    Choice(
        id="team", prompt="Which team should handle this ticket?", options=["billing", "engineering", "sales"],
        descriptions={"billing": "Charges, refunds, invoices", "engineering": "Errors, outages, bugs", "sales": "Quotes, plans, upgrades"},
    ),
    Noul(id="urgent", prompt="Does the customer need a resolution within 24 hours?"),
    Noul(id="refund", prompt="Is a refund explicitly requested?"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    # How many times the LLM judge is asked the same questions; with N>1 the vote shares become the
    # probabilities. Ignored by Jev, which already returns probabilities.
    parser.add_argument("--samples", type=int, default=1,
                        help="ask an LLM judge N times and use the vote shares as probabilities")
    args = parser.parse_args()

    verdict = make_judge(samples=args.samples).judge(TICKET, QUESTIONS)
    print(f"backend={verdict.backend}  latency={verdict.latency_s:.2f}s  cost=${verdict.cost_usd:.5f}")
    for qid, answer in verdict.answers.items():
        probs = {k: round(v, 2) for k, v in answer.probabilities.items()}
        extra = f"  -> {decide(answer)}" if qid != "team" else ""
        print(f"  {qid:8s} value={answer.value!s:12s} p={probs}{extra}")


if __name__ == "__main__":
    main()
