"""Select the judge backend from the environment.

    JUDGE_BACKEND=llm   (default) any LiteLLM model, see llm_factory (JUDGE_PROVIDER / JUDGE_MODEL)
    JUDGE_BACKEND=jev   hosted TypeSafe Jev (TYPESAFE_API_KEY, JEV_MODEL)
    JUDGE_FALLBACK=llm  if Jev is selected but no key/network, fall back to the LLM judge
"""

from __future__ import annotations

import os

from judge import Judge, LLMJudge
from llm_factory import resolve_model


def make_judge(samples: int = 1) -> Judge:
    backend = (os.getenv("JUDGE_BACKEND") or "llm").strip().lower()
    if backend == "jev":
        try:
            from jev_judge import JevJudge

            return JevJudge()
        except RuntimeError:
            if os.getenv("JUDGE_FALLBACK", "").lower() != "llm":
                raise
    elif backend != "llm":
        raise RuntimeError(f"Unsupported JUDGE_BACKEND '{backend}'. Use 'llm' or 'jev'.")
    return LLMJudge(resolve_model(role="judge"), samples=samples)
