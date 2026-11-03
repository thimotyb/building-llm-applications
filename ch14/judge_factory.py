"""Select the judge backend from the environment.

    JUDGE_BACKEND=llm   (default) any LiteLLM model, see llm_factory (JUDGE_PROVIDER / JUDGE_MODEL)
    JUDGE_BACKEND=jev   hosted TypeSafe Jev (TYPESAFE_API_KEY, JEV_MODEL)
    JUDGE_BACKEND=cascade  Jev first, the LLM judge only on unsure questions (CASCADE_THRESHOLD, default 0.7)
    JUDGE_BACKEND=opendecider  local open decision model (OPENDECIDER_MODEL, see opendecider_judge.py)
    JUDGE_FALLBACK=llm  if Jev is selected but no key/network, fall back to the LLM judge
"""

from __future__ import annotations

import os

from judge import Judge, LLMJudge
from llm_factory import resolve_model


def make_judge(samples: int = 1) -> Judge:
    """Return the judge object for the backend named in JUDGE_BACKEND (default: llm).

    Every returned object has the same method, judge(state, questions) -> Verdict, so the agentic loop
    can use any of them without knowing which one it got.
    """
    backend = (os.getenv("JUDGE_BACKEND") or "llm").strip().lower()
    if backend == "jev":
        # Hosted decision model. The imports are inside the branches so that a missing optional
        # dependency (e.g. the opendecider package) only matters when that backend is selected.
        try:
            from jev_judge import JevJudge

            return JevJudge()  # raises RuntimeError if TYPESAFE_API_KEY is not set
        except RuntimeError:
            # No key: fail loudly unless the user explicitly asked for a fallback. We never switch
            # judge silently, because a different judge has different calibration and cost.
            if os.getenv("JUDGE_FALLBACK", "").lower() != "llm":
                raise
            # JUDGE_FALLBACK=llm: fall through to the LLM judge at the bottom of the function
    elif backend == "cascade":
        # Cheap and fast judge first, stronger judge only for the questions it is unsure about
        from jev_judge import JevJudge
        from router import CascadeJudge

        tiers = [JevJudge(), LLMJudge(resolve_model(role="judge"), samples=samples)]
        return CascadeJudge(tiers, float(os.getenv("CASCADE_THRESHOLD", "0.7")))
    elif backend == "opendecider":
        # Local open-weight decision model: no API key, no per-call cost
        from opendecider_judge import OpenDeciderJudge

        return OpenDeciderJudge()
    elif backend != "llm":
        # A typo in the variable must not silently select the default judge
        raise RuntimeError(f"Unsupported JUDGE_BACKEND '{backend}'. Use 'llm', 'jev', 'cascade' or 'opendecider'.")
    # Default backend (also the Jev fallback): a general LLM prompted to answer the typed questions as JSON.
    # resolve_model(role="judge") reads JUDGE_PROVIDER / JUDGE_MODEL and, if unset, uses the worker's provider.
    return LLMJudge(resolve_model(role="judge"), samples=samples)
