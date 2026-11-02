"""Backend for OpenDecider, an open-weight decision model (https://github.com/manjunathshiva/opendecider).

Not a chat model: it reads the probability of an option letter (small) or a [MASK] score (nano), so it
must be driven through its own package, not through the JSON-prompt LLMJudge.

    pip install -r requirements-opendecider.txt
    OPENDECIDER_MODEL = manjunathshiva/opendecider-nano                       (~0.4B, CPU is fine)
                        manjunathshiva/opendecider-small                      (4B, transformers, GPU advised)
                        ollama:hf.co/manjunathshiva/opendecider-small-GGUF:Q8_0   (Ollama, needs logprobs)
The question schema is the same as Jev's (`system_one(state, questions)`), so the mapping is shared.
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

from jev_judge import _DEFAULT_LEVELS, _from_jev_answer, _to_jev_question
from judge import Question, Verdict

DEFAULT_MODEL = "manjunathshiva/opendecider-nano"


class OpenDeciderJudge:
    """`Judge` backed by a local OpenDecider model. Pass `model` (an object with `system_one`) in tests."""

    def __init__(self, model_id: str | None = None, model: Any = None):
        self.model_id = model_id or os.getenv("OPENDECIDER_MODEL", DEFAULT_MODEL)
        if model is None:
            try:
                from opendecider import load
            except ImportError as exc:
                raise RuntimeError("OpenDecider is not installed: pip install -r requirements-opendecider.txt") from exc
            model = load(self.model_id)
        self.model = model

    def judge(self, state: str, questions: Sequence[Question]) -> Verdict:
        start = time.perf_counter()
        result = self.model.system_one(state, {q.id: _to_jev_question(q) for q in questions})
        answers = {}
        for q in questions:
            raw = dict(result["answers"][q.id])
            if q.kind == "score" and "score" not in raw:  # derive the expected level from the distribution
                probs = raw.get("probabilities", {})
                raw["score"] = sum(float(k) * float(v) for k, v in probs.items())
            answers[q.id] = _from_jev_answer(q, raw)
        return Verdict(answers, time.perf_counter() - start, 0.0, self.model_id)
