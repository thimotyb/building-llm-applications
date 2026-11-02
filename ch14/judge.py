"""Typed judge interface ("System One" style) for chapter 14.

A judge receives a *state* (free text or JSON) and a list of typed *questions*,
and returns typed *answers with probabilities*. It never writes free text.
The decision logic (thresholds, routing) stays in code, not in the judge.

Backends implement the `Judge` protocol:
    LLMJudge   any LiteLLM model (DeepSeek, Gemini, Ollama...), included here
    FakeJudge  deterministic, for tests and recorded demos
    (a hosted decision-model backend can be added behind the same interface)
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, Field

from llm_factory import ModelSpec, complete, resolve_model


# --- questions -------------------------------------------------------------

class Choice(BaseModel):
    """Pick exactly one option."""

    kind: Literal["choice"] = "choice"
    id: str
    prompt: str
    options: list[str] = Field(min_length=2)
    descriptions: dict[str, str] | None = None  # option -> what it means (used by decision-model backends)


class Noul(BaseModel):
    """Yes/no question; the answer carries P(true)."""

    kind: Literal["noul"] = "noul"
    id: str
    prompt: str


class Score(BaseModel):
    """Numeric rating in [low, high]."""

    kind: Literal["score"] = "score"
    id: str
    prompt: str
    low: float = 0.0
    high: float = 1.0
    levels: list[str] | None = None  # ordered level descriptions, lowest first (decision-model backends)


Question = Choice | Noul | Score


# --- answers ---------------------------------------------------------------

@dataclass(frozen=True)
class Answer:
    question_id: str
    value: str | bool | float  # chosen option / True-False / score
    probabilities: dict[str, float] = field(default_factory=dict)  # option -> p ; noul: {"true": p, "false": 1-p}

    @property
    def confidence(self) -> float:
        """Probability of the returned value (score: 1.0, no distribution)."""
        if isinstance(self.value, bool):
            return self.probabilities.get("true" if self.value else "false", 1.0)
        return self.probabilities.get(str(self.value), 1.0)


@dataclass(frozen=True)
class Verdict:
    answers: dict[str, Answer]
    latency_s: float = 0.0
    cost_usd: float = 0.0
    backend: str = ""

    def __getitem__(self, question_id: str) -> Answer:
        return self.answers[question_id]


class Judge(Protocol):
    def judge(self, state: str, questions: Sequence[Question]) -> Verdict: ...


# --- LLM backend -----------------------------------------------------------

_SYSTEM = (
    "You are a calibrated judge, not an assistant. Answer ONLY the typed questions about the STATE. "
    "Never write explanations. Return JSON: {\"answers\": {<question id>: <answer>}} where "
    "choice -> {\"option\": <one of the options>, \"probabilities\": {<option>: p, ...}} (p sum to 1); "
    "noul -> {\"p_true\": p}; score -> {\"score\": number within the given range}. "
    "Probabilities must reflect real uncertainty: do not output 0 or 1 unless certain."
)


def _render_questions(questions: Sequence[Question]) -> str:
    lines = []
    for q in questions:
        if q.kind == "choice":
            lines.append(f"- {q.id} (choice): {q.prompt} Options: {q.options}")
        elif q.kind == "noul":
            lines.append(f"- {q.id} (noul): {q.prompt}")
        else:
            lines.append(f"- {q.id} (score {q.low}..{q.high}): {q.prompt}")
    return "\n".join(lines)


def _clamp(p: float) -> float:
    return min(1.0, max(0.0, float(p)))


def _parse_answer(q: Question, raw: object) -> Answer:
    if q.kind == "noul":
        p = _clamp(raw.get("p_true", 0.5) if isinstance(raw, dict) else raw)
        return Answer(q.id, p >= 0.5, {"true": p, "false": 1 - p})
    if q.kind == "score":
        value = raw.get("score") if isinstance(raw, dict) else raw
        return Answer(q.id, min(q.high, max(q.low, float(value))))
    option = raw.get("option") if isinstance(raw, dict) else raw
    if option not in q.options:
        raise ValueError(f"'{option}' is not a valid option for {q.id}: {q.options}")
    probs = raw.get("probabilities", {}) if isinstance(raw, dict) else {}
    probs = {o: _clamp(probs.get(o, 0.0)) for o in q.options}
    total = sum(probs.values())
    probs = {o: p / total for o, p in probs.items()} if total > 0 else {option: 1.0}
    return Answer(q.id, option, probs)


class LLMJudge:
    """Judge backed by any LiteLLM model. `samples>1` adds empirical probabilities.

    With samples=1 the probabilities are self-reported by the model (cheap, but
    poorly calibrated on small models). With samples>1 the question is asked
    several times at temperature>0 and the vote shares are blended in: this is
    slower and costlier, but measurable. Calibrate before trusting thresholds.
    """

    def __init__(self, spec: ModelSpec | None = None, samples: int = 1, temperature: float = 0.0):
        self.spec = spec or resolve_model(role="judge")
        self.samples = max(1, samples)
        self.temperature = temperature if self.samples == 1 else max(temperature, 0.7)

    def _ask(self, state: str, questions: Sequence[Question]) -> tuple[dict, float]:
        user = f"STATE:\n{state}\n\nQUESTIONS:\n{_render_questions(questions)}"
        response = complete(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            self.spec,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
        try:
            import litellm

            cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:  # unknown pricing (local models): cost is 0
            cost = 0.0
        return json.loads(text).get("answers", {}), cost

    def judge(self, state: str, questions: Sequence[Question]) -> Verdict:
        import time

        start = time.perf_counter()
        runs, total_cost = [], 0.0
        for _ in range(self.samples):
            answers, cost = self._ask(state, questions)
            runs.append(answers)
            total_cost += cost

        result: dict[str, Answer] = {}
        for q in questions:
            parsed = [_parse_answer(q, run[q.id]) for run in runs if q.id in run]
            if not parsed:
                raise ValueError(f"The judge returned no answer for question '{q.id}'.")
            result[q.id] = parsed[0] if len(parsed) == 1 else _aggregate(q, parsed)
        return Verdict(result, time.perf_counter() - start, total_cost, self.spec.label)


def _aggregate(q: Question, answers: list[Answer]) -> Answer:
    n = len(answers)
    if q.kind == "score":
        return Answer(q.id, sum(a.value for a in answers) / n)  # type: ignore[arg-type]
    if q.kind == "noul":
        p = sum(a.probabilities["true"] for a in answers) / n
        return Answer(q.id, p >= 0.5, {"true": p, "false": 1 - p})
    votes = Counter(a.value for a in answers)
    probs = {o: votes.get(o, 0) / n for o in q.options}
    winner = max(probs, key=probs.get)  # type: ignore[arg-type]
    return Answer(q.id, winner, probs)


# --- deterministic backend -------------------------------------------------

class FakeJudge:
    """Returns pre-set answers; used by tests and recorded demos."""

    def __init__(self, answers: dict[str, Answer], backend: str = "fake"):
        self._answers, self._backend = answers, backend

    def judge(self, state: str, questions: Sequence[Question]) -> Verdict:
        return Verdict({q.id: self._answers[q.id] for q in questions}, backend=self._backend)


# --- decision helper (logic stays in code) --------------------------------

def decide(answer: Answer, accept: float = 0.7, reject: float = 0.3) -> Literal["yes", "no", "unsure"]:
    """Map a Noul answer to a decision with an explicit uncertain band (human review)."""

    p = answer.probabilities.get("true", 0.5)
    return "yes" if p >= accept else "no" if p <= reject else "unsure"
