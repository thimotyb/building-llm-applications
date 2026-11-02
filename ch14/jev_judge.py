"""Backend for TypeSafe Jev, a hosted "System One" decision model (closed weights, API only).

Jev takes a `state` and named typed questions and returns typed answers with
probabilities, without generating free text. Docs: https://docs.typesafe.ai

    POST {base}/v1/systemone   Authorization: Bearer $TYPESAFE_API_KEY
    noul   -> {"type":"noul","noul": p_true}
    choice -> {"choice": top, "confidence": c, "probabilities": {option: p}}
    score  -> {"score": expected level index, "probabilities": {"0": p, ...}}

Environment:
    TYPESAFE_API_KEY   required (read from the repo-root .env)
    JEV_MODEL          default jev-latest; PIN a version (e.g. jev-1.14.0) once thresholds are calibrated
    TYPESAFE_BASE_URL  default https://api.typesafe.ai
Billing: input tokens only (USD 0.042 per million tokens at the time of writing, check the docs).
"""

from __future__ import annotations

import os
import time
from typing import Sequence

import httpx

import llm_factory  # noqa: F401  (loads the repo-root .env)
from judge import Answer, Question, Verdict

PRICE_PER_INPUT_TOKEN_USD = 0.042 / 1_000_000
_DEFAULT_LEVELS = ["very low", "low", "medium", "high", "very high"]


def _to_jev_question(q: Question) -> dict:
    if q.kind == "noul":
        return {"type": "noul", "instructions": q.prompt}
    if q.kind == "choice":
        criteria = q.descriptions or {option: option for option in q.options}
        return {"type": "choice", "instructions": q.prompt, "criteria": criteria}
    return {"type": "score", "instructions": q.prompt, "criteria": q.levels or _DEFAULT_LEVELS}


def _from_jev_answer(q: Question, raw: dict) -> Answer:
    if q.kind == "noul":
        p = min(1.0, max(0.0, float(raw["noul"])))
        return Answer(q.id, p >= 0.5, {"true": p, "false": 1 - p})
    if q.kind == "choice":
        probs = {o: float(raw.get("probabilities", {}).get(o, 0.0)) for o in q.options}
        return Answer(q.id, raw["choice"], probs)
    levels = q.levels or _DEFAULT_LEVELS
    fraction = float(raw["score"]) / max(1, len(levels) - 1)  # expected level index -> [0, 1]
    probs = {levels[int(k)]: float(v) for k, v in raw.get("probabilities", {}).items() if int(k) < len(levels)}
    return Answer(q.id, q.low + fraction * (q.high - q.low), probs)


class JevJudge:
    """`Judge` backed by the Jev API. Pass `client` (httpx.Client) to inject a transport in tests."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.Client | None = None,
        max_retries: int = 3,
    ):
        self.model = model or os.getenv("JEV_MODEL", "jev-latest")
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Missing TYPESAFE_API_KEY (set it in .env).")
        self.base_url = (base_url or os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai")).rstrip("/")
        self.client = client or httpx.Client(timeout=30)
        self.max_retries = max_retries

    def _post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries + 1):
            response = self.client.post(f"{self.base_url}/v1/systemone", json=body, headers=headers)
            if response.status_code in (429, 529) and attempt < self.max_retries:
                time.sleep(2**attempt)  # exponential backoff as recommended by the docs
                continue
            if response.status_code >= 400:
                # API failures must surface as errors, never as a negative answer
                raise RuntimeError(f"Jev API error {response.status_code}: {response.text[:300]}")
            return response.json()
        raise RuntimeError("unreachable")

    def judge(self, state: str, questions: Sequence[Question]) -> Verdict:
        start = time.perf_counter()
        body = {"model": self.model, "state": state, "questions": {q.id: _to_jev_question(q) for q in questions}}
        data = self._post(body)
        answers = {q.id: _from_jev_answer(q, data["answers"][q.id]) for q in questions}
        cost = data.get("usage", {}).get("input_tokens", 0) * PRICE_PER_INPUT_TOKEN_USD
        return Verdict(answers, time.perf_counter() - start, cost, data.get("model", self.model))
