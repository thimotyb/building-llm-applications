import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev_judge import JevJudge  # noqa: E402
from judge import Choice, Noul, Score  # noqa: E402
from judge_factory import make_judge  # noqa: E402

QUESTIONS = [
    Noul(id="urgent", prompt="Is it urgent?"),
    Choice(id="team", prompt="Which team?", options=["billing", "engineering"],
           descriptions={"billing": "money", "engineering": "bugs"}),
    Score(id="quality", prompt="Quality?", low=0, high=10),
]
RESPONSE = {
    "model": "jev-1.14.0",
    "answers": {
        "urgent": {"type": "noul", "noul": 0.93},
        "team": {"type": "choice", "choice": "engineering", "confidence": 0.8,
                 "probabilities": {"billing": 0.2, "engineering": 0.8}},
        "quality": {"type": "score", "score": 2.0, "confidence": 0.6,
                    "probabilities": {"1": 0.1, "2": 0.8, "3": 0.1}},
    },
    "usage": {"input_tokens": 1_000_000, "output_tokens": 30},
}


def judge_with(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return JevJudge(model="jev-1.14.0", api_key="secret", base_url="https://x.test", client=client, max_retries=1)


def test_request_shape_and_typed_answers():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=RESPONSE)

    verdict = judge_with(handler).judge("some state", QUESTIONS)
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["questions"]["team"]["criteria"] == {"billing": "money", "engineering": "bugs"}
    assert seen["body"]["questions"]["urgent"] == {"type": "noul", "instructions": "Is it urgent?"}
    assert len(seen["body"]["questions"]["quality"]["criteria"]) == 5
    assert verdict["urgent"].value is True and verdict["urgent"].confidence == pytest.approx(0.93)
    assert verdict["team"].value == "engineering"
    assert verdict["quality"].value == pytest.approx(5.0)  # level 2 of 0..4 mapped onto 0..10
    assert verdict.cost_usd == pytest.approx(0.042) and verdict.backend == "jev-1.14.0"


def test_rate_limit_is_retried_then_errors_are_not_swallowed(monkeypatch):
    monkeypatch.setattr("jev_judge.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429 if calls["n"] == 1 else 200, json=RESPONSE)

    assert judge_with(handler).judge("s", QUESTIONS)["urgent"].value is True
    assert calls["n"] == 2

    with pytest.raises(RuntimeError, match="401"):
        judge_with(lambda r: httpx.Response(401, text="bad key")).judge("s", QUESTIONS)


def test_factory_selects_backend_and_falls_back(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("JUDGE_BACKEND", "jev")
    monkeypatch.setattr("jev_judge.os.getenv", lambda k, d=None: d if k == "TYPESAFE_API_KEY" else __import__("os").environ.get(k, d))
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        make_judge()
    monkeypatch.setenv("JUDGE_FALLBACK", "llm")
    monkeypatch.setenv("LLM_FALLBACK_OLLAMA", "1")
    assert type(make_judge()).__name__ == "LLMJudge"
