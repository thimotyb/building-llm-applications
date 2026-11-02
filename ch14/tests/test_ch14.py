import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import judge as judge_mod  # noqa: E402
import llm_factory  # noqa: E402
from judge import Answer, Choice, FakeJudge, LLMJudge, Noul, Score, decide  # noqa: E402


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "LLM_PROVIDER", "JUDGE_PROVIDER", "JUDGE_MODEL", "LLM_FALLBACK_OLLAMA",
        "OLLAMA_MODEL", "DEEPSEEK_MODEL", "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_is_local_ollama():
    spec = llm_factory.resolve_model()
    assert spec.provider == "ollama"
    assert spec.model == "ollama_chat/gemma4:e4b"
    assert spec.api_key is None


def test_provider_switch_only_changes_model_string(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    spec = llm_factory.resolve_model()
    assert spec.model.startswith("deepseek/")
    assert spec.api_key == "k"


def test_missing_key_raises_or_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        llm_factory.resolve_model()
    monkeypatch.setenv("LLM_FALLBACK_OLLAMA", "1")
    assert llm_factory.resolve_model().provider == "ollama"


def test_judge_role_is_independent_from_worker(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("JUDGE_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert llm_factory.resolve_model().provider == "ollama"
    assert llm_factory.resolve_model(role="judge").provider == "gemini"


def _fake_completion(payload: str):
    message = SimpleNamespace(content=payload)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


QUESTIONS = [
    Choice(id="route", prompt="Where to send it?", options=["billing", "tech"]),
    Noul(id="urgent", prompt="Is it urgent?"),
    Score(id="quality", prompt="Quality?", low=0, high=10),
]
PAYLOAD = (
    '{"answers": {"route": {"option": "tech", "probabilities": {"billing": 0.2, "tech": 0.8}},'
    ' "urgent": {"p_true": 0.9}, "quality": 12}}'
)


def test_llm_judge_parses_typed_answers(monkeypatch):
    monkeypatch.setattr(judge_mod, "complete", lambda *a, **k: _fake_completion(PAYLOAD))
    verdict = LLMJudge(llm_factory.resolve_model()).judge("state", QUESTIONS)
    assert verdict["route"].value == "tech" and verdict["route"].confidence == pytest.approx(0.8)
    assert verdict["urgent"].value is True
    assert verdict["quality"].value == 10  # clamped to the declared range


def test_llm_judge_rejects_invalid_option(monkeypatch):
    bad = '{"answers": {"route": {"option": "sales"}, "urgent": {"p_true": 0.5}, "quality": 1}}'
    monkeypatch.setattr(judge_mod, "complete", lambda *a, **k: _fake_completion(bad))
    with pytest.raises(ValueError, match="not a valid option"):
        LLMJudge(llm_factory.resolve_model()).judge("s", QUESTIONS)


def test_sampling_produces_empirical_probabilities(monkeypatch):
    replies = iter(
        ['{"answers": {"route": "tech"}}'] * 3 + ['{"answers": {"route": "billing"}}']
    )
    monkeypatch.setattr(judge_mod, "complete", lambda *a, **k: _fake_completion(next(replies)))
    q = [QUESTIONS[0]]
    verdict = LLMJudge(llm_factory.resolve_model(), samples=4).judge("s", q)
    assert verdict["route"].value == "tech"
    assert verdict["route"].probabilities["tech"] == pytest.approx(0.75)


def test_decide_has_an_uncertain_band_for_human_review():
    def noul(p):
        return Answer("x", p >= 0.5, {"true": p, "false": 1 - p})

    assert decide(noul(0.9)) == "yes"
    assert decide(noul(0.1)) == "no"
    assert decide(noul(0.5)) == "unsure"


def test_fake_judge_is_deterministic():
    answers = {"urgent": Answer("urgent", True, {"true": 0.95, "false": 0.05})}
    verdict = FakeJudge(answers).judge("s", [QUESTIONS[1]])
    assert verdict["urgent"].confidence == 0.95


def test_missing_question_is_asked_again_alone(monkeypatch):
    calls = []

    def fake(messages, *a, **k):
        calls.append(messages[-1]["content"])
        if len(calls) == 1:  # first reply forgets "quality" and "urgent"
            return _fake_completion('{"answers": {"route": {"option": "tech", "probabilities": {"tech": 1}}}}')
        return _fake_completion('{"answers": {"urgent": {"p_true": 0.2}, "quality": 3}}')

    monkeypatch.setattr(judge_mod, "complete", fake)
    verdict = LLMJudge(llm_factory.resolve_model()).judge("state", QUESTIONS)
    assert len(calls) == 2 and "route" not in calls[1].split("QUESTIONS:")[1]
    assert verdict["urgent"].value is False and verdict["quality"].value == 3


def test_still_missing_after_retries_raises(monkeypatch):
    monkeypatch.setattr(judge_mod, "complete", lambda *a, **k: _fake_completion('{"answers": {}}'))
    with pytest.raises(ValueError, match="no answer"):
        LLMJudge(llm_factory.resolve_model()).judge("s", QUESTIONS)
