"""Fast checks that do not require Ollama or a running A2A server."""

import pytest

from ch13 import model
from ch13.a2a_server.agent import estimate_delivery


def test_delivery_estimate_is_deterministic() -> None:
    result = estimate_delivery(3.0, "europe")

    assert result["status"] == "ok"
    assert result["estimated_business_days"] == 5
    assert result["estimated_price_eur"] == 19.0


def test_delivery_estimate_rejects_unknown_zone() -> None:
    result = estimate_delivery(1.0, "moon")

    assert result["status"] == "error"
    assert "destination_zone" in str(result["message"])


def test_delivery_estimate_rejects_non_positive_weight() -> None:
    result = estimate_delivery(0, "domestic")

    assert result["status"] == "error"
    assert "weight_kg" in str(result["message"])


@pytest.mark.parametrize(
    ("provider", "environment", "expected_model"),
    [
        (
            "ollama",
            {"OLLAMA_MODEL": "gemma4:12b", "OLLAMA_BASE_URL": "http://ollama:11434"},
            "ollama_chat/gemma4:12b",
        ),
        (
            "deepseek",
            {"DEEPSEEK_MODEL": "deepseek-chat", "DEEPSEEK_API_KEY": "test-key"},
            "deepseek/deepseek-chat",
        ),
        (
            "gemini",
            {"GEMINI_MODEL": "gemini-flash-latest", "GEMINI_API_KEY": "test-key"},
            "gemini/gemini-flash-latest",
        ),
    ],
)
def test_build_model_selects_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    environment: dict[str, str],
    expected_model: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_lite_llm(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setenv("LLM_PROVIDER", provider)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(model, "LiteLlm", fake_lite_llm)

    model.build_model()

    assert captured["model"] == expected_model


def test_build_model_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
        model.build_model()
