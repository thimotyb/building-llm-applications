"""Fast checks that do not require Ollama or a running A2A server."""

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
