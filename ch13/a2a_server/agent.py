"""Expose an ADK shipping specialist as an A2A server."""

import os

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import Agent

try:
    # Uvicorn imports this module as ch13.a2a_server.agent from the repo root.
    from ch13.model import build_model
except ModuleNotFoundError as exc:
    if exc.name != "ch13":
        raise
    # ADK CLI adds ch13 itself to sys.path and imports a2a_server.agent.
    from model import build_model


ZONE_DAYS = {
    "local": 1,
    "domestic": 3,
    "europe": 5,
    "international": 8,
}


def estimate_delivery(weight_kg: float, destination_zone: str) -> dict[str, object]:
    """Return a deterministic teaching estimate for a parcel delivery.

    Args:
        weight_kg: Parcel weight in kilograms. Must be greater than zero.
        destination_zone: One of local, domestic, europe, or international.
    """

    zone = destination_zone.strip().lower()
    if weight_kg <= 0:
        return {"status": "error", "message": "weight_kg must be greater than zero"}
    if zone not in ZONE_DAYS:
        return {
            "status": "error",
            "message": "destination_zone must be local, domestic, europe, or international",
        }

    base_price = {
        "local": 5.0,
        "domestic": 8.0,
        "europe": 14.0,
        "international": 24.0,
    }[zone]
    extra_weight = max(0.0, weight_kg - 1.0)
    price_eur = round(base_price + extra_weight * 2.5, 2)
    return {
        "status": "ok",
        "destination_zone": zone,
        "weight_kg": weight_kg,
        "estimated_business_days": ZONE_DAYS[zone],
        "estimated_price_eur": price_eur,
        "disclaimer": "Teaching estimate only; it is not a carrier quotation.",
    }


root_agent = Agent(
    name="shipping_specialist",
    model=build_model(),
    description="Estimates parcel delivery time and price for supported destination zones.",
    instruction="""
You are a shipping specialist exposed to other agents through A2A. Ask for the
parcel weight and destination zone when either value is missing. Use
estimate_delivery for every estimate. Return its numbers exactly, include its
disclaimer, and never present the teaching estimate as a real carrier quote.
""",
    tools=[estimate_delivery],
)

A2A_HOST = os.getenv("A2A_SERVER_HOST", "127.0.0.1")
A2A_PORT = int(os.getenv("A2A_SERVER_PORT", "8001"))
A2A_PROTOCOL = os.getenv("A2A_SERVER_PROTOCOL", "http")

# to_a2a creates the A2A request handler, task stores, Starlette application,
# and an Agent Card available at /.well-known/agent-card.json.
a2a_app = to_a2a(
    root_agent,
    host=A2A_HOST,
    port=A2A_PORT,
    protocol=A2A_PROTOCOL,
)
