"""Consume the remote shipping specialist from an ADK client agent."""

import os

from google.adk.agents import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent

try:
    # Direct imports use the full chapter package from the repository root.
    from ch13.model import build_model
except ModuleNotFoundError as exc:
    if exc.name != "ch13":
        raise
    # ADK CLI adds ch13 itself to sys.path and imports a2a_client.agent.
    from model import build_model


AGENT_CARD_URL = os.getenv(
    "A2A_AGENT_CARD_URL",
    "http://127.0.0.1:8001/.well-known/agent-card.json",
)

shipping_specialist = RemoteA2aAgent(
    name="shipping_specialist",
    description="Remote A2A agent that estimates parcel delivery time and price.",
    agent_card=AGENT_CARD_URL,
    use_legacy=False,
)

root_agent = Agent(
    name="shipping_client",
    model=build_model(),
    description="Collects shipping requirements and delegates estimates to an A2A agent.",
    instruction="""
You are the client-side shipping assistant. Collect the parcel weight in
kilograms and one destination zone: local, domestic, europe, or international.
Delegate every estimate to the shipping_specialist sub-agent through A2A. Do
not calculate or alter the estimate yourself. Clearly identify the result as a
teaching estimate rather than a real carrier quotation.
""",
    sub_agents=[shipping_specialist],
)
