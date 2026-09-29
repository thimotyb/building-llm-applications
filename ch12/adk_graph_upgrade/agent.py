"""Flight-upgrade graph adapted from the ADK graph workflow illustration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from google.adk import Agent, Event, Workflow  # noqa: E402
from google.adk.agents.context import Context  # noqa: E402
from google.adk.events import RequestInput  # noqa: E402
from google.adk.models.lite_llm import LiteLlm  # noqa: E402
from google.adk.tools import FunctionTool  # noqa: E402


LOW_MILES_THRESHOLD = 5_000
HIGH_MILES_THRESHOLD = 20_000


class HistoryAnalysis(BaseModel):
    """Structured decision returned by the history-analysis agent."""

    approved: bool = Field(description="Whether the upgrade should be approved.")
    reason: str = Field(description="Short reason for the decision.")


def _text(value: Any) -> str:
    """Extract readable text from either a string or an ADK Content value."""
    if isinstance(value, str):
        return value
    parts = getattr(value, "parts", None) or []
    rendered = " ".join(
        part.text for part in parts if getattr(part, "text", None)
    )
    return rendered or str(value)


def parse_miles(value: Any) -> int:
    """Read the first integer from a user message such as '12,000 miles'."""
    match = re.search(r"\d[\d.,\s]*", _text(value))
    if not match:
        raise ValueError("Enter the member's miles, for example: 12000")
    digits = re.sub(r"\D", "", match.group(0))
    return int(digits)


def miles_route(miles: int) -> str:
    """Apply a total policy at the boundaries omitted by the source figure."""
    if miles < LOW_MILES_THRESHOLD:
        return "DENY"
    if miles <= HIGH_MILES_THRESHOLD:
        return "GET_CONSENT"
    return "AUTO_APPROVE"


def check_miles(node_input: Any) -> Event:
    """Function node: classify the request by loyalty-mile balance."""
    miles = parse_miles(node_input)
    request = {"customer_id": "CUST-1042", "miles": miles}
    route = miles_route(miles)
    reason = (
        "Fewer than 5,000 miles."
        if route == "DENY"
        else "The request is eligible for the next workflow step."
    )
    return Event(
        output={**request, "reason": reason},
        route=route,
        state={"upgrade_request": request},
    )


def get_consent(node_input: dict[str, Any]):
    """Human-input node: pause until the member accepts or rejects."""
    yield RequestInput(
        message=(
            f"You have {node_input['miles']:,} miles. "
            "Do you consent to using your flight history for this upgrade decision? "
            "Reply yes or no."
        ),
        payload=node_input,
        response_schema=str,
    )


def route_consent(ctx: Context, node_input: Any) -> Event:
    """Route the human reply and restore the request from session state."""
    accepted = _text(node_input).strip().lower() in {
        "y",
        "yes",
        "s",
        "si",
        "sì",
    }
    request = dict(ctx.state.get("upgrade_request", {}))
    if accepted:
        return Event(output=request, route="FETCH_HISTORY")
    return Event(
        output={**request, "reason": "The member did not grant consent."},
        route="REJECTED",
    )


def fetch_flight_history(customer_id: str, miles: int) -> dict[str, Any]:
    """Simulate a flight-history service used as an ADK FunctionTool node."""
    return {
        "customer_id": customer_id,
        "miles": miles,
        "loyalty_years": 7,
        "completed_flights_last_12_months": 14,
        "late_cancellations_last_12_months": 0,
        "prior_complaints_last_12_months": 0,
    }


fetch_flight_history_tool = FunctionTool(fetch_flight_history)

os.environ.setdefault("OLLAMA_API_BASE", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
ollama_model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")

analyze_history = Agent(
    name="analyze_history",
    model=LiteLlm(model=f"ollama_chat/{ollama_model}"),
    mode="single_turn",
    description="Reviews structured flight history for a loyalty upgrade.",
    instruction="""
Review the flight-history object received from the previous workflow node.
Approve only when the member has at least 10 completed flights in the last
12 months, no late cancellations, and no prior complaints in that period.
Return only the structured decision required by the output schema.
""",
    output_schema=HistoryAnalysis,
)


def route_analysis(ctx: Context, node_input: Any) -> Event:
    """Convert the LLM's structured decision into a graph route."""
    if isinstance(node_input, HistoryAnalysis):
        analysis = node_input
    else:
        analysis = HistoryAnalysis.model_validate(node_input)
    request = dict(ctx.state.get("upgrade_request", {}))
    output = {**request, "reason": analysis.reason}
    return Event(
        output=output,
        route="APPROVED" if analysis.approved else "REJECTED",
    )


def process_upgrade(node_input: dict[str, Any]) -> Event:
    """Function node: simulate the irreversible upgrade operation."""
    result = {
        "upgrade_id": f"UPG-{node_input['customer_id']}-001",
        "customer_id": node_input["customer_id"],
        "miles": node_input["miles"],
    }
    return Event(output=result, message="Upgrade processed; issuing the benefit.")


def issue_upgrade(node_input: dict[str, Any]) -> Event:
    """Terminal success node."""
    return Event(
        message=(
            f"Upgrade {node_input['upgrade_id']} issued for "
            f"{node_input['customer_id']}."
        )
    )


def deny_upgrade(node_input: dict[str, Any]) -> Event:
    """Terminal rejection node."""
    return Event(
        message=(
            f"Upgrade denied for {node_input.get('customer_id', 'the member')}: "
            f"{node_input.get('reason', 'policy requirements were not met')}"
        )
    )


root_agent = Workflow(
    name="flight_upgrade_workflow",
    description="Implements the flight-upgrade graph shown in the ADK documentation.",
    edges=[
        ("START", check_miles),
        (
            check_miles,
            {
                "DENY": deny_upgrade,
                "GET_CONSENT": get_consent,
                "AUTO_APPROVE": process_upgrade,
            },
        ),
        (get_consent, route_consent),
        (
            route_consent,
            {
                "REJECTED": deny_upgrade,
                "FETCH_HISTORY": fetch_flight_history_tool,
            },
        ),
        (fetch_flight_history_tool, analyze_history, route_analysis),
        (
            route_analysis,
            {
                "APPROVED": process_upgrade,
                "REJECTED": deny_upgrade,
            },
        ),
        (process_upgrade, issue_upgrade),
    ],
)
