"""Synthetic labelled support tickets for the judge benchmark (no personal data).

100 template-built tickets (labels true by construction) + 20 hand-written
tricky ones (negations, implicit refunds, urgency without keywords).
Tasks: team (billing|engineering|sales), urgent (needs resolution within 24h), refund (refund/credit requested).
Deterministic: `python dataset.py` rewrites dataset.jsonl.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

TEAMS = ["billing", "engineering", "sales"]
TEAM_DESCRIPTIONS = {
    "billing": "Charges, refunds, invoices, subscriptions",
    "engineering": "Errors, crashes, outages, access problems, bugs",
    "sales": "Quotes, plans, upgrades, discounts, contracts",
}
QUESTION_URGENT = "Does the customer need a resolution within 24 hours?"
QUESTION_REFUND = "Does the customer explicitly ask for a refund or a credit back?"
QUESTION_TEAM = "Which team should handle this ticket?"

_ISSUES = {
    "billing": [
        "I was charged twice for order {n}.", "Invoice {n} shows the wrong amount.",
        "The subscription fee was taken after I cancelled.", "A payment of 49 EUR appears on my statement and I do not recognise it.",
    ],
    "engineering": [
        "The checkout page returns a 500 error on order {n}.", "The mobile app crashes when I open my cart.",
        "The API has been timing out since this morning.", "I cannot log in and the reset email never arrives.",
    ],
    "sales": [
        "I would like a quote for 200 seats.", "What is the difference between the Pro and Team plans?",
        "We want to upgrade our plan before the end of the month.", "Do you offer a discount for non-profits?",
    ],
}
_URGENT = {
    True: ["I need this solved today.", "This blocks our launch tomorrow morning.", "Please answer within a few hours, I travel tonight."],
    False: ["No rush, whenever you have time.", "There is no deadline on my side.", "Take your time.", "", ""],
}
_REFUND = {
    True: ["Please refund the amount.", "I want my money back.", "I am requesting a full refund."],
    False: ["", "", "Please just fix it.", "I do not want a refund, I only want this resolved."],
}

_HAND = [  # (text, team, urgent, refund)
    ("I do not want a refund, I want the double charge on order 5521 explained. Next week is fine.", "billing", False, False),
    ("The refund you promised last month has still not arrived, and the bank closes my account tomorrow.", "billing", True, True),
    ("Our checkout is down right now and we are losing sales every minute.", "engineering", True, False),
    ("Could your engineers look at the slow dashboard when they get a chance? No refund needed.", "engineering", False, False),
    ("We were overcharged, please credit it back to our card.", "billing", False, True),
    ("Quote needed for 50 seats; the board votes this afternoon.", "sales", True, False),
    ("Upgrade request: we would like the Team plan from next quarter.", "sales", False, False),
    ("Thanks for the great service! Just confirming that our March invoice was paid.", "billing", False, False),
    ("The app crashes at payment. I want to cancel the order and get my money back, immediately.", "engineering", True, True),
    ("Wrong amount on invoice 88, my accountant needs the corrected one by end of day.", "billing", True, False),
    ("I would like to return the product and get a refund; there is no hurry.", "billing", False, True),
    ("The API key stopped working during a demo to our client in 30 minutes.", "engineering", True, False),
    ("Do you have a student discount? I am not in a hurry.", "sales", False, False),
    ("Please cancel my subscription and refund this month; I noticed it only today.", "billing", False, True),
    ("The reset email never arrives and I am locked out before tomorrow's audit.", "engineering", True, False),
    ("Interested in your enterprise offer, can someone call me this week?", "sales", False, False),
    ("You charged my card 3 times! Fix this now and refund two of them.", "billing", True, True),
    ("The export button does nothing since the last update.", "engineering", False, False),
    ("We need an urgent contract amendment before the signing at 5 pm.", "sales", True, False),
    ("I got a refund for the wrong order; please correct it, no rush.", "billing", False, False),
]


def build() -> list[dict]:
    rng = random.Random(14)
    items = []
    for i in range(100):
        team = TEAMS[i % 3]
        urgent, refund = rng.random() < 0.5, rng.random() < (0.1 if team == "sales" else 0.5)
        issue = rng.choice(_ISSUES[team]).format(n=rng.randint(1000, 9999))
        parts = [issue, rng.choice(_REFUND[refund]), rng.choice(_URGENT[urgent])]
        items.append({"text": " ".join(p for p in parts if p), "team": team, "urgent": urgent, "refund": refund})
    items += [{"text": t, "team": team, "urgent": u, "refund": r, "hard": True} for t, team, u, r in _HAND]
    for n, item in enumerate(items):
        item["id"] = n
    return items


def load() -> list[dict]:
    path = Path(__file__).with_name("dataset.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    data = build()
    Path(__file__).with_name("dataset.jsonl").write_text("\n".join(json.dumps(d) for d in data) + "\n")
    print(f"{len(data)} tickets written")
