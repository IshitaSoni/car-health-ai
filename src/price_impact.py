"""
car-health-ai - Week 7 stretch goal: expected repair cost
=========================================================
Expected repair cost over the next 10,000 km  =  sum over parts of
    (probability the part fails) x (cost to repair it)

The PROBABILITIES come from the models. The COSTS below are PLACEHOLDER
assumptions, not data: this project has no repair-cost dataset. They exist so
the dashboard can show how a risk turns into money. Replace them with local
garage quotes before relying on any figure (the dashboard lets you edit them).
Currency: INR. Use them to compare cars and fixes, not as a quote.
"""
import pandas as pd
import health_score as hs
from health_score import COMPONENTS

CURRENCY = "INR"
DEFAULT_COSTS = {          # PLACEHOLDERS: edit before use
    "engine": 60000,
    "brakes": 8000,
    "battery": 6000,
    "transmission": 50000,
    "suspension": 15000,
    "tires": 20000,
}


def expected_cost(risks, costs=None):
    """Returns {total, flagged_total, table}. Costs must be >= 0."""
    costs = costs or DEFAULT_COSTS
    if any(costs[c] < 0 for c in COMPONENTS):
        raise ValueError("repair costs must not be negative")
    rows = [{"Component": c.capitalize(), "Risk %": round(risks[c] * 100, 1),
             "Assumed repair cost": costs[c], "Expected cost": round(risks[c] * costs[c])}
            for c in COMPONENTS]
    t = pd.DataFrame(rows).sort_values("Expected cost", ascending=False).reset_index(drop=True)
    total = sum(risks[c] * costs[c] for c in COMPONENTS)
    flagged = sum(risks[c] * costs[c] for c in COMPONENTS if risks[c] >= hs.FLAG_THRESHOLD)
    return {"total": float(total), "flagged_total": float(flagged), "table": t}