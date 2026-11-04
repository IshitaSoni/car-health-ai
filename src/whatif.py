"""
car-health-ai - Week 7: what-if simulator
=========================================
"What would the score be if this part were fixed?" A fix replaces the relevant
inputs with healthy values and re-runs the same assessment.

Healthy values are taken from the dataset itself, not invented: the 75th
percentile where higher is better and the 25th where lower is better ("better
than three cars in four", not a perfect extreme). A fix is applied only when
the current value is worse than the dataset median; an already-healthy input is
left alone. Both rules exist because the tree models are noisy at the extremes
(e.g. 7% of cars have a service gap clipped to exactly 1.0 month, and the
battery model learned a spurious jump there), and a what-if that makes a car
look worse after a repair would mislead.

IMPORTANT: this is a counterfactual on the MODEL'S INPUTS. It shows how the
model's risk estimate would change, built on simulated wear rules. It is not a
promise about a real repair. Age and mileage cannot be changed by a repair, so
a fix never lowers risk below what those two inputs imply.
"""
import numpy as np
import pandas as pd
import dashboard_logic as L
import price_impact as pi
from health_score import COMPONENTS

# preset name -> list of (input, percentile of the dataset to set it to)
PRESETS = {
    "Replace brake pads": [("brake_pad_depth_mm", 75)],
    "Replace tires": [("tire_tread_depth_mm", 75)],
    "Replace battery": [("battery_voltage", 75)],
    "Full service (fluids and oil)": [("service_gap_months", 25), ("oil_quality_index", 75),
                                      ("transmission_fluid_quality", 75)],
    "Fix suspension / vibration": [("vibration_score", 25)],
}


def healthy_value(df, key, pct):
    v = float(np.percentile(df[key], pct))
    kind = L.SPEC[key][2]
    return int(round(v)) if kind == "int" else round(v, 2)


def apply_presets(inputs, names, df):
    """Copy of inputs with the chosen fixes applied. Unknown names raise KeyError."""
    new = dict(inputs)
    for name in names:
        for key, pct in PRESETS[name]:
            better = healthy_value(df, key, pct)
            median = float(np.percentile(df[key], 50))
            cur = new[key]
            if pct >= 50:                       # higher is better
                if cur < median:
                    new[key] = max(cur, better)
            else:                               # lower is better
                if cur > median:
                    new[key] = min(cur, better)
    return new


def compare(before, new_inputs, res, costs=None):
    """Assess the car as entered (`before` = an assess_car result) and after the fix."""
    after = L.assess_car(new_inputs, res)
    rows = []
    for c in COMPONENTS:
        b, a = before["risks"][c] * 100, after["risks"][c] * 100
        rows.append({"Component": c.capitalize(), "Before %": round(b, 1), "After %": round(a, 1),
                     "Change (points)": round(a - b, 1)})
    out = {
        "after": after,
        "table": pd.DataFrame(rows),
        "score_before": before["assessment"]["health_score"],
        "score_after": after["assessment"]["health_score"],
        "score_change": round(after["assessment"]["health_score"] - before["assessment"]["health_score"], 1),
        "rating_before": before["assessment"]["rating"],
        "rating_after": after["assessment"]["rating"],
    }
    if costs is not None:
        cb = pi.expected_cost(before["risks"], costs)["total"]
        ca = pi.expected_cost(after["risks"], costs)["total"]
        out["expected_cost_before"], out["expected_cost_after"] = round(cb), round(ca)
        out["expected_cost_change"] = round(ca - cb)
    return out