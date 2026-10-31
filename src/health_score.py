"""
car-health-ai - Health Score definition
=======================================
Turns six per-component failure probabilities (each P(fail within 10,000 km))
into a single 0-100 Health Score, a rating band, and a list of flagged parts.

    Health Score = 100 x (1 - sum_i  w_i * risk_i)

where risk_i is the *calibrated* failure probability of component i and the
weights w_i sum to 1. The score is therefore "100 minus the weighted expected
failure risk", which keeps it easy to explain: a car whose parts all carry a
20% risk scores about 80.

Weights are a judgement call, not something learned from data. They reflect
(a) safety criticality and (b) typical repair cost/downtime for a buyer:

    engine        0.22   expensive, can strand the car
    brakes        0.22   safety critical
    transmission  0.17   expensive to repair
    suspension    0.15   safety and ride quality
    tires         0.14   safety, but cheap and quick to replace
    battery       0.10   cheap, easy to replace

The weights are exposed as a dictionary so they can be changed (or exposed as
sliders in the dashboard). Week 4 includes a sensitivity check showing how much
the ranking of cars changes under different weightings.
"""

COMPONENTS = ["engine", "brakes", "battery", "transmission", "suspension", "tires"]

WEIGHTS = {
    "engine": 0.22,
    "brakes": 0.22,
    "transmission": 0.17,
    "suspension": 0.15,
    "tires": 0.14,
    "battery": 0.10,
}

# A component is flagged when its calibrated risk reaches this value. Chosen in
# Week 3/4 from the precision/recall trade-off: lower than 0.5 so that risky
# parts are not missed (inspection favours recall over avoiding false alarms).
FLAG_THRESHOLD = 0.30

# Rating bands: (minimum score, label). Checked from top to bottom.
# Cutoffs were chosen on development (out-of-fold training) data only, see
# results/band_tuning_dev.csv. Rule of thumb used: every band holds at least
# ~15% of cars, and average actual failures per car roughly triple from one
# band to the next (Good ~0.4, Fair ~1.1, Poor ~3).
BANDS = [
    (90.0, "Good"),
    (70.0, "Fair"),
    (0.0, "Poor"),
]

# Parts whose failure is a safety issue. If one of these is flagged the overall
# rating cannot be better than "Fair", however good the average looks.
SAFETY_CRITICAL = ("brakes", "suspension", "tires")


def _check_weights(weights):
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {total:.4f}")


def health_score(risks, weights=None):
    """risks: dict component -> probability (0-1). Returns score 0-100."""
    weights = weights or WEIGHTS
    _check_weights(weights)
    weighted_risk = sum(weights[c] * float(risks[c]) for c in weights)
    return 100.0 * (1.0 - weighted_risk)


def rating_for(score, flags=(), bands=None):
    """Band label for a score, with the safety-critical cap applied."""
    bands = bands or BANDS
    label = next(name for cutoff, name in bands if score >= cutoff)
    if label == "Good" and any(c in SAFETY_CRITICAL for c in flags):
        label = "Fair"
    return label


def flagged_components(risks, threshold=None):
    """Components at or above the flag threshold, highest risk first."""
    t = FLAG_THRESHOLD if threshold is None else threshold
    hits = [(c, float(r)) for c, r in risks.items() if r >= t]
    return [c for c, _ in sorted(hits, key=lambda x: -x[1])]


def assess(risks, weights=None, threshold=None):
    """Full assessment for one car: score, rating, flagged parts."""
    score = health_score(risks, weights)
    flags = flagged_components(risks, threshold)
    return {
        "health_score": round(score, 1),
        "rating": rating_for(score, flags),
        "flagged": flags,
        "risks": {c: round(float(risks[c]), 3) for c in COMPONENTS},
    }