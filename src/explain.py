"""
car-health-ai - Week 5: per-prediction explanations (Shapley values)
====================================================================
Explains WHY a component got its failure risk. For one car and one component
model it returns each input's contribution, in percentage points of risk:

    risk(car) = baseline risk + sum of contributions

Method: exact interventional Shapley values. Each component model uses at most
7 raw inputs, so all 2^n feature subsets can be enumerated. For a subset S the
value v(S) is the model's mean predicted risk when the inputs in S take the
car's values and the others are drawn from a background sample of training
cars. This is the quantity shap.Explainer(model, background) estimates; here it
is computed exactly, in probability units, with no extra dependency.
Inputs are grouped by RAW feature (e.g. "climate", not climate_hot) so a
categorical input gets one contribution.

Check: contributions always sum to (risk - baseline) (Shapley efficiency); the
tests assert this. src/shap_crosscheck.py compares against the real shap package.
"""
import math
import joblib
import numpy as np
import pandas as pd
from health_score import COMPONENTS

# How to say each input in plain words. Used by the report generator.
VALUE_TEXT = {
    "age_years": lambda v: f"{v:.1f} years old",
    "mileage_km": lambda v: f"{v:,.0f} km on the odometer",
    "service_gap_months": lambda v: f"{v:.0f} months since the last service",
    "missed_services": lambda v: f"{int(v)} missed services",
    "driving_conditions": lambda v: f"{v} driving conditions",
    "climate": lambda v: f"a {v} climate",
    "oil_quality_index": lambda v: f"oil quality index {v:.0f}",
    "engine_temp_c": lambda v: f"engine temperature {v:.0f} C",
    "brake_pad_depth_mm": lambda v: f"brake pad depth {v:.1f} mm",
    "battery_voltage": lambda v: f"battery voltage {v:.2f} V",
    "transmission_fluid_quality": lambda v: f"transmission fluid quality index {v:.0f}",
    "vibration_score": lambda v: f"vibration score {v:.1f}",
    "tire_tread_depth_mm": lambda v: f"tire tread depth {v:.1f} mm",
}


def load_artifacts(models_dir="models", model_set="calibrated"):
    """model_set: 'calibrated' (Week 3-4 models) or 'monotone' (Week 7, direction-constrained)."""
    return {c: joblib.load(f"{models_dir}/{c}_{model_set}.joblib") for c in COMPONENTS}


def encode(rows, art):
    """Raw rows -> the exact columns the model was trained on."""
    return pd.get_dummies(rows[art["raw_features"]]).reindex(columns=art["features"], fill_value=0)


def predict_risk(art, rows):
    return art["model"].predict_proba(encode(rows, art))[:, 1]


def shapley(art, car, background):
    """Exact Shapley contributions for one car (a one-row DataFrame)."""
    feats = list(art["raw_features"])
    n, B = len(feats), len(background)
    bg = background[feats].reset_index(drop=True)
    x = car[feats].iloc[0]
    frames = []
    for mask in range(1 << n):
        f = bg.copy()
        for i, name in enumerate(feats):
            if mask >> i & 1:
                f[name] = x[name]
        frames.append(f)
    preds = predict_risk(art, pd.concat(frames, ignore_index=True)).reshape(1 << n, B)
    v = preds.mean(axis=1)                       # v[mask] = value of that subset
    fact = math.factorial
    phi = np.zeros(n)
    for i in range(n):
        for mask in range(1 << n):
            if mask >> i & 1:
                continue
            k = bin(mask).count("1")
            w = fact(k) * fact(n - k - 1) / fact(n)
            phi[i] += w * (v[mask | (1 << i)] - v[mask])
    risk, base = float(v[(1 << n) - 1]), float(v[0])
    return {"risk": risk, "baseline": base, "contributions": dict(zip(feats, phi)),
            "efficiency_gap": abs(phi.sum() - (risk - base))}


def explain_car(car, artifacts, background, components=None):
    return {c: shapley(artifacts[c], car, background) for c in (components or COMPONENTS)}


def top_drivers(expl, car, n=3, min_pp=1):
    """Largest contributions as plain-language drivers (percentage points)."""
    rows = []
    for feat, phi in sorted(expl["contributions"].items(), key=lambda kv: -abs(kv[1])):
        pp = int(round(phi * 100))
        if abs(pp) < min_pp:
            continue
        val = car[feat].iloc[0]
        val = val.item() if hasattr(val, "item") else val
        rows.append({"feature": feat, "value_text": VALUE_TEXT[feat](val),
                     "effect_pp": abs(pp), "direction": "raises" if pp > 0 else "lowers"})
        if len(rows) == n:
            break
    return rows