"""
car-health-ai - Week 6: dashboard logic (no Streamlit imports)
==============================================================
Everything the dashboard computes lives here so it can be unit tested:
input specs, validation, range warnings, running one assessment, and the data
behind each chart. src/app.py only draws.
"""
import os
import numpy as np
import pandas as pd
import health_score as hs
import explain as ex
import report as rp
from health_score import COMPONENTS

# key, label, kind, min, max, step (or options for "cat"), group
INPUT_SPECS = [
    ("age_years", "Age (years)", "float", 0.0, 30.0, 0.1, "History"),
    ("mileage_km", "Mileage (km)", "int", 0, 500000, 1000, "History"),
    ("service_gap_months", "Months since last service", "float", 0.0, 60.0, 0.5, "History"),
    ("missed_services", "Missed services", "int", 0, 10, 1, "History"),
    ("driving_conditions", "Driving conditions", "cat", ["urban", "mixed", "highway"], None, None, "History"),
    ("climate", "Climate", "cat", ["temperate", "hot", "coastal"], None, None, "History"),
    ("oil_quality_index", "Oil quality index", "float", 0.0, 100.0, 1.0, "Sensors"),
    ("engine_temp_c", "Engine temperature (C)", "float", 60.0, 140.0, 0.5, "Sensors"),
    ("brake_pad_depth_mm", "Brake pad depth (mm)", "float", 0.0, 12.0, 0.1, "Sensors"),
    ("battery_voltage", "Battery voltage (V)", "float", 9.0, 14.5, 0.01, "Sensors"),
    ("transmission_fluid_quality", "Transmission fluid quality index", "float", 0.0, 100.0, 1.0, "Sensors"),
    ("vibration_score", "Vibration score", "float", 0.0, 10.0, 0.1, "Sensors"),
    ("tire_tread_depth_mm", "Tire tread depth (mm)", "float", 0.0, 10.0, 0.1, "Sensors"),
]
SPEC = {s[0]: s for s in INPUT_SPECS}
KEYS = [s[0] for s in INPUT_SPECS]


def pick_model_set(models_dir="models", model_set="auto"):
    """'auto' uses the Week 7 monotone models when all six files exist, else the Week 3-4 models."""
    if model_set != "auto":
        return model_set
    have = all(os.path.exists(f"{models_dir}/{c}_monotone.joblib") for c in COMPONENTS)
    return "monotone" if have else "calibrated"


def load_resources(data_path="data/car_health_ai_dataset.csv", models_dir="models",
                   test_scores="results/test_health_scores.csv", model_set="auto"):
    """Dataset, six model artifacts, a 100-car training background, sample cars."""
    model_set = pick_model_set(models_dir, model_set)
    for p in (data_path, test_scores, f"{models_dir}/engine_{model_set}.joblib"):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p}; run the Week 3-4 scripts from the project root first")
    df = pd.read_csv(data_path)
    t = pd.read_csv(test_scores)
    background = df[~df.car_id.isin(t.car_id)].sample(100, random_state=7)
    s = t.sort_values("score").reset_index(drop=True)
    samples = {"Poor car (8th percentile)": s.car_id.iloc[int(0.08 * len(s))],
               "Typical car (50th percentile)": s.car_id.iloc[int(0.50 * len(s))],
               "Good car (90th percentile)": s.car_id.iloc[int(0.90 * len(s))]}
    return {"df": df, "artifacts": ex.load_artifacts(models_dir, model_set), "background": background,
            "samples": samples, "model_set": model_set}


def default_inputs(df):
    """Median car of the dataset, cast to the types the form expects."""
    out = {}
    for k, _, kind, lo, hi, step, _g in INPUT_SPECS:
        if kind == "cat":
            out[k] = df[k].mode().iloc[0]
        elif kind == "int":
            out[k] = int(round(df[k].median()))
        else:
            out[k] = float(round(df[k].median(), 2))
    return out


def sample_inputs(df, car_id):
    row = df[df.car_id == car_id].iloc[0]
    out = {}
    for k, _, kind, *_ in INPUT_SPECS:
        out[k] = row[k] if kind == "cat" else (int(row[k]) if kind == "int" else float(row[k]))
    return out


def validate_inputs(inputs):
    """Hard errors: the car cannot be scored. Returns a list of messages."""
    problems = []
    for k, label, kind, lo, hi, _step, _g in INPUT_SPECS:
        if k not in inputs:
            problems.append(f"{label} is missing")
            continue
        v = inputs[k]
        if kind == "cat":
            if v not in lo:                       # for "cat", lo holds the options
                problems.append(f"{label} must be one of {', '.join(lo)}")
        else:
            try:
                f = float(v)
            except (TypeError, ValueError):
                problems.append(f"{label} must be a number")
                continue
            if np.isnan(f) or not (lo <= f <= hi):
                problems.append(f"{label} must be between {lo} and {hi}")
    return problems


def range_warnings(inputs, df):
    """Soft warnings: valid input, but outside the range the models were trained on."""
    out = []
    for k, label, kind, *_ in INPUT_SPECS:
        if kind == "cat":
            continue
        lo, hi = float(df[k].min()), float(df[k].max())
        if not (lo <= float(inputs[k]) <= hi):
            out.append(f"{label} is outside the training range ({lo:g} to {hi:g}); "
                       f"treat this result with extra caution.")
    return out


def car_frame(inputs):
    row = {k: inputs[k] for k in KEYS}
    row["car_id"] = "Entered car"
    return pd.DataFrame([row])


def assess_car(inputs, res):
    """Score one car. Raises ValueError for invalid input."""
    problems = validate_inputs(inputs)
    if problems:
        raise ValueError("; ".join(problems))
    car = car_frame(inputs)
    arts, bg = res["artifacts"], res["background"]
    risks = {c: float(ex.predict_risk(arts[c], car)[0]) for c in COMPONENTS}
    a = hs.assess(risks)
    payload = rp.build_payload(car, arts, bg, car_id="Entered car")
    expl = ex.explain_car(car, arts, bg, components=a["flagged"])
    return {"car": car, "risks": risks, "assessment": a, "payload": payload, "expl": expl,
            "warnings": range_warnings(inputs, res["df"])}


# ---------- data behind the charts ----------
def gauge_spec(score, rating):
    colors = {"Good": "#2E9E75", "Fair": "#E0A100", "Poor": "#D64B1A"}
    return {"value": score, "bar": colors[rating],
            "steps": [(0, hs.BANDS[1][0], "#F6D5C8"), (hs.BANDS[1][0], hs.BANDS[0][0], "#F8EBC2"),
                      (hs.BANDS[0][0], 100, "#CDEBDF")]}


def radar_data(risks):
    names = [c.capitalize() for c in COMPONENTS]
    vals = [round(risks[c] * 100, 1) for c in COMPONENTS]
    return {"names": names + names[:1], "risk_pct": vals + vals[:1],
            "threshold_pct": [round(hs.FLAG_THRESHOLD * 100)] * (len(names) + 1)}


def driver_table(expl_one, car):
    """All inputs of one component, largest effect first: for the 'Why' bar chart."""
    rows = []
    for feat, phi in expl_one["contributions"].items():
        v = car[feat].iloc[0]
        v = v.item() if hasattr(v, "item") else v
        rows.append({"input": feat.replace("_", " "), "value": ex.VALUE_TEXT[feat](v),
                     "effect_pp": round(float(phi) * 100, 1)})
    return pd.DataFrame(rows).sort_values("effect_pp", key=abs, ascending=False).reset_index(drop=True)


def risk_table(risks):
    rows = [{"Component": c.capitalize(), "Risk %": round(risks[c] * 100, 1),
             "Flagged": "yes" if risks[c] >= hs.FLAG_THRESHOLD else "no"} for c in COMPONENTS]
    return pd.DataFrame(rows).sort_values("Risk %", ascending=False).reset_index(drop=True)


def llm_available():
    """True if a LangChain chat model can be used (package + key present)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import langchain_anthropic  # noqa: F401
        import langchain_core       # noqa: F401
        return True
    except ImportError:
        return False


def write_report(payload, use_llm=False):
    llm_fn = None
    if use_llm and llm_available():
        from langchain_anthropic import ChatAnthropic
        llm_fn = rp.langchain_llm_fn(ChatAnthropic(model="claude-sonnet-5-5", temperature=0))
    return rp.generate_report(payload, llm_fn=llm_fn)