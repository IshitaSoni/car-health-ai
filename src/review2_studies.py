"""
car-health-ai - Review II design trade-off studies
==================================================
Two extra studies that back design decisions not covered by Weeks 3-4:
  A. Feature-set study: per-component feature lists vs all observable inputs
     vs history-only (no sensors).
  B. Inference latency: time to score ONE car (six models + Health Score),
     the cost that matters for an interactive dashboard.
Outputs: results/review2_feature_study.csv, results/review2_latency.csv
Usage: python src/review2_studies.py   (run from project root)
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib
import sys
sys.path.insert(0, "src")
from train_models import FEATURES, COMPONENTS, DATA_PATH, SEED
import health_score as hs

df = pd.read_csv(DATA_PATH)
SENSORS = ["oil_quality_index", "engine_temp_c", "brake_pad_depth_mm", "battery_voltage",
           "transmission_fluid_quality", "vibration_score", "tire_tread_depth_mm"]
HISTORY = ["age_years", "mileage_km", "avg_daily_km", "service_gap_months",
           "missed_services", "driving_conditions", "climate"]
ALL_OBS = HISTORY + SENSORS


def model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_depth=3, random_state=SEED)


def auc_for(cols, comp):
    X = pd.get_dummies(df[cols])
    y = df[f"{comp}_fail_within_10k"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    m = model().fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1]), X.shape[1]

rows = []
for c in COMPONENTS:
    a, n1 = auc_for(FEATURES[c], c)
    b, n2 = auc_for(ALL_OBS, c)
    h, n3 = auc_for(HISTORY, c)
    rows.append(dict(component=c, per_component_auc=a, per_component_n=n1,
                     all_observable_auc=b, all_observable_n=n2,
                     history_only_auc=h, history_only_n=n3))
fs = pd.DataFrame(rows)
fs.to_csv("results/review2_feature_study.csv", index=False)
print(fs.round(3).to_string(index=False))
print("MEAN", fs[["per_component_auc", "all_observable_auc", "history_only_auc"]].mean().round(3).to_dict())

# --- latency: score one car end to end
arts = {c: joblib.load(f"models/{c}_calibrated.joblib") for c in COMPONENTS}
one = df.iloc[[0]]
def score_one():
    risks = {}
    for c, a in arts.items():
        X = pd.get_dummies(one[a["raw_features"]])
        X = X.reindex(columns=a["features"], fill_value=0)
        risks[c] = a["model"].predict_proba(X)[:, 1][0]
    return hs.assess(risks)
score_one()
ts = []
for _ in range(200):
    t = time.perf_counter(); score_one(); ts.append((time.perf_counter() - t) * 1000)
lat = pd.DataFrame([dict(n_runs=200, median_ms=np.median(ts), p95_ms=np.percentile(ts, 95),
                         max_ms=max(ts))])
lat.to_csv("results/review2_latency.csv", index=False)
print(lat.round(2).to_string(index=False))