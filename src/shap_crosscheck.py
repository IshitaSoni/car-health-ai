"""
Run this on YOUR machine (needs `pip install shap`). Not run in the build sandbox,
which cannot install packages.
Compares the exact Shapley values in explain.py with the real shap package
(interventional TreeExplainer with the same background, probability output).
They should agree to about 1e-6 for the same background and model.
Usage (project root):  python src/shap_crosscheck.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, shap
import explain as ex
from health_score import COMPONENTS

df = pd.read_csv("data/car_health_ai_dataset.csv")
tst = pd.read_csv("results/test_health_scores.csv")
bg = df[~df.car_id.isin(tst.car_id)].sample(100, random_state=7)
arts = ex.load_artifacts()
cars = df[df.car_id.isin(tst.car_id)].sample(5, random_state=3)
worst = 0.0
for c in COMPONENTS:
    a = arts[c]
    expl = shap.TreeExplainer(a["model"], data=ex.encode(bg, a), feature_perturbation="interventional",
                              model_output="probability")
    for i in range(len(cars)):
        car = cars.iloc[[i]]
        sv = expl.shap_values(ex.encode(car, a))[0]
        cols = list(a["features"])
        # group one-hot columns back to raw features
        grouped = {}
        for raw in a["raw_features"]:
            grouped[raw] = sum(sv[j] for j, col in enumerate(cols) if col == raw or col.startswith(raw + "_"))
        mine = ex.shapley(a, car, bg)["contributions"]
        worst = max(worst, max(abs(grouped[k] - mine[k]) for k in mine))
print(f"largest difference between explain.py and shap: {worst:.2e}")