"""
car-health-ai - Week 3: baseline per-component failure-risk models
===================================================================
Trains one classifier per component (Engine, Brakes, Battery, Transmission,
Suspension, Tires) to predict P(failure within 10,000 km).

Three models are compared on every component:
  1. Logistic Regression          - simple sanity-check baseline
  2. GradientBoostingClassifier   - classic scikit-learn boosting
  3. HistGradientBoostingClassifier - faster histogram-based boosting

IMPORTANT - leakage guard:
  The dataset contains *_fail_risk_prob columns. Those are the probabilities
  the labels were sampled from, so using them as features would leak the
  answer. They are deliberately excluded. Only inputs an inspector could
  actually observe (vehicle history + sensor readings) are used.

Usage:
    python train_models.py
Outputs (in the working directory):
    models/<component>_model.joblib   best model per component
    results/metrics.csv               full comparison table
    results/*.png                     ROC curves, model comparison, importances
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, roc_curve, average_precision_score)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "data/car_health_ai_dataset.csv"
SEED = 42
THRESHOLD = 0.5

# Features an inspector can actually observe, chosen per component from the
# EDA findings (each component uses its own sensor + relevant history).
FEATURES = {
    "engine":       ["age_years", "mileage_km", "service_gap_months", "missed_services",
                     "climate", "oil_quality_index", "engine_temp_c"],
    "brakes":       ["age_years", "mileage_km", "service_gap_months", "missed_services",
                     "driving_conditions", "brake_pad_depth_mm"],
    "battery":      ["age_years", "service_gap_months", "missed_services",
                     "climate", "battery_voltage"],
    "transmission": ["mileage_km", "service_gap_months", "missed_services",
                     "driving_conditions", "transmission_fluid_quality"],
    "suspension":   ["mileage_km", "driving_conditions", "climate", "vibration_score"],
    "tires":        ["mileage_km", "driving_conditions", "tire_tread_depth_mm"],
}
COMPONENTS = list(FEATURES.keys())

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def make_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=3, random_state=SEED),
    }


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    }, proba


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    rows = []
    best_probas = {}      # for ROC plot
    best_models = {}
    test_sets = {}

    for comp in COMPONENTS:
        feats = FEATURES[comp]
        X = pd.get_dummies(df[feats], drop_first=False)
        y = df[f"{comp}_fail_within_10k"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED)

        print(f"\n--- {comp.upper()} ({len(X.columns)} features, "
              f"{y.mean():.1%} positive) ---")

        best_auc, best_name = -1, None
        for name, model in make_models().items():
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            metrics, proba = evaluate(model, X_test, y_test)
            metrics.update({"component": comp, "model": name, "train_time_s": train_time})
            rows.append(metrics)
            print(f"  {name:22s} AUC={metrics['roc_auc']:.3f}  "
                  f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
                  f"F1={metrics['f1']:.3f}  ({train_time:.2f}s)")

            if name != "Logistic Regression" and metrics["roc_auc"] > best_auc:
                best_auc, best_name = metrics["roc_auc"], name
                best_models[comp] = (name, model)
                best_probas[comp] = (y_test.values, proba)
                test_sets[comp] = (X_test, y_test)

        joblib.dump({"model": best_models[comp][1], "features": list(X.columns),
                     "raw_features": feats, "model_name": best_name},
                    f"models/{comp}_model.joblib")
        print(f"  -> saved best boosted model: {best_name}")

    results = pd.DataFrame(rows)[["component", "model", "roc_auc", "pr_auc",
                                  "precision", "recall", "f1", "train_time_s"]]
    results.to_csv("results/metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (test set, threshold = 0.5)")
    print("=" * 70)
    print(results.round(3).to_string(index=False))

    # ---------- noise ceiling + threshold sweep ----------
    # Labels were sampled from a known probability (with noise), so even a
    # perfect model cannot reach AUC 1.0. Scoring the true generating
    # probability against the labels gives that ceiling.
    print("\n" + "=" * 70)
    print("NOISE CEILING vs ACHIEVED (ROC-AUC) and THRESHOLD SWEEP")
    print("=" * 70)
    sweep_rows = []
    for comp in COMPONENTS:
        name, model = best_models[comp]
        X_test, y_test = test_sets[comp]
        oracle = roc_auc_score(df[f"{comp}_fail_within_10k"], df[f"{comp}_fail_risk_prob"])
        achieved = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        print(f"{comp:13s} ceiling={oracle:.3f}  achieved={achieved:.3f}  gap={oracle - achieved:.3f}")
        proba = model.predict_proba(X_test)[:, 1]
        for t in (0.5, 0.35, 0.25):
            pred = (proba >= t).astype(int)
            sweep_rows.append({
                "component": comp, "threshold": t,
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "oracle_auc": oracle, "achieved_auc": achieved})
    pd.DataFrame(sweep_rows).to_csv("results/threshold_sweep.csv", index=False)

    # ---------- plot 1: ROC curves (best boosted model per component) ----------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for comp in COMPONENTS:
        y_true, proba = best_probas[comp]
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{comp} (AUC {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curves - Best Boosted Model per Component")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/01_roc_curves.png")
    plt.close()

    # ---------- plot 2: model comparison on ROC-AUC ----------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot = results.pivot(index="component", columns="model", values="roc_auc").loc[COMPONENTS]
    pivot.plot(kind="bar", ax=ax, color=["#8E8E93", "#FF6B35", "#3FB68B"], width=0.8)
    lo = max(0.5, pivot.values.min() - 0.03)
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("ROC-AUC (test set)")
    ax.set_title("Model Comparison by Component")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig("results/02_model_comparison.png")
    plt.close()

    # ---------- plot 3: permutation importance per component ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, comp in zip(axes.flat, COMPONENTS):
        name, model = best_models[comp]
        X_test, y_test = test_sets[comp]
        imp = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                     n_repeats=10, random_state=SEED)
        s = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
        s.tail(6).plot(kind="barh", ax=ax, color="#FF6B35")
        ax.set_title(f"{comp} ({name})")
        ax.set_xlabel("Drop in ROC-AUC when shuffled")
    plt.tight_layout()
    plt.savefig("results/03_feature_importance.png")
    plt.close()

    print("\nSaved: results/metrics.csv, results/01_roc_curves.png, "
          "results/02_model_comparison.png, results/03_feature_importance.png")


if __name__ == "__main__":
    main()