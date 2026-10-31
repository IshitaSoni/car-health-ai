"""
car-health-ai - Week 4: calibration + Health Score
===================================================
Steps
  1. Common train/test split (so all six components are scored on the SAME cars,
     which Week 3 did not do - each component had its own split).
  2. Calibration check: are predicted probabilities trustworthy? Compare the raw
     model with sigmoid and isotonic calibration using out-of-fold predictions
     on the training data only (the test set is never used to choose anything).
  3. Choose the flag threshold from the same out-of-fold predictions.
  4. Combine the six calibrated risks into a 0-100 Health Score (health_score.py).
  5. Tune the rating bands on the development (out-of-fold) scores only.
  6. Validate once on the untouched test set: does a lower score really mean
     more failures? Plus a weight-sensitivity check.

Usage (from the project folder):
    python week4_health_score.py
Needs: data/car_health_ai_dataset.csv, train_models.py and health_score.py in
the same folder as this script.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, precision_score, recall_score

from train_models import FEATURES          # same per-component feature sets as Week 3
import health_score as hs

DATA_PATH = "data/car_health_ai_dataset.csv"
SEED = 42
COMPONENTS = hs.COMPONENTS
N_BINS = 10

# chart colours (validated: lightness band, chroma, CVD separation, contrast)
BLUE = "#2F6DB5"
ORANGE = "#E8541A"
INK = "#1C1C1E"
MUTED = "#6E6E73"
GRID = "#E5E5EA"

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "font.size": 10, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": GRID, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def base_model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                          max_depth=3, random_state=SEED)


def make_estimator(method):
    return base_model() if method == "none" else CalibratedClassifierCV(base_model(), method=method, cv=5)


def ece(y, p, n_bins=N_BINS):
    """Expected calibration error with equal-frequency bins."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"], n_bins, labels=False, duplicates="drop")
    g = d.groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((g["n"] * (g["p"] - g["y"]).abs()).sum() / len(d))


def reliability(y, p, n_bins=N_BINS):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"], n_bins, labels=False, duplicates="drop")
    g = d.groupby("bin").agg(p=("p", "mean"), y=("y", "mean"))
    return g["p"].values, g["y"].values


def score_frame(R, y_counts_df):
    """Scores, flags and ratings for a dataframe of calibrated risks."""
    w = np.array([hs.WEIGHTS[c] for c in COMPONENTS])
    scores = 100.0 * (1.0 - R[COMPONENTS].values @ w)
    flags = [hs.flagged_components(dict(zip(COMPONENTS, row))) for row in R[COMPONENTS].values]
    ratings = [hs.rating_for(s, f) for s, f in zip(scores, flags)]
    return pd.DataFrame({"score": scores, "rating": ratings, "n_flagged": [len(f) for f in flags]},
                        index=R.index)


def band_table(R, scores_df, Y):
    """Per-rating summary: cars, predicted vs actual failures."""
    safety = list(hs.SAFETY_CRITICAL)
    rows = []
    for label in [b[1] for b in hs.BANDS]:
        m = (scores_df["rating"] == label).values
        if m.sum() == 0:
            rows.append({"rating": label, "cars": 0})
            continue
        actual = Y.loc[m, COMPONENTS].sum(axis=1)
        rows.append({
            "rating": label, "cars": int(m.sum()),
            "avg_score": scores_df.loc[m, "score"].mean(),
            "predicted_failures": R.loc[m, COMPONENTS].sum(axis=1).mean(),
            "actual_failures": actual.mean(),
            "pct_any_failure": (actual >= 1).mean() * 100,
            "pct_safety_failure": (Y.loc[m, safety].sum(axis=1) >= 1).mean() * 100,
        })
    return pd.DataFrame(rows)


def main():
    style()
    df = pd.read_csv(DATA_PATH)
    Y = df[[f"{c}_fail_within_10k" for c in COMPONENTS]].copy()
    Y.columns = COMPONENTS
    any_fail = (Y.sum(axis=1) >= 1).astype(int)

    # ------------------------------------------------------------ 1. common split
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, stratify=any_fail, random_state=SEED)
    print(f"Common split: {len(idx_train)} train / {len(idx_test)} test cars")

    X = {c: pd.get_dummies(df[FEATURES[c]]) for c in COMPONENTS}
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # ------------------------------------------------- 2. calibration comparison
    methods = ["none", "sigmoid", "isotonic"]
    oof = {m: pd.DataFrame(index=idx_train, columns=COMPONENTS, dtype=float) for m in methods}
    rows = []
    print("\nCalibration comparison (out-of-fold on training data)")
    for c in COMPONENTS:
        for m in methods:
            p = cross_val_predict(make_estimator(m), X[c].loc[idx_train], Y.loc[idx_train, c],
                                  cv=skf, method="predict_proba")[:, 1]
            oof[m][c] = p
            rows.append({"component": c, "method": m,
                         "brier": brier_score_loss(Y.loc[idx_train, c], p),
                         "ece": ece(Y.loc[idx_train, c], p),
                         "roc_auc": roc_auc_score(Y.loc[idx_train, c], p)})
    cal_cmp = pd.DataFrame(rows)
    cal_cmp.to_csv("results/calibration_comparison.csv", index=False)
    summary = cal_cmp.groupby("method")[["brier", "ece", "roc_auc"]].mean().loc[methods]
    print(summary.round(4).to_string())

    best = summary["brier"].idxmin()
    if best != "none" and summary.loc[best, "brier"] > summary.loc["none", "brier"] - 0.0005:
        best = "none"      # gain too small to justify extra machinery
    print(f"-> chosen calibration: {best}")
    R_dev = oof[best].copy()

    # ----------------------------------------------- 3. flag threshold (dev only)
    thr_rows = []
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        yt = np.concatenate([Y.loc[idx_train, c].values for c in COMPONENTS])
        pt = np.concatenate([R_dev[c].values >= t for c in COMPONENTS]).astype(int)
        thr_rows.append({"threshold": t, "precision": precision_score(yt, pt),
                         "recall": recall_score(yt, pt),
                         "flag_rate": pt.mean()})
    thr = pd.DataFrame(thr_rows)
    thr.to_csv("results/flag_threshold_sweep.csv", index=False)
    print("\nFlag threshold sweep (pooled over components, dev data)")
    print(thr.round(3).to_string(index=False))

    # -------------------------------------------- final fit, test-set predictions
    final, base_fit = {}, {}
    R_test = pd.DataFrame(index=idx_test, columns=COMPONENTS, dtype=float)
    R_test_raw = pd.DataFrame(index=idx_test, columns=COMPONENTS, dtype=float)
    metric_rows = []
    for c in COMPONENTS:
        Xtr, Xte = X[c].loc[idx_train], X[c].loc[idx_test]
        ytr, yte = Y.loc[idx_train, c], Y.loc[idx_test, c]
        b = base_model().fit(Xtr, ytr)
        m = b if best == "none" else CalibratedClassifierCV(base_model(), method=best, cv=5).fit(Xtr, ytr)
        base_fit[c], final[c] = b, m
        raw, cal = b.predict_proba(Xte)[:, 1], m.predict_proba(Xte)[:, 1]
        R_test_raw[c], R_test[c] = raw, cal
        flag = (cal >= hs.FLAG_THRESHOLD).astype(int)
        metric_rows.append({
            "component": c,
            "brier_raw": brier_score_loss(yte, raw), "brier_final": brier_score_loss(yte, cal),
            "ece_raw": ece(yte, raw), "ece_final": ece(yte, cal),
            "roc_auc": roc_auc_score(yte, cal),
            "precision_at_flag": precision_score(yte, flag, zero_division=0),
            "recall_at_flag": recall_score(yte, flag, zero_division=0),
        })
        joblib.dump({"model": m, "base_model": b, "features": list(X[c].columns),
                     "raw_features": FEATURES[c], "calibration": best},
                    f"models/{c}_calibrated.joblib")
    test_metrics = pd.DataFrame(metric_rows)
    test_metrics.to_csv("results/calibration_metrics.csv", index=False)
    print("\nTest-set calibration and flag metrics (same 1,200 cars for every component)")
    print(test_metrics.round(3).to_string(index=False))

    # --------------------------------------------------- 4/5. score + band tuning
    S_dev = score_frame(R_dev, Y.loc[idx_train])
    print("\nDEV score percentiles (out-of-fold): "
          + ", ".join(f"p{q}={np.percentile(S_dev['score'], q):.1f}" for q in (5, 25, 50, 75, 95)))
    print(f"Current bands: {hs.BANDS}   flag threshold: {hs.FLAG_THRESHOLD}")
    print(band_table(R_dev, S_dev, Y.loc[idx_train]).round(2).to_string(index=False))

    # band-cutoff grid, development data only
    Yd = Y.loc[idx_train]
    flags_dev = [hs.flagged_components(dict(zip(COMPONENTS, r))) for r in R_dev[COMPONENTS].values]
    n_dev_fail = Yd.sum(axis=1).values
    grid = []
    for good in (80, 85, 88, 90, 92):
        for fair in (60, 65, 70, 75):
            bands = [(good, "Good"), (fair, "Fair"), (0.0, "Poor")]
            lab = np.array([hs.rating_for(s, f, bands) for s, f in zip(S_dev["score"], flags_dev)])
            row = {"good_cutoff": good, "fair_cutoff": fair}
            for name in ("Good", "Fair", "Poor"):
                m = lab == name
                row[f"pct_{name}"] = m.mean() * 100
                row[f"fail_{name}"] = n_dev_fail[m].mean() if m.any() else np.nan
            grid.append(row)
    grid = pd.DataFrame(grid)
    grid.to_csv("results/band_tuning_dev.csv", index=False)
    print("\nBand cutoff grid (dev data): share of cars and actual failures per car in each band")
    print(grid.round(2).to_string(index=False))

    # --------------------------------------------------------- 6. test validation
    S_test = score_frame(R_test, Y.loc[idx_test])
    Yt = Y.loc[idx_test]
    n_fail = Yt.sum(axis=1)
    rho = spearmanr(S_test["score"], n_fail).correlation
    auc_any = roc_auc_score((n_fail >= 1).astype(int), -S_test["score"])
    auc_two = roc_auc_score((n_fail >= 2).astype(int), -S_test["score"])
    auc_mile = roc_auc_score((n_fail >= 1).astype(int), df.loc[idx_test, "mileage_km"])
    auc_age = roc_auc_score((n_fail >= 1).astype(int), df.loc[idx_test, "age_years"])
    print("\nTEST validation of the Health Score")
    print(f"  Spearman(score, actual failures) = {rho:.3f}  (negative = lower score, more failures)")
    print(f"  ROC-AUC for 'any component fails'   : score {auc_any:.3f} | mileage alone {auc_mile:.3f} | age alone {auc_age:.3f}")
    print(f"  ROC-AUC for '2+ components fail'    : score {auc_two:.3f}")
    # noise ceiling: the same score built from the TRUE generating probabilities
    R_true = df.loc[idx_test, [f"{c}_fail_risk_prob" for c in COMPONENTS]]
    R_true.columns = COMPONENTS
    w_vec = np.array([hs.WEIGHTS[c] for c in COMPONENTS])
    oracle = 100 * (1 - R_true.values @ w_vec)
    print(f"  Noise ceiling (score from true probabilities): AUC any {roc_auc_score((n_fail >= 1).astype(int), -oracle):.3f}, "
          f"2+ {roc_auc_score((n_fail >= 2).astype(int), -oracle):.3f}, Spearman {spearmanr(oracle, n_fail).correlation:.3f}")

    bt = band_table(R_test, S_test, Yt)
    bt.to_csv("results/health_score_bands_test.csv", index=False)
    print("\nBand table (test set)")
    print(bt.round(2).to_string(index=False))

    # deciles of the score: predicted vs actual failures
    dec = pd.DataFrame({"score": S_test["score"], "pred": R_test.sum(axis=1), "actual": n_fail})
    dec["decile"] = pd.qcut(dec["score"], 10, labels=False)
    dg = dec.groupby("decile").agg(avg_score=("score", "mean"), predicted=("pred", "mean"),
                                   actual=("actual", "mean"))
    dg.to_csv("results/health_score_deciles_test.csv")
    print("\nDeciles (0 = lowest scores)")
    print(dg.round(2).to_string())

    # weight sensitivity
    variants = {
        "chosen weights": hs.WEIGHTS,
        "equal weights": {c: 1 / 6 for c in COMPONENTS},
        "safety-heavy": {"brakes": .30, "tires": .20, "suspension": .20, "engine": .10, "transmission": .10, "battery": .10},
        "cost-heavy": {"engine": .30, "transmission": .30, "brakes": .15, "suspension": .10, "tires": .05, "battery": .10},
    }
    sens = []
    chosen = S_test["score"]
    for name, w in variants.items():
        sc = 100 * (1 - R_test[COMPONENTS] @ pd.Series(w)[COMPONENTS])
        sens.append({"variant": name, "spearman_vs_chosen": spearmanr(sc, chosen).correlation,
                     "auc_any_failure": roc_auc_score((n_fail >= 1).astype(int), -sc),
                     "mean_score": sc.mean()})
    sc = 100 * (1 - R_test[COMPONENTS].max(axis=1))
    sens.append({"variant": "worst part only (1 - max risk)", "spearman_vs_chosen": spearmanr(sc, chosen).correlation,
                 "auc_any_failure": roc_auc_score((n_fail >= 1).astype(int), -sc), "mean_score": sc.mean()})
    sens = pd.DataFrame(sens)
    sens.to_csv("results/weight_sensitivity.csv", index=False)
    print("\nWeight sensitivity (test)")
    print(sens.round(3).to_string(index=False))

    out = pd.concat([df.loc[idx_test, ["car_id"]], R_test.add_suffix("_risk"), S_test,
                     n_fail.rename("actual_failures")], axis=1)
    out.round(4).to_csv("results/test_health_scores.csv", index=False)

    # example assessments: cars at the 8th, 50th and 90th percentile of the test scores
    # (representative cars, not the single worst/best outlier)
    order = S_test["score"].sort_values()
    pick = lambda q: order.index[int(q * (len(order) - 1))]
    picks = {"Poor example": pick(0.08), "Typical example": pick(0.50), "Good example": pick(0.90)}
    examples = {}
    for label, i in picks.items():
        a = hs.assess(R_test.loc[i].to_dict())
        a["car_id"] = df.loc[i, "car_id"]
        a["actual_failures"] = [c for c in COMPONENTS if Y.loc[i, c] == 1]
        examples[label] = a
    with open("results/example_assessments.json", "w") as f:
        json.dump(examples, f, indent=2)

    # ------------------------------------------------------------------- plots
    # 04: reliability curves
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True, sharey=True)
    for ax, c in zip(axes.flat, COMPONENTS):
        yte = Yt[c]
        ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1)
        if best != "none":
            px, py = reliability(yte, R_test_raw[c])
            ax.plot(px, py, "o-", color=BLUE, linewidth=2, markersize=6, label="Raw model")
        px, py = reliability(yte, R_test[c])
        ax.plot(px, py, "o-", color=ORANGE, linewidth=2, markersize=6,
                label="Calibrated" if best != "none" else "Model (uncalibrated)")
        ax.set_title(f"{c.capitalize()}   ECE {test_metrics.set_index('component').loc[c, 'ece_final']:.3f}",
                     loc="left", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("Predicted failure probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("Actual failure rate")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([0], [0], ls="--", color=MUTED)], l + ["Perfect calibration"],
               loc="lower center", ncol=3)
    fig.suptitle("Do predicted probabilities match reality? (test set, 10 equal-size bins)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig("results/04_calibration_curves.png")
    plt.close(fig)

    # 05: score distribution with bands
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(S_test["score"], bins=np.arange(0, 102, 2), color=BLUE, edgecolor="white", linewidth=1)
    cut = [b[0] for b in hs.BANDS if b[0] > 0]
    for x in cut:
        ax.axvline(x, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0, 101)
    top = ax.get_ylim()[1] * 1.28
    ax.set_ylim(0, top)
    edges = [101] + cut + [0]
    for hi, lo, (_, name) in zip(edges[:-1], edges[1:], hs.BANDS):
        n = int((S_test["rating"] == name).sum())
        ax.text((hi + lo) / 2, top * 0.98, f"{name}\n{n} cars", ha="center", va="top", fontsize=10, color=INK)
    ax.set_xlabel("Health Score (higher is healthier)")
    ax.set_ylabel("Number of cars")
    ax.set_title("Distribution of Health Scores on the test set", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/05_score_distribution.png")
    plt.close(fig)

    # 06: validation - predicted vs actual failures by decile, and by rating
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    xs = np.arange(1, 11)
    a1.plot(xs, dg["predicted"], "o-", color=BLUE, linewidth=2, markersize=6, label="Predicted failures per car")
    a1.plot(xs, dg["actual"], "o-", color=ORANGE, linewidth=2, markersize=6, label="Actual failures per car")
    a1.set_xticks(xs)
    a1.set_xticklabels([f"{v:.0f}" for v in dg["avg_score"]])
    a1.set_xlabel("Average Health Score of each group of 120 cars (lowest scores on the left)")
    a1.set_ylabel("Component failures per car")
    a1.set_title("Lower score, more failures", loc="left", fontsize=12)
    a1.legend(loc="upper right")
    labels = [b for b in bt["rating"]][::-1]
    vals = bt.set_index("rating").loc[labels, "actual_failures"].values
    cars = bt.set_index("rating").loc[labels, "cars"].values
    bars = a2.bar(labels, vals, color=BLUE, width=0.55)
    for b_, v, n in zip(bars, vals, cars):
        a2.text(b_.get_x() + b_.get_width() / 2, v + 0.03, f"{v:.2f}\n({n} cars)", ha="center", va="bottom", fontsize=10)
    a2.set_ylim(0, max(vals) * 1.3)
    a2.set_ylabel("Actual failures per car")
    a2.set_title("By rating band", loc="left", fontsize=12)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig("results/06_score_validation.png")
    plt.close(fig)

    # 07: example cars
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    for ax, (label, a) in zip(axes, examples.items()):
        r = [a["risks"][c] for c in COMPONENTS][::-1]
        names = [c.capitalize() for c in COMPONENTS][::-1]
        cols = [ORANGE if c.lower() in a["flagged"] else BLUE for c in names]
        ax.barh(names, r, color=cols, height=0.6)
        ax.axvline(hs.FLAG_THRESHOLD, color=MUTED, linestyle="--", linewidth=1)
        for i, v in enumerate(r):
            ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, 1)
        ax.set_title(f"{label}\nScore {a['health_score']:.0f} - {a['rating']}", loc="left", fontsize=11)
        ax.set_xlabel("Predicted failure risk (next 10,000 km)")
        ax.grid(axis="y", visible=False)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
               [f"Flagged (risk >= {hs.FLAG_THRESHOLD:.0%})", "Not flagged"], loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("results/07_example_assessments.png")
    plt.close(fig)

    print("\nSaved results/*.csv, results/*.json, plots 04-07 and models/*_calibrated.joblib")


if __name__ == "__main__":
    main()