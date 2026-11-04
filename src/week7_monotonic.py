"""
car-health-ai - Week 7: monotonic-constraint models (opt-in)
============================================================
Problem found in Week 7 testing: the unconstrained boosted models can be jagged,
e.g. one car's transmission risk was 12% at fluid quality 78 but 33% at 80, so a
"fix" in the what-if tool could RAISE a part's risk. Failure rate in the data
falls smoothly as fluid quality rises, so this is the model learning noise.

Fix: tell the boosting model which way each input may push risk
(monotonic_cst). The direction of each constraint is taken from the DATA
(sign of the Spearman correlation between the input and that part's failure
label), not assumed. Every numeric input is constrained, including weak ones:
a weakly related input left free is where the model learned spikes (e.g. the
battery model's service-gap input, which has |rho| < 0.05). Categorical inputs
(climate, driving conditions) stay unconstrained.

Trains with the same split, seed and settings as Week 3, so the comparison is
fair. Saves models/<component>_monotone.joblib (same layout as *_calibrated)
and results/week7_monotone_comparison.csv. Does NOT touch the Week 3-4 models.
Usage (project root):  python src/week7_monotonic.py
"""
import os, sys, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from train_models import FEATURES, DATA_PATH, SEED
import explain as ex

df = pd.read_csv(DATA_PATH)
rows = []
for comp, feats in FEATURES.items():
    y = df[f"{comp}_fail_within_10k"]
    # direction per raw numeric input, from the data
    sign = {}
    for f in feats:
        if df[f].dtype == object or str(df[f].dtype).startswith("str") or df[f].dtype.name == "category":
            sign[f] = 0
        else:
            rho = spearmanr(df[f], y)[0]
            sign[f] = 1 if rho > 0 else -1
    X = pd.get_dummies(df[feats])
    cst = []
    for col in X.columns:
        raw = next((f for f in feats if col == f), None)
        cst.append(sign[raw] if raw else 0)          # one-hot columns stay unconstrained
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    kw = dict(max_iter=200, learning_rate=0.05, max_depth=3, random_state=SEED)
    free = HistGradientBoostingClassifier(**kw).fit(Xtr, ytr)
    mono = HistGradientBoostingClassifier(monotonic_cst=cst, **kw).fit(Xtr, ytr)
    pf, pm = free.predict_proba(Xte)[:, 1], mono.predict_proba(Xte)[:, 1]
    rows.append({"component": comp, "constraints": {f: s for f, s in sign.items() if s},
                 "auc_free": roc_auc_score(yte, pf), "auc_monotone": roc_auc_score(yte, pm),
                 "brier_free": brier_score_loss(yte, pf), "brier_monotone": brier_score_loss(yte, pm)})
    # final artifact: same fitting rule as week4 (fit on the Week 4 training cars)
    t = pd.read_csv("results/test_health_scores.csv")
    tr = df[~df.car_id.isin(t.car_id)]
    final = HistGradientBoostingClassifier(monotonic_cst=cst, **kw).fit(
        pd.get_dummies(tr[feats]).reindex(columns=X.columns, fill_value=0), tr[f"{comp}_fail_within_10k"])
    joblib.dump({"model": final, "features": list(X.columns), "raw_features": feats,
                 "calibration": "none", "monotone": True}, f"models/{comp}_monotone.joblib")
out = pd.DataFrame(rows)
out.to_csv("results/week7_monotone_comparison.csv", index=False)
show = out.drop(columns="constraints").round(4)
print(show.to_string(index=False))
print("\nmean AUC free %.4f -> monotone %.4f | mean Brier free %.4f -> monotone %.4f" % (
    out.auc_free.mean(), out.auc_monotone.mean(), out.brier_free.mean(), out.brier_monotone.mean()))
for r in rows: print(r["component"], r["constraints"])