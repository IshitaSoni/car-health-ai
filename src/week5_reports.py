"""
car-health-ai - Week 5 run: explanations + example reports
==========================================================
Usage (project root):  python src/week5_reports.py
Optional, needs `pip install langchain-anthropic` and ANTHROPIC_API_KEY:
                       python src/week5_reports.py --llm
Outputs: results/week5_reports/*.md, results/week5_payloads.json,
         results/week5_global_importance.csv, results/08_explanations.png,
         results/week5_run_log.txt
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import explain as ex, report as rp
from health_score import COMPONENTS

BLUE, ORANGE, INK, MUTED, GRID = "#2F6DB5", "#E8541A", "#1C1C1E", "#6E6E73", "#E5E5EA"
USE_LLM = "--llm" in sys.argv
os.makedirs("results/week5_reports", exist_ok=True)

df = pd.read_csv("data/car_health_ai_dataset.csv")
tst = pd.read_csv("results/test_health_scores.csv")
background = df[~df.car_id.isin(tst.car_id)].sample(100, random_state=7)   # training cars only
arts = ex.load_artifacts()

llm_fn = None
if USE_LLM:
    from langchain_anthropic import ChatAnthropic
    llm_fn = rp.langchain_llm_fn(ChatAnthropic(model="claude-sonnet-5-5", temperature=0))

# ---- three example cars: 8th, 50th, 90th percentile of Health Score on the test set
s = tst.sort_values("score").reset_index(drop=True)
picks = {"poor_8th_pct": 0.08, "typical_50th_pct": 0.50, "good_90th_pct": 0.90}
payloads, log = {}, []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (name, q) in zip(axes, picks.items()):
    cid = s.car_id.iloc[int(q * len(s))]
    car = df[df.car_id == cid]
    t0 = time.perf_counter()
    p = rp.build_payload(car, arts, background)
    dt = time.perf_counter() - t0
    r = rp.generate_report(p, llm_fn=llm_fn)
    payloads[name] = p
    open(f"results/week5_reports/{name}.md", "w", encoding="utf-8").write(r["text"])
    log.append(f"{name}: car {cid} score {p['health_score']} {p['rating']} flagged {p['n_flagged']} "
               f"| payload {dt:.2f}s | report source {r['source']} | problems {r['problems']}")
    # chart: drivers of the highest-risk component (all 6 explained, pick top risk)
    full = ex.explain_car(car, arts, background)
    comp = max(full, key=lambda c: full[c]["risk"])
    contrib = sorted(full[comp]["contributions"].items(), key=lambda kv: kv[1])
    labels = [f.replace("_", " ") for f, _ in contrib]
    vals = [v * 100 for _, v in contrib]
    ax.barh(labels, vals, color=[ORANGE if v > 0 else BLUE for v in vals])
    for i, v in enumerate(vals):
        ax.text(v + (0.6 if v >= 0 else -0.6), i, ("0" if abs(v) < 0.5 else f"{v:+.0f}"), va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_title(f"{name.split('_')[0].capitalize()} car, highest-risk part: {comp}\nrisk {full[comp]['risk']*100:.0f}% "
                 f"(average car {full[comp]['baseline']*100:.0f}%)", fontsize=11, color=INK, loc="left")
    ax.set_xlabel("percentage points of failure risk", color=INK)
    lim = max(abs(min(vals)), abs(max(vals))) * 1.35 + 1
    ax.set_xlim(-lim, lim)
    ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=ORANGE, label="raises risk"), Patch(color=BLUE, label="lowers risk")],
           loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig("results/08_explanations.png", dpi=150); plt.close(fig)

json.dump(payloads, open("results/week5_payloads.json", "w"), indent=1)

# ---- global importance: mean |contribution| over 150 test cars, per component
test_cars = df[df.car_id.isin(tst.car_id)].sample(150, random_state=1)
rows = []
t0 = time.perf_counter()
for c in COMPONENTS:
    acc, eff = {}, 0.0
    for i in range(len(test_cars)):
        e = ex.shapley(arts[c], test_cars.iloc[[i]], background)
        eff = max(eff, e["efficiency_gap"])
        for f, v in e["contributions"].items():
            acc.setdefault(f, []).append(abs(v))
    for f, v in sorted(acc.items(), key=lambda kv: -np.mean(kv[1])):
        rows.append({"component": c, "feature": f, "mean_abs_pp": round(float(np.mean(v)) * 100, 2)})
    log.append(f"{c}: max efficiency gap over 150 cars = {eff:.2e}")
log.append(f"global importance for 150 cars x 6 components took {time.perf_counter()-t0:.0f}s")
g = pd.DataFrame(rows); g.to_csv("results/week5_global_importance.csv", index=False)
top = g.groupby("component").head(1)
log.append("top driver per component:\n" + top.to_string(index=False))
open("results/week5_run_log.txt", "w").write("\n".join(log))
print("\n".join(log))