# Car Health AI

AI-powered predictive health scoring system for pre-purchase used car inspection.
Final-year B.Tech project. 8-week build, weekly progress presentations.

## What this project does

Takes a car's age, mileage, service history, and diagnostic sensor readings, and
outputs a single **Health Score (0–100)** plus per-component risk flags for six
systems: **Engine, Brakes, Battery, Transmission, Suspension, Tires**. Flags
which components are likely to fail within the next 10,000 km, explains _why_
(via SHAP), and generates a natural-language inspection report (via an LLM).

## Tech stack

| Layer                                              | Tool                             |
| -------------------------------------------------- | -------------------------------- |
| Data generation & manipulation                     | Python, pandas, NumPy            |
| ML models (per-component failure-risk classifiers) | scikit-learn (Gradient Boosting) |
| Explainability                                     | SHAP (`TreeExplainer`)           |
| LLM report generation                              | LangChain                        |
| Dashboard                                          | Streamlit                        |

## System architecture

```
Inputs → ML Models → Health Score → Explainability → LLM Report → Dashboard
```

1. **Inputs** — age, mileage, service history, simulated sensor readings
2. **ML Models** — 6 separate Gradient Boosting classifiers, one per component,
   each predicting P(failure within 10,000 km)
3. **Health Score** — weighted aggregation of the 6 component risk scores into
   one 0–100 score
4. **Explainability** — SHAP values per component model, showing which input
   features drove each prediction
5. **LLM Report** — LangChain turns risk scores + SHAP output into a
   human-readable inspection report
6. **Dashboard** — Streamlit UI: input form, health score gauge, component
   radar chart, generated report tab

## Data

**No public dataset exists** pairing (age, mileage, service history, sensor
readings) with actual component failure outcomes — real fleet telemetry like
this is proprietary. This project uses a **synthetic dataset**, generated in
`generate_data.py`:

- 6,000 simulated cars
- Base features (age, mileage, driving conditions, climate, service history)
  drawn from realistic statistical distributions
- Diagnostic sensor readings (oil quality, engine temp, brake pad depth,
  battery voltage, transmission fluid quality, vibration score, tire tread
  depth) **derived** from base features using hand-designed degradation
  formulas — e.g. brake wear scales with mileage _and_ how much of that
  mileage was urban/stop-go driving; battery wear scales with age _and_ heat
  exposure
- Failure labels generated via a logistic function mapping each component's
  "stress" score to a failure probability, then sampled — not deterministic,
  includes realistic noise
- Component list and feature relationships are grounded in standard
  industry 200-point pre-purchase inspection (PPI) checklists, not arbitrary

Real-world reference datasets (Kaggle used-car listings, AI4I 2020 Predictive
Maintenance, NASA C-MAPSS) are used for **sanity-checking** synthetic ranges
and degradation patterns — not yet integrated into the pipeline.

This is disclosed explicitly in the project report as a known limitation and
design choice, not hidden.

## Project structure

```
car-health-ai/
├── data/
│   └── car_health_ai_dataset.csv     # generated dataset (not committed if large)
├── generate_data.py                   # synthetic data generator
├── notebooks/                         # EDA, model exploration (Week 2+)
├── models/                            # trained model artifacts (Week 3+)
├── src/                                # dashboard + pipeline code (Week 5+)
└── README.md
```

## Setup

```powershell
pip install pandas numpy scikit-learn shap langchain streamlit
mkdir data
python generate_data.py
```

## Current status

- [x] Domain research — component list grounded in PPI checklists
- [x] System architecture designed
- [x] Synthetic dataset generator built and working
- [ ] EDA on generated dataset
- [ ] Per-component Gradient Boosting models trained
- [ ] Health Score calibration
- [ ] SHAP explainability integrated
- [ ] LangChain report generator built
- [ ] Streamlit dashboard built
- [ ] End-to-end integration + testing
- [ ] Final report, demo video, presentation

## 8-week roadmap

| Week | Focus                                                                  |
| ---- | ---------------------------------------------------------------------- |
| 1    | Research, architecture, synthetic dataset generator — **done**         |
| 2    | EDA on dataset; cross-check against real reference datasets            |
| 3    | Baseline per-component Gradient Boosting models + evaluation metrics   |
| 4    | Health Score calibration (weighted aggregation across components)      |
| 5    | SHAP explainability + LangChain report generator                       |
| 6    | Streamlit dashboard v1                                                 |
| 7    | "What-if" simulator, stretch features (price-impact estimate), testing |
| 8    | Polish, deployment, final report, demo video, final presentation       |
