"""
car-health-ai - Week 6: Streamlit dashboard
===========================================
Run from the project root:
    python -m pip install streamlit plotly
    streamlit run src/app.py
The page only draws; all computation is in src/dashboard_logic.py.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import dashboard_logic as L
import whatif as W
import price_impact as PI

st.set_page_config(page_title="Car Health AI", layout="wide")
BLUE, ORANGE, INK = "#2F6DB5", "#E8541A", "#1C1C1E"


@st.cache_resource
def get_resources():
    return L.load_resources()


@st.cache_data
def run_assessment(items):
    return L.assess_car(dict(items), get_resources())


try:
    res = get_resources()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------- sidebar: inputs ----------------
defaults = L.default_inputs(res["df"])
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


for c, v in PI.DEFAULT_COSTS.items():
    st.session_state.setdefault(f"cost_{c}", int(v))


def apply_sample():
    choice = st.session_state["sample_choice"]
    vals = defaults if choice == "Median car" else L.sample_inputs(res["df"], res["samples"][choice])
    for k, v in vals.items():
        st.session_state[k] = v


st.sidebar.title("Car details")
st.sidebar.selectbox("Start from", ["Median car"] + list(res["samples"]), key="sample_choice",
                     on_change=apply_sample)
with st.sidebar.form("car_form"):
    group = None
    for key, label, kind, lo, hi, step, grp in L.INPUT_SPECS:
        if grp != group:
            st.markdown(f"**{grp}**")
            group = grp
        if kind == "cat":
            st.selectbox(label, lo, key=key)
        elif kind == "int":
            st.number_input(label, min_value=int(lo), max_value=int(hi), step=int(step), key=key)
        else:
            st.number_input(label, min_value=float(lo), max_value=float(hi), step=float(step), key=key)
    st.form_submit_button("Assess car", type="primary")

with st.sidebar.expander("Repair cost assumptions (placeholders: edit)"):
    st.caption("No repair-cost data exists in this project. Replace these with local garage quotes.")
    for c in PI.DEFAULT_COSTS:
        st.number_input(f"{c.capitalize()} ({PI.CURRENCY})", min_value=0, step=500, key=f"cost_{c}")

inputs = {k: st.session_state[k] for k in L.KEYS}
costs = {c: float(st.session_state[f"cost_{c}"]) for c in PI.DEFAULT_COSTS}

# ---------------- main page ----------------
st.title("Car Health AI")
st.caption("Pre-purchase inspection assistant. Scores come from models trained on simulated cars: "
           "use them as a screening aid, not a diagnosis.")

try:
    out = run_assessment(tuple(sorted(inputs.items())))
except ValueError as e:
    st.error(f"Cannot assess this car: {e}")
    st.stop()

a, risks, payload = out["assessment"], out["risks"], out["payload"]
for w in out["warnings"]:
    st.warning(w)

cost = PI.expected_cost(risks, costs)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Health Score", f"{a['health_score']:.1f} / 100")
m2.metric("Rating", a["rating"])
m3.metric("Flagged parts", len(a["flagged"]),
          help=f"Risk of failing within 10,000 km at or above {int(L.hs.FLAG_THRESHOLD * 100)}%")
m4.metric(f"Expected repair cost ({PI.CURRENCY})", f"{round(cost['total']):,}",
          help="Sum of (failure risk x assumed repair cost) over the next 10,000 km. Costs are placeholders.")
if payload["rating_capped_by_safety_part"]:
    st.info("Rating held at Fair because a safety-critical part is flagged.")

tab_overview, tab_why, tab_whatif, tab_report = st.tabs(["Overview", "Why", "What-if", "Report"])

with tab_overview:
    g = L.gauge_spec(a["health_score"], a["rating"])
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=g["value"], number={"font": {"color": INK}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": g["bar"]},
               "steps": [{"range": [lo, hi], "color": c} for lo, hi, c in g["steps"]]}))
    gauge.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=10))
    r = L.radar_data(risks)
    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=r["risk_pct"], theta=r["names"], fill="toself", name="Risk",
                                    line=dict(color=BLUE)))
    radar.add_trace(go.Scatterpolar(r=r["threshold_pct"], theta=r["names"], name="Flag level",
                                    line=dict(color=ORANGE, dash="dash")))
    radar.update_layout(height=340, margin=dict(l=40, r=40, t=30, b=30),
                        polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")),
                        legend=dict(orientation="h", y=-0.1))
    c1, c2 = st.columns(2)
    c1.plotly_chart(gauge, use_container_width=True)
    c2.plotly_chart(radar, use_container_width=True)
    st.subheader("Risk of failing within the next 10,000 km")
    st.dataframe(L.risk_table(risks), hide_index=True, use_container_width=True)
    st.subheader("Expected repair cost by part")
    st.dataframe(cost["table"], hide_index=True, use_container_width=True)
    st.caption("Repair costs are editable placeholders in the sidebar, not data.")

with tab_why:
    if not a["flagged"]:
        st.write("No part is flagged, so there is nothing to explain. "
                 "The highest-risk parts are listed on the Overview tab.")
    for comp in a["flagged"]:
        t = L.driver_table(out["expl"][comp], out["car"])
        st.subheader(f"{comp.capitalize()}: {round(risks[comp] * 100)}% risk")
        bar = go.Figure(go.Bar(
            x=t["effect_pp"][::-1], y=t["input"][::-1], orientation="h",
            marker_color=[ORANGE if v > 0 else BLUE for v in t["effect_pp"][::-1]],
            customdata=t["value"][::-1], hovertemplate="%{customdata}<br>%{x:+.1f} points<extra></extra>"))
        bar.update_layout(height=60 + 38 * len(t), margin=dict(l=10, r=10, t=10, b=30),
                          xaxis_title="percentage points of failure risk (orange raises, blue lowers)")
        st.plotly_chart(bar, use_container_width=True)

with tab_whatif:
    st.write("Pick fixes to see how the model's risk estimate would change. This is a what-if on the "
             "model's inputs, built on simulated wear rules, not a promise about a real repair.")
    choices = st.multiselect("Fixes to try", list(W.PRESETS), key="whatif_choices")
    if not choices:
        st.write("Choose one or more fixes above.")
    else:
        new_inputs = W.apply_presets(inputs, choices, res["df"])
        cmp = W.compare(out, new_inputs, res, costs)
        w1, w2, w3 = st.columns(3)
        w1.metric("Health Score", f"{cmp['score_after']:.1f}", delta=f"{cmp['score_change']:+.1f}")
        w2.metric("Rating", cmp["rating_after"], help=f"Before: {cmp['rating_before']}")
        w3.metric(f"Expected repair cost ({PI.CURRENCY})", f"{cmp['expected_cost_after']:,}",
                  delta=f"{cmp['expected_cost_change']:+,}", delta_color="inverse")
        t = cmp["table"]
        bars = go.Figure()
        bars.add_trace(go.Bar(x=t["Component"], y=t["Before %"], name="Before", marker_color=BLUE))
        bars.add_trace(go.Bar(x=t["Component"], y=t["After %"], name="After", marker_color=ORANGE))
        bars.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=10, b=30),
                           yaxis_title="risk of failing within 10,000 km (%)", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(bars, use_container_width=True)
        st.dataframe(t, hide_index=True, use_container_width=True)
        st.caption("Age and mileage cannot be changed by a repair, so risk never falls below what they imply.")

with tab_report:
    use_llm = False
    if L.llm_available():
        use_llm = st.checkbox("Write with a LangChain model (text is checked against the facts)")
    else:
        st.caption("Template report. To use a language model: pip install langchain-anthropic and set ANTHROPIC_API_KEY.")
    rep = L.write_report(payload, use_llm=use_llm)
    st.markdown(rep["text"])
    st.caption(f"Written by: {rep['source']}" + (f" | notes: {rep['problems']}" if rep["problems"] else ""))
    st.download_button("Download report (.md)", rep["text"], file_name="inspection_report.md")