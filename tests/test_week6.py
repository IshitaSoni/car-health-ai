"""Week 6 tests: dashboard logic and a smoke run of src/app.py against a Streamlit stub.
Run from the project root:  python -m unittest discover -s tests -v
The stub checks the page's flow and wiring only; it cannot check how the page looks
(Streamlit and Plotly are not needed to run these tests)."""
import os, sys, runpy, types, unittest
from unittest.mock import MagicMock
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import dashboard_logic as L
import health_score as hs


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = L.load_resources()
        cls.df = cls.res["df"]
        cls.default = L.default_inputs(cls.df)
        cls.test_scores = pd.read_csv("results/test_health_scores.csv").set_index("car_id")


class TestInputs(Base):
    def test_default_inputs_valid(self):
        self.assertEqual(L.validate_inputs(self.default), [])

    def test_all_specs_have_matching_dataset_columns(self):
        self.assertTrue(set(L.KEYS) <= set(self.df.columns))

    def test_sample_inputs_valid(self):
        for cid in self.res["samples"].values():
            self.assertEqual(L.validate_inputs(L.sample_inputs(self.df, cid)), [])

    def test_missing_key_rejected(self):
        bad = dict(self.default); del bad["climate"]
        self.assertTrue(any("missing" in p for p in L.validate_inputs(bad)))

    def test_bad_category_rejected(self):
        self.assertTrue(L.validate_inputs(dict(self.default, climate="arctic")))

    def test_out_of_bounds_rejected(self):
        self.assertTrue(L.validate_inputs(dict(self.default, age_years=99)))
        self.assertTrue(L.validate_inputs(dict(self.default, brake_pad_depth_mm=-1)))

    def test_nan_and_text_rejected(self):
        self.assertTrue(L.validate_inputs(dict(self.default, age_years=float("nan"))))
        self.assertTrue(L.validate_inputs(dict(self.default, age_years="old")))

    def test_spec_bounds_are_valid_inputs(self):
        for k, _l, kind, lo, hi, *_ in L.INPUT_SPECS:
            if kind != "cat":
                self.assertEqual(L.validate_inputs(dict(self.default, **{k: lo})), [], k)
                self.assertEqual(L.validate_inputs(dict(self.default, **{k: hi})), [], k)

    def test_assess_raises_on_invalid(self):
        with self.assertRaises(ValueError):
            L.assess_car(dict(self.default, climate="arctic"), self.res)

    def test_range_warning_only_outside_training_range(self):
        self.assertEqual(L.range_warnings(self.default, self.df), [])
        w = L.range_warnings(dict(self.default, mileage_km=400000), self.df)
        self.assertEqual(len(w), 1)
        self.assertIn("Mileage", w[0])


class TestAssessment(Base):
    def test_matches_week4_scores_for_sample_cars(self):
        for cid in self.res["samples"].values():
            out = L.assess_car(L.sample_inputs(self.df, cid), self.res)
            ref = self.test_scores.loc[cid]
            self.assertAlmostEqual(out["assessment"]["health_score"], ref["score"], delta=0.06, msg=cid)
            self.assertEqual(out["assessment"]["rating"], ref["rating"], cid)

    def test_flagged_matches_threshold(self):
        out = L.assess_car(self.default, self.res)
        want = sorted(c for c, r in out["risks"].items() if r >= hs.FLAG_THRESHOLD)
        self.assertEqual(sorted(out["assessment"]["flagged"]), want)

    def test_explanations_only_for_flagged(self):
        out = L.assess_car(L.sample_inputs(self.df, self.res["samples"]["Poor car (8th percentile)"]), self.res)
        self.assertEqual(set(out["expl"]), set(out["assessment"]["flagged"]))
        self.assertGreater(len(out["expl"]), 0)

    def test_thinner_brake_pads_raise_brake_risk(self):
        a = L.assess_car(dict(self.default, brake_pad_depth_mm=8.0), self.res)["risks"]["brakes"]
        b = L.assess_car(dict(self.default, brake_pad_depth_mm=1.0), self.res)["risks"]["brakes"]
        self.assertGreater(b, a)

    def test_out_of_range_car_still_scores_with_warning(self):
        out = L.assess_car(dict(self.default, mileage_km=450000), self.res)
        self.assertTrue(out["warnings"])
        self.assertTrue(0 <= out["assessment"]["health_score"] <= 100)

    def test_deterministic(self):
        a = L.assess_car(self.default, self.res)["assessment"]
        b = L.assess_car(self.default, self.res)["assessment"]
        self.assertEqual(a, b)


class TestChartData(Base):
    def test_driver_table_sorted_and_sums_to_gap(self):
        out = L.assess_car(L.sample_inputs(self.df, self.res["samples"]["Poor car (8th percentile)"]), self.res)
        comp = out["assessment"]["flagged"][0]
        t = L.driver_table(out["expl"][comp], out["car"])
        eff = t["effect_pp"].abs().tolist()
        self.assertEqual(eff, sorted(eff, reverse=True))
        e = out["expl"][comp]
        self.assertAlmostEqual(t["effect_pp"].sum(), (e["risk"] - e["baseline"]) * 100, delta=0.5)

    def test_radar_is_closed_polygon_with_threshold(self):
        r = L.radar_data(L.assess_car(self.default, self.res)["risks"])
        self.assertEqual(r["names"][0], r["names"][-1])
        self.assertEqual(r["risk_pct"][0], r["risk_pct"][-1])
        self.assertEqual(set(r["threshold_pct"]), {30})
        self.assertEqual(len(r["names"]), 7)

    def test_gauge_bands_cover_0_to_100(self):
        g = L.gauge_spec(80, "Fair")
        self.assertEqual(g["steps"][0][0], 0)
        self.assertEqual(g["steps"][-1][1], 100)
        for a, b in zip(g["steps"], g["steps"][1:]):
            self.assertEqual(a[1], b[0])

    def test_risk_table_sorted_and_flags_agree(self):
        out = L.assess_car(self.default, self.res)
        t = L.risk_table(out["risks"])
        self.assertEqual(t["Risk %"].tolist(), sorted(t["Risk %"].tolist(), reverse=True))
        self.assertEqual(set(t[t.Flagged == "yes"]["Component"].str.lower()), set(out["assessment"]["flagged"]))


class TestReportWiring(Base):
    def test_template_report_without_key(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            self.assertFalse(L.llm_available())
            out = L.assess_car(self.default, self.res)
            rep = L.write_report(out["payload"], use_llm=True)    # asked, but unavailable
            self.assertEqual(rep["source"], "template")
            self.assertIn("simulated", rep["text"])
        finally:
            if old is not None: os.environ["ANTHROPIC_API_KEY"] = old


def make_streamlit_stub(prefill):
    st = types.ModuleType("streamlit")
    calls = {"error": [], "columns": [], "metric": [], "subheader": [], "plotly": 0}
    st.session_state = dict(prefill)
    deco = lambda f=None, **k: f if callable(f) else (lambda g: g)
    st.cache_resource, st.cache_data = deco, deco
    m = MagicMock()
    st.sidebar = MagicMock()
    st.sidebar.form.return_value = MagicMock()
    st.sidebar.selectbox = lambda *a, **k: None
    st.set_page_config = lambda **k: None
    st.error = lambda msg: calls["error"].append(msg)
    def stop(): raise RuntimeError("st.stop called")
    st.stop = stop
    st.selectbox = lambda label, options, key=None, **k: st.session_state[key]
    st.number_input = lambda label, key=None, **k: st.session_state[key]
    st.form_submit_button = lambda *a, **k: False
    st.checkbox = lambda *a, **k: False
    def columns(n):
        cols = [MagicMock() for _ in range(n)]
        for c in cols:
            c.metric = lambda label, value, **k: calls["metric"].append((label, value))
            c.plotly_chart = lambda *a, **k: calls.__setitem__("plotly", calls["plotly"] + 1)
        calls["columns"].append(cols); return cols
    st.columns = columns
    st.tabs = lambda names: [MagicMock() for _ in names]
    st.subheader = lambda t: calls["subheader"].append(t)
    def plot(*a, **k): calls["plotly"] += 1
    st.plotly_chart = plot
    for name in ("title", "caption", "warning", "info", "markdown", "write", "dataframe", "download_button"):
        setattr(st, name, m)
    return st, calls


class TestAppSmoke(Base):
    def run_app(self, prefill):
        st, calls = make_streamlit_stub(prefill)
        old = {k: sys.modules.get(k) for k in ("streamlit", "plotly", "plotly.graph_objects")}
        sys.modules["streamlit"] = st
        sys.modules["plotly"] = MagicMock(); sys.modules["plotly.graph_objects"] = MagicMock()
        try:
            runpy.run_path(os.path.join(ROOT, "src", "app.py"), run_name="__main__")
        finally:
            for k, v in old.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
        return calls

    def test_median_car_page_renders(self):
        calls = self.run_app({})
        self.assertEqual(calls["error"], [])
        labels = [m[0] for m in calls["metric"]]
        self.assertEqual(labels, ["Health Score", "Rating", "Flagged parts"])
        self.assertGreaterEqual(calls["plotly"], 3)          # gauge, radar, 1+ driver chart

    def test_poor_car_draws_one_driver_chart_per_flagged_part(self):
        cid = self.res["samples"]["Poor car (8th percentile)"]
        calls = self.run_app(L.sample_inputs(self.df, cid))
        flagged = L.assess_car(L.sample_inputs(self.df, cid), self.res)["assessment"]["flagged"]
        self.assertEqual(calls["plotly"], 2 + len(flagged))
        self.assertEqual(len(calls["subheader"]), 1 + len(flagged))

    def test_invalid_input_shows_error_and_stops(self):
        bad = dict(self.default, climate="arctic")
        with self.assertRaises(RuntimeError):
            self.run_app(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)