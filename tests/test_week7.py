"""Week 7 tests: what-if, repair cost, random-input invariants, and a fresh end-to-end run.
Run from the project root:  python -m unittest discover -s tests -v"""
import os, sys, time, unittest
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import dashboard_logic as L, whatif as W, price_impact as PI
import health_score as hs, explain as ex, report as rp, generate_data as gd
from train_models import FEATURES
from health_score import COMPONENTS


MONO = all(os.path.exists(f"models/{c}_monotone.joblib") for c in COMPONENTS)
SKIP_MSG = "run `python src/week7_monotonic.py` first"


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = L.load_resources()
        cls.df = cls.res["df"]
        cls.cars = {k: L.sample_inputs(cls.df, v) for k, v in cls.res["samples"].items()}
        cls.default = L.default_inputs(cls.df)


class TestWhatIf(Base):
    def test_no_presets_leaves_inputs_unchanged(self):
        self.assertEqual(W.apply_presets(self.default, [], self.df), self.default)

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            W.apply_presets(self.default, ["Teleport car"], self.df)

    def test_every_preset_gives_valid_inputs(self):
        for name in W.PRESETS:
            new = W.apply_presets(self.default, [name], self.df)
            self.assertEqual(L.validate_inputs(new), [], name)

    def test_healthy_values_come_from_the_dataset(self):
        new = W.apply_presets(dict(self.default, brake_pad_depth_mm=1.0), ["Replace brake pads"], self.df)
        self.assertAlmostEqual(new["brake_pad_depth_mm"], np.percentile(self.df["brake_pad_depth_mm"], 75), places=1)

    def test_already_healthy_inputs_are_left_alone(self):
        good = dict(self.default, brake_pad_depth_mm=9.0, vibration_score=0.5, service_gap_months=1.8)
        new = W.apply_presets(good, ["Replace brake pads", "Fix suspension / vibration", "Full service (fluids and oil)"], self.df)
        for k in ("brake_pad_depth_mm", "vibration_score", "service_gap_months"):
            self.assertEqual(new[k], good[k], k)

    def test_regression_service_fix_does_not_hit_the_clipped_gap_artifact(self):
        # a typical car with a 1.8-month gap used to be moved to the clipped 1.0 and got a spurious battery jump
        inp = self.cars["Typical car (50th percentile)"]
        new = W.apply_presets(inp, ["Full service (fluids and oil)"], self.df)
        self.assertEqual(new["service_gap_months"], inp["service_gap_months"])

    def test_brake_fix_changes_only_brake_risk(self):
        inp = self.cars["Poor car (8th percentile)"]
        before = L.assess_car(inp, self.res)
        c = W.compare(before, W.apply_presets(inp, ["Replace brake pads"], self.df), self.res)
        for comp in COMPONENTS:
            if comp != "brakes":
                self.assertEqual(c["after"]["risks"][comp], before["risks"][comp], comp)
        self.assertLess(c["after"]["risks"]["brakes"], before["risks"]["brakes"])
        self.assertGreater(c["score_change"], 0)

    @unittest.skipUnless(MONO, SKIP_MSG)
    def test_all_fixes_never_lower_score_on_sample_cars(self):
        for name, inp in self.cars.items():
            before = L.assess_car(inp, self.res)
            c = W.compare(before, W.apply_presets(inp, list(W.PRESETS), self.df), self.res)
            self.assertGreaterEqual(c["score_after"], c["score_before"], name)

    def test_compare_reports_cost_change_when_costs_given(self):
        inp = self.cars["Poor car (8th percentile)"]
        before = L.assess_car(inp, self.res)
        c = W.compare(before, W.apply_presets(inp, list(W.PRESETS), self.df), self.res, PI.DEFAULT_COSTS)
        self.assertLess(c["expected_cost_after"], c["expected_cost_before"])
        self.assertEqual(c["expected_cost_change"], c["expected_cost_after"] - c["expected_cost_before"])

    def test_compare_without_costs_has_no_cost_keys(self):
        inp = self.default
        c = W.compare(L.assess_car(inp, self.res), inp, self.res)
        self.assertNotIn("expected_cost_before", c)
        self.assertEqual(c["score_change"], 0)


@unittest.skipUnless(MONO, SKIP_MSG)
class TestMonotoneModels(Base):
    """Week 7 finding: unconstrained models could show a part's risk RISING after a fix."""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mres = L.load_resources(model_set="monotone")

    def test_auto_picks_monotone_when_files_exist(self):
        self.assertEqual(L.pick_model_set(), "monotone")
        self.assertEqual(L.pick_model_set(model_set="calibrated"), "calibrated")

    def test_no_fix_ever_raises_any_part_risk_or_lowers_score(self):
        rng = np.random.default_rng(0)
        for cid in rng.choice(self.df.car_id.values, 40, replace=False):
            inp = L.sample_inputs(self.df, cid)
            b = L.assess_car(inp, self.mres)
            c = W.compare(b, W.apply_presets(inp, list(W.PRESETS), self.df), self.mres)
            self.assertGreaterEqual(c["score_after"], c["score_before"], cid)
            for k in COMPONENTS:
                self.assertLessEqual(c["after"]["risks"][k], b["risks"][k] + 1e-9, (cid, k))

    def test_risk_is_monotone_in_each_sensor(self):
        # (input, component, direction of risk as the input rises)
        cases = [("transmission_fluid_quality", "transmission", -1), ("brake_pad_depth_mm", "brakes", -1),
                 ("tire_tread_depth_mm", "tires", -1), ("battery_voltage", "battery", -1),
                 ("oil_quality_index", "engine", -1), ("vibration_score", "suspension", +1)]
        for cid in list(self.mres["samples"].values()):
            base = L.sample_inputs(self.df, cid)
            for key, comp, d in cases:
                lo, hi = float(self.df[key].min()), float(self.df[key].max())
                risks = [float(ex.predict_risk(self.mres["artifacts"][comp],
                                               L.car_frame(dict(base, **{key: float(v)})))[0])
                         for v in np.linspace(lo, hi, 12)]
                diffs = np.diff(risks) * d
                self.assertTrue((diffs >= -1e-9).all(), (cid, key))

    def test_comparison_file_shows_no_accuracy_loss(self):
        t = pd.read_csv("results/week7_monotone_comparison.csv")
        self.assertGreaterEqual(t.auc_monotone.mean(), t.auc_free.mean() - 0.005)


class TestPriceImpact(unittest.TestCase):
    def test_zero_risk_zero_cost(self):
        self.assertEqual(PI.expected_cost({c: 0.0 for c in COMPONENTS})["total"], 0)

    def test_certain_failure_sums_all_costs(self):
        self.assertAlmostEqual(PI.expected_cost({c: 1.0 for c in COMPONENTS})["total"],
                               sum(PI.DEFAULT_COSTS.values()))

    def test_formula_for_one_part(self):
        r = {c: 0.0 for c in COMPONENTS}; r["engine"] = 0.5
        self.assertAlmostEqual(PI.expected_cost(r, PI.DEFAULT_COSTS)["total"], 0.5 * PI.DEFAULT_COSTS["engine"])

    def test_negative_cost_rejected(self):
        with self.assertRaises(ValueError):
            PI.expected_cost({c: 0.1 for c in COMPONENTS}, dict(PI.DEFAULT_COSTS, brakes=-1))

    def test_flagged_total_not_more_than_total_and_table_sorted(self):
        r = {c: 0.1 for c in COMPONENTS}; r["brakes"] = 0.6
        out = PI.expected_cost(r)
        self.assertLessEqual(out["flagged_total"], out["total"])
        self.assertEqual(out["flagged_total"], 0.6 * PI.DEFAULT_COSTS["brakes"])
        e = out["table"]["Expected cost"].tolist()
        self.assertEqual(e, sorted(e, reverse=True))

    def test_defaults_cover_every_component(self):
        self.assertEqual(set(PI.DEFAULT_COSTS), set(COMPONENTS))


class TestRandomInputs(Base):
    """150 random valid cars, each assessed once and shared by the tests below."""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        rng = np.random.default_rng(0)
        cls.rand = []
        for _ in range(150):
            car = {}
            for k, _l, kind, lo, hi, *_ in L.INPUT_SPECS:
                if kind == "cat":
                    car[k] = lo[rng.integers(len(lo))]
                else:
                    a, b = float(cls.df[k].min()), float(cls.df[k].max())
                    v = rng.uniform(a, b)
                    car[k] = int(round(v)) if kind == "int" else round(float(v), 2)
            cls.rand.append(car)
        cls.outs = [L.assess_car(c, cls.res) for c in cls.rand]

    def test_all_random_cars_valid_and_scored_in_range(self):
        for car, out in zip(self.rand, self.outs):
            self.assertEqual(L.validate_inputs(car), [])
            a = out["assessment"]
            self.assertTrue(0 <= a["health_score"] <= 100)
            self.assertIn(a["rating"], {"Good", "Fair", "Poor"})

    def test_flags_and_cap_consistent(self):
        for out in self.outs:
            a, r = out["assessment"], out["risks"]
            self.assertEqual(sorted(a["flagged"]), sorted(c for c in COMPONENTS if r[c] >= hs.FLAG_THRESHOLD))
            if any(c in hs.SAFETY_CRITICAL for c in a["flagged"]):
                self.assertNotEqual(a["rating"], "Good")

    def test_explanations_stay_exact_on_random_cars(self):
        for out in self.outs[:60]:
            for c, e in out["expl"].items():
                self.assertLess(e["efficiency_gap"], 1e-9, c)

    def test_reports_for_random_cars_are_grounded(self):
        for out in self.outs[:60]:
            p = out["payload"]
            self.assertEqual(rp.check_report(rp.render_template(p), p), [])

    def test_assessment_is_fast_enough_for_interactive_use(self):
        t = time.perf_counter()
        for car in self.rand[:10]:
            L.assess_car(car, self.res)
        self.assertLess((time.perf_counter() - t) / 10, 1.0)      # seconds per car


class TestFreshEndToEnd(unittest.TestCase):
    """Generate new data -> train -> score -> explain -> report, with nothing from earlier weeks' files."""
    @classmethod
    def setUpClass(cls):
        gd.RNG = np.random.default_rng(123)
        cls.df = gd.build_dataset(1800)
        n_train = 1400
        cls.train, cls.test = cls.df.iloc[:n_train], cls.df.iloc[n_train:]
        cls.arts, cls.aucs = {}, {}
        for c, feats in FEATURES.items():
            X = pd.get_dummies(cls.train[feats])
            m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=3, random_state=42)
            m.fit(X, cls.train[f"{c}_fail_within_10k"])
            cls.arts[c] = {"model": m, "features": list(X.columns), "raw_features": feats}
            Xt = pd.get_dummies(cls.test[feats]).reindex(columns=X.columns, fill_value=0)
            cls.aucs[c] = roc_auc_score(cls.test[f"{c}_fail_within_10k"], m.predict_proba(Xt)[:, 1])

    def test_dataset_shape(self):
        self.assertEqual(self.df.shape, (1800, 27))

    def test_models_beat_chance_on_every_component(self):
        for c, a in self.aucs.items():
            self.assertGreater(a, 0.65, c)

    def test_pipeline_scores_track_failures(self):
        scores, fails = [], []
        for i in range(len(self.test)):
            car = self.test.iloc[[i]]
            risks = {c: float(ex.predict_risk(self.arts[c], car)[0]) for c in COMPONENTS}
            scores.append(hs.assess(risks)["health_score"])
            fails.append(sum(int(car[f"{c}_fail_within_10k"].iloc[0]) for c in COMPONENTS))
        self.assertLess(spearmanr(scores, fails)[0], -0.4)

    def test_explain_and_report_on_fresh_models(self):
        car = self.test.iloc[[0]]
        bg = self.train.sample(60, random_state=7)
        p = rp.build_payload(car, self.arts, bg)
        self.assertEqual(rp.check_report(rp.render_template(p), p), [])
        for e in ex.explain_car(car, self.arts, bg).values():
            self.assertLess(e["efficiency_gap"], 1e-9)

    def test_generator_is_reproducible_for_a_given_seed(self):
        gd.RNG = np.random.default_rng(5); a = gd.build_dataset(200)
        gd.RNG = np.random.default_rng(5); b = gd.build_dataset(200)
        pd.testing.assert_frame_equal(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)