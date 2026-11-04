"""car-health-ai test suite (unittest). Run from project root:
    python -m unittest discover -s tests -v
Needs data/car_health_ai_dataset.csv and models/*_calibrated.joblib."""
import os, sys, unittest
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

import health_score as hs
import generate_data as gd
from train_models import FEATURES, DATA_PATH

COMPS = hs.COMPONENTS


class TestHealthScore(unittest.TestCase):          # unit tests, module health_score
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(hs.WEIGHTS.values()), 1.0, places=9)

    def test_all_zero_risk_scores_100(self):
        self.assertEqual(hs.health_score({c: 0.0 for c in COMPS}), 100.0)

    def test_all_one_risk_scores_0(self):
        self.assertAlmostEqual(hs.health_score({c: 1.0 for c in COMPS}), 0.0, places=9)

    def test_uniform_20pct_scores_80(self):
        self.assertAlmostEqual(hs.health_score({c: 0.2 for c in COMPS}), 80.0, places=9)

    def test_score_decreases_when_risk_rises(self):
        base = {c: 0.1 for c in COMPS}
        worse = dict(base, engine=0.6)
        self.assertLess(hs.health_score(worse), hs.health_score(base))

    def test_engine_weighs_more_than_battery(self):
        a = hs.health_score(dict({c: 0 for c in COMPS}, engine=0.5))
        b = hs.health_score(dict({c: 0 for c in COMPS}, battery=0.5))
        self.assertLess(a, b)

    def test_bad_weights_rejected(self):
        with self.assertRaises(ValueError):
            hs.health_score({c: 0.1 for c in COMPS}, weights={c: 0.5 for c in COMPS})

    def test_band_boundaries(self):
        self.assertEqual(hs.rating_for(90.0), "Good")
        self.assertEqual(hs.rating_for(89.99), "Fair")
        self.assertEqual(hs.rating_for(70.0), "Fair")
        self.assertEqual(hs.rating_for(69.99), "Poor")

    def test_safety_cap_downgrades_good(self):
        self.assertEqual(hs.rating_for(95, flags=["brakes"]), "Fair")

    def test_non_safety_flag_does_not_cap(self):
        self.assertEqual(hs.rating_for(95, flags=["battery"]), "Good")

    def test_flag_threshold_inclusive_and_sorted(self):
        r = {"engine": 0.30, "brakes": 0.55, "battery": 0.29, "transmission": 0.1,
             "suspension": 0.0, "tires": 0.4}
        self.assertEqual(hs.flagged_components(r), ["brakes", "tires", "engine"])

    def test_assess_output_shape(self):
        out = hs.assess({c: 0.1 for c in COMPS})
        self.assertEqual(set(out), {"health_score", "rating", "flagged", "risks"})
        self.assertEqual(set(out["risks"]), set(COMPS))


class TestDataGenerator(unittest.TestCase):         # unit/integration, generate_data
    @classmethod
    def setUpClass(cls):
        cls.df = gd.build_dataset(500)

    def test_row_and_column_count(self):
        self.assertEqual(self.df.shape, (500, 27))

    def test_no_missing_values(self):
        self.assertEqual(int(self.df.isna().sum().sum()), 0)

    def test_unique_car_ids(self):
        self.assertTrue(self.df["car_id"].is_unique)

    def test_labels_binary_and_probs_in_range(self):
        for c in COMPS:
            self.assertTrue(set(self.df[f"{c}_fail_within_10k"].unique()) <= {0, 1})
            p = self.df[f"{c}_fail_risk_prob"]
            self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_quality_indices_clipped_0_100(self):
        for col in ("oil_quality_index", "transmission_fluid_quality"):
            self.assertTrue(self.df[col].between(0, 100).all())

    def test_categories_valid(self):
        self.assertTrue(set(self.df.driving_conditions) <= {"urban", "highway", "mixed"})
        self.assertTrue(set(self.df.climate) <= {"hot", "temperate", "coastal"})

    def test_physical_ranges(self):
        self.assertTrue((self.df.age_years >= 0).all())
        self.assertTrue((self.df.mileage_km >= 0).all())
        self.assertTrue((self.df.tire_tread_depth_mm >= 0).all())

    def test_wear_relationship_brakes(self):
        # more mileage -> thinner pads (negative correlation expected)
        self.assertLess(self.df.mileage_km.corr(self.df.brake_pad_depth_mm), 0)


class TestLeakageGuard(unittest.TestCase):
    def test_no_probability_or_label_features(self):
        for comp, feats in FEATURES.items():
            for f in feats:
                self.assertNotIn("fail", f, f"{comp} uses leaky column {f}")

    def test_features_exist_in_dataset(self):
        cols = set(pd.read_csv(DATA_PATH, nrows=1).columns)
        for feats in FEATURES.values():
            self.assertTrue(set(feats) <= cols)

    def test_six_components(self):
        self.assertEqual(sorted(FEATURES), sorted(COMPS))


class TestModelArtifacts(unittest.TestCase):         # integration: models + score
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)
        cls.arts = {c: joblib.load(f"models/{c}_calibrated.joblib") for c in COMPS}

    def risks_for(self, rows):
        out = {}
        for c, a in self.arts.items():
            X = pd.get_dummies(rows[a["raw_features"]]).reindex(columns=a["features"], fill_value=0)
            out[c] = a["model"].predict_proba(X)[:, 1]
        return out

    def test_artifact_keys(self):
        for a in self.arts.values():
            self.assertTrue({"model", "features", "raw_features"} <= set(a))

    def test_probabilities_valid(self):
        r = self.risks_for(self.df.head(200))
        for c in COMPS:
            self.assertTrue(((r[c] >= 0) & (r[c] <= 1)).all())

    def test_single_row_inference_end_to_end(self):
        r = self.risks_for(self.df.iloc[[0]])
        out = hs.assess({c: v[0] for c, v in r.items()})
        self.assertTrue(0 <= out["health_score"] <= 100)
        self.assertIn(out["rating"], {"Good", "Fair", "Poor"})

    def test_missing_dummy_column_handled(self):
        # a one-row input only has one climate category; reindex must fill the rest
        r = self.risks_for(self.df.iloc[[3]])
        self.assertEqual(len(r["engine"]), 1)

    def test_worn_car_scores_lower_than_new_car(self):
        d = self.df
        new = d.nsmallest(100, "mileage_km")
        old = d.nlargest(100, "mileage_km")
        def s(x):
            r = self.risks_for(x)
            return np.mean([hs.health_score({c: r[c][i] for c in COMPS}) for i in range(len(x))])
        self.assertGreater(s(new), s(old))


class TestScoreValidation(unittest.TestCase):        # system-level, held-out results
    @classmethod
    def setUpClass(cls):
        cls.t = pd.read_csv("results/test_health_scores.csv")

    def test_score_range(self):
        self.assertTrue(self.t["score"].between(0, 100).all())

    def test_score_negatively_correlated_with_failures(self):
        rho = spearmanr(self.t["score"], self.t["actual_failures"])[0]
        self.assertLess(rho, -0.5)

    def test_band_failure_rates_increase(self):
        m = self.t.groupby("rating")["actual_failures"].mean()
        self.assertLess(m["Good"], m["Fair"])
        self.assertLess(m["Fair"], m["Poor"])


if __name__ == "__main__":
    unittest.main(verbosity=2)