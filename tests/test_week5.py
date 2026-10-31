"""Week 5 tests: explanations and report generation. Run from project root:
    python -m unittest discover -s tests -v"""
import os, sys, unittest
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
import explain as ex, report as rp
from health_score import COMPONENTS


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv("data/car_health_ai_dataset.csv")
        t = pd.read_csv("results/test_health_scores.csv")
        cls.bg = cls.df[~cls.df.car_id.isin(t.car_id)].sample(100, random_state=7)
        cls.arts = ex.load_artifacts()
        s = t.sort_values("score")
        pick = lambda q: cls.df[cls.df.car_id == s.car_id.iloc[int(q * len(s))]]
        cls.worst, cls.mid, cls.best = pick(0.08), pick(0.5), pick(0.9)


class TestShapley(Base):
    def test_efficiency_all_components_three_cars(self):
        for car in (self.worst, self.mid, self.best):
            for c, e in ex.explain_car(car, self.arts, self.bg).items():
                self.assertLess(e["efficiency_gap"], 1e-9, c)

    def test_risk_matches_model_prediction(self):
        e = ex.shapley(self.arts["brakes"], self.worst, self.bg)
        self.assertAlmostEqual(e["risk"], ex.predict_risk(self.arts["brakes"], self.worst)[0], places=9)

    def test_contribution_keys_are_raw_features(self):
        e = ex.shapley(self.arts["engine"], self.mid, self.bg)
        self.assertEqual(set(e["contributions"]), set(self.arts["engine"]["raw_features"]))

    def test_thin_brake_pads_raise_brake_risk(self):
        car = self.df.nsmallest(1, "brake_pad_depth_mm")
        e = ex.shapley(self.arts["brakes"], car, self.bg)
        self.assertGreater(e["contributions"]["brake_pad_depth_mm"], 0.05)

    def test_thick_pads_lower_brake_risk(self):
        car = self.df.nlargest(1, "brake_pad_depth_mm")
        e = ex.shapley(self.arts["brakes"], car, self.bg)
        self.assertLess(e["contributions"]["brake_pad_depth_mm"], 0)

    def test_top_drivers_sorted_and_plain(self):
        e = ex.shapley(self.arts["tires"], self.worst, self.bg)
        d = ex.top_drivers(e, self.worst)
        eff = [x["effect_pp"] for x in d]
        self.assertEqual(eff, sorted(eff, reverse=True))
        self.assertTrue(all(x["direction"] in ("raises", "lowers") for x in d))


class TestReport(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.p = {k: rp.build_payload(c, cls.arts, cls.bg) for k, c in
                 {"worst": cls.worst, "mid": cls.mid, "best": cls.best}.items()}

    def test_template_passes_grounding_check(self):
        for k, p in self.p.items():
            self.assertEqual(rp.check_report(rp.render_template(p), p), [], k)

    def test_payload_matches_assess(self):
        p = self.p["worst"]
        self.assertEqual(p["rating"], "Poor")
        self.assertEqual(p["n_flagged"], len(p["flagged"]))

    def test_check_catches_invented_number(self):
        p = self.p["worst"]
        bad = rp.render_template(p) + " Repairs cost 47000."
        self.assertTrue(any("numbers" in x for x in rp.check_report(bad, p)))

    def test_check_catches_missing_flagged_part(self):
        p = self.p["worst"]
        import re
        bad = re.sub("brake", "xxx", rp.render_template(p), flags=re.I)
        self.assertTrue(any("not mentioned" in x for x in rp.check_report(bad, p)))

    def test_check_catches_absolute_claim(self):
        p = self.p["worst"]
        self.assertTrue(rp.check_report(rp.render_template(p) + " The engine will fail.", p))

    def test_check_requires_simulated_note(self):
        p = self.p["best"]
        bad = rp.render_template(p).replace(rp.SIM_NOTE, "")
        self.assertTrue(any("simulated" in x for x in rp.check_report(bad, p)))

    def test_no_llm_uses_template(self):
        r = rp.generate_report(self.p["mid"])
        self.assertEqual(r["source"], "template")

    def test_good_llm_text_is_used(self):
        p = self.p["mid"]
        good = rp.render_template(p)
        r = rp.generate_report(p, llm_fn=lambda facts: good)
        self.assertEqual(r["source"], "llm")

    def test_bad_llm_text_falls_back(self):
        p = self.p["worst"]
        r = rp.generate_report(p, llm_fn=lambda facts: "Everything is fine, 99% safe.")
        self.assertEqual(r["source"], "template")
        self.assertTrue(r["problems"])
        self.assertEqual(rp.check_report(r["text"], p), [])

    def test_llm_exception_falls_back(self):
        def boom(facts): raise RuntimeError("no key")
        r = rp.generate_report(self.p["mid"], llm_fn=boom)
        self.assertEqual(r["source"], "template")

    def test_llm_receives_facts_json_only(self):
        seen = {}
        def spy(facts): seen["f"] = facts; return "x"
        rp.generate_report(self.p["mid"], llm_fn=spy, max_tries=1)
        self.assertIn('"health_score"', seen["f"])
        self.assertNotIn("age_years", seen["f"].split('"drivers"')[0])   # no raw dataset columns


if __name__ == "__main__":
    unittest.main(verbosity=2)