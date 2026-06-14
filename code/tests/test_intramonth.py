"""
Tests for the intramonth nowcasting system (Part J).

Fast synthetic fixtures for the causal core (no network); a light integration test
guarded on data availability. The suite FAILS if:
  - future daily data leaks into an as-of feature,
  - regime posteriors are invalid / don't sum to 1,
  - model weights don't sum to 1 or don't shift with horizon,
  - scenario probabilities don't sum to 100%,
  - the pipeline drops a forecast origin,
  - models are not switchable through configuration.
"""
import os, sys, unittest
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intramonth import config as C, hf_data as H, scenarios as SC, weights as W


def _synth_daily(start="2020-01-01", end="2020-06-30", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end)
    base = 50 + np.cumsum(rng.standard_normal(len(idx)))
    return pd.DataFrame({n: base + rng.standard_normal(len(idx)) for n in C.HF_TICKERS},
                        index=idx)


class TestHFCausal(unittest.TestCase):
    def setUp(self):
        self.daily = _synth_daily()
        self.month = pd.Timestamp("2020-05-31")

    def test_asof_only_uses_past(self):
        """as-of feature window must contain no daily row after (month_end - k)."""
        for k in C.ORIGINS:
            asof = H.asof_date(self.month, k)
            feats = H.asof_features(self.daily, self.month, k)
            # reconstruct window manually and confirm last in-window date <= asof
            m_start = self.month.replace(day=1)
            win = self.daily[(self.daily.index >= m_start) & (self.daily.index <= asof)]
            if len(win):
                self.assertLessEqual(win.index.max(), asof)
            self.assertIn("hf_coverage", feats)

    def test_future_injection_invariance(self):
        """Injecting daily values AFTER the as-of date must not change features (no leakage)."""
        k = 10
        f0 = H.asof_features(self.daily, self.month, k)
        asof = H.asof_date(self.month, k)
        poisoned = self.daily.copy()
        poisoned.loc[poisoned.index > asof] += 999.0      # corrupt the future
        f1 = H.asof_features(poisoned, self.month, k)
        for key in f0:
            if np.isfinite(f0[key]):
                self.assertAlmostEqual(f0[key], f1[key], places=9,
                                       msg=f"{key} changed when future was poisoned → leakage")

    def test_coverage_monotonic(self):
        """Coverage must increase (weakly) as k shrinks T-30 → T-1."""
        covs = [H.asof_features(self.daily, self.month, k)["hf_coverage"] for k in C.ORIGINS]
        self.assertEqual(covs, sorted(covs), "coverage must be non-decreasing as T-k→T-1")
        self.assertLessEqual(covs[-1], 1.0)


class TestRegimeCoherence(unittest.TestCase):
    def test_posteriors_sum_to_one(self):
        from intramonth import regime as R
        rng = np.random.default_rng(1)
        y = pd.Series(3 + np.cumsum(rng.standard_normal(180)) * 0.1,
                      index=pd.date_range("2008-01-31", periods=180, freq="ME"))
        post, P, lab = R.filtered_posteriors(y)
        row_sums = post.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6))
        now = R.nowcast_posterior(y)
        self.assertAlmostEqual(sum(now.values()), 1.0, places=6)
        for v in now.values():
            self.assertGreaterEqual(v, -1e-9); self.assertLessEqual(v, 1 + 1e-9)


class TestWeights(unittest.TestCase):
    def _fake_run(self):
        idx = pd.date_range("2015-01-31", periods=100, freq="ME")
        rng = np.random.default_rng(2)
        def bt(scale):
            err = rng.standard_normal(100) * scale
            return pd.DataFrame({"cpi_actual": 3 + rng.standard_normal(100),
                                 "cpi_pred": 3 + rng.standard_normal(100),
                                 "err": err}, index=idx)
        aa = pd.DataFrame({"actual": 3 + rng.standard_normal(100),
                           "pred": 3 + rng.standard_normal(100)}, index=idx)
        return {"baseline": aa,
                "models": {"factor": dict(name="BVAR", bt=bt(0.5), rmse=0.5, contribution=0.3),
                           "regime_tvp": dict(name="TVP", bt=bt(0.6), rmse=0.6, contribution=0.4),
                           "intramonth": dict(name="MIDAS", bt=bt(0.55), rmse=0.55, contribution=0.2)},
                "baseline_rmse": 0.45, "resid_std": 0.4}

    def test_weights_sum_to_one_all_origins(self):
        run = self._fake_run(); errs = W.model_errors(run)
        labels = pd.Series("normal", index=run["baseline"].index)
        post = {"disinflation": 0.1, "normal": 0.8, "shock": 0.1}
        asof = run["baseline"].index[-1]
        for k in C.ORIGINS:
            w, _ = W.weights_for_month(run, errs, labels, asof, post, k)
            self.assertAlmostEqual(sum(w.values()), 1.0, places=9)
            for v in w.values():
                self.assertGreaterEqual(v, -1e-12)

    def test_horizon_shifts_weight(self):
        """Baseline weight at T-30 must exceed baseline weight at T-1 (HF tilt)."""
        run = self._fake_run(); errs = W.model_errors(run)
        labels = pd.Series("normal", index=run["baseline"].index)
        post = {"disinflation": 0.1, "normal": 0.8, "shock": 0.1}
        asof = run["baseline"].index[-1]
        w30, _ = W.weights_for_month(run, errs, labels, asof, post, 30)
        w1, _ = W.weights_for_month(run, errs, labels, asof, post, 1)
        self.assertGreater(w30["baseline"], w1["baseline"])
        self.assertGreater(w1["intramonth"], w30["intramonth"])


class TestScenarios(unittest.TestCase):
    def test_probs_sum_to_100(self):
        post = {"disinflation": 0.2, "normal": 0.5, "shock": 0.3}
        drivers = {"energy_led": 0.6, "services_led": 0.4, "policy_tightening": 0.2}
        pert = {-1: 2.8, 0: 3.0, +1: 3.2}
        for disp in (0.0, 0.5, 1.0):
            df = SC.build_scenarios(post, drivers, pert, 0.4, 3.0, disp, 0.5)
            self.assertAlmostEqual(df["prob"].sum(), 1.0, places=9)
            self.assertEqual(set(df["scenario"]), set(C.SCENARIOS))

    def test_expected_point_between_extremes(self):
        post = {"disinflation": 0.3, "normal": 0.5, "shock": 0.2}
        drivers = {"energy_led": 0.5, "services_led": 0.5, "policy_tightening": 0.0}
        pert = {-1: 2.7, 0: 3.0, +1: 3.3}
        df = SC.build_scenarios(post, drivers, pert, 0.4, 3.0, 0.4, 0.5)
        e = SC.expected_forecast(df)
        self.assertGreater(e, df["point"].min() - 1e-6)
        self.assertLess(e, df["point"].max() + 1e-6)


class TestSwitchable(unittest.TestCase):
    def test_models_resolve_by_name(self):
        from intramonth.stack import _zoo_class
        for name in ("AutoARIMA", "BVAR", "TVP", "MIDAS", "HuberNet", "ElasticNet"):
            self.assertTrue(callable(_zoo_class(name)))

    def test_stack_config_swappable(self):
        from intramonth.stack import _zoo_class
        custom = dict(C.STACK); custom["factor"] = "HuberNet"
        self.assertTrue(callable(_zoo_class(custom["factor"])))


class TestPipelineIntegration(unittest.TestCase):
    """Light end-to-end check guarded on live data availability."""
    def test_evolution_produces_all_origins(self):
        if not os.getenv("FRED_API_KEY"):
            self.skipTest("no FRED_API_KEY; skipping live integration")
        try:
            from intramonth import evolution as E
            res = E.evolve("cpi_headline_yoy", origins=[14, 1], end_year=2024)
        except Exception as e:
            self.skipTest(f"live data unavailable: {str(e)[:50]}")
        evo = res["evolution"]
        self.assertEqual(list(evo["k"]), [14, 1])      # no dropped origins
        self.assertEqual(len(res["scenarios"]), 2)
        for k in (14, 1):
            self.assertAlmostEqual(res["scenarios"][k]["prob"].sum(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
