import unittest
from unittest import mock

import pandas as pd
import numpy as np

import factors as F


class TestGasEu(unittest.TestCase):
    def test_gas_eu_in_registry(self):
        self.assertIn("gas_eu", F.REGISTRY)

    def test_gas_eu_fields(self):
        e = F.REGISTRY["gas_eu"]
        self.assertEqual(e["transform"], "logret")
        self.assertEqual(e["pub_lag"], 0)
        self.assertTrue(e["candidate"])
        self.assertIsNotNone(e["fetch"])
        self.assertIsNone(e.get("region"))  # must NOT be tagged US

    def test_gas_hh_is_tagged_us(self):
        self.assertEqual(F.REGISTRY["gas_hh"].get("region"), "US")


import uk_model_zoo as Z


class TestRollingWindow(unittest.TestCase):
    def _make_df(self):
        idx = pd.date_range("2010-01-31", periods=180, freq="ME")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "f1": rng.standard_normal(180),
            "f2": rng.standard_normal(180),
            "cpi_yoy": rng.standard_normal(180) * 0.5 + 3.0,
        }, index=idx)
        return df

    def test_base_model_has_window_none(self):
        self.assertIsNone(Z.BaseModel.WINDOW)

    def test_expanding_window_train_grows_each_year(self):
        df = self._make_df()

        sizes = []

        class DummyModel(Z.BaseModel):
            name = "dummy"
            def _fit_predict_year(self, train, test, factors, target):
                sizes.append(len(train))
                return np.zeros(len(test))

        DummyModel().backtest(df, ["f1", "f2"], "cpi_yoy", start_year=2015)
        self.assertEqual(sizes, sorted(sizes))  # must grow monotonically

    def test_rolling_window_caps_train_size(self):
        df = self._make_df()

        sizes = []

        class RollingDummy(Z.BaseModel):
            name = "rolling_dummy"
            WINDOW = 24
            def _fit_predict_year(self, train, test, factors, target):
                sizes.append(len(train))
                return np.zeros(len(test))

        RollingDummy().backtest(df, ["f1", "f2"], "cpi_yoy", start_year=2015)
        # With WINDOW=24 and min_train=24, train should be at most 24 months
        for sz in sizes:
            self.assertLessEqual(sz, 24)


class TestAllModels(unittest.TestCase):
    def test_all_models_count(self):
        models = Z.all_models()
        self.assertEqual(len(models), 13)  # RegimeEns moved to experimental (2020-21 blowup)

    def test_experimental_models_exist(self):
        exp = Z.experimental_models()
        self.assertEqual(len(exp), 9)
        names = {m.name for m in exp}
        self.assertIn("RAMM-LGBM", names)
        self.assertIn("HMM", names)
        self.assertIn("RegimeEns", names)

    def test_windowed_models(self):
        models = Z.all_models()
        windowed = {m.name: m.WINDOW for m in models if m.WINDOW is not None}
        # No rolling-window models remain in all_models() after LSTAR moved to experimental
        for name, w in windowed.items():
            self.fail(f"Unexpected WINDOW on {name}={w}")

    def test_all_model_names_unique(self):
        models = Z.all_models()
        names = [m.name for m in models]
        self.assertEqual(len(names), len(set(names)), "duplicate model names found")


class TestElasticNet(unittest.TestCase):
    def _make_df(self):
        idx = pd.date_range("2005-01-31", periods=240, freq="ME")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "f1": rng.standard_normal(240),
            "f2": rng.standard_normal(240),
            "cpi_yoy": rng.standard_normal(240) * 0.5 + 3.0,
        }, index=idx)
        return df

    def test_elastic_net_in_all_models(self):
        names = [m.name for m in Z.all_models()]
        self.assertIn("ElasticNet", names)

    def test_elastic_net_backtest_runs(self):
        df = self._make_df()
        m = Z.ElasticNet()
        bt = m.backtest(df, ["f1", "f2"], "cpi_yoy", start_year=2015)
        self.assertGreater(len(bt), 0)
        self.assertIn("actual", bt.columns)
        self.assertIn("pred", bt.columns)


class TestScreenCandidates(unittest.TestCase):
    def _make_df(self):
        idx = pd.date_range("2005-01-31", periods=200, freq="ME")
        rng = np.random.default_rng(7)
        target = rng.standard_normal(200) * 0.5 + 3.0
        df = pd.DataFrame({
            "f1": target * 0.8 + rng.standard_normal(200) * 0.1,
            "f_noise": rng.standard_normal(200),
            "cpi_yoy": target,
        }, index=idx)
        return df

    def test_screen_candidates_returns_list(self):
        df = self._make_df()
        with unittest.mock.patch.dict(F.REGISTRY, {
            "f1": dict(candidate=True, transform="level", pub_lag=0, fetch=None),
            "f_noise": dict(candidate=True, transform="level", pub_lag=0, fetch=None),
        }):
            result = F.screen_candidates(df, "cpi_yoy", threshold=0.001)
        self.assertIsInstance(result, list)

    def test_screen_candidates_keeps_informative_drops_noise(self):
        df = self._make_df()
        with unittest.mock.patch.dict(F.REGISTRY, {
            "f1": dict(candidate=True, transform="level", pub_lag=0, fetch=None),
            "f_noise": dict(candidate=True, transform="level", pub_lag=0, fetch=None),
        }):
            result = F.screen_candidates(df, "cpi_yoy", threshold=0.01)
        self.assertIn("f1", result)
        self.assertNotIn("f_noise", result)


import main as NC


class TestModelGate(unittest.TestCase):
    def _make_bt(self, rmse_scale):
        """Create a fake backtest DataFrame with noise at given scale."""
        idx = pd.date_range("2015-01-31", periods=60, freq="ME")
        rng = np.random.default_rng(0)
        actual = rng.standard_normal(60) + 3.0
        noise = rng.standard_normal(60) * rmse_scale
        pred = actual + noise - noise.mean()
        return pd.DataFrame({"actual": actual, "pred": pred}, index=idx)

    def test_greedy_subset_excludes_models_above_ar1(self):
        bt_good = self._make_bt(0.1)
        bt_bad  = self._make_bt(2.0)
        ar1_rmse = 0.5

        err_good = (bt_good["actual"] - bt_good["pred"]).rename("good")
        err_bad  = (bt_bad["actual"]  - bt_bad["pred"]).rename("bad")
        corr_mat = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]],
                                 index=["good", "bad"], columns=["good", "bad"])
        bt_dict = {"good": bt_good, "bad": bt_bad}

        result = NC.greedy_uncorrelated_subset(corr_mat, bt_dict,
                                               rho_threshold=0.5, ar1_rmse=ar1_rmse)
        self.assertIn("good", result)
        self.assertNotIn("bad", result)

    def test_greedy_subset_returns_empty_without_ar1_rmse(self):
        bt = self._make_bt(0.1)
        corr_mat = pd.DataFrame([[1.0]], index=["m"], columns=["m"])
        bt_dict = {"m": bt}
        result = NC.greedy_uncorrelated_subset(corr_mat, bt_dict,
                                               rho_threshold=0.5, ar1_rmse=None)
        self.assertEqual(result, [])


class TestRegulatoryEventFactors(unittest.TestCase):
    def test_mpc_rate_change_in_registry(self):
        self.assertIn("mpc_rate_change", F.REGISTRY)

    def test_mpc_rate_change_fields(self):
        e = F.REGISTRY["mpc_rate_change"]
        self.assertEqual(e["pub_lag"], 0)
        self.assertFalse(e["candidate"])
        self.assertEqual(e["transform"], "level")
        self.assertIsNotNone(e.get("fetch"))

    def test_mpc_vote_split_in_registry(self):
        self.assertIn("mpc_vote_split", F.REGISTRY)

    def test_mpc_vote_split_fields(self):
        e = F.REGISTRY["mpc_vote_split"]
        self.assertEqual(e["pub_lag"], 0)
        self.assertFalse(e["candidate"])
        self.assertEqual(e["transform"], "level")
        self.assertEqual(e["csv"], "mpc_vote_split.csv")

    def test_ofgem_cap_delta_in_registry(self):
        self.assertIn("ofgem_cap_delta", F.REGISTRY)

    def test_ofgem_cap_delta_fields(self):
        e = F.REGISTRY["ofgem_cap_delta"]
        self.assertEqual(e["pub_lag"], 0)
        self.assertFalse(e["candidate"])
        self.assertEqual(e["transform"], "diff")
        self.assertIsNone(e["fetch"])
        self.assertEqual(e["csv"], "ofgem_cap.csv")

    def test_budget_event_in_registry(self):
        self.assertIn("budget_event", F.REGISTRY)

    def test_budget_event_fields(self):
        e = F.REGISTRY["budget_event"]
        self.assertEqual(e["pub_lag"], 0)
        self.assertFalse(e["candidate"])
        self.assertEqual(e["transform"], "level")
        self.assertIsNone(e["fetch"])
        self.assertEqual(e["csv"], "budget_event.csv")

    def test_mpc_vote_split_loads_and_forward_fills(self):
        """_mpc_vote_split() returns a ffilled monthly series with values in [-9, 9]."""
        import tempfile, os
        csv_content = "date,hike_votes,hold_votes,cut_votes\n2022-02-03,5,4,0\n2022-03-17,8,1,0\n2022-05-05,6,3,0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            tmp = f.name
        try:
            orig = F.DATA_DIR
            import unittest.mock as mock
            with mock.patch.object(F, "DATA_DIR", os.path.dirname(tmp)):
                dest = os.path.join(os.path.dirname(tmp), "mpc_vote_split.csv")
                os.rename(tmp, dest)
                s = F._mpc_vote_split()
                self.assertEqual(s.loc["2022-02-28"], 5)
                self.assertEqual(s.loc["2022-03-31"], 8)
                self.assertEqual(s.loc["2022-04-30"], 8)
                self.assertTrue((s >= -9).all() and (s <= 9).all())
        finally:
            try:
                os.remove(dest)
            except Exception:
                pass

    def test_ofgem_cap_delta_spikes_oct_2022(self):
        """ofgem_cap_delta series shows large positive spike in Oct 2022."""
        s, status = F.load_factor("ofgem_cap_delta")
        if status == "unavailable":
            self.skipTest("data/ofgem_cap.csv not yet created")
        oct_2022 = s.loc["2022-10-31"] if "2022-10-31" in s.index else None
        self.assertIsNotNone(oct_2022)
        self.assertGreater(oct_2022, 500)

    def test_budget_event_sep_oct_2022(self):
        """budget_event = 1 for Sep 2022 (mini-budget) and Oct 2022 (reversal)."""
        s, status = F.load_factor("budget_event")
        if status == "unavailable":
            self.skipTest("data/budget_event.csv not yet created")
        self.assertEqual(s.loc["2022-09-30"], 1)
        self.assertEqual(s.loc["2022-10-31"], 1)


if __name__ == "__main__":
    unittest.main()
