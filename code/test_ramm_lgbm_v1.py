import os
import unittest
from unittest import mock

import pandas as pd
import numpy as np

import ramm_lgbm_v1 as m


class TestRAMMLGBM(unittest.TestCase):

    def test_download_market_data_selects_close_series(self):
        idx = pd.date_range("2020-01-01", periods=91, freq="D")
        brent_cols = pd.MultiIndex.from_tuples([("Close", "BZ=F")], names=["Price", "Ticker"])
        vix_cols = pd.MultiIndex.from_tuples([("Close", "^VIX")], names=["Price", "Ticker"])

        brent_df = pd.DataFrame(
            [[1.0]] * 31 + [[2.0]] * 29 + [[3.0]] * 31,
            index=idx,
            columns=brent_cols,
        )
        vix_df = pd.DataFrame(
            [[10.0]] * 31 + [[11.0]] * 29 + [[12.0]] * 31,
            index=idx,
            columns=vix_cols,
        )

        with mock.patch.object(m.yf, "download", side_effect=[brent_df, vix_df]):
            brent_m, vix_m = m.download_market_data()

        self.assertEqual(brent_m.name, "brent_ret")
        self.assertEqual(vix_m.name, "vix")
        self.assertEqual(len(brent_m), 3)
        self.assertEqual(len(vix_m), 3)
        self.assertTrue(pd.isna(brent_m.iloc[0]))
        self.assertAlmostEqual(brent_m.iloc[1], 1.0)
        self.assertAlmostEqual(vix_m.iloc[2], 12.0)

    def test_add_regimes_creates_regime_column(self):
        idx = pd.date_range("2020-01-31", periods=24, freq="ME")
        sample = pd.DataFrame(
            {
                "brent_vol_6m": range(24),
                "vix": range(12, 36),
                "be5": range(24, 48),
                "be10": range(36, 60),
            },
            index=idx,
        )

        result, kmeans = m.add_regimes(sample)

        self.assertIn("regime", result.columns)
        self.assertEqual(len(result), 24)
        # regime is int (0/1) — accept int or float kinds
        self.assertIn(result["regime"].dropna().dtype.kind, ("i", "u", "f"))
        # kmeans is None for causal regime implementation
        self.assertIsNone(kmeans)

    def test_fred_monthly_requires_api_key(self):
        env_key = os.environ.pop("FRED_API_KEY", None)
        try:
            with self.assertRaises(OSError):
                m.fred_monthly("CPILFESL")
        finally:
            if env_key is not None:
                os.environ["FRED_API_KEY"] = env_key


if __name__ == "__main__":
    unittest.main()
