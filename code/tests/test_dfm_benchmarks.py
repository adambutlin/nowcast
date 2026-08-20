"""Benchmarks read frozen production constants rather than restating them."""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import benchmarks as BM


class TestFrozenConstants:
    def test_reads_lambda_and_weights_from_targets(self):
        from targets import HEADLINE
        assert BM.PROD_LAMBDA == HEADLINE.lam
        assert BM.PROD_WEIGHTS == HEADLINE.overlay_weights

    def test_reads_lgbm_params_from_production(self):
        from production.model import LGB
        assert BM.LGB_PARAMS == LGB

    def test_expected_frozen_values(self):
        """Guards against a silent governance change going unnoticed."""
        assert BM.PROD_LAMBDA == 0.5
        assert BM.PROD_WEIGHTS == {"tvp": 0.5, "lgbm": 0.5}


class TestAR1:
    def test_ar1_backtest_shape(self):
        idx = pd.date_range("2010-01-31", "2018-12-31", freq="ME")
        rng = np.random.default_rng(0)
        y = pd.Series(np.cumsum(rng.normal(size=len(idx))) * 0.1 + 2.0, index=idx)
        bt = BM.ar1_backtest(y, start_year=2015, end_year=2018)
        assert len(bt) == 48
        assert {"actual", "pred"}.issubset(bt.columns)

    def test_ar1_is_causal(self):
        """Poisoning the future must not change any earlier prediction."""
        idx = pd.date_range("2010-01-31", "2018-12-31", freq="ME")
        rng = np.random.default_rng(1)
        y = pd.Series(rng.normal(size=len(idx)), index=idx)
        base = BM.ar1_backtest(y, start_year=2015, end_year=2016)
        y2 = y.copy()
        y2[y2.index > pd.Timestamp("2017-01-31")] = 1e6
        after = BM.ar1_backtest(y2, start_year=2015, end_year=2016)
        pd.testing.assert_frame_equal(base, after)


@pytest.mark.slow
class TestProductionReplication:
    def test_reproduces_the_documented_rel_rmse(self):
        """AA ~0.467 and production ~0.416 on 2015-2024 (README model table)."""
        res = BM.production_backtest()
        aa = np.sqrt(((res["aa_pred"] - res["actual"]) ** 2).mean())
        prod = np.sqrt(((res["prod_pred"] - res["actual"]) ** 2).mean())
        assert 0.40 < aa < 0.53, f"AA RMSE {aa:.3f} outside the documented range"
        assert prod / aa < 0.98, f"production rel-RMSE {prod/aa:.3f} shows no edge"
