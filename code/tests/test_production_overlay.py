"""
Production LGBM overlay — purged + embargoed training.

The frozen production model fits its LGBM overlay on the AutoARIMA-residual
history, then predicts the first unreleased month `nd`. Because the residual
target (cpi_yoy) is a 12-month difference, residuals within 12 months of `nd`
share its YoY window. production.model._lgbm_resid purges that label-horizon
(+1m embargo) so those autocorrelated residuals do not drive the overlay.

Spec: docs/superpowers/specs/2026-06-22-purged-embargo-backtest-design.md
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "new_factors"))

from production import model as PM
import two_stage as TS


def _monthly(start, n):
    return pd.date_range(start, periods=n, freq="ME")


def _frame():
    idx = _monthly("2010-01-31", 174)          # 2010-01 .. 2024-06
    rng = np.random.default_rng(0)
    live = ["f1", "f2"]
    df = pd.DataFrame({c: rng.normal(0, 1, len(idx)) for c in live}, index=idx)
    resid = pd.Series(rng.normal(0, 0.3, len(idx)), index=idx, name="resid")
    nd = idx[-1]
    resid.loc[nd] = np.nan                       # nd is unreleased
    return df, live, resid, nd


def _month(idx, nd, k):
    """The index entry exactly k calendar months before nd (day-of-month safe)."""
    hits = idx[idx.to_period("M") == nd.to_period("M") - k]
    assert len(hits) == 1
    return hits[0]


class TestProductionLGBMPurge:
    def test_poisoning_purged_residual_does_not_change_overlay(self):
        df, live, resid, nd = _frame()
        base = PM._lgbm_resid(df, live, resid, nd)
        poisoned = resid.copy()
        poisoned.loc[_month(resid.index, nd, 6)] += 1000.0   # inside the 12+1 window
        after = PM._lgbm_resid(df, live, poisoned, nd)
        assert np.isclose(base, after, atol=1e-9)

    def test_poisoning_unpurged_residual_does_change_overlay(self):
        """Sanity: a residual well outside the purge window still influences the fit."""
        df, live, resid, nd = _frame()
        base = PM._lgbm_resid(df, live, resid, nd)
        poisoned = resid.copy()
        poisoned.loc[_month(resid.index, nd, 30)] += 1000.0
        after = PM._lgbm_resid(df, live, poisoned, nd)
        assert not np.isclose(base, after, atol=1e-9)

    def test_boundary_is_exactly_horizon_plus_embargo(self):
        """gap = PURGE_HORIZON + EMBARGO: month nd-gap is dropped, nd-(gap+1) kept."""
        df, live, resid, nd = _frame()
        base = PM._lgbm_resid(df, live, resid, nd)
        gap = TS.PURGE_HORIZON + TS.EMBARGO
        dropped = resid.copy()
        dropped.loc[_month(resid.index, nd, gap)] += 1000.0          # last purged month
        assert np.isclose(base, PM._lgbm_resid(df, live, dropped, nd), atol=1e-9)
        kept = resid.copy()
        kept.loc[_month(resid.index, nd, gap + 1)] += 1000.0         # first kept month
        assert not np.isclose(base, PM._lgbm_resid(df, live, kept, nd), atol=1e-9)
