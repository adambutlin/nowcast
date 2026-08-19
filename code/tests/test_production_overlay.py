"""
Production overlays — training window and prediction month.

The production model fits its overlays on the AutoARIMA-residual history, then
predicts the first unreleased month `nd`.

Purge/embargo is a BACKTEST device: because the residual target (cpi_yoy) is a
12-month difference, training rows within 12 months of a *test* month share its
YoY window. `_lgbm_resid` therefore takes purge_horizon/embargo arguments, which
a walk-forward passes and the live path leaves off — every residual up to the
last released CPI month is genuinely observable in real time (governance
decision 2026-08-19, docs/final_model.md §11).

Spec: docs/superpowers/specs/2026-06-22-purged-embargo-backtest-design.md
"""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "new_factors"))

from production import model as PM
import two_stage as TS
import uk_model_zoo as Z
import validation as V


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


def _purged(df, live, resid, nd):
    """_lgbm_resid with the backtest purge/embargo switched on."""
    return PM._lgbm_resid(df, live, resid, nd, TS.PURGE_HORIZON, TS.EMBARGO)


class TestProductionLGBMPurge:
    """purge_horizon/embargo, when passed, must drop exactly the overlapping window."""

    def test_poisoning_purged_residual_does_not_change_overlay(self):
        df, live, resid, nd = _frame()
        base = _purged(df, live, resid, nd)
        poisoned = resid.copy()
        poisoned.loc[_month(resid.index, nd, 6)] += 1000.0   # inside the 12+1 window
        after = _purged(df, live, poisoned, nd)
        assert np.isclose(base, after, atol=1e-9)

    def test_poisoning_unpurged_residual_does_change_overlay(self):
        """Sanity: a residual well outside the purge window still influences the fit."""
        df, live, resid, nd = _frame()
        base = _purged(df, live, resid, nd)
        poisoned = resid.copy()
        poisoned.loc[_month(resid.index, nd, 30)] += 1000.0
        after = _purged(df, live, poisoned, nd)
        assert not np.isclose(base, after, atol=1e-9)

    def test_boundary_is_exactly_horizon_plus_embargo(self):
        """gap = PURGE_HORIZON + EMBARGO: month nd-gap is dropped, nd-(gap+1) kept."""
        df, live, resid, nd = _frame()
        base = _purged(df, live, resid, nd)
        gap = TS.PURGE_HORIZON + TS.EMBARGO
        dropped = resid.copy()
        dropped.loc[_month(resid.index, nd, gap)] += 1000.0          # last purged month
        assert np.isclose(base, _purged(df, live, dropped, nd), atol=1e-9)
        kept = resid.copy()
        kept.loc[_month(resid.index, nd, gap + 1)] += 1000.0         # first kept month
        assert not np.isclose(base, _purged(df, live, kept, nd), atol=1e-9)


class TestLiveOverlayUsesRecentResiduals:
    """The live path must NOT purge: recent residuals are observable in real time."""

    def test_lgbm_default_trains_on_the_month_before_nd(self):
        df, live, resid, nd = _frame()
        base = PM._lgbm_resid(df, live, resid, nd)
        poisoned = resid.copy()
        poisoned.loc[_month(resid.index, nd, 1)] += 1000.0    # last released month
        assert not np.isclose(base, PM._lgbm_resid(df, live, poisoned, nd), atol=1e-9)

    def test_tvp_default_trains_on_the_month_before_nd(self):
        df, live, resid, nd = _frame()
        base = PM._tvp_resid(df, live, resid, nd)
        poisoned = resid.copy()
        poisoned.loc[_month(resid.index, nd, 1)] += 10.0
        assert not np.isclose(base, PM._tvp_resid(df, live, poisoned, nd), atol=1e-9)


class TestProductionTVPMonth:
    """The TVP leg must forecast `nd`, not the month its embargoed target implies.

    production.model embargoes the residual column before handing it to the TVP,
    so the column's last non-NaN month is PURGE+EMBARGO months before `nd`.
    BaseModel._nowcast_row infers the prediction month from that column, which
    would silently point the TVP at a 13-month-stale month (and its stale factor
    row). _tvp_resid must pin the prediction month to `nd` instead.
    """

    def _embargoed(self, resid, nd):
        return V.embargo_series(resid, nd, TS.PURGE_HORIZON, TS.EMBARGO)

    def test_overlay_responds_to_target_month_factors(self):
        df, live, resid, nd = _frame()
        emb = self._embargoed(resid, nd)
        base = PM._tvp_resid(df, live, emb, nd)
        moved = df.copy()
        moved.loc[nd, live] = moved.loc[nd, live] + 5.0
        assert not np.isclose(base, PM._tvp_resid(moved, live, emb, nd), atol=1e-9)

    def test_overlay_ignores_the_stale_inferred_month(self):
        """nd-(PURGE+EMBARGO) is neither the prediction row nor a training row."""
        df, live, resid, nd = _frame()
        emb = self._embargoed(resid, nd)
        gap = TS.PURGE_HORIZON + TS.EMBARGO
        base = PM._tvp_resid(df, live, emb, nd)
        stale = df.copy()
        stale.loc[_month(df.index, nd, gap), live] = stale.loc[_month(df.index, nd, gap), live] + 5.0
        assert np.isclose(base, PM._tvp_resid(stale, live, emb, nd), atol=1e-9)

    def test_zoo_nowcast_date_default_is_unchanged(self):
        """Omitting nowcast_date keeps the historic inference from the target column."""
        df, live, resid, nd = _frame()
        d = df.copy(); d["resid"] = self._embargoed(resid, nd)
        _, inferred = Z.TVP().nowcast(d, live, "resid")
        gap = TS.PURGE_HORIZON + TS.EMBARGO
        assert pd.Timestamp(inferred) == _month(df.index, nd, gap)
        _, pinned = Z.TVP().nowcast(d, live, "resid", nowcast_date=nd)
        assert pd.Timestamp(pinned) == nd


class TestOverlayDegradesInsteadOfReturningNaN:
    """A dead leg must not poison the forecast.

    BaseModel._nowcast_row blocks the TVP whenever a PINNED factor is past its
    ffill budget — which happens routinely to the quarterly uk_quarterly_gdp late
    in a quarter, and produced a NaN August-2026 headline forecast before the
    overlay learned to renormalise over its surviving members.
    """

    def _setup(self, monkeypatch, tvp, lgbm, step=0.0):
        from targets import Target
        import production.model as PM
        df, live, resid, nd = _frame()
        df = df.copy()
        df["y"] = np.nan                                   # target column, all unreleased
        target = Target(name="y", label="stub", pinned=tuple(live),
                        overlay_weights={"tvp": 0.5, "lgbm": 0.5}, lam=0.5,
                        aa_start=2001, start=2015, end=None, reg=(),
                        reg_event_layer=True)

        class _AA:
            def nowcast(self, *a, **k):
                return 2.5, nd

        monkeypatch.setattr(PM.Z, "AutoARIMA", _AA)
        monkeypatch.setattr(PM, "_residual_history", lambda *a, **k: resid)
        monkeypatch.setattr(PM, "_tvp_resid", lambda *a, **k: tvp)
        monkeypatch.setattr(PM, "_lgbm_resid", lambda *a, **k: lgbm)
        monkeypatch.setattr(PM.REG, "reg_event_step",
                            lambda *a, **k: (step, {"reason": "stub"}))
        return PM.nowcast(df, live, target)

    def test_surviving_leg_takes_the_full_overlay_weight(self, monkeypatch):
        nc = self._setup(monkeypatch, tvp=float("nan"), lgbm=0.4)
        assert nc["dead_legs"] == ["tvp"]
        assert nc["overlay"] == pytest.approx(0.4)          # renormalised, not halved
        assert nc["lgbm_contribution"] == pytest.approx(0.2)
        assert nc["tvp_contribution"] == 0.0
        assert nc["forecast"] == pytest.approx(2.5 + 0.5 * 0.4, abs=1e-3)

    def test_both_legs_alive_is_unchanged(self, monkeypatch):
        nc = self._setup(monkeypatch, tvp=0.2, lgbm=0.4)
        assert nc["dead_legs"] == []
        assert nc["overlay"] == pytest.approx(0.3)
        assert nc["forecast"] == pytest.approx(2.5 + 0.5 * 0.3, abs=1e-3)

    def test_both_legs_dead_falls_back_to_the_anchor(self, monkeypatch):
        nc = self._setup(monkeypatch, tvp=float("nan"), lgbm=float("nan"), step=0.25)
        assert nc["dead_legs"] == ["lgbm", "tvp"]
        assert nc["overlay"] == 0.0
        assert nc["forecast"] == pytest.approx(2.75, abs=1e-3)   # AA + RegEvent only
        assert np.isfinite(nc["forecast"])
