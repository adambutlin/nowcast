"""
Deliverable 6 — event-panel integrity, timing alignment, look-ahead bias.
Tests fail if future information can enter the panel.
"""

import numpy as np
import pandas as pd
import pytest

from rates import sources as S, event_panel as EP, config as C


def _daily_rates_fixture(start="2014-12-01", n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    lvl = 2.0 + np.cumsum(rng.standard_normal(n) * 0.01)
    return pd.DataFrame({"ois_1y": lvl, "gilt_2y": lvl + 0.2,
                         "gilt_5y": lvl + 0.4, "gilt_10y": lvl + 0.6}, index=idx)


class TestRateMoves:
    def test_move_uses_only_release_and_prior_day(self):
        rd = _daily_rates_fixture()
        release = pd.Timestamp("2015-03-18")
        mv = S.rate_moves(rd, [release])
        idx = rd.index
        t1 = idx[idx >= release][0]
        t0 = idx[idx < release][-1]
        expect = (rd.at[t1, "gilt_2y"] - rd.at[t0, "gilt_2y"]) * 100
        assert mv.loc[release, "uk_2y_gilt_move"] == pytest.approx(expect, abs=1e-9)

    def test_future_rates_do_not_change_past_move(self):
        """Appending rate data AFTER a release must not change that release's move."""
        rd = _daily_rates_fixture()
        release = pd.Timestamp("2015-03-18")
        m1 = S.rate_moves(rd, [release]).loc[release, "uk_2y_gilt_move"]
        future = rd.copy()
        future.loc[pd.Timestamp("2015-03-19"):] += 5.0   # huge future shock
        m2 = S.rate_moves(future, [release]).loc[release, "uk_2y_gilt_move"]
        assert m1 == pytest.approx(m2, abs=1e-9), "future rate leaked into past move"

    def test_move_sign_convention_bp(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2015-01-01", periods=10)
        lvl = pd.Series(np.linspace(2.0, 2.05, 10), index=idx)  # rising 5bp total
        rd = pd.DataFrame({"gilt_2y": lvl})
        rel = idx[5]
        mv = S.rate_moves(rd, [rel]).loc[rel, "uk_2y_gilt_move"]
        assert mv > 0   # yields up -> positive (hawkish) move, in bp
        assert abs(mv) < 5   # one-day move, in bp not pp


class TestPanelIntegrity:
    def _patch_sources(self, monkeypatch):
        rng = np.random.default_rng(1)
        months = pd.date_range("2015-01-31", periods=60, freq="ME")
        pred = pd.Series(2 + rng.standard_normal(60) * 0.3, index=months, name="my_nowcast")
        act  = pd.Series(2 + rng.standard_normal(60) * 0.3, index=months, name="actual_cpi")
        monkeypatch.setattr(S, "my_nowcast", lambda model=None, fallback=None: (pred, act))
        monkeypatch.setattr(S, "ucl_nowcast", lambda: pd.Series(dtype=float))
        monkeypatch.setattr(S, "economist_consensus", lambda: pd.Series(dtype=float))
        monkeypatch.setattr(S, "market_implied", lambda: pd.Series(dtype=float))
        monkeypatch.setattr(S, "daily_rates", lambda: _daily_rates_fixture(n=2000))
        monkeypatch.setattr(S, "mpc_dates", lambda: list(months))
        return months

    def test_schema_complete(self, monkeypatch):
        self._patch_sources(monkeypatch)
        p = EP.build_event_panel(save=False)
        for col in EP.SCHEMA:
            assert col in p.columns, f"missing schema column {col}"

    def test_release_after_reference_month(self, monkeypatch):
        months = self._patch_sources(monkeypatch)
        p = EP.build_event_panel(save=False)
        # release_date must be AFTER the reference month-end (no same/earlier dating)
        assert (p["release_date"] > p.index).all()

    def test_my_nowcast_is_causal_source(self, monkeypatch):
        """my_nowcast in the panel must equal the (causal) backtest pred, never actual."""
        self._patch_sources(monkeypatch)
        p = EP.build_event_panel(save=False)
        # my_nowcast and actual differ (would be identical only if actual leaked in)
        assert not np.allclose(p["my_nowcast"].dropna(),
                               p["actual_cpi_mom"].reindex(p["my_nowcast"].dropna().index))

    def test_naive_rw_anchor_is_lagged(self, monkeypatch):
        """naive_rw baseline must be the PREVIOUS month's actual (shift +1), never current."""
        self._patch_sources(monkeypatch)
        p = EP.build_event_panel(save=False)
        assert p.attrs["anchor_mode"] == "naive_rw"
        expect = p["actual_cpi_mom"].shift(1)
        assert np.allclose(p["baseline_expectation"].dropna(),
                           expect.reindex(p["baseline_expectation"].dropna().index), atol=1e-9)
