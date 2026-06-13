"""
Deliverable 6 — Gate 1 / Gate 2 validity + walk-forward look-ahead guards.
"""

import numpy as np
import pandas as pd
import pytest

from rates import synth, gates as G, config as C, mvp as M, signal as SIG, stage1 as S1


class TestStage1MechanicalGuard:
    def _slow_anchor_panel(self):
        """my_nowcast ~ actual (good forecast, volatile CPI), market_implied a
        slow horizon-mismatched anchor -> gap == CPI level -> Stage 1 must be
        flagged INVALID rather than a (mechanical) PASS."""
        rng = np.random.default_rng(0)
        idx = pd.date_range("2014-01-31", periods=120, freq="ME")
        cpi = 2 + 4 * np.sin(np.arange(120) / 10.0) + rng.standard_normal(120) * 0.3
        market = 2.5 + rng.standard_normal(120) * 0.2          # slow, ~constant vs CPI
        myn = cpi + rng.standard_normal(120) * 0.25            # good forecast of CPI
        p = pd.DataFrame(index=idx)
        p["actual_cpi_mom"] = cpi
        p["my_nowcast"] = myn
        p["market_implied_expectation"] = market
        p["my_surprise"] = myn - market
        p["actual_surprise"] = cpi - market
        p["ldi_event"] = 0
        p.attrs["anchor_mode"] = "market_implied"
        return p

    def test_flags_mechanical_identity(self):
        r = S1.stage1_test(self._slow_anchor_panel(), plot=False)
        assert r["verdict"] == "INVALID_MECHANICAL"
        assert r["mechanical_identity"] is True
        # the constant-anchor placebo reproduces the slope -> anchor adds nothing
        assert r["slope_rel_change_vs_placebo"] < 0.15

    def test_passes_on_matched_anchor(self):
        r = S1.stage1_test(synth.make_synthetic_panel(incremental=True, seed=0), plot=False)
        assert r["verdict"] == "PASS"
        assert r["mechanical_identity"] is False
        assert r["b"] > 0


class TestGate2Discrimination:
    def test_passes_on_incremental_signal(self):
        verdicts = []
        for seed in range(5):
            p = synth.make_synthetic_panel(incremental=True, seed=seed)
            verdicts.append(G.gate2_incremental(p, target="uk_2y_gilt_move")["verdict"])
        assert verdicts.count("PASS") >= 4   # robust across seeds

    def test_fails_on_null_signal(self):
        verdicts = []
        for seed in range(5):
            p = synth.make_synthetic_panel(incremental=False, seed=seed)
            verdicts.append(G.gate2_incremental(p, target="uk_2y_gilt_move")["verdict"])
        assert verdicts.count("PASS") == 0   # no false positives

    def test_incremental_beta_positive_and_significant(self):
        p = synth.make_synthetic_panel(incremental=True, seed=0)
        r = G.gate2_incremental(p, target="uk_2y_gilt_move")
        assert r["beta_my_surprise"] > 0
        assert r["t_my_surprise_HAC"] >= C.GATE2_T_THRESHOLD
        assert r["oos_corr"] >= C.GATE2_OOS_CORR_MIN


class TestWalkForwardNoLookahead:
    def test_oos_prediction_excludes_current_point(self, monkeypatch):
        """The walk-forward fit at row i must use strictly rows < i. We verify by
        corrupting the TARGET at the last row and confirming earlier OOS preds
        are unchanged (future cannot reach back)."""
        p = synth.make_synthetic_panel(incremental=True, seed=3)
        bt1, _ = M.walk_forward_mvp(p, target="uk_2y_gilt_move", min_train=24)
        p2 = p.copy()
        p2.iloc[-1, p2.columns.get_loc("uk_2y_gilt_move")] += 1000.0   # corrupt final move
        bt2, _ = M.walk_forward_mvp(p2, target="uk_2y_gilt_move", min_train=24)
        common = bt1.index.intersection(bt2.index)[:-1]   # all but the corrupted last
        assert np.allclose(bt1.loc[common, "pred_move"],
                           bt2.loc[common, "pred_move"], atol=1e-9)

    def test_design_drops_all_nan_controls(self):
        """An entirely-NaN control (e.g. absent market_implied) must not null the sample."""
        p = synth.make_synthetic_panel(incremental=True, seed=0)
        p["market_implied_expectation"] = np.nan        # wipe a control
        p["ucl_nowcast"] = np.nan
        y, X = G._design(p, "uk_2y_gilt_move")
        assert len(y) > 100                              # rows survive
        assert "my_surprise" in X.columns
        assert "market_implied_expectation" not in X.columns


class TestGate2ExcludesLDI:
    def test_ldi_rows_excluded(self):
        p = synth.make_synthetic_panel(incremental=True, seed=0)
        p.iloc[10:13, p.columns.get_loc("ldi_event")] = 1
        p.iloc[10:13, p.columns.get_loc("uk_2y_gilt_move")] = 999  # poison
        y, X = G._design(p, "uk_2y_gilt_move", exclude_ldi=True)
        assert (y < 500).all()   # poisoned LDI rows removed


class TestSignalGating:
    def test_signal_respects_deadband_and_costs(self):
        p = synth.make_synthetic_panel(incremental=True, seed=0)
        bt, _ = M.walk_forward_mvp(p, target="uk_2y_gilt_move")
        trades, m = SIG.backtest_signal(bt, p, deadband_bp=1.0, tcost_bp=0.5)
        assert m["n_traded"] <= m["n_events"]            # deadband can only reduce trades
        # zero-prediction rows must carry zero position
        flat = trades[trades["pred_move"].abs() < 1.0]
        assert (flat["position"].abs() < 1e-9).all()
