"""
Production-layer tests: regime classification, confidence suppression, risk
controls, model switching, no-lookahead in sizing, attribution integrity.
"""

import numpy as np
import pandas as pd
import pytest

from rates import (synth, regime as R, prod_signal as PS, risk as RK,
                   production as P, config as C)


def _panel():
    return synth.make_synthetic_panel(incremental=True, seed=0)


class TestRegime:
    def test_adds_columns_and_trust_range(self):
        p = R.classify_regimes(_panel())
        for c in ["policy_regime", "infl_regime", "regime", "regime_trust"]:
            assert c in p.columns
        assert p["regime_trust"].between(0, 1).all()

    def test_shock_axis_from_high_cpi(self):
        p = _panel().copy()
        p["actual_cpi_mom"] = 9.0          # well above SHOCK_CPI_YOY
        out = R.classify_regimes(p)
        # row 0 is NaN by design (shift(1), causal); everything after = shock
        assert (out["infl_regime"].iloc[1:] == "shock").all()

    def test_regime_is_causal(self):
        """A future CPI spike must not change earlier regime labels (shift(1))."""
        p = _panel()
        r1 = R.classify_regimes(p)["regime_trust"].values
        p2 = p.copy(); p2.iloc[-1, p2.columns.get_loc("actual_cpi_mom")] = 50.0
        r2 = R.classify_regimes(p2)["regime_trust"].values
        assert np.allclose(r1[:-2], r2[:-2])   # earlier rows unchanged


class TestConfidence:
    def test_confidence_in_unit_interval(self):
        p = PS.build_signals(_panel())
        assert p["confidence"].dropna().between(0, 1).all()

    def test_low_trust_lowers_confidence(self):
        p = _panel().copy()
        p["mpc_regime"] = "pinned"          # trust 0.2
        lo = PS.build_signals(p)["confidence"].mean()
        p["mpc_regime"] = "hiking"          # trust 0.8
        hi = PS.build_signals(p)["confidence"].mean()
        assert hi > lo


class TestRisk:
    def test_ldi_and_budget_excluded(self):
        p = PS.build_signals(_panel())
        p.iloc[5, p.columns.get_loc("ldi_event")] = 1
        p.iloc[6, p.columns.get_loc("budget_event")] = 1
        rk = RK.apply_risk(p)
        assert not rk["tradeable"].iloc[5] and rk["reason"].iloc[5] == "ldi"
        assert not rk["tradeable"].iloc[6] and rk["reason"].iloc[6] == "budget"

    def test_low_confidence_suppressed(self):
        p = PS.build_signals(_panel()).copy()
        p["confidence"] = 0.0
        rk = RK.apply_risk(p)
        assert (~rk["tradeable"]).all()
        assert (rk["size_mult"] == 0).all()


class TestPositions:
    def _mvp_like(self, p):
        # fabricate an mvp_bt aligned to the panel (pred ~ gap, realized = 2Y move)
        return pd.DataFrame({"pred_move": p["my_surprise"] * 20.0,
                             "realized_move": p["uk_2y_gilt_move"]}, index=p.index).dropna()

    def test_zero_position_when_not_tradeable(self):
        p = PS.build_signals(_panel())
        bt = self._mvp_like(p)
        pos = P.build_positions(p, bt)
        flat = pos[pos["size_mult"] == 0.0]
        assert (flat["position"].abs() < 1e-9).all()

    def test_sizing_is_causal(self):
        """Corrupting the final realized move must not change earlier positions."""
        p = PS.build_signals(_panel())
        bt = self._mvp_like(p)
        pos1 = P.build_positions(p, bt)
        bt2 = bt.copy(); bt2.iloc[-1, bt2.columns.get_loc("realized_move")] += 1000.0
        pos2 = P.build_positions(p, bt2)
        common = pos1.index.intersection(pos2.index)[:-1]
        assert np.allclose(pos1.loc[common, "position"], pos2.loc[common, "position"])

    def test_attribution_sums_to_total(self):
        p = PS.build_signals(_panel())
        pos = P.build_positions(p, self._mvp_like(p))
        attr = P.attribution(pos)
        assert attr["by_regime"]["total_pnl_bp"].sum() == pytest.approx(
            pos["pnl_bp"].sum(), abs=1e-6)


class TestModelSwitch:
    def test_pipeline_runs_for_each_model_column(self):
        """Switching the my_nowcast series reruns the whole position machinery."""
        base = _panel()
        outs = {}
        for shift in [0.0, 0.5]:          # two 'models' = two nowcast series
            p = base.copy()
            p["my_nowcast"] = base["my_nowcast"] + shift
            p["my_surprise"] = p["my_nowcast"] - p["baseline_expectation"]
            p = PS.build_signals(p)
            bt = pd.DataFrame({"pred_move": p["my_surprise"] * 20.0,
                               "realized_move": p["uk_2y_gilt_move"]}, index=p.index).dropna()
            outs[shift] = P.backtest_metrics(P.build_positions(p, bt))
        assert set(outs) == {0.0, 0.5}
        assert all("sharpe_ann" in v for v in outs.values())
