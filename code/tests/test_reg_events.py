"""Regulatory-event layer — arithmetic, gating, and the Ofgem parsing.

Offline: every test builds its own cap/contribution series or feeds a scraped
table in as a literal, so nothing here touches the network.
"""
import numpy as np
import pandas as pd
import pytest

from production import reg_events as REG


def _cap_series(pairs):
    idx = pd.to_datetime([d for d, _ in pairs]) + pd.offsets.MonthEnd(0)
    return pd.Series([v for _, v in pairs], index=idx)


class TestCapPctChange:
    def test_keeps_reset_months_only(self):
        cap = _cap_series([("2026-01-31", 1000), ("2026-02-28", 1000), ("2026-03-31", 1000),
                           ("2026-04-30", 1100), ("2026-05-31", 1100), ("2026-06-30", 1200)])
        pct = REG.cap_pct_change(cap)
        assert list(pct.index.month) == [4]              # June is not a reset month
        assert pct.iloc[0] == pytest.approx(10.0)

    def test_ignores_rounding_sized_changes(self):
        cap = _cap_series([("2026-01-31", 1000), ("2026-04-30", 1000.0001)])
        assert len(REG.cap_pct_change(cap)) == 0


class TestPassthrough:
    def _fixture(self, beta=0.02):
        pct = pd.Series([10.0, -20.0, 30.0, 5.0],
                        index=pd.to_datetime(["2024-01-31", "2024-04-30",
                                              "2024-07-31", "2024-10-31"]))
        return pct, pct * beta

    def test_recovers_beta_through_the_origin(self):
        pct, contrib = self._fixture(0.02)
        beta, n = REG.passthrough(pct=pct, contrib=contrib)
        assert beta == pytest.approx(0.02)
        assert n == 4

    def test_before_excludes_later_resets(self):
        pct, contrib = self._fixture(0.02)
        contrib = contrib.copy()
        contrib.iloc[-1] = 99.0                          # would wreck an unrestricted fit
        beta, n = REG.passthrough(before="2024-10-01", pct=pct, contrib=contrib)
        assert beta == pytest.approx(0.02)
        assert n == 3

    def test_no_history_returns_nan(self):
        pct, contrib = self._fixture()
        beta, n = REG.passthrough(before="2000-01-01", pct=pct, contrib=contrib)
        assert np.isnan(beta) and n == 0


class TestRegEvent:
    def _series(self, n=12, beta=0.02):
        months = pd.date_range("2022-01-31", periods=n, freq="QE")
        pct = pd.Series(np.linspace(5, 20, n), index=months)
        return pct, pct * beta

    def test_zero_when_no_reset_this_month(self):
        pct, contrib = self._series()
        step, d = REG.reg_event_step("2026-08-31", pct=pct, contrib=contrib)
        assert step == 0.0 and "no cap reset" in d["reason"]

    def test_zero_until_enough_prior_resets(self):
        pct, contrib = self._series()
        early = pct.index[1]
        step, d = REG.reg_event_step(early, pct=pct, contrib=contrib)
        assert step == 0.0 and "beta not trusted" in d["reason"]

    def test_applies_kappa_times_beta_times_cap_pct(self):
        pct, contrib = self._series(beta=0.02)
        target = pct.index[-1]
        step, d = REG.reg_event_step(target, kappa=0.75, min_prior=4, pct=pct, contrib=contrib)
        assert d["reason"] == "applied"
        assert step == pytest.approx(0.75 * 0.02 * pct.iloc[-1], rel=1e-6)
        assert d["cap_pct"] == pytest.approx(pct.iloc[-1], rel=1e-6)

    def test_step_is_signed_like_the_cap_change(self):
        pct, contrib = self._series()
        pct = pct.copy(); pct.iloc[-1] = -12.0
        step, _ = REG.reg_event_step(pct.index[-1], min_prior=4, pct=pct, contrib=contrib)
        assert step < 0


class TestOfgemParsing:
    RATES = dict(periods=["1 April to 30 June 2026", "1 July to 30 September 2026"],
                 elec=[(24.67, 57.21), (26.11, 57.19)],
                 gas=[(5.74, 29.09), (7.33, 29.04)])

    def test_period_label_maps_to_effective_month_end(self):
        assert REG._period_start("1 July to 30 September 2026") == pd.Timestamp("2026-07-31")
        assert REG._period_start("1 January to 31 March 2025") == pd.Timestamp("2025-01-31")

    def test_implied_pct_matches_published_change(self):
        pct, detail = REG.implied_cap_pct(self.RATES)
        assert pct == pytest.approx(13.5, abs=0.2)       # Ofgem announced "13%"
        assert detail["prev_bill"] == pytest.approx(1641, abs=1)

    def test_implied_pct_is_immune_to_a_consumption_rebasing(self, monkeypatch):
        """A TDCV revision must not show up as a price change."""
        base, _ = REG.implied_cap_pct(self.RATES)
        monkeypatch.setattr(REG, "TDCV_KWH", dict(elec=2000, gas=9000))
        rebased, _ = REG.implied_cap_pct(self.RATES)
        assert rebased != pytest.approx(base, abs=1e-9)  # different basket, different %
        assert abs(rebased - base) < 3.0                 # but not a fabricated step

    def test_refresh_appends_only_missing_quarters(self, tmp_path, monkeypatch):
        csv = tmp_path / "ofgem_cap.csv"
        csv.write_text("date,value\n2026-04-30,1641.18\n")
        monkeypatch.setattr(REG, "CAP_CSV", str(csv))
        added = REG.refresh_cap_csv(path=str(csv), rates=self.RATES)
        assert [m for m, _ in added] == ["2026-07-31"]   # April already present
        assert REG.refresh_cap_csv(path=str(csv), rates=self.RATES) == []
