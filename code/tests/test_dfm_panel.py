"""Panel specs and block map (as-of matrix is tested in Task 6)."""
import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import panel as P
from dfm import config as C


class TestSpecs:
    def test_thirty_two_series_plus_reg_block(self):
        assert len(P.PANEL_SERIES) == 32
        assert len(P.REG_SERIES) == 3

    def test_every_series_has_a_block_and_a_transform(self):
        for name, spec in P.PANEL_SERIES.items():
            assert spec["block"] in P.BLOCKS, f"{name} has unknown block {spec['block']}"
            assert spec["transform"] in {"level", "logret", "yoy", "diff"}
            assert spec["freq"] in {"daily", "monthly", "quarterly"}

    def test_block_sizes_are_three_to_eight(self):
        for block, members in P.block_members().items():
            assert 3 <= len(members) <= 8, f"{block} has {len(members)} members"

    def test_six_blocks_plus_global_equals_k_factors(self):
        assert len(P.BLOCKS) + 1 == C.K_FACTORS

    def test_all_series_includes_target_and_reg(self):
        allser = P.all_series()
        assert "cpi_index" in allser
        assert set(P.REG_SERIES).issubset(allser)
        assert len(allser) == 32 + 3 + 1

    def test_daily_series_are_marked_daily(self):
        from dfm import daily as D
        for name, spec in P.PANEL_SERIES.items():
            if name in D.FETCHERS:
                assert spec["freq"] == "daily", f"{name} has a daily fetcher but freq={spec['freq']}"

    def test_quarterly_series_is_gdp_only(self):
        q = [n for n, s in P.PANEL_SERIES.items() if s["freq"] == "quarterly"]
        assert q == ["uk_quarterly_gdp"]

    def test_no_lseg_series_in_headline_panel(self):
        lseg = {"uk_pmi_services", "uk_pmi_composite", "uk_infl_swap_1y", "uk_corp_profits"}
        assert not (set(P.PANEL_SERIES) & lseg)


import numpy as np
import pandas as pd


@pytest.fixture
def fake_panel(monkeypatch):
    """A 3-series panel with a hand-built calendar, no network."""
    months = pd.date_range("2023-01-31", "2024-06-30", freq="ME")
    monthly = pd.DataFrame({
        "uk_ppi_input": np.linspace(1.0, 2.0, len(months)),
        "imf_all_commodity": np.linspace(-0.5, 0.5, len(months)),
    }, index=months)

    daily_idx = pd.date_range("2022-12-01", "2024-06-30", freq="D")
    brent = pd.Series(np.linspace(50.0, 90.0, len(daily_idx)), index=daily_idx)

    cal = pd.concat([
        P.RC.build_calendar(["uk_ppi_input", "imf_all_commodity", "oil_brent"], months),
    ], ignore_index=True)

    monkeypatch.setattr(P, "_monthly_matrix", lambda names, start=None: monthly)
    monkeypatch.setattr(P.D, "load_daily", lambda name, force=False: brent)
    monkeypatch.setattr(P, "load_calendar_cached", lambda: cal)
    specs = {
        "uk_ppi_input": dict(block="nominal", transform="yoy", freq="monthly"),
        "imf_all_commodity": dict(block="energy", transform="logret", freq="monthly"),
        "oil_brent": dict(block="energy", transform="logret", freq="daily"),
    }
    monkeypatch.setattr(P, "PANEL_SERIES", specs)
    monkeypatch.setattr(P, "REG_SERIES", [])
    return months


class TestAsOf:
    def test_returns_one_row_per_month_and_one_column_per_series(self, fake_panel):
        m = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        assert list(m.columns) == ["uk_ppi_input", "imf_all_commodity", "oil_brent"]
        assert m.index[-1] == pd.Timestamp("2024-06-30")

    def test_unpublished_month_is_missing(self, fake_panel):
        """uk_ppi_input for month T publishes T+16d, so at T it is not yet out."""
        m = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        assert np.isnan(m.loc[pd.Timestamp("2024-06-30"), "uk_ppi_input"])
        assert not np.isnan(m.loc[pd.Timestamp("2024-05-31"), "uk_ppi_input"])

    def test_market_series_is_available_at_month_end(self, fake_panel):
        m = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        assert not np.isnan(m.loc[pd.Timestamp("2024-06-30"), "oil_brent"])

    def test_earlier_origin_has_no_more_information(self, fake_panel):
        late = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        early = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-09"))
        assert early.notna().sum().sum() <= late.notna().sum().sum()

    def test_no_future_leak(self, fake_panel, monkeypatch):
        """Poisoning daily values dated after asof must not change a single cell."""
        base = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-10"))
        orig = P.D.load_daily("oil_brent")

        def poisoned(name, force=False):
            s = orig.copy()
            s[s.index > pd.Timestamp("2024-06-10")] = 1e6
            return s

        monkeypatch.setattr(P.D, "load_daily", poisoned)
        P._CACHE.clear()
        after = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-10"))
        pd.testing.assert_frame_equal(base, after)

    def test_reg_zero_fill_respects_the_publication_mask(self, monkeypatch):
        """An unreleased reg month must stay NaN, not be asserted as "no event".

        The zero-fill used to run after the publication mask and overwrite it, so
        at an intra-month as-of date the current month's event dummies were set to
        0 before the month had ended — claiming knowledge of an event that had not
        been observed. That is docs/postmortem_2026_07.md §3 inverted.
        """
        months = pd.date_range("2023-01-31", "2024-06-30", freq="ME")
        monthly = pd.DataFrame({"ofgem_cap_delta": np.zeros(len(months))}, index=months)
        cal = P.RC.build_calendar(["ofgem_cap_delta"], months)
        monkeypatch.setattr(P, "_monthly_matrix", lambda names, start=None: monthly)
        monkeypatch.setattr(P, "load_calendar_cached", lambda: cal)
        monkeypatch.setattr(P, "PANEL_SERIES", {})
        monkeypatch.setattr(P, "REG_SERIES", ["ofgem_cap_delta"])
        P._CACHE.clear()

        # ofgem_cap_delta publishes at month-end (offset 0), so mid-month the
        # current month is not yet observable.
        mid = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-12"))
        assert np.isnan(mid.loc[pd.Timestamp("2024-06-30"), "ofgem_cap_delta"])
        assert mid.loc[pd.Timestamp("2024-05-31"), "ofgem_cap_delta"] == 0.0

        end = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        assert end.loc[pd.Timestamp("2024-06-30"), "ofgem_cap_delta"] == 0.0

    def test_standardize_is_causal(self, fake_panel):
        """Means and sds come from the training rows only, never the test row."""
        m = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        z, mu, sd = P.standardize(m, through=pd.Timestamp("2024-03-31"))
        train = m.loc[:pd.Timestamp("2024-03-31")]
        assert mu["imf_all_commodity"] == pytest.approx(train["imf_all_commodity"].mean())
        assert sd["imf_all_commodity"] == pytest.approx(train["imf_all_commodity"].std())
