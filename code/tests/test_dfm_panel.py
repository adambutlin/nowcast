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
