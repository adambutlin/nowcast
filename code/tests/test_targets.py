"""Target config: HEADLINE must mirror the frozen params; CORE declared (AA-only until race)."""
import targets as T


def test_headline_matches_frozen_params():
    h = T.HEADLINE
    assert h.name == "cpi_yoy"
    assert h.lam == 0.5
    assert h.overlay_weights == {"tvp": 0.5, "lgbm": 0.5}
    assert h.aa_start == 2001 and h.start == 2015 and h.end is None
    assert set(h.pinned) == {
        "oil_brent", "gas_eu", "uk_quarterly_gdp", "imf_all_commodity",
        "global_supply_chain_pressure", "mpc_rate_change", "ofgem_cap_delta",
        "uk_ppi_input", "deep_sea_freight"}
    assert h.reg == ("mpc_rate_change", "ofgem_cap_delta")


def test_core_target_declared():
    c = T.CORE
    assert c.name == "uk_core_cpi"
    assert c.lam == 0.5              # prior; may change after calibration (T6)
    assert isinstance(c.pinned, tuple)   # empty -> AA-only until the race (T5) fills it
    assert c.overlay_weights == {"tvp": 0.5, "lgbm": 0.5}
