"""Per-target model configuration.

HEADLINE mirrors the frozen two_stage/model params byte-for-byte; CORE is filled in by the
core research phases (data -> screen -> race -> calibrate). Keeping all target-specific config
in one place lets two_stage.py and production/model.py be parameterized by target while the
headline path stays byte-identical (regression-locked in tests/test_headline_regression.py).
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict


@dataclass(frozen=True)
class Target:
    name: str                       # target column / factor-registry series name
    label: str                      # human label for printing
    pinned: Tuple[str, ...]         # pinned overlay factors ( () -> AA-only )
    overlay_weights: Dict[str, float]
    lam: float                      # shrinkage haircut on the overlay
    aa_start: int                   # AutoARIMA backtest first year
    start: int                      # walk-forward scoring first year
    end: Optional[int]              # None -> residuals run to the latest actual
    reg: Tuple[str, ...] = ("mpc_rate_change", "ofgem_cap_delta")  # fill-0 regulatory factors


HEADLINE = Target(
    name="cpi_yoy", label="Headline CPI YoY",
    pinned=("oil_brent", "gas_eu", "uk_quarterly_gdp", "imf_all_commodity",
            "global_supply_chain_pressure", "mpc_rate_change", "ofgem_cap_delta",
            "uk_ppi_input", "deep_sea_freight"),
    overlay_weights={"tvp": 0.5, "lgbm": 0.5}, lam=0.5,
    aa_start=2001, start=2015, end=None)

# CORE pinned set from the factor race (docs/core_factor_race.md, 2026-07-24 with LSEG factors):
# exogenous candidates that beat AA-only on a walk-forward (univ_rel_rmse<1), are live-timely, and
# are non-redundant — one per economic channel:
#   uk_pmi_services (0.959) demand/services · uk_ppi_output (0.966) cost-push · uk_infl_swap_1y (0.992)
#   inflation expectations · uk_corp_profits (0.991) profit-led/'greedflation' · gbp_eer (0.996) FX.
# Dropped: uk_pmi_composite (redundant with services PMI), uk_monthly_gdp (data dead, ends 2024-03),
# uk_cpih (circular with core). LSEG factors (PMI/swap/profits) need Workspace running for the live
# nowcast; they degrade to unavailable otherwise. The gate re-validates this set (see docs).
CORE = Target(
    name="uk_core_cpi", label="Core CPI YoY (ex energy, food, alcohol & tobacco)",
    pinned=("uk_pmi_services", "uk_ppi_output", "uk_infl_swap_1y", "uk_corp_profits", "gbp_eer"),
    overlay_weights={"tvp": 0.5, "lgbm": 0.5}, lam=0.5,
    aa_start=2001, start=2015, end=None)
