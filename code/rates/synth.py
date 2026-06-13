"""
rates/synth.py — synthetic event panel with a KNOWN ground truth, so Gate 2 is
runnable and testable without the (network-gated) UCL / daily-rates data.

DGP (front-end move is driven by the part of the realized surprise the market
did NOT already price via UCL):

  u  ~ public component  (UCL sees it)
  m  ~ my-unique component
  actual_surprise = u + incr*m + eps_a
  ucl_surprise    = u + eps_u
  my_surprise     = rho*u + incr*m + eps_my
  move(2y)        = beta * (actual_surprise - ucl_surprise) + market_noise
                  = beta * (incr*m + eps_a - eps_u) + market_noise

When incremental=True, my_surprise carries m -> predicts the move after
controlling for ucl_surprise (Gate 2 PASSES). When False, my_surprise carries
only u+noise -> orthogonal to the move given ucl_surprise (Gate 2 FAILS).
"""

import numpy as np
import pandas as pd


def make_synthetic_panel(n=140, incremental=True, beta=25.0, rho=0.7,
                         seed=0, start="2013-01-31"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="ME")
    incr = 1.0 if incremental else 0.0

    u   = rng.standard_normal(n) * 0.30          # public surprise component (pp)
    m   = rng.standard_normal(n) * 0.20          # my unique component (pp)
    eps_a  = rng.standard_normal(n) * 0.05
    eps_u  = rng.standard_normal(n) * 0.05
    eps_my = rng.standard_normal(n) * 0.05
    mkt_noise = rng.standard_normal(n) * 4.0     # bp

    baseline = 2.0 + rng.standard_normal(n) * 0.5
    actual_surprise = u + incr * m + eps_a
    ucl_surprise    = u + eps_u
    my_surprise     = rho * u + incr * m + eps_my

    actual = baseline + actual_surprise
    ucl    = baseline + ucl_surprise
    myn    = baseline + my_surprise
    move2y = beta * (actual_surprise - ucl_surprise) + mkt_noise

    p = pd.DataFrame(index=idx)
    p.index.name = "ref_month"
    p["release_date"]               = idx + pd.offsets.Day(18)
    p["actual_cpi_mom"]             = actual
    p["my_nowcast"]                 = myn
    p["ucl_nowcast"]                = ucl
    p["economist_consensus"]        = baseline       # anchor = consensus
    p["market_implied_expectation"] = baseline
    p["baseline_expectation"]       = baseline
    p["my_surprise"]     = p["my_nowcast"]  - baseline
    p["ucl_surprise"]    = p["ucl_nowcast"] - baseline
    p["market_surprise"] = p["market_implied_expectation"] - baseline
    p["actual_surprise"] = p["actual_cpi_mom"] - baseline
    p["boe_1y_ois_move"]  = move2y * 0.9 + rng.standard_normal(n) * 2
    p["uk_2y_gilt_move"]  = move2y
    p["uk_5y_gilt_move"]  = move2y * 0.7 + rng.standard_normal(n) * 3
    p["uk_10y_gilt_move"] = move2y * 0.4 + rng.standard_normal(n) * 5
    regimes = rng.choice(["pinned", "hiking", "hold", "cutting"], size=n)
    p["mpc_regime"]   = regimes
    p["days_to_mpc"]  = rng.integers(1, 45, size=n)
    p["budget_event"] = 0
    p["ldi_event"]    = 0
    p.attrs["anchor_mode"] = "consensus"
    p.attrs["truth_beta"]  = beta if incremental else 0.0
    return p
