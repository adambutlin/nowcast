"""
rates/regime.py — Part B. Regime layer.

Two causal axes per release (using only data <= release eve):
  policy_regime : hiking / cutting / hold / pinned  (already on the panel,
                  derived from Bank Rate trail in sources.mpc_regime)
  infl_regime   : shock / normal  (CPI YoY level or trailing-12m vol)

Combined `regime` label + `regime_trust` in [0,1] = how much the inflation
signal should be trusted in that state (encodes the research finding that the
signal is regime-dependent and near-zero when anchored).
"""

import numpy as np
import pandas as pd

from . import config as C


def classify_regimes(panel):
    """Return panel with added columns: policy_regime, infl_regime, regime,
    regime_trust. Causal: CPI features use the LAST PUBLISHED print (shift(1))."""
    p = panel.copy()
    policy = p["mpc_regime"] if "mpc_regime" in p.columns else pd.Series("unknown", index=p.index)

    # inflation-shock axis from last-published CPI YoY level + trailing vol
    cpi = p["actual_cpi_mom"].shift(1)                 # known at release eve
    cpi_vol = cpi.rolling(12, min_periods=6).std()
    shock = (cpi.abs() >= C.SHOCK_CPI_YOY) | (cpi_vol >= C.SHOCK_CPI_VOL)
    infl = pd.Series(np.where(shock.fillna(False), "shock", "normal"), index=p.index)

    p["policy_regime"] = policy.values
    p["infl_regime"]   = infl.values
    p["regime"]        = [f"{a}|{b}" for a, b in zip(policy.values, infl.values)]
    p["regime_trust"]  = _trust(policy.values, infl.values)
    return p


def _trust(policy, infl):
    out = []
    for pol, inf in zip(policy, infl):
        base = C.REGIME_TRUST.get(pol, 0.3)
        if inf == "shock":
            base = min(1.0, base + C.SHOCK_TRUST_BONUS)
        out.append(base)
    return np.array(out, dtype=float)


def regime_summary(panel):
    """Counts + mean realized 2Y move by regime (diagnostic)."""
    p = classify_regimes(panel) if "regime" not in panel.columns else panel
    tgt = C.TARGET_PRIMARY
    g = p.groupby("regime")
    return pd.DataFrame({
        "n": g.size(),
        "trust": g["regime_trust"].first(),
        "mean_move_bp": g[tgt].mean() if tgt in p.columns else np.nan,
        "std_move_bp": g[tgt].std() if tgt in p.columns else np.nan,
    }).sort_values("trust", ascending=False)
