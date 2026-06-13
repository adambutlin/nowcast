"""
rates/production.py — Parts D/E/G/H. Repricing -> position -> backtest -> attribution.

Repricing reuses mvp.walk_forward_mvp (walk-forward, regime + days_to_mpc
conditioning via gates._design). Positions layer confidence + risk on top so
exposure shrinks automatically when the signal is untrusted.
"""

import numpy as np
import pandas as pd

from . import config as C
from . import mvp as M
from . import prod_signal as PS
from . import risk as RK
from . import regime as R


# ── Part D: regime-conditioned, walk-forward expected move ────────────────────
def forecast_repricing(panel, target=None):
    target = target or C.TARGET_PRIMARY
    bt, metrics = M.walk_forward_mvp(panel, target=target,
                                     exclude_ldi=C.EXCLUDE_LDI)
    return bt, metrics


# ── Part E: confidence- and risk-weighted position ───────────────────────────
def build_positions(panel, mvp_bt, target=None):
    target = target or C.TARGET_PRIMARY
    p = PS.build_signals(panel)                       # confidence, regime, signals
    rk = RK.apply_risk(p, target=target)

    bt = mvp_bt.join(p[["confidence", "regime", "policy_regime", "regime_trust"]], how="left")
    bt = bt.join(rk[["tradeable", "size_mult", "reason"]], how="left")

    # causal trailing vol of the realized move for vol-targeting
    vol = bt["realized_move"].rolling(C.VOL_WINDOW, min_periods=4).std().shift(1)
    vol = vol.fillna(bt["realized_move"].expanding().std().shift(1)).bfill().clip(lower=1.0)

    base = (bt["pred_move"] / (C.VOL_K * vol)).clip(-C.POS_CAP, C.POS_CAP)
    base = base.where(bt["pred_move"].abs() >= C.DEADBAND_BP, 0.0)     # deadband
    bt["position"] = (base * bt["size_mult"].fillna(0.0))             # confidence x risk
    dpos = bt["position"].diff().abs().fillna(bt["position"].abs())
    bt["pnl_bp"] = bt["position"] * bt["realized_move"] - C.TCOST_BP * dpos
    bt["turnover"] = dpos
    return bt


# ── Part H: metrics ──────────────────────────────────────────────────────────
def backtest_metrics(bt):
    pnl = bt["pnl_bp"]
    traded = bt[bt["position"].abs() > 1e-9]
    ann = np.sqrt(12.0)
    mu, sd = pnl.mean(), pnl.std()
    cum = pnl.cumsum()
    dd = float((cum - cum.cummax()).min())
    return dict(
        n_events=int(len(bt)), n_traded=int(len(traded)),
        total_pnl_bp=float(pnl.sum()),
        sharpe_ann=float(mu / sd * ann) if sd and np.isfinite(sd) else np.nan,
        hit_rate=float((traded["pnl_bp"] > 0).mean()) if len(traded) else np.nan,
        avg_turnover=float(bt["turnover"].mean()),
        max_dd_bp=dd,
    )


# ── Part G: attribution ──────────────────────────────────────────────────────
def attribution(bt):
    """PnL decomposition by regime (and policy regime)."""
    def _agg(by):
        g = bt.groupby(by)
        out = pd.DataFrame({
            "n": g.size(),
            "n_traded": g.apply(lambda d: int((d["position"].abs() > 1e-9).sum())),
            "total_pnl_bp": g["pnl_bp"].sum(),
            "mean_pnl_bp": g["pnl_bp"].mean(),
            "hit_rate": g.apply(lambda d: float((d.loc[d["position"].abs() > 1e-9, "pnl_bp"] > 0).mean())
                                if (d["position"].abs() > 1e-9).any() else np.nan),
        })
        return out.sort_values("total_pnl_bp", ascending=False)
    return {"by_regime": _agg("regime"), "by_policy": _agg("policy_regime")}


def run_model_comparison(panel_builder, models=None, target=None):
    """Part G model contribution: run the production backtest per MODEL, stack
    Sharpe/PnL. panel_builder(model)->panel."""
    models = models or C.MODELS
    rows = []
    for m in models:
        try:
            panel = panel_builder(m)
            bt, _ = forecast_repricing(panel, target=target)
            if bt is None or len(bt) == 0:
                rows.append(dict(model=m, status="no_repricing")); continue
            pos = build_positions(panel, bt, target=target)
            met = backtest_metrics(pos)
            rows.append(dict(model=m, **{k: met[k] for k in
                        ("n_traded", "total_pnl_bp", "sharpe_ann", "hit_rate", "max_dd_bp")}))
        except Exception as e:
            rows.append(dict(model=m, status=f"error: {str(e)[:40]}"))
    return pd.DataFrame(rows).set_index("model")
