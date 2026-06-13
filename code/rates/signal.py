"""
rates/signal.py — Deliverable 5. Predicted front-end move -> tradeable position.

Event strategy (one trade per CPI release, ~1-day hold, no overlap):
  position > 0  == positioned for yields UP (short the gilt / pay fixed).
  pnl_bp        == position * realized_move_bp  - tcost*|position|

Sizing: position = clip( pred_move / (vol_k * trailing_move_vol), -cap, cap ),
with a deadband that flattens when |pred_move| is below `deadband_bp`. Volatility
targeting uses ONLY trailing realized move vol (causal).
"""

import numpy as np
import pandas as pd

from . import config as C


def backtest_signal(mvp_bt, panel, target=None, vol_k=1.0, cap=2.0,
                    deadband_bp=1.0, tcost_bp=0.5, vol_window=12):
    """mvp_bt: DataFrame[index ref_month, pred_move, realized_move].
    Returns (trades DataFrame, metrics dict)."""
    target = target or C.PRIMARY_MOVE
    if mvp_bt is None or len(mvp_bt) == 0:
        return pd.DataFrame(), {"status": "no_mvp"}
    df = mvp_bt.copy()
    # trailing realized-move vol (causal: shift so vol_t excludes the current move)
    roll_vol = df["realized_move"].rolling(vol_window, min_periods=4).std().shift(1)
    roll_vol = roll_vol.fillna(df["realized_move"].expanding().std().shift(1)).bfill().clip(lower=1.0)

    raw = df["pred_move"] / (vol_k * roll_vol)
    pos = raw.clip(-cap, cap)
    pos = pos.where(df["pred_move"].abs() >= deadband_bp, 0.0)   # deadband
    dpos = pos.diff().abs().fillna(pos.abs())                    # turnover (per event)
    pnl = pos * df["realized_move"] - tcost_bp * dpos

    trades = pd.DataFrame({
        "pred_move": df["pred_move"], "realized_move": df["realized_move"],
        "position": pos, "pnl_bp": pnl,
    }, index=df.index)
    traded = trades[trades["position"].abs() > 1e-9]
    ann = np.sqrt(12.0)   # ~12 releases/yr
    mu, sd = pnl.mean(), pnl.std()
    metrics = dict(
        n_events=int(len(trades)), n_traded=int(len(traded)),
        total_pnl_bp=float(pnl.sum()),
        mean_pnl_bp=float(mu), pnl_vol_bp=float(sd),
        IR_annualized=float(mu / sd * ann) if sd and np.isfinite(sd) else np.nan,
        hit_rate=float((traded["pnl_bp"] > 0).mean()) if len(traded) else np.nan,
        avg_turnover=float(dpos.mean()),
    )
    trades.to_csv(C.SIGNAL_CSV)
    return trades, metrics
