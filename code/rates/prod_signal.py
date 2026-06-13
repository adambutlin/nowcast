"""
rates/prod_signal.py — Part C. Signal layer (causal).

For the active MODEL's nowcast already on the panel as my_nowcast / my_surprise:
  forecast_gap_z : expanding-standardized (my_nowcast - consensus)   [reuse stage1]
  revision_z     : expanding-standardized month-over-month change in my_nowcast
  confidence     : regime_trust * signal_strength  in [0,1]

signal_strength = squashed |forecast_gap_z| (more conviction when the gap is
large relative to its own history). confidence feeds position sizing so exposure
shrinks automatically when the regime is untrusted or the gap is weak.
"""

import numpy as np
import pandas as pd

from . import config as C
from . import stage1 as S1
from . import regime as R


def build_signals(panel):
    """Return panel with: forecast_gap_z, revision_z, signal_strength,
    confidence (all causal). Adds regime columns if absent."""
    p = R.classify_regimes(panel) if "regime_trust" not in panel.columns else panel.copy()

    gaps = S1.causal_gaps(p)                       # gap_raw / gap_z / gap_vol
    p["forecast_gap_z"] = gaps["gap_z"]

    rev = p["my_nowcast"].diff()                   # revision vs last month's nowcast
    mu = rev.expanding(min_periods=12).mean().shift(1)
    sd = rev.expanding(min_periods=12).std().shift(1)
    p["revision_z"] = (rev - mu) / sd

    # signal strength in [0,1): saturating function of |gap_z|
    p["signal_strength"] = np.tanh(np.abs(p["forecast_gap_z"].fillna(0.0)) / 1.5)
    p["confidence"] = (p["regime_trust"] * p["signal_strength"]).clip(0.0, 1.0)
    return p
