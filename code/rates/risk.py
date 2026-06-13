"""
rates/risk.py — Part I. Risk controls -> per-event tradeable mask + size multiplier.

Controls (all causal):
  * LDI-style exclusion       (config.EXCLUDE_LDI, panel ldi_event)
  * fiscal-event exclusion    (config.EXCLUDE_BUDGET, panel budget_event)
  * volatility kill switch     (trailing realized move-vol > KILL_VOL_BP)
  * low-confidence suppression (confidence < MIN_CONFIDENCE)
"""

import numpy as np
import pandas as pd

from . import config as C


def apply_risk(panel, target=None):
    """Return DataFrame[tradeable(bool), size_mult(float), reason] indexed like panel.
    panel must already carry `confidence` (prod_signal) + event flags."""
    target = target or C.TARGET_PRIMARY
    idx = panel.index
    tradeable = pd.Series(True, index=idx)
    reason = pd.Series("", index=idx, dtype=object)

    def block(mask, label):
        nonlocal tradeable, reason
        hit = mask & tradeable
        reason[hit] = label
        tradeable = tradeable & ~mask

    if C.EXCLUDE_LDI and "ldi_event" in panel:
        block(panel["ldi_event"].fillna(0).astype(bool), "ldi")
    if C.EXCLUDE_BUDGET and "budget_event" in panel:
        block(panel["budget_event"].fillna(0).astype(bool), "budget")

    # volatility kill switch: trailing realized move-vol (causal, shift(1))
    if target in panel:
        vol = panel[target].rolling(C.VOL_WINDOW, min_periods=4).std().shift(1)
        block(vol > C.KILL_VOL_BP, "vol_kill")

    if "confidence" in panel:
        block(panel["confidence"].fillna(0.0) < C.MIN_CONFIDENCE, "low_confidence")

    size_mult = panel["confidence"].fillna(0.0) if "confidence" in panel else pd.Series(1.0, index=idx)
    size_mult = size_mult.where(tradeable, 0.0)
    return pd.DataFrame({"tradeable": tradeable, "size_mult": size_mult, "reason": reason})
