"""
Purged + embargoed cross-validation helpers for the walk-forward backtest.

The production overlays (TVP, LGBM) are trained on the AutoARIMA-residual history
whose target, ``cpi_yoy``, is a 12-month difference. In an expanding-window
backtest the training set abuts the test fold with no gap, so the last ≤12
training months share YoY information with the first test months and leak across
the boundary, optimistically inflating OOS metrics.

``purge_embargo`` removes that overlap:
  - PURGE    drops training rows whose label-horizon (``horizon`` months) overlaps
             the test fold.
  - EMBARGO  drops a further ``embargo`` months so regime-shift information cannot
             leak via residual autocorrelation just before the fold.

This is a backtest-honesty tool. It is intentionally NOT applied to the live
nowcast: there the future test label is unknown and residuals up to the last
release are legitimately usable, so purging them would discard real information
rather than prevent leakage.

Spec: docs/superpowers/specs/2026-06-22-purged-embargo-backtest-design.md
"""
import pandas as pd


def purge_embargo(train, test_start, horizon=12, embargo=1):
    """Drop training rows within ``horizon + embargo`` months before ``test_start``.

    Parameters
    ----------
    train : DataFrame indexed by a DatetimeIndex (month-end or month-start).
    test_start : Timestamp — the first timestamp of the test fold.
    horizon : int — label-horizon in months to purge (YoY ⇒ 12).
    embargo : int — extra months of gap after the purge window.

    Returns
    -------
    DataFrame — ``train`` restricted to rows strictly before the cutoff
    ``test_start - (horizon + embargo)`` months. ``horizon=0, embargo=0`` is the
    identity for a causal training set (all rows already precede ``test_start``).
    """
    gap = int(horizon) + int(embargo)
    if gap <= 0:
        return train
    # Month-granular so the result is independent of day-of-month (the index may be
    # month-start or month-end, and DateOffset preserves the day): keep training
    # months strictly earlier than `gap` calendar months before the test fold.
    cutoff = pd.Period(pd.Timestamp(test_start), "M") - gap
    keep = train.index.to_period("M") < cutoff
    return train[keep]
