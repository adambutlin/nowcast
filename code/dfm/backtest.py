"""
dfm/backtest.py — expanding-window walk-forward with purge, embargo and resume.

One fit per test month, matching the TVP/LGBM backtest convention in
new_factors/two_stage.py. The test month's target is blanked before the model
sees the panel, so the ragged edge in the backtest is exactly the ragged edge in
a live nowcast.

Purge/embargo reuse validation.purge_embargo — the same helper production uses,
so leakage discipline is identical, not merely similar.

Every (tag, month) result is checkpointed to JSON so a multi-hour sweep survives
interruption and resumes where it stopped. Only the scalar result row is
checkpointed, never the posterior: a resumed sweep therefore starts its first
refit without a warm start, which changes the Gibbs initialisation (not the
target of the chain) relative to an uninterrupted run.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import validation as V
from dfm import config as C
from dfm.panel import standardize

SHOCK_YEARS = (2022, 2023)
MIN_TRAIN_MONTHS = 60      # refuse to fit a 7-factor DFM on less than five years

# Nowcast entries that live in the target's z-units and must be rescaled by the
# target's own mu/sd. `sd` is a scale, so it takes sd but not mu; `month` and any
# diagnostic entries pass through untouched.
_LOCATION_KEYS = ("mean", "median", "p05", "p95", "draws")


def _checkpoint_path(directory, tag, month):
    return os.path.join(directory, f"{tag}_{pd.Timestamp(month):%Y-%m}.json")


def _rescale(nc, mu, sd):
    """Map a nowcast dict from z-units back into the target's own units."""
    out = {k: (v * sd + mu if k in _LOCATION_KEYS else v) for k, v in nc.items()}
    if "sd" in nc:
        out["sd"] = nc["sd"] * sd
    return out


def walk_forward(panel, target, model_cls, model_cfg=None, start_year=None,
                 end_year=None, purge=None, embargo=None, checkpoint_dir=None,
                 tag="run", warm_start=True, min_train=MIN_TRAIN_MONTHS,
                 verbose=False):
    """
    Expanding-window walk-forward over month-end rows of `panel`.

    Returns a DataFrame indexed by test month with actual, pred, interval and
    the wall-clock seconds each fit took.
    """
    start_year = C.EVAL_START if start_year is None else start_year
    end_year = C.EVAL_END if end_year is None else end_year
    purge = C.PURGE_STANDALONE if purge is None else purge
    embargo = C.EMBARGO_STANDALONE if embargo is None else embargo
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    months = [m for m in panel.index
              if start_year <= m.year <= end_year and np.isfinite(panel.loc[m, target])]
    rows, prev_post = [], None
    for m in months:
        if checkpoint_dir:
            cp = _checkpoint_path(checkpoint_dir, tag, m)
            if os.path.exists(cp):
                with open(cp) as fh:
                    rows.append(json.load(fh))
                continue

        train = panel.loc[:m].copy()
        actual = float(train.loc[m, target])
        train.loc[m, target] = np.nan                 # never show the model the answer
        if purge or embargo:
            # The purge applies to history only. The test month's FEATURE row is
            # observable at nowcast time and must survive, so it is held out of
            # purge_embargo and concatenated back with its target already blanked.
            hist = V.purge_embargo(train.iloc[:-1], m, purge, embargo)
            train = pd.concat([hist, train.loc[[m]]])
        if len(train) < min_train:
            continue

        # Causal standardisation. The model does not standardise internally —
        # doing so would need the training cut point, which only the caller knows.
        # Moments come from months strictly before the test month, so the test
        # month's own values never influence its own scaling. The nowcast comes
        # back in z-units and is rescaled by the target's own mu/sd.
        through = m - pd.offsets.MonthEnd(1)
        ztrain, mu, sd = standardize(train, through=through)
        tmu, tsd = float(mu[target]), float(sd[target])
        if not (np.isfinite(tmu) and np.isfinite(tsd)):
            # standardize() defends sd (0 -> 1, NaN -> 1) but not mu: a target
            # with no observations before `through` yields mu = NaN, which would
            # rescale every prediction to NaN. Fail loudly rather than emit them.
            raise ValueError(
                f"target {target!r} has no usable history before {through:%Y-%m}; "
                "start the evaluation window after the target's first observation")

        t0 = time.time()
        model = model_cls(model_cfg) if model_cfg is not None else model_cls()
        model.fit(ztrain, target=target, warm_start=prev_post if warm_start else None)
        nc = _rescale(model.nowcast(ztrain, target=target), tmu, tsd)
        secs = time.time() - t0
        if warm_start:
            try:
                prev_post = model.posterior_mean()
            except Exception:
                prev_post = None

        row = dict(month=str(pd.Timestamp(m).date()), actual=actual,
                   pred=float(nc["mean"]), p05=float(nc["p05"]), p95=float(nc["p95"]),
                   sd=float(nc["sd"]), fit_seconds=round(secs, 3))
        rows.append(row)
        if checkpoint_dir:
            with open(_checkpoint_path(checkpoint_dir, tag, m), "w") as fh:
                json.dump(row, fh)
        if verbose:
            print(f"  {row['month']}  pred={row['pred']:+.3f}  actual={row['actual']:+.3f}"
                  f"  ({secs:.1f}s)", flush=True)

    if not rows:
        return pd.DataFrame(columns=["actual", "pred", "p05", "p95", "sd", "fit_seconds"])
    out = pd.DataFrame(rows)
    out["month"] = pd.to_datetime(out["month"])
    return out.set_index("month").sort_index()


def _window_mask(idx, window):
    if window == "full":
        return np.ones(len(idx), dtype=bool)
    if window == "shock":
        return idx.year.isin(SHOCK_YEARS)
    if window == "ex_shock":
        return ~idx.year.isin(SHOCK_YEARS)
    if window == "pre_2020":
        return idx.year < 2020
    raise ValueError(f"unknown window {window!r}")


def metrics(bt, windows=("full", "shock", "ex_shock", "pre_2020")):
    """RMSE / MAE / n by evaluation window.

    The windows are deliberately NOT a partition: `pre_2020` is a subset of
    `ex_shock`. Both are reported because they answer different questions —
    "how does it do outside the shock" and "how did it do in the calm decade".
    """
    rows = []
    for w in windows:
        mask = _window_mask(bt.index, w)
        sub = bt[mask].dropna(subset=["actual", "pred"])
        err = sub["pred"] - sub["actual"]
        rows.append(dict(window=w, n=len(sub),
                         rmse=float(np.sqrt((err ** 2).mean())) if len(sub) else np.nan,
                         mae=float(err.abs().mean()) if len(sub) else np.nan,
                         bias=float(err.mean()) if len(sub) else np.nan))
    return pd.DataFrame(rows).set_index("window")


def dm_test(e1, e2, h=1):
    """Diebold-Mariano on squared-error differentials. Returns (stat, p)."""
    from scipy.stats import t as tdist
    d = np.asarray(e1, dtype=float) ** 2 - np.asarray(e2, dtype=float) ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dbar = d.mean()
    gamma0 = ((d - dbar) ** 2).mean()
    var = gamma0
    for lag in range(1, h):
        cov = ((d[lag:] - dbar) * (d[:-lag] - dbar)).mean()
        var += 2 * (1 - lag / h) * cov
    stat = dbar / np.sqrt(var / n) if var > 0 else np.nan
    p = 2 * (1 - tdist.cdf(abs(stat), n - 1)) if np.isfinite(stat) else np.nan
    return float(stat), float(p)
