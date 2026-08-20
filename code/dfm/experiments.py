"""
dfm/experiments.py — assemble the model panel, probe runtime, run an arm.

Cost control is a first-class task, not an afterthought (spec §5.3): the timing
probe runs on the REAL panel before any sweep, and its projection decides whether
the documented fallbacks are needed.

Three rules this module exists to enforce, all of them silent failures otherwise:

1. `run_arm` hands `backtest.walk_forward` a RAW panel. walk_forward standardises
   each fold causally and rescales the nowcast itself, so standardising here would
   z-score the panel twice and every number would come back wrong but plausible.
2. `to_yoy` cannot convert a month without 12 months of prior index history. The
   drop is COUNTED and warned about rather than swallowed, because a summary table
   built from 108 of 120 months looks exactly like one built from 120.
3. `summarise` refuses to mix MoM and YoY frames. Both spaces produce a table of
   the same shape with the same column names, and only the magnitudes differ, so a
   mix is invisible at the point it matters. Frames carry their space in
   `DataFrame.attrs["space"]`; a conflict raises.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd

from dfm import backtest as B
from dfm import config as C
from dfm import panel as P
from dfm import target as T
from dfm.model import DFMConfig, DFMNowcaster

N_CONFIGS_FULL = 12          # 6 arms x 2 modes
N_MONTHS_FULL = 120          # 2015-2024
TARGET = "cpi_mom"

SPACE_ATTR = "space"         # DataFrame.attrs key: "mom" or "yoy"


# ── panel assembly ───────────────────────────────────────────────────────────
def model_panel(ref_month, asof_ts, start=None, target=TARGET):
    """
    Observation matrix + the CPI MoM target column, aligned on month-end.

    `asof_ts` is the information standpoint. Stage 1 calls this as
    `model_panel(ref, ref)` — as-of equals the reference month-end, the d = T
    origin, which is production's standpoint and the only one used for the
    headline comparison. The T-30 … R-1 origin ladder is Plan 4.

    CAVEAT — the target column is NOT publication-masked. `P.asof` masks every
    observation series by its release date, but CPI itself is inserted straight
    from `T.cpi_index()`, so the row for `ref_month` carries that month's MoM
    whenever the vintage already has it. That is correct for `walk_forward`,
    which blanks each test month's target before the model sees it and needs the
    value present to score against, but it means this frame must NOT be fed to a
    single-shot live nowcast without blanking the reference month first.
    """
    obs = P.asof(ref_month, asof_ts, start=start)
    if target in obs.columns:
        raise ValueError(
            f"{target!r} is already an observation column; it would be overwritten "
            "in place instead of appended last")
    mom = T.mom_from_index(T.cpi_index()).reindex(obs.index)
    n_obs = int(np.isfinite(mom.to_numpy(dtype=float)).sum())
    if n_obs == 0:
        raise ValueError(
            f"the CPI target has no finite value on the observation index "
            f"({obs.index[0]:%Y-%m}..{obs.index[-1]:%Y-%m}); the two indices are "
            "misaligned (month-end vs month-start?) rather than merely ragged")
    out = obs.copy()
    out[target] = mom
    out.attrs[SPACE_ATTR] = "mom"
    out.attrs["n_target_obs"] = n_obs
    return out


# ── timing ───────────────────────────────────────────────────────────────────
def timing_probe(panel, n_months=3, arm="E1", n_configs=N_CONFIGS_FULL,
                 n_months_full=N_MONTHS_FULL, draws=None, target=TARGET,
                 mode="standalone"):
    """
    Time a handful of real fits and project the full sweep.

    The probe mirrors what `walk_forward` actually does per fold — blank the test
    month's target, standardise causally through the previous month, fit, nowcast,
    carry the posterior mean forward as the next warm start — because an
    unstandardised panel puts the sampler in a different numerical regime (the
    stationarity rejection loop in `sampler.draw_factor_ar` is data-dependent) and
    would not time the thing we are about to run. It does not apply purge/embargo:
    that drops two rows out of ~330, which is below the noise floor of the
    measurement.

    Returns, all in seconds unless named otherwise:
      seconds_first_fit      the cold fit (no warm start)
      seconds_per_warm_fit   mean of the warm-started fits, or the cold fit if
                             `n_months` was 1 and there were none
      seconds_per_fit        mean over every probe fit
      projected_hours_full_sweep
                             n_configs * (one cold fit + (n_months_full - 1) warm
                             fits), AT THE PROBE'S DRAW COUNT. This is the honest
                             number for how `run_arm` runs today: walk_forward
                             takes a single DFMConfig and uses `cfg.draws` for
                             every fit, so C.DRAWS_FIRST is never reached.
      projected_hours_if_first_fit_uses_draws_first
                             the same projection with the cold fit rescaled to
                             C.DRAWS_FIRST draws (the Gibbs loop is a flat loop
                             over draws, so cost is linear in them). Reported so
                             the default `draws=C.DRAWS_WARM` cannot be mistaken
                             for a 2000-draw first fit.
    """
    if draws is None:
        draws = C.DRAWS_WARM
    cfg = DFMConfig(arm=arm, mode=mode, draws=draws,
                    burn=min(C.BURN_WARM, draws // 4))

    panel = panel.sort_index()
    col = panel[target].to_numpy(dtype=float)
    months = list(panel.index[np.isfinite(col)][-n_months:])
    if not months:
        raise ValueError(f"no month of {panel.index.size} has a finite {target!r}")

    per_fit, prev, warm_used = [], None, 0
    for m in months:
        train = panel.loc[:m].copy()
        train.loc[m, target] = np.nan            # never show the model the answer
        ztrain, _, _ = P.standardize(train, through=m - pd.offsets.MonthEnd(1))
        t0 = time.perf_counter()
        model = DFMNowcaster(cfg).fit(ztrain, target=target, warm_start=prev)
        model.nowcast(ztrain, target=target)
        per_fit.append(time.perf_counter() - t0)
        warm_used += prev is not None
        try:
            prev = model.posterior_mean() or None
        except Exception:
            prev = None

    first = per_fit[0]
    warm = float(np.mean(per_fit[1:])) if len(per_fit) > 1 else first
    sweep = n_configs * (first + (n_months_full - 1) * warm)
    first_at_draws_first = first * (C.DRAWS_FIRST / draws)
    return dict(
        seconds_per_fit=float(np.mean(per_fit)),
        seconds_first_fit=float(first),
        seconds_per_warm_fit=float(warm),
        warm_starts_used=int(warm_used),
        n_probe_months=len(months),
        probe_months=[str(pd.Timestamp(m).date()) for m in months],
        n_train_rows=int(len(panel.loc[:months[-1]])),
        n_series=int(panel.shape[1]),
        projected_hours_full_sweep=sweep / 3600.0,
        projected_hours_if_first_fit_uses_draws_first=(
            n_configs * (first_at_draws_first + (n_months_full - 1) * warm) / 3600.0),
        arm=arm, mode=mode, draws=cfg.draws, burn=cfg.burn,
        n_configs=n_configs, n_months_full=n_months_full)


# ── one arm, end to end ──────────────────────────────────────────────────────
def run_arm(arm="E1", mode="standalone", panel=None, start_year=None, end_year=None,
            draws=None, burn=None, target=TARGET, verbose=True):
    """
    Walk-forward one arm and write its results CSV. Returns (frame, path).

    The panel handed to `walk_forward` is RAW. walk_forward standardises every
    fold causally against months strictly before the test month and rescales the
    nowcast back with the target's own mu/sd; pre-standardising here would apply
    a second z-score on top and quietly shrink every prediction.

    `draws` defaults to C.DRAWS_WARM and applies to EVERY fit, including the
    first: walk_forward takes one DFMConfig for the whole sweep, so there is no
    per-fit draw schedule and C.DRAWS_FIRST is not used unless passed explicitly.
    """
    end = end_year or C.EVAL_END
    if panel is None:
        edge = pd.Timestamp(f"{end}-12-31") + pd.offsets.MonthEnd(0)
        panel = model_panel(edge, edge, target=target)
    cfg = DFMConfig(arm=arm, mode=mode,
                    draws=draws or C.DRAWS_WARM, burn=burn or C.BURN_WARM)
    overlay = mode == "overlay"
    purge = C.PURGE_OVERLAY if overlay else C.PURGE_STANDALONE
    embargo = C.EMBARGO_OVERLAY if overlay else C.EMBARGO_STANDALONE
    tag = f"{arm}_{mode}"
    bt = B.walk_forward(panel, target=target, model_cls=DFMNowcaster, model_cfg=cfg,
                        start_year=start_year, end_year=end_year,
                        purge=purge, embargo=embargo,
                        checkpoint_dir=C.CHECKPOINTS, tag=tag, verbose=verbose)
    bt.attrs[SPACE_ATTR] = "mom"
    path = os.path.join(C.DATA, f"backtest_{tag}.csv")
    bt.to_csv(path)
    return bt, path


# ── MoM -> YoY ───────────────────────────────────────────────────────────────
_YOY_COLS = ("actual", "pred", "p05", "p95")


def to_yoy(bt, index=None, warn=True):
    """
    Convert a MoM backtest to YoY space so it is comparable with production.

    YoY_T = 100 * ((1 + c_T/100) * exp(pi_T/100) - 1) with the carry c_T known at
    T, so the map is strictly increasing in pi_T; p05/p95 are converted too when
    present and remain the 5th/95th percentiles.

    `T.carry` needs I_{T-1} and I_{T-12}, so a month whose index history is short
    or gappy cannot be converted. Those months are DROPPED, but the count, the
    months themselves and the reason land in `.attrs` and (unless `warn=False`)
    in a UserWarning — a summary silently computed over fewer months than the
    caller believes is the exact failure this guards.
    """
    index = T.cpi_index() if index is None else index
    cols = [c for c in _YOY_COLS if c in bt.columns]
    if "actual" not in cols or "pred" not in cols:
        raise KeyError(
            f"to_yoy needs 'actual' and 'pred'; frame has {list(bt.columns)}")

    rows, dropped = [], []
    for m, r in bt.iterrows():
        try:
            row = {c: T.yoy_from_mom(float(r[c]), index, m) for c in cols}
        except ValueError as exc:
            dropped.append((str(pd.Timestamp(m).date()), str(exc)))
            continue
        row["month"] = m
        rows.append(row)

    out = (pd.DataFrame(rows).set_index("month").sort_index() if rows
           else pd.DataFrame(columns=list(cols), index=pd.DatetimeIndex([], name="month")))
    out.attrs[SPACE_ATTR] = "yoy"
    out.attrs["n_in"] = int(len(bt))
    out.attrs["n_dropped"] = len(dropped)
    out.attrs["dropped_months"] = [m for m, _ in dropped]
    if dropped and warn:
        warnings.warn(
            f"to_yoy dropped {len(dropped)}/{len(bt)} month(s) with too little CPI "
            f"index history to form the carry: {[m for m, _ in dropped]} "
            f"(first reason: {dropped[0][1]})", UserWarning, stacklevel=2)
    return out


# ── results table ────────────────────────────────────────────────────────────
def summarise(results, space=None, windows=("full", "shock", "ex_shock", "pre_2020")):
    """
    name -> backtest frame  ==>  one metrics row per model.

    Every frame must be in the SAME space. `to_yoy`, `model_panel` and `run_arm`
    stamp `DataFrame.attrs["space"]`; benchmark frames generally carry nothing and
    are taken on trust. Two frames that declare DIFFERENT spaces raise, because a
    MoM arm compared against a YoY benchmark yields a table that is well-formed,
    ordered and completely meaningless.
    """
    declared = {}
    for name, bt in results.items():
        s = getattr(bt, "attrs", {}).get(SPACE_ATTR)
        if s:
            declared[name] = s
    seen = set(declared.values()) | ({space} if space else set())
    if len(seen) > 1:
        raise ValueError(
            f"summarise was handed a mix of spaces {sorted(seen)}: {declared}"
            + (f" against space={space!r}" if space else "")
            + ". Convert the MoM frames with to_yoy() first.")

    rows = {}
    for name, bt in results.items():
        missing = {"actual", "pred"} - set(bt.columns)
        if missing:
            raise KeyError(
                f"{name!r} is missing {sorted(missing)}; it has {list(bt.columns)}. "
                "Rename the prediction column (production_backtest calls it "
                "'prod_pred') before summarising.")
        if not len(bt):
            raise ValueError(f"{name!r} is empty; nothing to summarise")
        m = B.metrics(bt, windows=windows)
        row = {f"{stat}_{w}": float(m.loc[w, stat])
               for w in m.index for stat in ("rmse", "mae", "bias")}
        row["n"] = int(m.loc["full", "n"])
        rows[name] = row

    tab = pd.DataFrame.from_dict(rows, orient="index")
    tab.attrs[SPACE_ATTR] = (seen.pop() if seen else None)
    return tab
