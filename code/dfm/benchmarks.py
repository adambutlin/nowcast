"""
dfm/benchmarks.py — AutoARIMA, AR(1) and a frozen-production replication.

Production exposes nowcast() for a single live month, not a walk-forward, so the
replication is composed here from the FROZEN constants, imported rather than
restated: targets.HEADLINE carries lambda / overlay weights / pinned set, and
production.model.LGB carries the LightGBM parameters. A governance change to
production therefore changes this benchmark too, instead of silently diverging.

Everything here is READ-ONLY with respect to production code.
"""
import numpy as np
import pandas as pd

import factors as F
import uk_model_zoo as Z
import validation as V
from production.model import LGB as LGB_PARAMS
from targets import HEADLINE

from dfm import config as C

PROD_LAMBDA = HEADLINE.lam
PROD_WEIGHTS = dict(HEADLINE.overlay_weights)
PROD_PINNED = list(HEADLINE.pinned)
TARGET = HEADLINE.name


def _weighted_overlay(members):
    """
    NaN-safe weighted overlay, renormalising over the members that produced a
    finite prediction — the same rule two_stage._weighted_combo applies, so a
    dead leg degrades the overlay instead of nulling the whole forecast.
    """
    num = 0.0
    wsum = 0.0
    for name, value in members.items():
        if value is None or not np.isfinite(value):
            continue
        w = PROD_WEIGHTS[name]
        num += w * float(value)
        wsum += w
    return num / wsum if wsum > 0 else np.nan


def ar1_backtest(y, start_year=None, end_year=None):
    """Expanding-window AR(1) on a monthly series. The floor benchmark."""
    start_year = start_year or C.EVAL_START
    end_year = end_year or C.EVAL_END
    y = y.dropna()
    rows = []
    for m in y.index:
        if not (start_year <= m.year <= end_year):
            continue
        hist = y[y.index < m]
        if len(hist) < 24:
            continue
        x, t = hist.values[:-1], hist.values[1:]
        xc, tc = x - x.mean(), t - t.mean()
        phi = float(xc @ tc / (xc @ xc)) if xc @ xc > 0 else 0.0
        const = float(t.mean() - phi * x.mean())
        rows.append(dict(month=m, actual=float(y.loc[m]),
                         pred=const + phi * float(hist.iloc[-1])))
    return pd.DataFrame(rows).set_index("month") if rows else pd.DataFrame()


def load_production_matrix():
    """The frozen headline factor matrix, pub-lagged exactly as production builds it."""
    raw, status = F.build_matrix(names=PROD_PINNED + [TARGET])
    live = [n for n in PROD_PINNED if status.get(n) != "unavailable"]
    raw = raw[raw.index.year >= 1997]
    df = F.apply_publication_lags(raw, live)
    for rf in HEADLINE.reg:
        if rf in df.columns:
            observed = df[rf].dropna()
            if len(observed):
                span = df.index <= observed.index[-1]
                df.loc[span, rf] = df.loc[span, rf].fillna(0)
    return df.resample("ME").last(), live


def autoarima_backtest(df=None, start_year=None, end_year=None):
    """AutoARIMA anchor — the model production is judged against."""
    if df is None:
        df, _ = load_production_matrix()
    aa = Z.AutoARIMA().backtest(df, [], TARGET, start_year=HEADLINE.aa_start,
                                end_year=end_year or C.EVAL_END)
    lo = start_year or C.EVAL_START
    return aa[(aa.index.year >= lo) & (aa.index.year <= (end_year or C.EVAL_END))]


def production_backtest(start_year=None, end_year=None):
    """
    Walk-forward replication of AA + 0.25*TVP + 0.25*LGBM on this harness.

    RegEvent is NOT included: it is a live arithmetic layer keyed to announced cap
    resets, and the frozen backtest numbers in README/final_model.md predate it.
    Its contribution is reported separately in Plan 4.
    """
    import lightgbm as lgb

    start_year = start_year or C.EVAL_START
    end_year = end_year or C.EVAL_END
    df, live = load_production_matrix()
    aa = autoarima_backtest(df, start_year, end_year)
    # The residual column spans the WHOLE AA backtest (aa_start onward), not just the
    # scored window, so the overlays have their full pre-2015 training history; it is
    # blanked from each test month onward below to keep the walk-forward causal.
    full_aa = Z.AutoARIMA().backtest(df, [], TARGET, start_year=HEADLINE.aa_start,
                                     end_year=end_year)
    resid_all = (full_aa["actual"] - full_aa["pred"]).rename("resid")

    rows = []
    for m in aa.index:
        d = df.copy()
        d["resid"] = resid_all.reindex(d.index)
        d.loc[d.index >= m, "resid"] = np.nan          # only past residuals are known
        try:
            tvp, _ = Z.TVP().nowcast(d, live, "resid", nowcast_date=m)
        except Exception:
            tvp = np.nan
        data = pd.DataFrame(d[live]).join(d["resid"]).dropna(subset=["resid"])
        train = V.purge_embargo(data, m, C.PURGE_OVERLAY, C.EMBARGO_OVERLAY)
        try:
            gbm = float(lgb.LGBMRegressor(**LGB_PARAMS)
                        .fit(train[live], train["resid"]).predict(d.loc[[m], live])[0])
        except Exception:
            gbm = np.nan
        overlay = _weighted_overlay({"tvp": tvp, "lgbm": gbm})
        pred = float(aa.loc[m, "pred"]) + (PROD_LAMBDA * overlay
                                           if np.isfinite(overlay) else 0.0)
        rows.append(dict(month=m, actual=float(aa.loc[m, "actual"]),
                         aa_pred=float(aa.loc[m, "pred"]),
                         tvp_resid=tvp, lgbm_resid=gbm, prod_pred=pred))
    return pd.DataFrame(rows).set_index("month")
