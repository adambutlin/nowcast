"""
intramonth/panel.py — intramonth panel builder (Part B/I).

build_panel(target_key, k) assembles, for forecast origin T-k, a causal monthly panel:
  - target column (actual value, used for walk-forward training/eval)
  - monthly macro factors  (pub-lagged via factors.apply_publication_lags)
  - HF as-of features       (partial-month aggregation truncated at month_end - k days)

Every row M only contains information knowable at origin T-k of month M, so a
walk-forward fit (train rows < M, predict M) is leakage-free at every origin.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factors as F
from intramonth import config as C, targets as T, hf_data as H

_FACTOR_CACHE = {}


def _monthly_factors():
    """Causal monthly macro factor matrix (pub-lagged). Cached."""
    if "m" in _FACTOR_CACHE:
        return _FACTOR_CACHE["m"]
    names = C.MONTHLY_FACTORS
    raw, status = F.build_matrix(names=names)
    live = [n for n in names if status.get(n) != "unavailable"]
    raw = raw[raw.index.year >= C.TRAIN_FROM - 2]
    df = F.apply_publication_lags(raw, live)
    # reg-event factors: 0 before their data starts (no event = 0)
    for rf in ["mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]:
        if rf in df.columns:
            df[rf] = df[rf].fillna(0)
    df = df.resample("ME").last()
    _FACTOR_CACHE["m"] = (df, live)
    return df, live


def build_panel(target_key=None, k=1, daily=None):
    """
    Causal intramonth panel for (target, origin T-k).
    Returns (panel DataFrame, meta dict). meta has target_status, hf_cols, factor_cols.
    """
    target_key = target_key or C.DEFAULT_TARGET
    y, y_status = T.resolve(target_key)
    if y is None:
        raise RuntimeError(f"target {target_key} unavailable — cannot build panel")
    y = y.resample("ME").last()

    fac, live = _monthly_factors()
    daily = H.get_daily() if daily is None else daily

    # union month index from target start (HF era) to last target month
    months = y.index[(y.index.year >= C.TRAIN_FROM)]
    hf = H.hf_panel(months, k, daily=daily)

    panel = pd.DataFrame(index=months)
    panel.index.name = "date"
    panel[target_key] = y.reindex(months)
    for c in fac.columns:
        panel[c] = fac[c].reindex(months)
    for c in hf.columns:
        panel[c] = hf[c].reindex(months)

    meta = dict(target=target_key, target_status=y_status, origin_k=k,
                hf_cols=list(hf.columns), factor_cols=list(fac.columns),
                live_factors=live)
    return panel, meta


def extend_to_nowcast(panel, meta, k, daily=None):
    """
    Append the next (unreleased) reference month as a target=NaN row carrying the
    causal HF as-of features at origin T-k and forward-filled monthly factors.
    Returns (extended panel, nowcast_date).
    """
    daily = H.get_daily() if daily is None else daily
    last = panel.index[-1]
    nd = (last + pd.offsets.MonthEnd(1))
    row = {c: np.nan for c in panel.columns}
    # monthly factors: last published value (causal ffill)
    for c in meta["factor_cols"]:
        col = panel[c].dropna()
        row[c] = float(col.iloc[-1]) if len(col) else np.nan
    # HF as-of features for the nowcast month at origin k
    hf = H.asof_features(daily, nd, k)
    for c in meta["hf_cols"]:
        row[c] = hf.get(c, np.nan)
    row[meta["target"]] = np.nan
    ext = pd.concat([panel, pd.DataFrame([row], index=[nd])])
    ext.index.name = "date"
    return ext, nd


def factor_columns(meta, hf=True, monthly=True):
    """Feature columns for the model stack at this origin (target excluded)."""
    cols = []
    if monthly:
        cols += meta["factor_cols"]
    if hf:
        cols += [c for c in meta["hf_cols"] if not c.startswith("hf_coverage")]
    return cols


if __name__ == "__main__":
    for k in (30, 1):
        p, meta = build_panel(C.DEFAULT_TARGET, k=k)
        feats = factor_columns(meta)
        last = p.dropna(subset=[meta["target"]]).index[-1]
        print(f"T-{k}: panel {p.shape}, target_status={meta['target_status']}, "
              f"{len(feats)} feats, last target {last.date()}={p.loc[last, meta['target']]:.2f}")
        print("    hf row (last):", p[meta["hf_cols"]].iloc[-1].round(4).to_dict())
