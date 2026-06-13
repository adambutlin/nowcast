"""
rates/sources.py — data adapters. Every loader returns a tidy, month-end- or
date-indexed pd.Series/DataFrame and NEVER raises on missing data: absent
sources yield empty/NaN so the event panel always builds (degraded, not broken).

CSV drop-in is the primary path (mirrors the factor registry idiom). Live
fetchers are best-effort and used only when a key/network is available.

CAUSALITY CONTRACT
  Predictor sources (my_nowcast, ucl, consensus, market_implied) must be
  values knowable on the release EVE (T-1). Outcome sources (daily rates)
  are the only place release-day data is read, and only via a signed move
  = level(release) - level(prev_business_day).
"""

import os
import datetime
import numpy as np
import pandas as pd

from . import config as C


# ─────────────────────────────────────────────────────────────────────────────
# generic CSV loader  [date, value] -> month-end Series
# ─────────────────────────────────────────────────────────────────────────────

def _load_value_csv(path, name):
    if not os.path.exists(path):
        return pd.Series(dtype=float, name=name)
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date", df.columns[0])
    vcol = cols.get("value", df.columns[-1])
    s = (df.assign(**{dcol: pd.to_datetime(df[dcol])})
           .set_index(dcol)[vcol].astype(float).sort_index())
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s.resample("ME").last().rename(name)


# ─────────────────────────────────────────────────────────────────────────────
# my nowcast — causal walk-forward preds from the CPI backtest artifact
# ─────────────────────────────────────────────────────────────────────────────

def my_nowcast(model=None, fallback=None):
    """Month-end Series of my as-of-T-1 nowcast (YoY %). Reuses the CPI
    backtest's per-year-refit predictions, which are causal (fit on < test
    year). Falls back per-month to `fallback` model where `model` is missing."""
    model    = model or C.MY_MODEL
    fallback = fallback or C.MY_MODEL_FALLBACK
    if not os.path.exists(C.BACKTEST_CSV):
        return pd.Series(dtype=float, name="my_nowcast"), pd.Series(dtype=float, name="actual_cpi")
    bt = pd.read_csv(C.BACKTEST_CSV, parse_dates=["date"])
    def _wide(col):
        return (bt.pivot_table(index="date", columns="model", values=col, aggfunc="first")
                  .sort_index())
    pred_w, act_w = _wide("pred"), _wide("actual")
    pred = pred_w[model] if model in pred_w.columns else pd.Series(index=pred_w.index, dtype=float)
    if fallback in pred_w.columns:
        pred = pred.fillna(pred_w[fallback])
    # actual is identical across models for a given date; take row-wise first non-NaN
    actual = act_w.bfill(axis=1).iloc[:, 0] if act_w.shape[1] else pd.Series(dtype=float)
    return pred.rename("my_nowcast"), actual.rename("actual_cpi")


# ─────────────────────────────────────────────────────────────────────────────
# public / market anchors  (CSV drop-in; empty if absent)
# ─────────────────────────────────────────────────────────────────────────────

def ucl_nowcast():        return _load_value_csv(C.UCL_CSV, "ucl_nowcast")
def economist_consensus():return _load_value_csv(C.CONSENSUS_CSV, "economist_consensus")
def market_implied():     return _load_value_csv(C.MKT_IMPLIED_CSV, "market_implied_expectation")


# ─────────────────────────────────────────────────────────────────────────────
# daily UK front-end rates  (CSV drop-in primary; BoE fetch best-effort)
# ─────────────────────────────────────────────────────────────────────────────

def daily_rates():
    """Daily DataFrame, percent, columns subset of C.RATE_COLS. CSV drop-in
    first; BoE live fetch only if CSV absent and network/certs available."""
    if os.path.exists(C.RATES_DAILY_CSV):
        df = pd.read_csv(C.RATES_DAILY_CSV, parse_dates=["date"]).set_index("date").sort_index()
        keep = [c for c in C.RATE_COLS if c in df.columns]
        return df[keep].astype(float)
    df = _fetch_boe_daily()           # best-effort
    if df is not None and len(df):
        return df
    return pd.DataFrame(columns=C.RATE_COLS)


def _fetch_boe_daily():
    """Best-effort BoE daily nominal gilt + OIS curves -> 1y/2y/5y/10y.
    Mirrors factors._boe_spot_5y parsing. Returns None on any failure
    (offline/sandbox). Populate data/uk_rates_daily.csv to make this unneeded."""
    try:
        import io, zipfile, requests
        try:
            import certifi
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass
        base = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
        out = {}
        specs = [("glcnominalddata.zip", "4. spot curve",
                  {"gilt_2y": 2.0, "gilt_5y": 5.0, "gilt_10y": 10.0}),
                 ("oisddata.zip", "4. spot curve", {"ois_1y": 1.0})]
        for zip_name, sheet, want in specs:
            try:
                r = requests.get(base + zip_name, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
                z = zipfile.ZipFile(io.BytesIO(r.content))
            except Exception:
                continue
            frames = []
            for fname in sorted(z.namelist()):
                try:
                    xl = pd.ExcelFile(z.open(fname))
                except Exception:
                    continue
                if sheet not in xl.sheet_names:
                    continue
                d = xl.parse(sheet, header=None)
                mats = d.iloc[3, 1:].values.astype(float)
                body = d.iloc[5:].copy()
                body.columns = ["date"] + list(mats)
                body = body[body["date"].apply(
                    lambda x: isinstance(x, (pd.Timestamp, datetime.datetime)))]
                body["date"] = pd.to_datetime(body["date"])
                body = body.set_index("date").apply(pd.to_numeric, errors="coerce")
                sel = {}
                for col, mat in want.items():
                    nearest = min(mats, key=lambda x: abs(x - mat))
                    sel[col] = body[nearest]
                frames.append(pd.DataFrame(sel))
            if frames:
                merged = pd.concat(frames).sort_index()
                for c in merged.columns:
                    out[c] = merged[c]
        if not out:
            return None
        return pd.DataFrame(out).sort_index()
    except Exception:
        return None


def rate_moves(rates_daily, release_dates):
    """For each release date, signed move (bp) = level(release_day or first
    trading day >= release) - level(previous trading day). Uses ONLY the
    release day and the prior business day — no forward window."""
    cols = [c for c in C.RATE_COLS if c in rates_daily.columns]
    out = {C.MOVE_COLS[c]: {} for c in cols}
    if not cols:
        return pd.DataFrame()
    idx = rates_daily.index
    for d in release_dates:
        d = pd.Timestamp(d)
        on_or_after = idx[idx >= d]
        before      = idx[idx < d]
        if len(on_or_after) == 0 or len(before) == 0:
            continue
        t1, t0 = on_or_after[0], before[-1]
        for c in cols:
            v1, v0 = rates_daily.at[t1, c], rates_daily.at[t0, c]
            if np.isfinite(v1) and np.isfinite(v0):
                out[C.MOVE_COLS[c]][d] = float((v1 - v0) * 100.0)  # percent -> bp
    return pd.DataFrame({k: pd.Series(v) for k, v in out.items()})


# ─────────────────────────────────────────────────────────────────────────────
# MPC dates, regime, calendar events
# ─────────────────────────────────────────────────────────────────────────────

def mpc_dates():
    if os.path.exists(C.MPC_DATES_CSV):
        s = pd.read_csv(C.MPC_DATES_CSV)
        col = s.columns[0]
        return pd.to_datetime(s[col]).sort_values().tolist()
    # fall back to Bank-Rate-change CSV months as a coarse meeting proxy
    s = _load_value_csv(C.BANKRATE_CSV, "x")
    return list(s.index)


def days_to_next_mpc(release_dates, meetings):
    meetings = sorted(pd.Timestamp(m) for m in meetings)
    out = {}
    for d in release_dates:
        d = pd.Timestamp(d)
        future = [m for m in meetings if m >= d]
        out[d] = int((future[0] - d).days) if future else np.nan
    return pd.Series(out, name="days_to_mpc")


def bank_rate_level():
    """Reconstruct Bank Rate level (%) from monthly bp-change CSV (cumsum)."""
    chg = _load_value_csv(C.BANKRATE_CSV, "chg")   # bp change per month
    if chg.empty:
        return chg
    return (chg.fillna(0).cumsum() / 100.0).rename("bank_rate")  # bp -> %  (relative level)


def mpc_regime(release_dates, anchor_rate=None):
    """Coarse policy regime per release from trailing 6m Bank-Rate change and
    level. {hiking, cutting, hold, pinned}. Causal: uses only data <= release."""
    lvl = bank_rate_level()
    out = {}
    for d in release_dates:
        d = pd.Timestamp(d)
        if lvl.empty:
            out[d] = "unknown"; continue
        hist = lvl[lvl.index <= d]
        if len(hist) < 7:
            out[d] = "unknown"; continue
        chg6 = hist.iloc[-1] - hist.iloc[-7]
        level = hist.iloc[-1]
        if   chg6 >=  0.40: out[d] = "hiking"
        elif chg6 <= -0.40: out[d] = "cutting"
        elif level <= 0.75: out[d] = "pinned"
        else:               out[d] = "hold"
    return pd.Series(out, name="mpc_regime")


def budget_flag(release_dates):
    s = _load_value_csv(C.BUDGET_CSV, "budget")
    out = {}
    for d in release_dates:
        d = pd.Timestamp(d)
        m = d + pd.offsets.MonthEnd(0)
        out[d] = int(s.get(m, 0) or 0) if not s.empty else 0
    return pd.Series(out, name="budget_event")


def ldi_flag(release_dates):
    lo, hi = pd.Timestamp(C.LDI_WINDOW[0]), pd.Timestamp(C.LDI_WINDOW[1])
    return pd.Series({pd.Timestamp(d): int(lo <= pd.Timestamp(d) <= hi)
                      for d in release_dates}, name="ldi_event")
