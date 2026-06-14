"""
intramonth/hf_data.py — high-frequency (daily) data engine with AS-OF truncation (Part B).

This is the causal core of the system. Existing repo code aggregates daily data to
month-end ONCE (loses intramonth detail). Here we keep the raw daily series and build
features AS OF an intramonth date, so a forecast at origin T-k for reference month M
sees ONLY daily observations with timestamp <= (month_end(M) - k days).

asof_features(month, k) returns, for each HF series, the partial-month aggregation:
  {n}_ret : log return of last-in-window vs last value of the PRIOR full month
  {n}_lvl : mean level over [month_start, asof]
and a scalar hf_coverage = elapsed-days / days-in-month (how much HF info has accrued).

No future leakage: injecting daily values dated after `asof` must not change the output
(verified in tests). No future standardization: levels are raw; z-scoring is causal and
happens later inside each model's walk-forward fit.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intramonth import config as C

_DAILY_CACHE = {}


def daily_hf(tickers=None, start=None, force=False):
    """Download daily HF series from yfinance → DataFrame[date × short_name]. Cached."""
    tickers = tickers or C.HF_TICKERS
    start = start or C.HF_START
    key = (tuple(sorted(tickers.items())), start)
    if not force and key in _DAILY_CACHE:
        return _DAILY_CACHE[key]
    out = None
    try:
        import yfinance as yf
        cols = {}
        for name, tkr in tickers.items():
            try:
                raw = yf.download(tkr, start=start, auto_adjust=True, progress=False)
                c = (raw[("Close", tkr)] if isinstance(raw.columns, pd.MultiIndex)
                     else raw["Close"])
                cols[name] = c.rename(name)
            except Exception:
                pass
        if cols:
            out = pd.concat(cols.values(), axis=1).sort_index()
            out.index = pd.to_datetime(out.index)
    except Exception:
        out = None
    _DAILY_CACHE[key] = out
    return out


def _csv_fallback():
    """If yfinance is blocked, load data/intramonth/hf_daily.csv [date, brent, gas, gbp, vix]."""
    p = os.path.join(C.DIR_INTRA_DATA, "hf_daily.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
        return df
    return None


def get_daily(force=False):
    """Daily HF frame, yfinance primary, CSV fallback, else None (pipeline still runs)."""
    d = daily_hf(force=force)
    if d is None or d.empty:
        d = _csv_fallback()
    if d is not None and not d.empty:
        # persist a snapshot so offline reruns are reproducible
        try:
            d.reset_index().rename(columns={"index": "date"}).to_csv(
                os.path.join(C.DIR_INTRA_DATA, "hf_daily.csv"), index=False)
        except Exception:
            pass
    return d


def asof_date(month_end, k):
    """As-of timestamp for origin T-k: k calendar days before reference month-end."""
    return pd.Timestamp(month_end) - pd.Timedelta(days=int(k))


def asof_features(daily, month_end, k):
    """
    Partial-month HF features for reference month `month_end`, AS OF (month_end - k days).
    Causal: uses only daily rows with index <= asof. Returns dict of features + coverage.
    """
    month_end = pd.Timestamp(month_end)
    m_start = month_end.replace(day=1)
    asof = asof_date(month_end, k)
    feats = {}
    if daily is None or daily.empty:
        for n in C.HF_TICKERS:
            feats[f"{n}_ret"] = np.nan
            feats[f"{n}_lvl"] = np.nan
        feats["hf_coverage"] = np.nan
        return feats

    in_win = daily[(daily.index >= m_start) & (daily.index <= asof)]
    prior  = daily[daily.index < m_start]
    days_in_month = (month_end - m_start).days + 1
    elapsed = max((asof - m_start).days + 1, 0)
    feats["hf_coverage"] = min(elapsed / days_in_month, 1.0)

    for n in C.HF_TICKERS:
        win_vals = in_win[n].dropna() if n in in_win.columns else pd.Series(dtype=float)
        pri_vals = prior[n].dropna()  if n in prior.columns  else pd.Series(dtype=float)
        if len(win_vals) and len(pri_vals):
            last_win = float(win_vals.iloc[-1]); last_prior = float(pri_vals.iloc[-1])
            feats[f"{n}_ret"] = (np.log(last_win / last_prior)
                                 if last_win > 0 and last_prior > 0 else np.nan)
            feats[f"{n}_lvl"] = float(win_vals.mean())
        else:
            feats[f"{n}_ret"] = np.nan
            feats[f"{n}_lvl"] = float(win_vals.mean()) if len(win_vals) else np.nan
    return feats


def hf_panel(months, k, daily=None):
    """
    Build a causal HF feature panel for a list of reference month-ends at origin T-k.
    Each row uses only that month's daily data up to k-days-before-its-own-end.
    """
    daily = get_daily() if daily is None else daily
    rows = {}
    for m in months:
        rows[pd.Timestamp(m)] = asof_features(daily, m, k)
    panel = pd.DataFrame(rows).T.sort_index()
    panel.index.name = "date"
    return panel


if __name__ == "__main__":
    d = get_daily()
    print("daily HF:", None if d is None else f"{d.shape} {d.index.min().date()}..{d.index.max().date()}")
    if d is not None:
        m = pd.Timestamp("2026-05-31")
        for k in C.ORIGINS:
            f = asof_features(d, m, k)
            print(f"  T-{k:<2} asof {asof_date(m,k).date()} cover={f['hf_coverage']:.2f} "
                  f"brent_ret={f['brent_ret']:+.4f} gas_ret={f['gas_ret']:+.4f}")
