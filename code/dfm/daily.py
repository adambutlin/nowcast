"""
dfm/daily.py — raw DAILY series, cached to disk.

factors.py resamples to month-end inside every fetcher, so the registry cannot
supply intraday resolution and must not be modified (production isolation). This
module fetches the daily series the as-of panel needs and caches each one to
data/dfm/daily/<name>.csv so a multi-hour sweep is reproducible offline.

partial_month() is the mixed-frequency aggregator: given daily observations and
an as-of timestamp, it returns the month's value USING ONLY days <= asof.
Levels take the last value in the window; returns are measured against the last
value of the prior month. Days after asof are invisible by construction.
"""
import os

import numpy as np
import pandas as pd

from dfm import config as C

LONG_START = "1989-01-01"


def _fred_daily(series_id):
    from fredapi import Fred
    s = Fred(api_key=os.getenv("FRED_API_KEY")).get_series(
        series_id, observation_start=LONG_START)
    s.index = pd.to_datetime(s.index)
    return s.dropna().sort_index()


def _yf_daily(ticker):
    import yfinance as yf
    raw = yf.download(ticker, start=LONG_START, auto_adjust=True, progress=False)
    c = (raw[("Close", ticker)] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"])
    return c.dropna().sort_index()


def _gbp_eur_daily():
    """EUR per GBP, derived exactly as factors._gbp_eur does but without the
    month-end resample: USD-per-GBP divided by USD-per-EUR, both FRED daily."""
    gbp = _fred_daily("DEXUSUK")
    eur = _fred_daily("DEXUSEU").reindex(gbp.index).ffill()
    return (gbp / eur).dropna()


# Daily-resolution members of the panel. Everything else is natively monthly or
# lower and is read through factors.build_matrix in panel.py.
#
# Verified against FRED on 2026-08-19 (median observation gap over 2024):
#   DCOILBRENTEU 1d, DHHNGSP 1d, DEXUSUK 1d, DEXUSEU 1d  -> genuinely daily
#   IRLTLT01GBM156N 31d, PCOPPUSDM 31d, PWHEAMTUSDM 31d  -> MONTHLY at source
# so uk_gilt_10y, copper_price and wheat_price are NOT daily despite the
# registry describing them as market rates, and are marked freq="monthly" in
# panel.py. Sources here mirror the registry's own fetchers so the DFM panel and
# production see the same underlying data.
#
# gas_eu is deliberately absent: factors._gas_eu_ttf blends daily TTF (2017+)
# onto IMF monthly (1960+), so a daily-only fetch would silently drop 1997-2016
# — the shock windows this model most needs. It stays monthly, which costs
# intra-month gas detail at the T-30..T-7 origins but nothing at d=T, where the
# headline comparison against production is made. Revisit in Plan 4.
FETCHERS = {
    "oil_brent": lambda: _fred_daily("DCOILBRENTEU"),
    "gas_hh":    lambda: _fred_daily("DHHNGSP"),
    "gbpusd":    lambda: _fred_daily("DEXUSUK"),
    "gbp_eur":   _gbp_eur_daily,
    "vix":       lambda: _yf_daily("^VIX"),
}


def _cache_path(name):
    return os.path.join(C.DAILY_CACHE, f"{name}.csv")


def load_daily(name, force=False):
    """Daily series for `name`, from cache when available."""
    path = _cache_path(name)
    if not force and os.path.exists(path):
        s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"]
        return s.rename(name).sort_index()
    if name not in FETCHERS:
        raise KeyError(f"{name!r} has no daily fetcher — add one to dfm.daily.FETCHERS")
    s = FETCHERS[name]().rename(name)
    s.index.name = "date"
    s.rename("value").to_frame().to_csv(path)
    s = s.sort_index()
    s.index = pd.DatetimeIndex(s.index, name="date", freq=None)
    return s


def partial_month(series, ref_month, asof, how="level"):
    """
    Month value for `ref_month` using only observations dated <= `asof`.

    how="level"  -> last value inside [month_start, asof]
    how="logret" -> log(last value inside window / last value of the prior month)

    Returns NaN when the window holds no observation (the cell is then MISSING in
    the panel and the Kalman filter simply skips its update).
    """
    end = pd.Timestamp(ref_month) + pd.offsets.MonthEnd(0)
    start = end.replace(day=1)
    asof = pd.Timestamp(asof)
    win = series[(series.index >= start) & (series.index <= min(asof, end))].dropna()
    if len(win) == 0:
        return np.nan
    last = float(win.iloc[-1])
    if how == "level":
        return last
    if how == "logret":
        prior = series[series.index < start].dropna()
        if len(prior) == 0 or float(prior.iloc[-1]) <= 0 or last <= 0:
            return np.nan
        return float(np.log(last / float(prior.iloc[-1])))
    raise ValueError(f"unknown aggregation {how!r}")
