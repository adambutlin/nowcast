"""
dfm/panel.py — the mixed-frequency observation panel.

32 series in 6 economic blocks (spec §1.6), plus CPI as MoM log-change and a
3-series observed exogenous reg block. Every series loads on the pervasive
`global` factor; `block` names the second factor it loads on under the
structured-loading arms (E5/E6).

`freq` decides how a month's value is formed at an as-of timestamp:
  daily      partial-month aggregation over days <= asof   (dfm.daily)
  monthly    the whole month, visible only once published  (release calendar)
  quarterly  Mariano-Murasawa 3-month aggregation; intra-quarter months missing

Only five series are genuinely daily at source — verified against FRED on
2026-08-19. uk_gilt_10y, copper_price and wheat_price are monthly at source
despite the registry describing them as market rates, and gas_eu blends daily
TTF (2017+) onto IMF monthly (1960+), so it stays monthly rather than silently
losing 1997-2016. See the comment in dfm/daily.py FETCHERS.

No LSEG series appear here: a multi-hour sweep must not depend on the Workspace
desktop app being open (spec §1.7).
"""
import numpy as np
import pandas as pd

import factors as F
from dfm import config as C
from dfm import daily as D
from dfm import release_calendar as RC

BLOCKS = ["activity", "nominal", "energy", "supply", "fx", "food"]

# name -> block, transform, frequency
PANEL_SERIES = {
    # ── activity (7) ─────────────────────────────────────────────────────────
    "uk_quarterly_gdp":  dict(block="activity", transform="yoy",    freq="quarterly"),
    "uk_retail_sales":   dict(block="activity", transform="yoy",    freq="monthly"),
    "uk_vacancies":      dict(block="activity", transform="yoy",    freq="monthly"),
    "uk_awg":            dict(block="activity", transform="yoy",    freq="monthly"),
    "uk_house_prices":   dict(block="activity", transform="yoy",    freq="monthly"),
    "vix":               dict(block="activity", transform="level",  freq="daily"),
    "move_index":        dict(block="activity", transform="level",  freq="monthly"),
    # ── nominal (8) ──────────────────────────────────────────────────────────
    "uk_ppi_output":     dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_ppi_input":      dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_core_cpi":       dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_services_cpi":   dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_rents":          dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_be5":            dict(block="nominal",  transform="level",  freq="monthly"),
    "uk_gilt_10y":       dict(block="nominal",  transform="diff",   freq="monthly"),
    "slope_2s10s":       dict(block="nominal",  transform="level",  freq="monthly"),
    # ── energy (5) ───────────────────────────────────────────────────────────
    "oil_brent":         dict(block="energy",   transform="logret", freq="daily"),
    "gas_eu":            dict(block="energy",   transform="logret", freq="monthly"),
    "gas_hh":            dict(block="energy",   transform="logret", freq="daily"),
    "imf_all_commodity": dict(block="energy",   transform="logret", freq="monthly"),
    "oil_vol_6m":        dict(block="energy",   transform="level",  freq="monthly"),
    # ── supply (6) ───────────────────────────────────────────────────────────
    "deep_sea_freight":  dict(block="supply",   transform="logret", freq="monthly"),
    "semiconductors_ppi":dict(block="supply",   transform="logret", freq="monthly"),
    "chemicals_ppi":     dict(block="supply",   transform="logret", freq="monthly"),
    "metals_index":      dict(block="supply",   transform="logret", freq="monthly"),
    "copper_price":      dict(block="supply",   transform="logret", freq="monthly"),
    "timber_price":      dict(block="supply",   transform="logret", freq="monthly"),
    # ── fx (3) ───────────────────────────────────────────────────────────────
    "gbp_eer":           dict(block="fx",       transform="diff",   freq="monthly"),
    "gbpusd":            dict(block="fx",       transform="logret", freq="daily"),
    "gbp_eur":           dict(block="fx",       transform="logret", freq="daily"),
    # ── food (3) — the only genuinely new channel vs production ──────────────
    "food_price_index":  dict(block="food",     transform="logret", freq="monthly"),
    "wheat_price":       dict(block="food",     transform="logret", freq="monthly"),
    "vegetable_oil_price": dict(block="food",   transform="logret", freq="monthly"),
}

# Observed exogenous shifters. Sparse, mostly-zero administered events cannot
# identify a latent factor, so they enter the measurement equation directly.
REG_SERIES = ["mpc_rate_change", "ofgem_cap_delta", "budget_event"]

TARGET_SERIES = "cpi_index"


def block_members():
    """block -> list of series names loading on that block factor."""
    out = {b: [] for b in BLOCKS}
    for name, spec in PANEL_SERIES.items():
        out[spec["block"]].append(name)
    return out


def all_series():
    """Every series the release calendar must cover."""
    return [TARGET_SERIES] + list(PANEL_SERIES) + list(REG_SERIES)


_CACHE = {}


def load_calendar_cached():
    """The dated release table, loaded once per process."""
    if "cal" not in _CACHE:
        _CACHE["cal"] = RC.load_calendar()
    return _CACHE["cal"]


def _monthly_matrix(names, start=None):
    """Month-end matrix for the natively-monthly/quarterly series (raw, NO pub-lag).

    Publication timing is applied by `asof` from the dated calendar, so the
    integer pub_lag shift must NOT also be applied here — that would double-count
    the delay.
    """
    key = ("monthly", tuple(sorted(names)))
    if key in _CACHE:
        return _CACHE[key]
    raw, status = F.build_matrix(names=list(names))
    dead = [n for n, s in status.items() if s == "unavailable"]
    if dead:
        raise RuntimeError(f"panel series unavailable: {dead}")
    raw = raw[raw.index.year >= (start or C.PANEL_START)].resample("ME").last()
    _CACHE[key] = raw
    return raw


def _release_lookup():
    key = "rel"
    if key not in _CACHE:
        cal = load_calendar_cached()
        _CACHE[key] = {(r.series, pd.Timestamp(r.ref_period).to_period("M")): r.release_ts
                       for r in cal.itertuples()}
    return _CACHE[key]


def _released_by(series, ref_month, asof_ts):
    """Has `series`' observation of `ref_month` been published by `asof_ts`?"""
    ts = _release_lookup().get((series, pd.Timestamp(ref_month).to_period("M")))
    if ts is None:
        # No calendar row (e.g. a month outside the built range): fall back to the
        # documented offset rather than silently admitting the observation.
        ts, _ = RC.series_release(series, ref_month)
    return pd.Timestamp(ts) <= pd.Timestamp(asof_ts)


def asof(ref_month, asof_ts, start=None):
    """
    Observation matrix visible at `asof_ts`, for months up to and including
    `ref_month`. Unpublished cells are NaN — the Kalman filter skips them.

    Daily series are aggregated over days <= asof_ts within their month
    (dfm.daily.partial_month). Quarterly series carry their value in the last
    month of the quarter, with intra-quarter months NaN (Mariano-Murasawa).
    """
    ref_month = pd.Timestamp(ref_month) + pd.offsets.MonthEnd(0)
    asof_ts = pd.Timestamp(asof_ts)
    names = list(PANEL_SERIES) + list(REG_SERIES)

    monthly_names = [n for n in names if PANEL_SERIES.get(n, {}).get("freq") != "daily"]
    monthly = _monthly_matrix(monthly_names, start=start)
    index = monthly.index[monthly.index <= ref_month]

    cols = {}
    for name in names:
        spec = PANEL_SERIES.get(name, dict(freq="monthly", transform="level"))
        if spec["freq"] == "daily":
            series = D.load_daily(name)
            how = "level" if spec["transform"] in ("level", "diff") else "logret"
            vals = [D.partial_month(series, m, asof_ts, how=how) for m in index]
            col = pd.Series(vals, index=index, dtype=float)
            if spec["transform"] == "diff":
                col = col.diff()
        else:
            col = monthly[name].reindex(index).astype(float) if name in monthly.columns \
                else pd.Series(np.nan, index=index)
        # publication mask — the whole point of the module
        mask = [_released_by(name, m, asof_ts) for m in index]
        col = col.where(pd.Series(mask, index=index), np.nan)
        cols[name] = col

    out = pd.DataFrame(cols, index=index)

    # Reg series are event dummies: NaN inside their observed span means "no event"
    # (0), but NOT past their last observation — that is how a +13% Ofgem cap reset
    # entered the July-2026 nowcast as a 0 (docs/postmortem_2026_07.md §3).
    for rf in REG_SERIES:
        if rf in out.columns and rf in monthly.columns:
            observed = monthly[rf].dropna()
            if len(observed):
                span = out.index <= observed.index[-1]
                out.loc[span, rf] = out.loc[span, rf].fillna(0.0)

    # Quarterly: keep the value only in the quarter's final month.
    for name, spec in PANEL_SERIES.items():
        if spec["freq"] == "quarterly" and name in out.columns:
            keep = out.index.month.isin([3, 6, 9, 12])
            out.loc[~keep, name] = np.nan
    return out


def standardize(mat, through):
    """Z-score with means/sds computed on rows <= `through` only (causal)."""
    train = mat.loc[:pd.Timestamp(through)]
    mu = train.mean()
    sd = train.std().replace(0.0, 1.0).fillna(1.0)
    return (mat - mu) / sd, mu, sd


def staleness_report(asof_ts):
    """Series whose last observation is >6 months before `asof_ts`."""
    mat = asof(asof_ts, asof_ts)
    rows = []
    for name in mat.columns:
        obs = mat[name].dropna()
        last = obs.index[-1] if len(obs) else pd.NaT
        stale = np.nan if pd.isna(last) else (pd.Timestamp(asof_ts).to_period("M")
                                              - last.to_period("M")).n
        rows.append(dict(series=name, last_obs=last, months_stale=stale,
                         status="DEAD" if (pd.isna(stale) or stale > 6) else "LIVE"))
    return pd.DataFrame(rows).set_index("series")
