"""
factors.py — pluggable factor registry for UK CPI YoY nowcasting.

Two tiers:
  LIVE   — fetched programmatically (FRED long history, yfinance, dbnomics ONS).
  CSV    — gated/paywalled series. Drop a file in data/<name>.csv with columns
           [date, value] and it is auto-loaded into the registry. This is how
           you "temporarily test new factors": export the series once, drop the
           CSV, and every model + the Shapley screen picks it up automatically.

Every factor returns a monthly (month-end) pd.Series after its `transform`.
Long history (post-1992) comes from FRED spot series, NOT yfinance futures
(which only start ~2007 for Brent, ~2003 for GBP).

Registry entry fields:
  fetch    : callable -> raw pd.Series (DatetimeIndex), or None for CSV-only
  transform: "level" | "yoy" | "mom" | "logret" | "diff"
  pub_lag  : int — months of publication delay. 0 = known before CPI release
             (financial data); 1 = published same month as CPI (ONS releases).
             apply_publication_lags() shifts each factor by this value.
  csv      : filename in data/ that overrides/supplies the series if present
  candidate: True → screened; False → always included
  note     : provenance / how to obtain if gated

Mixed-frequency discipline:
  CPI(T) published ~16th of month T+1.
  pub_lag=0: full month-T value available when CPI(T) is released (oil, FX, VIX, PMI flash).
  pub_lag=1: published same day as or after CPI(T) — use month T-1 value only.
  apply_publication_lags() enforces this before data is fed to models.
"""

import os
import io
import zipfile
import datetime
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
FRED_KEY    = os.getenv("FRED_API_KEY")
BOE_BASE    = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
LONG_START  = "1989-01-01"          # CPI YoY (D7G7) starts 1989; pre-1992 used for warmup


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def _fred(series_id):
    """FRED series -> month-end series. Requires FRED_API_KEY."""
    from fredapi import Fred
    s = Fred(api_key=FRED_KEY).get_series(series_id, observation_start=LONG_START)
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last()


def _yf(ticker):
    import yfinance as yf
    raw = yf.download(ticker, start=LONG_START, auto_adjust=True, progress=False)
    c = (raw[("Close", ticker)] if isinstance(raw.columns, pd.MultiIndex)
         else raw["Close"])
    return c.resample("ME").last()


def _dbnomics(provider, dataset, series):
    from dbnomics import fetch_series
    df = fetch_series(provider, dataset, series)
    df["date"] = pd.to_datetime(df["original_period"])
    s = df.set_index("date")["value"].sort_index()
    return s.resample("ME").last()


def _boe_spot_5y(zip_url):
    r = requests.get(zip_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    frames = []
    for fname in sorted(z.namelist()):
        xl = pd.ExcelFile(z.open(fname))
        if "4. spot curve" not in xl.sheet_names:
            continue
        df = xl.parse("4. spot curve", header=None)
        mats = df.iloc[3, 1:].values.astype(float)
        data = df.iloc[5:].copy()
        data.columns = ["date"] + list(mats)
        data = data[data["date"].apply(
            lambda x: isinstance(x, (pd.Timestamp, datetime.datetime)))]
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date").apply(pd.to_numeric, errors="coerce")
        col_5y = min(mats, key=lambda x: abs(x - 5.0))
        frames.append(data[[col_5y]].rename(columns={col_5y: "y5"}))
    if not frames:
        return pd.Series(dtype=float)
    return pd.concat(frames).sort_index()["y5"]


def _uk_breakeven():
    nom = _boe_spot_5y(BOE_BASE + "glcinflationmonthedata.zip")
    real = _boe_spot_5y(BOE_BASE + "glcrealmonthedata.zip")
    be = (nom - real)
    be.index = be.index.to_period("M").to_timestamp("M")
    return be.resample("ME").last()


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def _apply_transform(s, kind):
    s = s.astype(float)
    if kind == "level":
        return s
    if kind == "yoy":
        return s.pct_change(12) * 100
    if kind == "mom":
        return s.pct_change()
    if kind == "logret":
        return np.log(s).diff()
    if kind == "diff":
        return s.diff()
    raise ValueError(f"unknown transform {kind}")


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# candidate=True  → screened, kept only if Shapley value clears threshold.
# candidate=False → core factor, always kept.
# pub_lag         → publication delay in months (0=contemporaneous, 1=one month lag).

REGISTRY = {
    # ── target ──────────────────────────────────────────────────────────────
    "cpi_yoy": dict(
        fetch=lambda: _dbnomics("ONS", "MM23", "D7G7.M"),
        transform="level", pub_lag=1, candidate=False, csv="cpi_yoy.csv",
        note="ONS CPI All Items YoY (D7G7) via dbnomics. TARGET."),

    # ── core live factors: pub_lag=0 (financial, available before CPI release) ──
    "oil_brent": dict(
        fetch=lambda: _fred("DCOILBRENTEU"), transform="logret",
        pub_lag=0, candidate=False, csv="oil_brent.csv",
        note="FRED Brent spot (DCOILBRENTEU) from 1987. pub_lag=0: end-month price "
             "available before CPI release (~16th of following month)."),
    "gbpusd": dict(
        fetch=lambda: _fred("DEXUSUK"), transform="logret",
        pub_lag=0, candidate=False, csv="gbpusd.csv",
        note="FRED USD/GBP (DEXUSUK) from 1971. pub_lag=0: financial series."),
    "uk_be5": dict(
        fetch=_uk_breakeven, transform="level",
        pub_lag=0, candidate=False, csv="uk_be5.csv",
        note="BoE gilt-implied 5Y breakeven (nominal-real spot). pub_lag=0: daily data."),

    # ── candidate live factors: pub_lag=0 (financial) ───────────────────────
    "vix": dict(
        fetch=lambda: _yf("^VIX"), transform="level",
        pub_lag=0, candidate=True, csv="vix.csv",
        note="CBOE VIX (yfinance ^VIX) from 1990. pub_lag=0: daily financial data."),
    "gas_hh": dict(
        fetch=lambda: _fred("DHHNGSP"), transform="logret",
        pub_lag=0, candidate=True, region="US", csv="gas_hh.csv",
        note="FRED Henry Hub gas (DHHNGSP). US proxy; UK NBP/TTF not free. pub_lag=0. "
             "region=US — excluded from UK-only runs."),
    "gas_eu": dict(
        fetch=lambda: _fred("PNGASEUUSDM"), transform="logret",
        pub_lag=0, candidate=True, csv="gas_eu.csv",
        note="IMF/FRED European natural gas price (PNGASEUUSDM, USD/mmBtu, 1960-). "
             "UK imported LNG proxy — more relevant to UK CPI than Henry Hub. "
             "pub_lag=0. TTF front-month futures (yfinance TTF=F) preferred post-2009; "
             "override by dropping data/gas_eu.csv."),
    "oil_vol_6m": dict(
        fetch=lambda: np.log(_fred("DCOILBRENTEU")).diff().rolling(6).std(),
        transform="level", pub_lag=0, candidate=True, csv="oil_vol_6m.csv",
        note="6m rolling std of Brent log-return. Derived from DCOILBRENTEU. pub_lag=0."),
    "gbpusd_vol_6m": dict(
        fetch=lambda: np.log(_fred("DEXUSUK")).diff().rolling(6).std(),
        transform="level", pub_lag=0, candidate=True, csv="gbpusd_vol_6m.csv",
        note="6m rolling std of GBP/USD log-return. Import uncertainty proxy. pub_lag=0."),

    # ── US macro proxies via FRED: pub_lag=0 (released before UK CPI) ───────
    "us_ism_pmi": dict(
        fetch=lambda: _fred("NAPM"), transform="level",
        pub_lag=0, candidate=True, region="US", csv="us_ism_pmi.csv",
        note="ISM Manufacturing PMI (FRED NAPM). Released 1st business day of T+1. "
             "pub_lag=0. region=US — excluded from UK-only runs."),
    "us_ppi_all": dict(
        fetch=lambda: _fred("PPIACO").pct_change(12) * 100, transform="level",
        pub_lag=0, candidate=True, region="US", csv="us_ppi_all.csv",
        note="US PPI All Commodities YoY (FRED PPIACO pct_change×12). "
             "pub_lag=0. region=US — excluded from UK-only runs."),

    # ── ONS factors: pub_lag=1 (published same day as CPI or after) ─────────
    "uk_rents": dict(
        fetch=lambda: _dbnomics("ONS", "MM23", "L522.M"), transform="yoy",
        pub_lag=1, candidate=True, csv="uk_rents.csv",
        note="ONS actual-rentals CPI sub-component (L522, ~7-8% basket). pub_lag=1: "
             "published same day as CPI — use apply_publication_lags() to avoid "
             "contemporaneous leakage. Leakage lift at lag=0: +0.209pp RMSE."),
    "uk_rents_lag1": dict(
        fetch=lambda: _dbnomics("ONS", "MM23", "L522.M").pct_change(12).mul(100).shift(1),
        transform="level",
        pub_lag=0, candidate=False, csv="uk_rents_lag1.csv",
        note="ONS rents (L522.M) YoY lagged 1 month — real-time safe. pub_lag=0 "
             "(the 1-month lag is baked into the fetch). Spearman rho=0.922 at lag=1."),
    "uk_vacancies": dict(
        fetch=lambda: _dbnomics("ONS", "UNEM", "AP2Y.M"), transform="logret",
        pub_lag=1, candidate=True, csv="uk_vacancies.csv",
        note="ONS vacancies (thousands). pub_lag=1: typically 4-6 weeks after reference month."),
    "uk_house_prices": dict(
        fetch=lambda: _dbnomics("ONS", "HPSSA", "HPI.M"), transform="yoy",
        pub_lag=2, candidate=True, csv="uk_house_prices.csv",
        note="ONS House Price Index. pub_lag=2: published ~6-8 weeks after reference month."),
    "uk_paye": dict(
        fetch=lambda: _dbnomics("ONS", "RTI", "median_pay.M"), transform="yoy",
        pub_lag=1, candidate=True, csv="uk_paye.csv",
        note="ONS PAYE RTI median pay. pub_lag=1."),

    # ── UK CPI components: pub_lag=1 (released same day as headline CPI) ────
    "uk_cpih": dict(
        fetch=lambda: _dbnomics("ONS", "MM23", "L55O.M"), transform="yoy",
        pub_lag=1, candidate=True, csv="uk_cpih.csv",
        note="ONS CPIH All Items (L55O in MM23). pub_lag=1. Verify series code against "
             "ONS MM23 — drop data/uk_cpih.csv [date, value=index] if fetch fails."),
    "uk_services_cpi": dict(
        fetch=lambda: _dbnomics("ONS", "MM23", "D7G9.M"), transform="level",
        pub_lag=1, candidate=True, csv="uk_services_cpi.csv",
        note="ONS Services CPI YoY (D7G9 in MM23, verify code). pub_lag=1. "
             "Services inflation is sticky and a leading indicator of future CPI. "
             "Drop data/uk_services_cpi.csv if fetch fails."),
    "uk_core_cpi": dict(
        fetch=None, transform="level",
        pub_lag=1, candidate=True, csv="uk_core_cpi.csv",
        note="ONS CPI ex food and energy YoY. CSV drop-in only. "
             "Source: ONS CPI release Table 2. Series code in MM23: varies by vintage. "
             "Drop data/uk_core_cpi.csv [date, value=YoY%]."),
    "uk_ppi_output": dict(
        fetch=None, transform="yoy",
        pub_lag=1, candidate=True, csv="uk_ppi_output.csv",
        note="ONS PPI Output prices YoY. pub_lag=1: typically 3-4 weeks after reference "
             "month but same window as CPI. CSV drop-in. Source: ONS PPI release. "
             "Drop data/uk_ppi_output.csv [date, value=index level]."),
    "uk_trimmed_mean_cpi": dict(
        fetch=None, transform="level",
        pub_lag=1, candidate=True, csv="uk_trimmed_mean_cpi.csv",
        note="BoE/ONS trimmed-mean CPI (5% each tail). pub_lag=1. "
             "Source: BoE working papers or ONS experimental stats. "
             "Drop data/uk_trimmed_mean_cpi.csv [date, value=YoY%]."),

    # ── Survey/PMI factors: pub_lag=0 (flash PMI released before CPI) ───────
    "uk_pmi_composite": dict(
        fetch=None, transform="level",
        pub_lag=0, candidate=True, csv="uk_pmi_composite.csv",
        note="S&P Global UK Composite PMI. pub_lag=0: flash released last week of "
             "reference month, final first week of T+1 — before CPI. "
             "Licensed (S&P Global / Markit). Drop data/uk_pmi_composite.csv."),
    "uk_pmi_manufacturing": dict(
        fetch=None, transform="level",
        pub_lag=0, candidate=True, csv="uk_pmi_manufacturing.csv",
        note="S&P Global UK Manufacturing PMI. pub_lag=0. Licensed. "
             "Drop data/uk_pmi_manufacturing.csv [date, value=index]."),
    "uk_pmi_services": dict(
        fetch=None, transform="level",
        pub_lag=0, candidate=True, csv="uk_pmi_services.csv",
        note="S&P Global UK Services PMI. pub_lag=0. Licensed. "
             "Drop data/uk_pmi_services.csv [date, value=index]."),
    "uk_pmi_input_prices": dict(
        fetch=None, transform="level",
        pub_lag=0, candidate=True, csv="uk_pmi_input_prices.csv",
        note="S&P Global UK PMI Input Prices sub-index. pub_lag=0. Licensed. "
             "Cost-push signal 1-2 months ahead of CPI. "
             "Drop data/uk_pmi_input_prices.csv [date, value=index]."),

    # ── Earnings/analyst signals: pub_lag=0 (released before CPI) ────────────
    "ibes_revisions_12m": dict(
        fetch=None, transform="level",
        pub_lag=0, candidate=True, csv="ibes_revisions_12m.csv",
        note="IBES/FactSet UK analyst earnings revision ratio (upgrades/downgrades "
             "trailing 12m). pub_lag=0: daily update. Demand-expectations proxy. "
             "Source: Refinitiv IBES or FactSet. Drop data/ibes_revisions_12m.csv "
             "[date, value=revision_ratio]."),

    # ── Household cost / welfare indices: pub_lag=1 ──────────────────────────
    "uk_hciall": dict(
        fetch=None, transform="yoy",
        pub_lag=1, candidate=True, csv="uk_hciall.csv",
        note="ONS Household Cost Indices: All Households (HCIALL). Experimental "
             "quarterly statistics measuring actual housing costs. pub_lag=1. "
             "Source: ONS HCIS experimental stats release (quarterly — interpolate). "
             "Drop data/uk_hciall.csv [date, value=index level]."),
    "vimes_boots_index": dict(
        fetch=None, transform="yoy",
        pub_lag=1, candidate=True, csv="vimes_boots_index.csv",
        note="Vimes Watchtower Boots Index — tracks relative price of low-cost vs "
             "premium essential goods (inspired by Pratchett's Boots Theory of "
             "socioeconomic unfairness). Measures cost pressure on lower-income "
             "households. pub_lag=1. No standard free source; construct from ONS "
             "Consumer Price Microdata or use JRF Minimum Income Standard. "
             "Drop data/vimes_boots_index.csv [date, value=index or YoY%]."),

    # ── Legacy gated factors (existing) ──────────────────────────────────────
    "uk_infl_swap_1y": dict(
        fetch=None, transform="level", pub_lag=0, candidate=True,
        csv="uk_infl_swap_1y.csv",
        note="UK 1Y RPI/CPI inflation swap. Bloomberg/Refinitiv. pub_lag=0. Drop CSV."),
    "uk_pmi_old": dict(
        fetch=None, transform="level", pub_lag=0, candidate=True,
        csv="uk_pmi_composite.csv",  # same CSV as uk_pmi_composite
        note="Alias for uk_pmi_composite. Use uk_pmi_composite instead."),
    "uk_ashe_pay": dict(
        fetch=None, transform="yoy", pub_lag=1, candidate=True,
        csv="uk_ashe_pay.csv",
        note="ONS ASHE earnings (annual survey). pub_lag=1. Manual export. Drop CSV."),
    "oil_gas_curve_gbp": dict(
        fetch=None, transform="logret", pub_lag=0, candidate=True,
        csv="oil_gas_curve_gbp.csv",
        note="Sterling oil & gas futures curve (front-12m avg in GBP). ICE. pub_lag=0."),
    "ons_econ_activity": dict(
        fetch=None, transform="level", pub_lag=1, candidate=True,
        csv="ons_econ_activity.csv",
        note="ONS Economic Activity & Social Change bulletin index. pub_lag=1. Drop CSV."),
    "sdf_daily_spend": dict(
        fetch=None, transform="yoy", pub_lag=0, candidate=True,
        csv="sdf_daily_spend.csv",
        note="Smart Data Foundry daily spending (monthly agg). pub_lag=0 (real-time). "
             "Gated. Drop CSV."),
}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLICATION LAG APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def apply_publication_lags(df, factor_names, registry=None):
    """
    Apply publication lags to factor columns.

    Each factor in the registry has a pub_lag field:
      0 → financial data available before CPI release (no shift needed)
      1 → ONS factor published same day as CPI (shift by 1 month)
      2 → published 2+ months after reference period (shift by 2)

    After this transform, row T of the returned df contains the
    REAL-TIME INFORMATION SET for forecasting CPI(T): every factor column
    holds the most recent value actually observable before CPI(T) is released.

    Note: BVAR uses its own internal lag structure (shift 1,2,3 within _make_X).
    With pre-shifted factors, BVAR's lag=1 block for a pub_lag=1 factor gives
    factor(T-2) rather than factor(T-1). This is conservative (one extra lag)
    but does eliminate the lag=0 leakage.
    """
    registry = registry or REGISTRY
    df_out = df.copy()
    for f in factor_names:
        entry = registry.get(f, {})
        lag = entry.get("pub_lag", 1)
        if lag > 0:
            df_out[f] = df_out[f].shift(lag)
    return df_out


def screen_candidates(df, target, threshold=0.001):
    """
    Shapley-based candidate factor screening.

    Fits a quick LightGBM on all candidate factors present in df, computes
    mean |SHAP| per factor, and returns those above threshold.

    Args:
        df:        DataFrame containing factor columns and target column.
        target:    Name of the target column in df.
        threshold: Minimum mean |SHAP| to retain a candidate (default 0.001).

    Returns:
        List of factor names (candidates only) with mean |SHAP| >= threshold.
        Non-candidate (core) factors are never in this list.
    """
    import shap
    from lightgbm import LGBMRegressor

    candidates = [n for n in df.columns
                  if n != target and REGISTRY.get(n, {}).get("candidate")]
    if not candidates:
        return []

    sub = df[candidates + [target]].dropna()
    if len(sub) < 30:
        return candidates  # not enough data to screen; keep all

    X = sub[candidates]
    y = sub[target]

    m = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                      num_leaves=4, min_child_samples=30,
                      reg_alpha=2.0, reg_lambda=2.0,
                      random_state=42, verbose=-1)
    m.fit(X, y)

    sv = shap.TreeExplainer(m).shap_values(X)
    importance = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)

    kept = list(importance[importance >= threshold].index)
    dropped = [f for f in candidates if f not in kept]
    if dropped:
        print(f"  [screen_candidates] dropped {dropped} (mean |SHAP| < {threshold})")
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path, transform):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    dcol = cols.get("date", df.columns[0])
    vcol = cols.get("value", df.columns[-1])
    df[dcol] = pd.to_datetime(df[dcol])
    s = df.set_index(dcol)[vcol].sort_index().resample("ME").last()
    return _apply_transform(s, transform)


def load_factor(name):
    """Return (series, status). CSV drop-in takes priority over live fetch."""
    entry = REGISTRY[name]
    csv_path = os.path.join(DATA_DIR, entry["csv"]) if entry.get("csv") else None

    if csv_path and os.path.exists(csv_path):
        try:
            return _load_csv(csv_path, entry["transform"]).rename(name), "csv"
        except Exception as e:
            print(f"  [{name}] CSV load failed: {e}")

    if entry["fetch"] is not None:
        try:
            raw = entry["fetch"]()
            return _apply_transform(raw, entry["transform"]).rename(name), "live"
        except Exception as e:
            print(f"  [{name}] live fetch failed: {e}")

    return pd.Series(dtype=float, name=name), "unavailable"


def build_matrix(names=None, include_unavailable=False):
    """
    Assemble a month-end factor DataFrame.
    Returns (df, status_dict). Unavailable factors are dropped unless requested.
    Note: pub_lags are NOT applied here — call apply_publication_lags() separately.
    """
    if names is None:
        names = list(REGISTRY.keys())
    series, status = [], {}
    for n in names:
        s, st = load_factor(n)
        status[n] = st
        if st != "unavailable" or include_unavailable:
            series.append(s)
    df = pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()
    return df, status


def candidate_factors():
    return [n for n, e in REGISTRY.items() if e.get("candidate")]


def core_factors():
    return [n for n, e in REGISTRY.items()
            if not e.get("candidate") and n != "cpi_yoy"]


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    df, status = build_matrix()
    print("\nFactor availability:")
    for n, st in status.items():
        role = "TARGET" if n == "cpi_yoy" else ("core" if n in core_factors() else "candidate")
        flag = {"live": "✓ live", "csv": "✓ csv", "unavailable": "✗ ----"}[st]
        lag_str = f"pub_lag={REGISTRY[n].get('pub_lag','?')}"
        print(f"  {flag:<8} {role:<9} {lag_str:<10} {n:<25} {REGISTRY[n]['note'][:55]}")
    if len(df):
        print(f"\nMatrix (raw, pre-lag): {df.index.min().date()} → {df.index.max().date()}  "
              f"({df.shape[0]} months, {df.shape[1]} factors)")
        live_facs = [n for n, s in status.items() if s != "unavailable" and n != "cpi_yoy"]
        df_rt = apply_publication_lags(df, live_facs)
        print(f"Matrix (pub-lag applied): {df_rt.dropna().index.min().date()} → "
              f"{df_rt.dropna().index.max().date()}")
