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

DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FRED_KEY    = os.getenv("FRED_API_KEY")
BOE_BASE    = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
LONG_START  = "1989-01-01"          # CPI YoY (D7G7) starts 1989; pre-1992 used for warmup


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def _fred(series_id, start=None):
    """FRED series -> month-end series. Requires FRED_API_KEY."""
    from fredapi import Fred
    s = Fred(api_key=FRED_KEY).get_series(series_id, observation_start=start or LONG_START)
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


def _rents_lag1():
    """
    ONS L522.M rents YoY lagged 1 month, extended by 1 forward row.

    The shift(1) makes 2026-04-30 hold March YoY, leaving May with no entry.
    Appending the unshifted current-month YoY at +1 month means _nowcast_row's
    ffill picks up April YoY (not March) for the May nowcast date.
    """
    raw = _dbnomics("ONS", "MM23", "L522.M")
    yoy = raw.pct_change(12).mul(100)
    shifted = yoy.shift(1)
    # Add one-month-ahead row with the most-recently-published rents YoY
    next_idx = shifted.index[-1] + pd.DateOffset(months=1)
    extension = pd.Series([yoy.iloc[-1]], index=[next_idx])
    return pd.concat([shifted, extension]).resample("ME").last()


def _ons_timeseries(code, section):
    """
    Fetch a monthly ONS timeseries via the ONS website JSON API.

    section examples:
      'employmentandlabourmarket/peopleinwork/earningsandworkinghours'
      'economy/inflationandpriceindices'
    Returns month-end pd.Series of float values.
    """
    url = (f"https://www.ons.gov.uk/{section}"
           f"/timeseries/{code}/data")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    months = r.json().get("months", [])
    if not months:
        return pd.Series(dtype=float)
    rows = []
    month_map = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
                 "July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
    for m in months:
        try:
            yr  = int(m["year"])
            mon = month_map.get(m["month"], 0)
            val = float(m["value"].replace(",", ""))
            if mon:
                rows.append((pd.Timestamp(yr, mon, 1) + pd.offsets.MonthEnd(0), val))
        except (ValueError, KeyError):
            continue
    if not rows:
        return pd.Series(dtype=float)
    idx, vals = zip(*rows)
    return pd.Series(vals, index=idx, dtype=float).sort_index()


def _ons_awe_kab9():
    """ONS AWE KAB9 (whole economy weekly pay, SA level £) with FRED fallback."""
    try:
        s = _ons_timeseries(
            "KAB9",
            "employmentandlabourmarket/peopleinwork/earningsandworkinghours",
        )
        if len(s) > 24:
            return s
    except Exception:
        pass
    return _fred("LCEAMN01GBM661S")


def _ons_vacancies():
    """
    ONS VACS01 (Job Vacancy Survey) AP2Y — total vacancies, SA, thousands.

    Fetches direct from ONS xlsx (bypasses dbnomics ingestion lag).
    The series is a 3-month rolling average; each row is labelled by the
    last month of the 3-month window, resampled to month-end.
    Falls back to dbnomics if the ONS file is not yet published.
    """
    import io, re

    months_3  = ["jan","feb","mar","apr","may","jun",
                 "jul","aug","sep","oct","nov","dec"]
    months_map = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
                  "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

    base = ("https://www.ons.gov.uk/file?uri=/employmentandlabourmarket"
            "/peoplenotinwork/unemployment/datasets"
            "/vacanciesandunemploymentvacs01/current/vacs01{mon}{yr}.xlsx")

    def parse_3m(s):
        m = re.match(r"(\w{3})-\s*(\w{3})\s+(\d{4})", str(s))
        if m:
            end_m = months_map.get(m.group(2))
            if end_m:
                return (pd.Timestamp(int(m.group(3)), end_m, 1)
                        + pd.offsets.MonthEnd(0))
        return pd.NaT

    now = pd.Timestamp.now()
    for offset in range(4):
        ts  = now - pd.DateOffset(months=offset)
        url = base.format(mon=months_3[ts.month - 1], yr=ts.year)
        try:
            r = requests.get(url, timeout=25,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            xl = pd.ExcelFile(io.BytesIO(r.content))
            if "VACS01" not in xl.sheet_names:
                continue
            df = xl.parse("VACS01", header=None)
            idx  = df.iloc[6:, 0].apply(parse_3m)
            vacs = pd.to_numeric(df.iloc[6:, 2], errors="coerce")
            s    = pd.Series(vacs.values, index=idx).dropna()
            if len(s) > 0:
                return s.resample("ME").last()
        except Exception:
            continue

    return _dbnomics("ONS", "UNEM", "AP2Y.M")


def _gas_eu_ttf():
    """
    European natural gas: daily TTF front-month (2017+) blended onto IMF monthly
    (1960+). TTF is rescaled to IMF units over the overlap period so log-returns
    are continuous across the splice point.
    """
    imf = _fred("PNGASEUUSDM")
    try:
        import yfinance as yf
        raw = yf.download("TTF=F", start="2017-01-01", auto_adjust=True,
                          progress=False)
        c = (raw[("Close", "TTF=F")] if isinstance(raw.columns, pd.MultiIndex)
             else raw["Close"])
        ttf = c.resample("ME").last().dropna()
        overlap = imf.index.intersection(ttf.index)
        scale = (imf[overlap].mean() / ttf[overlap].mean()
                 if len(overlap) >= 12 else 1.0)
        combined = imf.copy()
        combined[ttf.index] = ttf * scale
        return combined.dropna()
    except Exception:
        return imf


def _cpi_yoy_long():
    """UK CPI YoY from 1956: OECD GBRCPIALLMINMEI (index) YoY%, spliced with ONS D7G7 (1989+)."""
    try:
        oecd = _fred("GBRCPIALLMINMEI", start="1948-01-01")
        oecd_yoy = oecd.pct_change(12).mul(100).resample("ME").last()
    except Exception:
        oecd_yoy = pd.Series(dtype=float)

    try:
        ons = _dbnomics("ONS", "MM23", "D7G7.M")
        ons = ons.resample("ME").last()
    except Exception:
        ons = pd.Series(dtype=float)

    if len(ons) > 0 and len(oecd_yoy) > 0:
        # Rescale OECD to match ONS units over first 24-month overlap
        overlap = oecd_yoy.index.intersection(ons.dropna().index)[:24]
        if len(overlap) >= 6:
            ratio = ons[overlap].mean() / oecd_yoy[overlap].mean()
        else:
            ratio = 1.0
        combined = (oecd_yoy * ratio).copy()
        combined.update(ons)          # ONS overwrites OECD wherever available
        return combined.dropna()

    return (ons if len(ons) > 0 else oecd_yoy).dropna()


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


def _gbp_eur():
    gbp = _fred("DEXUSUK")                              # USD per GBP
    eur = _fred("DEXUSEU").reindex(gbp.index).ffill()  # USD per EUR, aligned
    return (gbp / eur).dropna()                         # EUR per GBP cross-rate


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
    "cpi_yoy_long": dict(
        fetch=_cpi_yoy_long,
        transform="level", pub_lag=1, candidate=False, csv="cpi_yoy_long.csv",
        note="UK CPI YoY 1956+: OECD GBRCPIALLMINMEI spliced with ONS D7G7 (1989+). "
             "For extended training. Use --target cpi_yoy_long with --train-from 1956."),
    "cpi_3m_chg": dict(
        fetch=None, transform="level", pub_lag=0, candidate=True, csv="cpi_3m_chg.csv",
        note="3-month change in lagged CPI YoY: target.shift(1).diff(3). "
             "Computed in main.py from the target column; pub_lag=0 (uses cpi_yoy T-1 to T-4, "
             "all released before CPI(T)). Candidate=True: subject to SHAP screening so it "
             "can be dropped if it adds no signal beyond oil_brent/gas_eu. "
             "Drop data/cpi_3m_chg.csv to override."),

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
        fetch=_gas_eu_ttf, transform="logret",
        pub_lag=0, candidate=True, csv="gas_eu.csv",
        note="European natural gas: daily TTF front-month (2017+, yfinance TTF=F) "
             "rescaled onto IMF PNGASEUUSDM (1960+). pub_lag=0: daily market price "
             "available same day. Override with data/gas_eu.csv."),
    "uk_gilt_10y": dict(
        fetch=lambda: _fred("IRLTLT01GBM156N", start="1960-01-01"),
        transform="diff", pub_lag=0, candidate=True, csv="uk_gilt_10y.csv",
        note="UK 10Y government bond yield (FRED IRLTLT01GBM156N) from 1960. "
             "diff transform for stationarity. pub_lag=0: daily market rate."),
    "oil_vol_6m": dict(
        fetch=lambda: np.log(_fred("DCOILBRENTEU")).diff().rolling(6).std(),
        transform="level", pub_lag=0, candidate=True, csv="oil_vol_6m.csv",
        note="6m rolling std of Brent log-return. Derived from DCOILBRENTEU. pub_lag=0."),
    "gbpusd_vol_6m": dict(
        fetch=lambda: np.log(_fred("DEXUSUK")).diff().rolling(6).std(),
        transform="level", pub_lag=0, candidate=True, csv="gbpusd_vol_6m.csv",
        note="6m rolling std of GBP/USD log-return. Import uncertainty proxy. pub_lag=0."),
    # ── 3-month cumulative log-returns: capture sustained cost-push channel ────
    # Monthly logret shows the period change; 3m cumulative captures whether
    # prices are still elevated 3 months after a spike (CPI lag ≈ 2-4 months).
    "oil_brent_3m": dict(
        fetch=lambda: np.log(_fred("DCOILBRENTEU")).diff(3),
        transform="level", pub_lag=0, candidate=True, csv="oil_brent_3m.csv",
        note="Brent 3-month cumulative log-return. Captures sustained energy cost-push "
             "vs one-period spike. Complements oil_brent (1m). pub_lag=0."),
    "gas_eu_3m": dict(
        fetch=lambda: _gas_eu_ttf().pipe(np.log).diff(3),
        transform="level", pub_lag=0, candidate=True, csv="gas_eu_3m.csv",
        note="European gas 3-month cumulative log-return (TTF/IMF blend). Captures "
             "sustained gas cost-push with 2-4 month CPI transmission lag. pub_lag=0."),
    "gbpusd_3m": dict(
        fetch=lambda: np.log(_fred("DEXUSUK")).diff(3),
        transform="level", pub_lag=0, candidate=True, csv="gbpusd_3m.csv",
        note="GBP/USD 3-month cumulative log-return. Import price pressure proxy "
             "over quarter horizon. Complements gbpusd (1m). pub_lag=0."),

    # ── GBP cross-rates and effective exchange rate ──────────────────────────
    # GBP/EUR separates true UK import-price pressure from USD-cycle effects.
    # A USD depreciation episode raises oil_brent (priced in USD) and lowers
    # gbpusd (GBP strengthens vs USD) simultaneously, creating partially
    # offsetting signals. GBP/EUR isolates the Europe-UK trade channel which
    # directly sets import prices for ~40% of UK goods trade.
    "gbp_eur": dict(
        fetch=_gbp_eur,
        transform="logret", pub_lag=0, candidate=True, csv="gbp_eur.csv",
        note="GBP/EUR cross-rate (FRED DEXUSUK / DEXUSEU, EUR per GBP). 1999+. "
             "pub_lag=0: daily market rate. Isolates Europe↔UK import price channel "
             "from global USD cycle. UK ~40% of goods trade with EU. "
             "Falls back to data/gbp_eur.csv."),
    "gbp_eer": dict(
        fetch=lambda: _fred("RBGBBIS"),
        transform="diff", pub_lag=0, candidate=True, csv="gbp_eer.csv",
        note="UK real broad effective exchange rate, BIS (FRED RBGBBIS, 2020=100). "
             "1994+. pub_lag=0: monthly index from BIS. Captures import price "
             "pressure across ALL UK trading partners, not just USD or EUR. "
             "diff transform for stationarity. Complement to gbpusd (bilateral USD)."),

    # ── Tech / semiconductor input costs ─────────────────────────────────────
    "semiconductors_ppi": dict(
        fetch=lambda: _fred("PCU334413334413"),
        transform="logret", pub_lag=0, candidate=True, csv="semiconductors_ppi.csv",
        note="US BLS PPI for semiconductor devices (FRED PCU334413334413, 1967+). "
             "pub_lag=0: globally priced in USD. Semiconductors are key input to UK "
             "electronics, autos (EV), machinery. Persistent deflationary trend "
             "(Moore's Law) interrupted by supply shocks (2020-22 chip shortage). "
             "Falls back to data/semiconductors_ppi.csv."),
    "battery_metals_proxy": dict(
        fetch=None, transform="level",
        pub_lag=0, candidate=True, csv="battery_metals_proxy.csv",
        note="Rare earth / battery metal input costs proxy. No free monthly series. "
             "Recommended sources: "
             "(1) Benchmark Mineral Intelligence lithium carbonate price (subscription); "
             "(2) USGS Mineral Commodity Summaries (annual only); "
             "(3) LME cobalt official price (LME website, monthly avg). "
             "Covers: lithium carbonate, cobalt sulfate, graphite (anode-grade), "
             "silicon metal, neodymium (NdFeB magnets), dysprosium. "
             "pub_lag=0 for futures/spot; pub_lag=1 for USGS. "
             "Drop data/battery_metals_proxy.csv [date, value=index or price_level]."),

    # ── Global shipping / supply chain ────────────────────────────────────────
    "deep_sea_freight": dict(
        fetch=lambda: _fred("PCU483111483111"),
        transform="logret", pub_lag=0, candidate=True, csv="deep_sea_freight.csv",
        note="US BLS PPI for deep sea freight transportation (FRED PCU483111483111, "
             "1988+). pub_lag=0. Globally priced — reflects container and bulk "
             "shipping cost changes. Strong correlation with Baltic Dry Index (BDI). "
             "Supply-chain bottleneck signal: rose 3× in 2021-22, fell 2023. "
             "Transmits into UK goods CPI with 2-4 month lag. "
             "Fallback: data/deep_sea_freight.csv."),
    "global_supply_chain_pressure": dict(
        fetch=None, transform="level",
        pub_lag=0, candidate=True, csv="global_supply_chain_pressure.csv",
        note="NY Fed Global Supply Chain Pressure Index (GSCPI). Standardised "
             "composite of BDI, air freight, PMI backlogs, cross-country PPI "
             "divergence. Monthly, 1997+. Not on FRED; fetch from NY Fed directly: "
             "https://www.newyorkfed.org/research/policy/gscpi (Excel download). "
             "pub_lag=0. Validated predictor of global goods inflation 1-3 months "
             "ahead. Drop data/global_supply_chain_pressure.csv [date, value=index]."),

    # ── Global commodity input costs: pub_lag=0 (market prices, contemporaneous) ─
    # Key non-energy intermediary inputs for UK manufacturing supply chain.
    # All individual metals confirmed on FRED (PCOPPUSDM, PALUMUSDM, PNICKUSDM,
    # PZINCUSDM, PIORECRUSDM). Transmission lag to UK CPI: 1-3 months via
    # import→PPI→CPI channel. IMF composite index not available on FRED;
    # metals_index computed as equal-weight average log-return across 5 metals.
    "metals_index": dict(
        fetch=lambda: pd.concat([
            np.log(_fred("PCOPPUSDM")).diff(),
            np.log(_fred("PALUMUSDM")).diff(),
            np.log(_fred("PNICKUSDM")).diff(),
            np.log(_fred("PZINCUSDM")).diff(),
            np.log(_fred("PIORECRUSDM")).diff(),
        ], axis=1).mean(axis=1, skipna=True),
        transform="level", pub_lag=0, candidate=True, csv="metals_index.csv",
        note="Equal-weight avg log-return: copper, aluminium, nickel, zinc, iron ore "
             "(FRED PCOPPUSDM/PALUMUSDM/PNICKUSDM/PZINCUSDM/PIORECRUSDM). 1992+. "
             "pub_lag=0. Covers UK manufacturing inputs: autos, construction, packaging, "
             "industrial goods. Falls back to data/metals_index.csv."),
    "copper_price": dict(
        fetch=lambda: _fred("PCOPPUSDM"), transform="logret",
        pub_lag=0, candidate=True, csv="copper_price.csv",
        note="LME copper grade-A cathode spot (FRED PCOPPUSDM, USD/metric ton). "
             "pub_lag=0. Leading economic cycle indicator; UK key input: construction, "
             "wiring, machinery. 1992+."),
    "nickel_price": dict(
        fetch=lambda: _fred("PNICKUSDM"), transform="logret",
        pub_lag=0, candidate=True, csv="nickel_price.csv",
        note="LME nickel spot (FRED PNICKUSDM, USD/metric ton). pub_lag=0. "
             "Stainless steel, battery production. 1980+."),
    "iron_ore_price": dict(
        fetch=lambda: _fred("PIORECRUSDM"), transform="logret",
        pub_lag=0, candidate=True, csv="iron_ore_price.csv",
        note="Iron ore (FRED PIORECRUSDM, USD/dry metric ton). pub_lag=0. "
             "Steel production input — upstream of construction and auto costs. 1980+."),
    "timber_price": dict(
        fetch=lambda: _fred("WPU081"), transform="logret",
        pub_lag=0, candidate=True, csv="timber_price.csv",
        note="US BLS PPI for lumber & wood products (FRED WPU081, index 1982=100). "
             "pub_lag=0. Globally priced commodity — strong correlation with UK "
             "construction material costs (Spearman ~0.7). 1926+. "
             "Fallback: data/timber_price.csv."),
    "chemicals_ppi": dict(
        fetch=lambda: _fred("WPU061"), transform="logret",
        pub_lag=0, candidate=True, csv="chemicals_ppi.csv",
        note="US BLS PPI for industrial chemicals and allied products (FRED WPU061, "
             "index 1982=100). pub_lag=0. Globally priced (USD-denominated feedstock "
             "markets). Proxy for UK chemical intermediary import costs. 1933+. "
             "Transmits into consumer goods, food packaging, plastics. "
             "Fallback: data/chemicals_ppi.csv."),

    # ── UK domestic activity and costs: pub_lag=1 ────────────────────────────
    "uk_monthly_gdp": dict(
        fetch=lambda: _fred("GBRPROINDMISMEI"), transform="yoy",
        pub_lag=1, candidate=True, csv="uk_monthly_gdp.csv",
        note="UK industrial production index, SA, YoY% (FRED GBRPROINDMISMEI, OECD). "
             "pub_lag=1: released ~4-6 weeks after reference month, before CPI. "
             "Activity proxy — above-trend output → domestic demand → services CPI. "
             "True monthly GDP (ONS ABMI series) available as CSV drop-in. "
             "Drop data/uk_monthly_gdp.csv [date, value=YoY%] for exact GDP."),
    "uk_awg": dict(
        fetch=lambda: _ons_awe_kab9(),
        transform="yoy", pub_lag=1, candidate=True, csv="uk_awg.csv",
        note="ONS AWE: Whole Economy weekly pay, SA level (£), series KAB9. "
             "YoY transform gives nominal wage growth %. 2000+, current to ~6 weeks lag. "
             "pub_lag=1: ONS AWE released same window as CPI. "
             "Critical for services CPI — wages dominate domestic services inflation. "
             "Falls back to FRED LCEAMN01GBM661S (OECD, 1990+, slightly stale). "
             "Override: data/uk_awg.csv [date, value=YoY%]."),
    "uk_ppi_input": dict(
        fetch=None, transform="yoy",
        pub_lag=1, candidate=True, csv="uk_ppi_input.csv",
        note="ONS PPI Input prices YoY (materials/fuels purchased by UK manufacturers). "
             "pub_lag=1: released ~3-4 weeks after reference month. "
             "Direct cost-push signal: input→output→CPI with 1-3m lag. "
             "No free live API: ONS timeseries API decommissioned; MM22 dbnomics unavailable. "
             "Download from ONS: https://www.ons.gov.uk/economy/inflationandpriceindices/"
             "bulletins/producerpriceinflationnewformat/latest (series K37R, total input). "
             "Drop data/uk_ppi_input.csv [date, value=YoY%]."),
    "uk_ftse250": dict(
        fetch=lambda: _yf("^FTMC"), transform="logret",
        pub_lag=0, candidate=True, csv="uk_ftse250.csv",
        note="FTSE 250 mid-cap index (yfinance ^FTMC, 1990+). pub_lag=0. "
             "UK domestic corporate profit proxy: ~80% of FTSE 250 revenues are UK-based "
             "vs ~25% for FTSE 100 (which is dominated by BP, HSBC, miners, international). "
             "Rising FTSE 250 → improving UK domestic demand and corporate pricing power "
             "→ demand-pull inflation signal with 2-4 month lag to CPI. "
             "Different signal to uk_be5 (expectations) or uk_gilt_10y (rates)."),
    "uk_ftse100": dict(
        fetch=lambda: _yf("^FTSE"), transform="logret",
        pub_lag=0, candidate=True, csv="uk_ftse100.csv",
        note="FTSE 100 large-cap index (yfinance ^FTSE, 1990+). pub_lag=0. "
             "Captures global earnings cycle and UK financial conditions. "
             "~75% revenues international — use uk_ftse250 for domestic profit signal. "
             "FTSE 100 vs FTSE 250 divergence can signal domestic/international decoupling. "
             "Complement to uk_be5 (inflation expectations) and gbpusd (FX conditions)."),

    # ── Global food commodity prices: pub_lag=0 ──────────────────────────────
    # COLLINEARITY WARNING: food IS ~10-15% of UK CPI basket. Using CONTEMPORANEOUS
    # UK food CPI as a factor would be circular (leakage). However, global commodity
    # prices are appropriate because:
    # (1) They are set on futures markets BEFORE retail prices adjust (2-4 month lag).
    # (2) They are pub_lag=0 (available before CPI release), not published with CPI.
    # (3) Models are evaluated OOS — SHAP screen validates incremental signal.
    # Risk check: run probe_leakage() on these factors before live use.
    "food_price_index": dict(
        fetch=lambda: _fred("PFOODINDEXM"),
        transform="logret", pub_lag=0, candidate=True, csv="food_price_index.csv",
        note="IMF Food and Beverage Price Index (FRED PFOODINDEXM, USD, 2016=100, 1992+). "
             "pub_lag=0: IMF IFS monthly. Covers cereals, vegetable oils, meat, seafood, "
             "sugar, bananas, oranges. UK food CPI transmission lag ~2-4 months. "
             "COLLINEARITY WARNING: food is ~10-15% of UK CPI; use only if "
             "probe_leakage() confirms no contemporaneous circular signal. "
             "Fallback: data/food_price_index.csv."),
    "wheat_price": dict(
        fetch=lambda: _fred("PWHEAMTUSDM"),
        transform="logret", pub_lag=0, candidate=True, csv="wheat_price.csv",
        note="IMF wheat price (FRED PWHEAMTUSDM, USD/metric ton, 1980+). pub_lag=0. "
             "Wheat → flour → bread/pasta: 3-5 month transmission to UK retail. "
             "Better granularity than food_price_index for bread/cereals sub-category. "
             "2022 Ukraine war spike (+90%) captured here 3 months before UK food CPI. "
             "Fallback: data/wheat_price.csv."),
    "vegetable_oil_price": dict(
        fetch=lambda: _fred("PSOYBUSDM"),
        transform="logret", pub_lag=0, candidate=True, csv="vegetable_oil_price.csv",
        note="IMF soybean price (FRED PSOYBUSDM, USD/metric ton, 1980+). pub_lag=0. "
             "Proxy for vegetable oil complex (soy, palm, rapeseed): key UK food "
             "manufacturing input (cooking oil, margarine, processed food). "
             "Also use PVOILUSDM (palm oil) if available. Fallback: data/vegetable_oil_price.csv."),

    # ── US macro proxies via FRED: pub_lag=0 (released before UK CPI) ───────
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
        fetch=_rents_lag1,
        transform="level",
        pub_lag=0, candidate=False, csv="uk_rents_lag1.csv",
        note="ONS rents (L522.M) YoY lagged 1 month — real-time safe. pub_lag=0 "
             "(the 1-month lag is baked into the fetch). Extended by one forward row "
             "so _nowcast_row ffills the most-recently-published rents YoY. "
             "Spearman rho=0.922 at lag=1."),
    "uk_vacancies": dict(
        fetch=_ons_vacancies, transform="logret",
        pub_lag=1, candidate=True, csv="uk_vacancies.csv",
        note="ONS vacancies (thousands), 3-month rolling SA (AP2Y). "
             "Fetched directly from ONS VACS01 xlsx (bypasses dbnomics lag). "
             "pub_lag=1: vacancy survey published ~5 weeks after reference month."),
    "uk_house_prices": dict(
        fetch=lambda: (
            _fred("QGBR628BIS")           # BIS quarterly real UK residential HPI
            .resample("ME").ffill()        # forward-fill quarters → monthly
        ),
        transform="yoy", pub_lag=2, candidate=True, csv="uk_house_prices.csv",
        note="BIS real residential property price index for UK (FRED QGBR628BIS, "
             "2010=100, 1968+). Quarterly source forward-filled to monthly. "
             "pub_lag=2: ONS HPI published ~6-8 weeks after reference month. "
             "ONS timeseries API decommissioned; dbnomics ONS/HPSSA unavailable. "
             "Drop data/uk_house_prices.csv [date, value=YoY%] for monthly ONS HPI "
             "(download from: https://www.gov.uk/government/collections/"
             "uk-house-price-index-reports)."),
    "uk_paye": dict(
        fetch=None,
        transform="yoy", pub_lag=1, candidate=True, csv="uk_paye.csv",
        note="HMRC RTI payroll count data (employees on payroll, monthly). "
             "No fetch function: RTI data is not available via a free public API. "
             "NOTE: KAB9 (ONS AWE whole economy weekly pay) is NOT this series — "
             "KAB9 is average weekly earnings and is used for uk_awg. "
             "uk_paye is a placeholder until HMRC RTI API access is added. "
             "Drop data/uk_paye.csv [date, value=payroll count or YoY%]."),

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
        note="ONS PPI Output prices YoY (home sales, series L3DW in MM22). "
             "pub_lag=1: released ~3-4 weeks after reference month. "
             "No free live API available (ONS timeseries API decommissioned). "
             "Download from ONS PPI bulletin (series L3DW). "
             "Drop data/uk_ppi_output.csv [date, value=YoY%]."),
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
