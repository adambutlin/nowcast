"""
Backtest: RAMM-LGBM (CPI MoM + 2Y repricing) vs DFM
Benchmarks: AR(1), market-implied 5Y breakeven, Cleveland Fed 1Y model

Train: 2005-2024  |  Holdout: 2025

Units:
  core_cpi_mom        – monthly decimal  (0.003 = 0.3% MoM)
  two_year_repricing  – pp change in 2Y yield
  CPI_YoY             – percent YoY  (2.5 = 2.5%)

ALFRED real-time discipline:
  CPILFESL and PAYEMS are revised.  For each forecast year yr, training data
  uses the vintage as-of Jan 1 of yr (what was actually known before forecasting).
  Test-period actuals use first-release values (unrevised).
  Market data (Brent, VIX, T5YIE, T10YIE, DGS2) has no meaningful vintages.
  UK ONS CPI and BoE data have no ALFRED equivalent — kept as-is.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fredapi import Fred
from lightgbm import LGBMRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap

warnings.filterwarnings("ignore")

FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY:
    sys.exit("Set FRED_API_KEY environment variable.")

fred      = Fred(api_key=FRED_KEY)
START     = "2005-01-01"
HOLDOUT   = 2025


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fred_m(sid):
    s = fred.get_series(sid)
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last()


def metrics_block(actual, pred, label):
    a = np.asarray(actual, float)
    p = np.asarray(pred,   float)
    ok = ~(np.isnan(a) | np.isnan(p))
    a, p = a[ok], p[ok]
    n = len(a)
    if n == 0:
        return {"label": label, "n": 0}
    rmse    = np.sqrt(mean_squared_error(a, p))
    mae     = mean_absolute_error(a, p)
    dir_acc = np.mean(np.sign(a) == np.sign(p)) * 100
    return dict(label=label, n=n, rmse=rmse, mae=mae, dir_acc=dir_acc)


def print_table(rows):
    print(f"\n  {'Model':<40} {'n':>4} {'RMSE':>10} {'MAE':>10} {'Dir%':>7}")
    print("  " + "-" * 75)
    for r in rows:
        if r["n"] == 0:
            print(f"  {r['label']:<40}  no data")
            continue
        print(f"  {r['label']:<40} {r['n']:>4} {r['rmse']:>10.5f} {r['mae']:>10.5f} {r['dir_acc']:>6.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# AR(1) EXPANDING-WINDOW BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def ar1_backtest(series, min_train=60, start_year=2015):
    """1-step-ahead AR(1) with expanding window, refitted each year."""
    rows = []
    for yr in sorted(series.index.year.unique()):
        if yr < start_year:
            continue
        train = series[series.index.year < yr].dropna()
        test  = series[series.index.year == yr].dropna()
        if len(train) < min_train or len(test) == 0:
            continue
        try:
            res   = ARIMA(train, order=(1, 0, 0)).fit()
            c     = res.params.get("const", 0.0)
            phi   = res.params.get("ar.L1", 0.0)
            # 1-step-ahead: ŷ_t = c + φ·y_{t-1}
            # Use actual previous value (oracle 1-step, standard AR eval)
            prev = train.iloc[-1]
            for date, actual in zip(test.index, test.values):
                rows.append(dict(date=date, actual=actual,
                                 pred=c + phi * prev, year=yr))
                prev = actual          # update with realised value
        except Exception as e:
            print(f"  AR(1) failed for {yr}: {e}")
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# ALFRED VINTAGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

ALFRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def get_alfred_full(series_id, start="2000-01-01"):
    """All vintage rows from ALFRED for series_id (one API call)."""
    params = {
        "series_id": series_id, "api_key": FRED_KEY, "file_type": "json",
        "realtime_start": start, "realtime_end": "9999-12-31",
        "observation_start": start,
    }
    r = requests.get(ALFRED_URL, params=params, timeout=90)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["observations"])
    df["date"]           = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["value"]          = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "realtime_start", "value"]].dropna(subset=["value"])


def as_of_series(alfred_df, realtime_date):
    """Series as it appeared on realtime_date: latest vintage ≤ date per obs."""
    rt = pd.Timestamp(realtime_date)
    avail = alfred_df[alfred_df["realtime_start"] <= rt]
    if avail.empty:
        return pd.Series(dtype=float)
    idx = avail.groupby("date")["realtime_start"].idxmax()
    s = avail.loc[idx].set_index("date")["value"].sort_index()
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last()


def first_release(alfred_df):
    """First-published value for each observation date (pre-revision)."""
    idx = alfred_df.groupby("date")["realtime_start"].idxmin()
    s = alfred_df.loc[idx].set_index("date")["value"].sort_index()
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last()


def build_vintage_data(rt_date, mkt_base, alfred_cpi, alfred_pay):
    """
    Full feature+target matrix for a given real-time date.
    mkt_base: market-data DataFrame (brent_ret, vix, be5, be10, two_year_repricing).
    CPI target and payroll feature use ALFRED vintages as of rt_date.
    """
    pay_rt  = as_of_series(alfred_pay, rt_date).pct_change().shift(1).rename("payroll_growth")
    cpi_rt  = as_of_series(alfred_cpi, rt_date).pct_change().rename("core_cpi_mom")
    df = mkt_base.copy()
    df["payroll_growth"] = pay_rt
    df["brent_vol_6m"]   = df["brent_ret"].rolling(6).std()
    df["vix_change"]     = df["vix"].pct_change()
    df["be_slope"]       = df["be10"] - df["be5"]
    df["regime"]         = (df["vix"] > df["vix"].expanding(12).median()).astype(int)
    df["core_cpi_mom"]   = cpi_rt
    return df


def lgbm_backtest_alfred(target_col, mkt_base, alfred_cpi, alfred_pay, cpi_actuals,
                          start_year=2015, min_train=60):
    """
    Expanding-window LGBM with ALFRED vintages.
    Train vintage: as-of Jan 1 of forecast year (no within-year revisions leak in).
    Test actuals : first-release for CPI; market data for 2Y repricing.
    """
    rows = []
    for yr in sorted(range(start_year, pd.Timestamp.now().year + 2)):
        rt_date = f"{yr}-01-01"
        d = build_vintage_data(rt_date, mkt_base, alfred_cpi, alfred_pay)
        train = d[d.index.year < yr].dropna(subset=FEATURES + [target_col])
        test  = d[d.index.year == yr].dropna(subset=FEATURES)
        if len(train) < min_train or len(test) == 0:
            continue
        m = lgbm()
        m.fit(train[FEATURES], train[target_col])
        preds = m.predict(test[FEATURES])
        for date, pred in zip(test.index, preds):
            actual = (cpi_actuals.get(date, np.nan)
                      if target_col == "core_cpi_mom"
                      else mkt_base["two_year_repricing"].get(date, np.nan))
            if pd.notna(actual):
                rows.append(dict(date=date, actual=float(actual),
                                 pred=float(pred), year=yr))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def ar1_backtest_alfred(alfred_cpi, cpi_actuals, start_year=2015, min_train=60):
    """AR(1) with ALFRED vintage training, evaluated on first-release actuals."""
    rows = []
    for yr in sorted(range(start_year, pd.Timestamp.now().year + 2)):
        rt_date    = f"{yr}-01-01"
        cpi_rt     = as_of_series(alfred_cpi, rt_date).pct_change().dropna()
        train      = cpi_rt[cpi_rt.index.year < yr]
        test_dates = sorted([d for d in cpi_actuals.index if d.year == yr])
        if len(train) < min_train or not test_dates:
            continue
        try:
            res  = ARIMA(train, order=(1, 0, 0)).fit()
            c    = res.params.get("const", 0.0)
            phi  = res.params.get("ar.L1",  0.0)
            prev = train.iloc[-1]
            for date in test_dates:
                actual = cpi_actuals.get(date, np.nan)
                if pd.notna(actual):
                    rows.append(dict(date=date, actual=float(actual),
                                     pred=float(c + phi * prev), year=yr))
                    prev = float(actual)
        except Exception:
            pass
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# BREAKEVEN & CLEVELAND FED BENCHMARKS (for CPI MoM)
# ─────────────────────────────────────────────────────────────────────────────

def market_breakeven_benchmark(target_series, be5_annual):
    """
    Monthly CPI forecast = lagged 5Y breakeven / 12 / 100.
    be5_annual in % per year (e.g. 2.5 = 2.5 %/yr); target in decimal MoM.
    """
    monthly_implied = be5_annual.resample("ME").last().shift(1) / 12 / 100
    aligned = target_series.to_frame("actual").join(monthly_implied.rename("pred"))
    aligned["year"] = aligned.index.year
    return aligned.dropna()


def cleveland_benchmark(target_series, cf1y_series):
    """
    Monthly CPI forecast = lagged Cleveland Fed 1Y expectation / 12 / 100.
    cf1y_series in % per year; target in decimal MoM.
    """
    monthly_implied = cf1y_series.resample("ME").last().shift(1) / 12 / 100
    aligned = target_series.to_frame("actual").join(monthly_implied.rename("pred"))
    aligned["year"] = aligned.index.year
    return aligned.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# LGBM SETUP (identical to RAMM_LGBM_v1.py)
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = ["brent_ret","payroll_growth","be5","be10","vix",
            "brent_vol_6m","vix_change","be_slope","regime"]
MONOTONE = [1, 1, 1, 1, 0, 0, 0, 0, 0]

def lgbm():
    return LGBMRegressor(
        objective="regression", n_estimators=500, learning_rate=0.02,
        num_leaves=15, max_depth=4, subsample=0.8, colsample_bytree=0.8,
        monotone_constraints=MONOTONE, random_state=42, verbose=-1
    )

def lgbm_backtest(data, target_col, start_year=2015, min_train=60):
    rows = []
    for yr in sorted(data.index.year.unique()):
        if yr < start_year: continue
        train = data[data.index.year < yr]
        test  = data[data.index.year == yr]
        if len(train) < min_train or len(test) == 0: continue
        m = lgbm()
        m.fit(train[FEATURES], train[target_col])
        for date, actual, pred in zip(test.index,
                                      test[target_col],
                                      m.predict(test[FEATURES])):
            rows.append(dict(date=date, actual=actual, pred=pred, year=yr))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

print("Fetching market & macro data …")

brent_raw = yf.download("BZ=F",    start=START, auto_adjust=True, progress=False)
vix_raw   = yf.download("^VIX",    start=START, auto_adjust=True, progress=False)

brent_c = (brent_raw[("Close","BZ=F")] if isinstance(brent_raw.columns, pd.MultiIndex)
           else brent_raw["Close"])
vix_c   = (vix_raw[("Close","^VIX")] if isinstance(vix_raw.columns, pd.MultiIndex)
           else vix_raw["Close"])

brent_m = brent_c.resample("ME").last().pct_change().rename("brent_ret")
vix_m   = vix_c.resample("ME").last().rename("vix")

payrolls       = fred_m("PAYEMS")
payroll_growth = payrolls.pct_change().shift(1).rename("payroll_growth")
be5            = fred_m("T5YIE").rename("be5")
be10           = fred_m("T10YIE").rename("be10")
cf1y           = fred_m("EXPINF1YR")           # Cleveland Fed 1Y expected CPI

features = pd.concat([brent_m, payroll_growth, be5, be10, vix_m], axis=1)
features["brent_vol_6m"] = features["brent_ret"].rolling(6).std()
features["vix_change"]   = features["vix"].pct_change()
features["be_slope"]     = features["be10"] - features["be5"]
features["regime"]       = (features["vix"] > features["vix"].expanding(12).median()).astype(int)

cpi_target    = fred_m("CPILFESL").pct_change().rename("core_cpi_mom")
repricing_tgt = fred_m("DGS2").diff().rename("two_year_repricing")

data = pd.concat([features, cpi_target, repricing_tgt], axis=1).dropna()

print(f"  LGBM dataset: {data.index[0].date()} → {data.index[-1].date()}  ({len(data)} months)")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — CORE CPI MoM
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning CPI MoM backtests …")

lgbm_cpi  = lgbm_backtest(data, "core_cpi_mom")
ar1_cpi   = ar1_backtest(data["core_cpi_mom"])
be_cpi    = market_breakeven_benchmark(data["core_cpi_mom"], be5.rename())
cf_cpi    = cleveland_benchmark(data["core_cpi_mom"], cf1y)


def cpi_rows(bt, suffix=""):
    full  = metrics_block(bt["actual"], bt["pred"], f"  {suffix} (full 2015–now)")
    h     = bt[bt["year"] == HOLDOUT]
    hold  = metrics_block(h["actual"], h["pred"], f"  {suffix} ({HOLDOUT})") if len(h) else {"label": f"  {suffix} ({HOLDOUT})", "n": 0}
    return [full, hold]


print("\n══ Core CPI MoM ══════════════════════════════════════════════")
rows_cpi = (
    cpi_rows(lgbm_cpi,  "RAMM-LGBM") +
    cpi_rows(ar1_cpi,   "AR(1)") +
    cpi_rows(be_cpi,    "Market breakeven (T5YIE/12)") +
    cpi_rows(cf_cpi,    "Cleveland Fed 1Y/12")
)
print_table(rows_cpi)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — 2Y TREASURY REPRICING
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning 2Y repricing backtests …")

lgbm_rates = lgbm_backtest(data, "two_year_repricing")
ar1_rates  = ar1_backtest(data["two_year_repricing"])
# Zero-forecast (mean reversion): predict 0 change every period
zero_rows  = [dict(date=d, actual=a, pred=0.0, year=d.year)
              for d, a in data["two_year_repricing"].items()
              if d.year >= 2015]
zero_rates = pd.DataFrame(zero_rows).set_index("date")


def rates_rows(bt, suffix=""):
    full = metrics_block(bt["actual"], bt["pred"], f"  {suffix} (full 2015–now)")
    h    = bt[bt["year"] == HOLDOUT]
    hold = metrics_block(h["actual"], h["pred"], f"  {suffix} ({HOLDOUT})") if len(h) else {"label": f"  {suffix} ({HOLDOUT})", "n": 0}
    return [full, hold]


print("\n══ 2Y Treasury Repricing ═════════════════════════════════════")
rows_rates = (
    rates_rows(lgbm_rates, "RAMM-LGBM") +
    rates_rows(ar1_rates,  "AR(1)") +
    rates_rows(zero_rates, "Zero (mean-reversion)")
)
print_table(rows_rates)


# ─────────────────────────────────────────────────────────────────────────────
# BoE GILT-IMPLIED UK BREAKEVEN (nominal - real spot, 5Y)
# RPI-linked; historically ~1pp above UK CPI YoY — best free proxy available.
#
# CONVERGENCE CAVEAT: 5Y breakeven at time t reflects the market's expectation
# for average inflation over t → t+5y. By construction it converges toward
# realised inflation as the horizon elapses. Using BE5/12 as a 1-month-ahead
# forecast therefore benefits from the fact that markets already know last
# month's CPI print and revise the breakeven accordingly — the benchmark
# partially proxies AR(1) persistence. In a stricter real-time backtest you
# would use ALFRED vintages of T5YIE (US) or the first-release BoE BE series
# (UK) to isolate what was actually observable when the forecast was made.
# ─────────────────────────────────────────────────────────────────────────────

def get_boe_spot_5y(zip_url):
    import zipfile, io as _io
    r = requests.get(zip_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    z = zipfile.ZipFile(_io.BytesIO(r.content))
    frames = []
    for fname in sorted(z.namelist()):
        xl = pd.ExcelFile(z.open(fname))
        if "4. spot curve" not in xl.sheet_names:
            continue
        df = xl.parse("4. spot curve", header=None)
        mats = df.iloc[3, 1:].values.astype(float)
        data = df.iloc[5:].copy()
        data.columns = ["date"] + list(mats)
        import datetime as _dt
        data = data[data["date"].apply(lambda x: isinstance(x, (pd.Timestamp, _dt.datetime)))]
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date").apply(pd.to_numeric, errors="coerce")
        col_5y = min(mats, key=lambda x: abs(x - 5.0))
        frames.append(data[[col_5y]].rename(columns={col_5y: "y5"}))
    if not frames:
        return pd.Series(dtype=float, name="y5")
    return pd.concat(frames).sort_index()["y5"]

print("Fetching BoE gilt-implied UK breakeven …")
_base = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
_nom5  = get_boe_spot_5y(_base + "glcinflationmonthedata.zip")
_real5 = get_boe_spot_5y(_base + "glcrealmonthedata.zip")
uk_be5 = (_nom5 - _real5).rename("uk_be5")
uk_be5.index = uk_be5.index.to_period("M").to_timestamp("M")   # month-end align
print(f"  UK BE5: {uk_be5.index[0].date()} → {uk_be5.index[-1].date()}  "
      f"latest={uk_be5.iloc[-1]:.2f}%  (RPI-implied; ~1pp above CPI)")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — DFM: UK CPI YoY
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning DFM backtests …")

cpi_ons = None   # set inside try; used by MODEL 4 outside it
fx_c    = None

try:
    from dbnomics import fetch_series

    cpi_ons = fetch_series("ONS", "MM23", "D7G7.M")
    cpi_ons["original_period"] = pd.to_datetime(cpi_ons["original_period"])
    cpi_ons = (cpi_ons.set_index("original_period")[["value"]]
               .rename(columns={"value": "CPI_YoY"})
               .resample("ME").last().sort_index())

    oil_c = brent_c          # reuse already downloaded
    fx_raw  = yf.download("GBPUSD=X", start=START, auto_adjust=True, progress=False)
    fx_c    = (fx_raw[("Close","GBPUSD=X")] if isinstance(fx_raw.columns, pd.MultiIndex)
               else fx_raw["Close"])

    prices_m = pd.DataFrame({"Oil": oil_c.resample("ME").last(),
                              "GBPUSD": fx_c.resample("ME").last()})
    rets_m   = np.log(prices_m).diff()
    df_dfm   = rets_m.join(cpi_ons, how="outer").sort_index()
    mu, sigma = df_dfm.mean(), df_dfm.std()
    df_z      = ((df_dfm - mu) / sigma).dropna(how="all")

    # DFM expanding-window backtest
    dfm_rows = []
    for yr in range(2015, df_z.index.year.max() + 1):
        train_z = df_z[df_z.index.year < yr]
        test_z  = df_z[df_z.index.year == yr]
        if len(train_z) < 48 or len(test_z) == 0:
            continue
        try:
            res = DynamicFactor(train_z, k_factors=1, factor_order=1,
                                error_order=1).fit(maxiter=200, disp=False)
            fc_z = res.forecast(steps=len(test_z))
            fc_cpi_z = (fc_z["CPI_YoY"].values if hasattr(fc_z, "columns")
                        else fc_z.predicted_mean["CPI_YoY"].values)
            actual_yoy = test_z["CPI_YoY"].values * sigma["CPI_YoY"] + mu["CPI_YoY"]
            pred_yoy   = fc_cpi_z                 * sigma["CPI_YoY"] + mu["CPI_YoY"]
            for date, a, p in zip(test_z.index, actual_yoy, pred_yoy):
                if not np.isnan(a):
                    dfm_rows.append(dict(date=date, actual=a, pred=p, year=yr))
        except Exception as e:
            pass

    dfm_bt = pd.DataFrame(dfm_rows).set_index("date") if dfm_rows else pd.DataFrame()

    # AR(1) on UK CPI YoY
    uk_cpi_series = cpi_ons["CPI_YoY"].dropna()
    ar1_uk = ar1_backtest(uk_cpi_series)

    # UK gilt breakeven benchmark (lagged 1 month, same units as CPI YoY = % per year)
    # RPI-implied; adjust down ~1pp for CPI basis (historical RPI-CPI wedge)
    RPI_CPI_WEDGE = 1.0
    uk_be_adj = (uk_be5 - RPI_CPI_WEDGE).shift(1)
    be_uk_aligned = (uk_cpi_series.to_frame("actual")
                     .join(uk_be_adj.rename("pred"))
                     .dropna())
    be_uk_aligned["year"] = be_uk_aligned.index.year
    be_uk_aligned = be_uk_aligned[be_uk_aligned["year"] >= 2015]

    print("\n══ DFM: UK CPI YoY ═══════════════════════════════════════════")
    print("  (BoE breakeven is RPI-linked; adjusted -1pp to CPI basis)")
    rows_dfm = []
    if len(dfm_bt):
        rows_dfm += [metrics_block(dfm_bt["actual"], dfm_bt["pred"], "  DFM (full 2015–now)")]
        h = dfm_bt[dfm_bt["year"] == HOLDOUT]
        rows_dfm += [metrics_block(h["actual"], h["pred"], f"  DFM ({HOLDOUT})") if len(h)
                     else {"label": f"  DFM ({HOLDOUT})", "n": 0}]
    if len(ar1_uk):
        rows_dfm += [metrics_block(ar1_uk["actual"], ar1_uk["pred"], "  AR(1) (full 2015–now)")]
        h = ar1_uk[ar1_uk["year"] == HOLDOUT]
        rows_dfm += [metrics_block(h["actual"], h["pred"], f"  AR(1) ({HOLDOUT})") if len(h)
                     else {"label": f"  AR(1) ({HOLDOUT})", "n": 0}]
    if len(be_uk_aligned):
        rows_dfm += [metrics_block(be_uk_aligned["actual"], be_uk_aligned["pred"],
                                   "  BoE gilt breakeven -1pp (full)")]
        h = be_uk_aligned[be_uk_aligned["year"] == HOLDOUT]
        rows_dfm += [metrics_block(h["actual"], h["pred"], f"  BoE gilt breakeven -1pp ({HOLDOUT})") if len(h)
                     else {"label": f"  BoE gilt breakeven -1pp ({HOLDOUT})", "n": 0}]
    print_table(rows_dfm)

    # DFM factor loadings — refit on full dataset
    # Loading shows how strongly each observable is driven by the common factor.
    # For CPI YoY this is the Shapley-equivalent: how much of CPI variance
    # is explained by the latent factor constructed from Oil + GBPUSD + CPI.
    try:
        dfm_full = DynamicFactor(df_z.dropna(), k_factors=1, factor_order=1,
                                 error_order=1).fit(maxiter=200, disp=False)
        loading_params = {k: v for k, v in dfm_full.params.items()
                          if k.startswith("loading.")}
        print("\n── DFM factor loadings (full sample) ──────────────────────")
        print("  (Shapley-equivalent: marginal contribution of each series to")
        print("   the common latent factor; larger |loading| = more co-movement)")
        print(f"  {'Series':<20} {'loading':>12}  {'|loading|':>10}")
        print(f"  {'-'*46}")
        sorted_loadings = sorted(loading_params.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, val in sorted_loadings:
            series_name = name.replace("loading.f1.", "")
            bar = "▲" if val > 0 else "▼"
            print(f"  {series_name:<20} {val:>+12.4f}  {abs(val):>10.4f}  {bar}")
    except Exception as _e:
        print(f"  DFM factor loadings failed: {_e}")

except Exception as e:
    print(f"  DFM section failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4 — UK RAMM-LGBM (direct comparison to DFM)
# Same target (UK CPI YoY), richer feature set: oil, GBP/USD, uk_be5, VIX.
# ─────────────────────────────────────────────────────────────────────────────

uk_lgbm_bt = pd.DataFrame()   # populated if UK data available

if cpi_ons is not None and fx_c is not None:
    print("\nRunning UK RAMM-LGBM backtests …")

    UK_FEATURES = ["oil_ret", "gbpusd_ret", "oil_vol_6m", "gbpusd_vol_6m",
                   "uk_be5", "vix", "vix_change", "regime_uk"]
    UK_MONOTONE = [1, -1, 0, 0, 1, 0, 0, 0]

    oil_m_uk   = np.log(brent_c.resample("ME").last()).diff().rename("oil_ret")
    gbpusd_m   = np.log(fx_c.resample("ME").last()).diff().rename("gbpusd_ret")

    uk_feat = pd.concat([oil_m_uk, gbpusd_m, vix_m], axis=1)
    uk_feat["oil_vol_6m"]    = uk_feat["oil_ret"].rolling(6).std()
    uk_feat["gbpusd_vol_6m"] = uk_feat["gbpusd_ret"].rolling(6).std()
    uk_feat["vix_change"]    = uk_feat["vix"].pct_change()
    uk_feat["uk_be5"]        = uk_be5
    exp_med = uk_feat["vix"].expanding(min_periods=12).median()
    uk_feat["regime_uk"]     = (uk_feat["vix"] > exp_med).astype(int)

    uk_data = uk_feat.join(cpi_ons["CPI_YoY"], how="inner").dropna(subset=UK_FEATURES)

    def lgbm_uk():
        return LGBMRegressor(
            objective="regression", n_estimators=500, learning_rate=0.02,
            num_leaves=15, max_depth=4, subsample=0.8, colsample_bytree=0.8,
            monotone_constraints=UK_MONOTONE, random_state=42, verbose=-1,
        )

    rows_uk = []
    for yr in sorted(range(2015, pd.Timestamp.now().year + 1)):
        train = uk_data[uk_data.index.year < yr].dropna(subset=UK_FEATURES + ["CPI_YoY"])
        test  = uk_data[uk_data.index.year == yr].dropna(subset=UK_FEATURES)
        if len(train) < 60 or len(test) == 0:
            continue
        m = lgbm_uk()
        m.fit(train[UK_FEATURES], train["CPI_YoY"])
        for date, pred in zip(test.index, m.predict(test[UK_FEATURES])):
            actual = uk_data.loc[date, "CPI_YoY"] if date in uk_data.index else np.nan
            if pd.notna(actual):
                rows_uk.append(dict(date=date, actual=float(actual),
                                    pred=float(pred), year=yr))
    uk_lgbm_bt = pd.DataFrame(rows_uk).set_index("date") if rows_uk else pd.DataFrame()

    print("\n══ UK CPI YoY — DFM vs RAMM-LGBM UK ═════════════════════════")
    print("  (BoE breakeven is RPI-linked; adjusted -1pp to CPI basis)")
    rows_uk_cmp = []

    def uk_rows(bt, label):
        if not len(bt): return []
        full = metrics_block(bt["actual"], bt["pred"], f"  {label} (full 2015–now)")
        h = bt[bt["year"] == HOLDOUT]
        hold = (metrics_block(h["actual"], h["pred"], f"  {label} ({HOLDOUT})")
                if len(h) else {"label": f"  {label} ({HOLDOUT})", "n": 0})
        return [full, hold]

    if len(dfm_bt):     rows_uk_cmp += uk_rows(dfm_bt,       "DFM")
    if len(uk_lgbm_bt): rows_uk_cmp += uk_rows(uk_lgbm_bt,   "RAMM-LGBM UK")
    if len(ar1_uk):     rows_uk_cmp += uk_rows(ar1_uk,        "AR(1)")
    if len(be_uk_aligned): rows_uk_cmp += uk_rows(be_uk_aligned, "BoE breakeven -1pp")
    print_table(rows_uk_cmp)

    # SHAP for UK LGBM (full-sample fit)
    mdl_uk_full = lgbm_uk()
    bt_uk_data  = uk_data.dropna(subset=UK_FEATURES + ["CPI_YoY"])
    mdl_uk_full.fit(bt_uk_data[UK_FEATURES], bt_uk_data["CPI_YoY"])
    shap_bt_X   = bt_uk_data.loc[bt_uk_data.index.year >= 2015, UK_FEATURES]

    exp_uk = shap.TreeExplainer(mdl_uk_full)
    sv_uk  = exp_uk.shap_values(shap_bt_X)
    mean_abs_uk = pd.Series(np.abs(sv_uk).mean(axis=0), index=shap_bt_X.columns)
    latest_sv   = pd.Series(sv_uk[-1], index=shap_bt_X.columns)
    print("\n── SHAP: UK CPI YoY (RAMM-LGBM UK) ──────────────────────────")
    print(f"  {'Feature':<22} {'mean |SHAP|':>12}  {'latest SHAP':>13}  dir")
    print(f"  {'-'*58}")
    for feat in mean_abs_uk.sort_values(ascending=False).index:
        bar = "▲" if latest_sv[feat] > 0 else "▼"
        print(f"  {feat:<22} {mean_abs_uk[feat]:>12.4f}  {latest_sv[feat]:>+13.4f}  {bar}")

    latest_uk_x = bt_uk_data[UK_FEATURES].iloc[[-1]]
    print(f"\n  UK CPI YoY nowcast : {mdl_uk_full.predict(latest_uk_x)[0]:+.2f}%  "
          f"(as of {bt_uk_data.index[-1].date()})")
else:
    print("\n  MODEL 4 skipped — UK data not available (dbnomics/BoE fetch failed).")


# ─────────────────────────────────────────────────────────────────────────────
# LATEST NOWCAST + SHAP (ALL MODELS)
# ─────────────────────────────────────────────────────────────────────────────

print("\n══ Latest nowcast ════════════════════════════════════════════")
mdl_cpi   = lgbm(); mdl_cpi.fit(data[FEATURES],   data["core_cpi_mom"])
mdl_rates = lgbm(); mdl_rates.fit(data[FEATURES], data["two_year_repricing"])
latest_x  = data[FEATURES].iloc[[-1]]
as_of     = data.index[-1].date()

print(f"  As of: {as_of}")
print(f"  Core CPI MoM nowcast : {mdl_cpi.predict(latest_x)[0]:+.4f}  ({mdl_cpi.predict(latest_x)[0]*100:+.2f}%)")
print(f"  2Y repricing nowcast : {mdl_rates.predict(latest_x)[0]:+.4f} pp")
print(f"  Cleveland Fed 1Y     : {cf1y.iloc[-1]:.2f}%  (as of {cf1y.index[-1].date()})")
print(f"  5Y breakeven         : {be5.iloc[-1]:.2f}%  (as of {be5.index[-1].date()})")


def shap_table(model, X, label):
    """Print mean |SHAP| over backtest period + latest-month SHAP per feature."""
    exp = shap.TreeExplainer(model)
    sv  = exp.shap_values(X)                                   # (n_samples, n_features)
    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    latest   = pd.Series(sv[-1],                  index=X.columns)
    order    = mean_abs.sort_values(ascending=False).index
    print(f"\n── SHAP: {label} ──────────────────────────────────────────")
    print(f"  {'Feature':<20} {'mean |SHAP|':>12}  {'latest SHAP':>13}  dir")
    print(f"  {'-'*56}")
    for feat in order:
        bar = "▲" if latest[feat] > 0 else "▼"
        print(f"  {feat:<20} {mean_abs[feat]:>12.6f}  {latest[feat]:>+13.6f}  {bar}")


# Backtest-period rows only (2015+) for stable mean |SHAP| estimate
bt_X = data.loc[data.index.year >= 2015, FEATURES]
shap_table(mdl_cpi,   bt_X, "Core CPI MoM (RAMM-LGBM)")
shap_table(mdl_rates, bt_X, "2Y Repricing (RAMM-LGBM)")


# ─────────────────────────────────────────────────────────────────────────────
# 2026 YTD PLOTS
# ─────────────────────────────────────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    YTD = 2026
    STYLE = {
        "Actual":             dict(color="black",   lw=2.5, ls="-",  marker="o", ms=7, zorder=5),
        "RAMM-LGBM":          dict(color="#1f77b4", lw=1.8, ls="--", marker="s", ms=5),
        "AR(1)":              dict(color="#ff7f0e", lw=1.5, ls=":",  marker="^", ms=5),
        "T5YIE/12":           dict(color="#2ca02c", lw=1.5, ls="-.", marker="v", ms=5),
        "Cleveland Fed":      dict(color="#9467bd", lw=1.5, ls="-.", marker="D", ms=5),
        "Zero":               dict(color="#d62728", lw=1.2, ls=":",  marker="x", ms=5),
        "DFM":                dict(color="#1f77b4", lw=1.8, ls="--", marker="s", ms=5),
        "RAMM-LGBM UK":       dict(color="#e377c2", lw=1.8, ls="--", marker="D", ms=5),
        "BoE BE −1pp":        dict(color="#2ca02c", lw=1.5, ls="-.", marker="v", ms=5),
    }

    def ytd(df):
        return df[df.index.year == YTD] if len(df) else df

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(f"{YTD} YTD: Model Predictions vs Benchmarks", fontsize=13, fontweight="bold")

    # ── Panel 1: Core CPI MoM ────────────────────────────────────────────────
    ax = axes[0]
    s = ytd(lgbm_cpi)
    if len(s):
        ax.plot(s.index, s["actual"] * 100, label="Actual (CPILFESL)", **STYLE["Actual"])
        ax.plot(s.index, s["pred"]   * 100, label="RAMM-LGBM",         **STYLE["RAMM-LGBM"])
    for bt, lbl in [(ar1_cpi, "AR(1)"), (be_cpi, "T5YIE/12"), (cf_cpi, "Cleveland Fed")]:
        s2 = ytd(bt)
        if len(s2):
            ax.plot(s2.index, s2["pred"] * 100, label=lbl, **STYLE[lbl])
    ax.set_title("US Core CPI MoM")
    ax.set_ylabel("% MoM")
    ax.legend(fontsize=8, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.3)

    # ── Panel 2: 2Y Repricing ────────────────────────────────────────────────
    ax = axes[1]
    s = ytd(lgbm_rates)
    if len(s):
        ax.plot(s.index, s["actual"], label="Actual (DGS2 Δ)", **STYLE["Actual"])
        ax.plot(s.index, s["pred"],   label="RAMM-LGBM",        **STYLE["RAMM-LGBM"])
    for bt, lbl, sty in [
        (ar1_rates,  "AR(1)", STYLE["AR(1)"]),
        (zero_rates, "Zero",  STYLE["Zero"]),
    ]:
        s2 = ytd(bt)
        if len(s2):
            ax.plot(s2.index, s2["pred"], label=lbl, **sty)
    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=0)
    ax.set_title("2Y Treasury Repricing")
    ax.set_ylabel("pp change in yield")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.3)

    # ── Panel 3: UK CPI YoY ──────────────────────────────────────────────────
    ax = axes[2]
    try:
        s = ytd(dfm_bt)
        if len(s):
            ax.plot(s.index, s["actual"], label="Actual (ONS)", **STYLE["Actual"])
            ax.plot(s.index, s["pred"],   label="DFM",           **STYLE["DFM"])
        s2 = ytd(ar1_uk)
        if len(s2):
            ax.plot(s2.index, s2["pred"], label="AR(1)", **STYLE["AR(1)"])
        s3 = ytd(be_uk_aligned)
        if len(s3):
            ax.plot(s3.index, s3["pred"], label="BoE BE −1pp", **STYLE["BoE BE −1pp"])
        if len(uk_lgbm_bt):
            s4 = ytd(uk_lgbm_bt)
            if len(s4):
                ax.plot(s4.index, s4["pred"], label="RAMM-LGBM UK", **STYLE["RAMM-LGBM UK"])
    except NameError:
        ax.text(0.5, 0.5, "DFM data unavailable", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    ax.set_title("UK CPI YoY")
    ax.set_ylabel("% YoY")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "backtest_2026ytd.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.show()

except Exception as _plot_err:
    print(f"  Plotting failed: {_plot_err}")

print("\nDone.")
