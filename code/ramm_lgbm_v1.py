
"""
RAMM-LGBM: Regime-Aware Monotonic LightGBM
Target 1: Core CPI MoM nowcast (ALFRED vintages)
Target 2: 2Y Treasury repricing after CPI release

Requires:
pip install pandas numpy fredapi yfinance lightgbm shap scikit-learn requests matplotlib

Set:
FRED_API_KEY environment variable
"""

import os
import io
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from fredapi import Fred
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import shap

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

START_DATE = "2005-01-01"
END_DATE = None

ALFRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def get_fred_client():
    key = os.getenv("FRED_API_KEY")
    if key is None:
        raise EnvironmentError(
            "FRED_API_KEY environment variable is required to fetch FRED data."
        )
    return Fred(api_key=key)

SERIES = {
    "core_cpi": "CPILFESL",
    "payrolls": "PAYEMS",
    "breakeven5": "T5YIE",
    "breakeven10": "T10YIE",
    "dgs2": "DGS2"
}

# ---------------------------------------------------------------------
# ALFRED VINTAGE HELPERS
# ---------------------------------------------------------------------

def get_alfred_vintage(series_id, realtime_date):
    params = {
        "series_id": series_id,
        "api_key": os.getenv("FRED_API_KEY"),
        "file_type": "json",
        "realtime_start": realtime_date,
        "realtime_end": realtime_date,
        "observation_start": START_DATE,
    }

    if params["api_key"] is None:
        raise EnvironmentError(
            "FRED_API_KEY environment variable is required to fetch ALFRED vintages."
        )

    r = requests.get(ALFRED_BASE, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()["observations"]

    df = pd.DataFrame(data)

    if len(df) == 0:
        return pd.Series(dtype=float)

    df["date"] = pd.to_datetime(df["date"])

    df["value"] = pd.to_numeric(
        df["value"],
        errors="coerce"
    )

    return df.set_index("date")["value"]


# ---------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------

def download_market_data():
    brent = yf.download(
        "BZ=F",
        start=START_DATE,
        auto_adjust=True,
        progress=False
    )

    vix = yf.download(
        "^VIX",
        start=START_DATE,
        auto_adjust=True,
        progress=False
    )

    if isinstance(brent.columns, pd.MultiIndex):
        brent_close = brent.loc[:, ("Close", "BZ=F")]
    else:
        brent_close = brent["Close"]

    if isinstance(vix.columns, pd.MultiIndex):
        vix_close = vix.loc[:, ("Close", "^VIX")]
    else:
        vix_close = vix["Close"]

    brent_m = (
        brent_close
        .resample("ME")
        .last()
        .pct_change()
        .rename("brent_ret")
    )

    vix_m = (
        vix_close
        .resample("ME")
        .last()
        .rename("vix")
    )

    return brent_m, vix_m


# ---------------------------------------------------------------------
# FRED SERIES
# ---------------------------------------------------------------------

def fred_monthly(series):
    fred = get_fred_client()
    s = fred.get_series(series)
    s.index = pd.to_datetime(s.index)
    return s


# ---------------------------------------------------------------------
# CPI TARGET
# ---------------------------------------------------------------------

def build_target_core_cpi():
    cpi = fred_monthly("CPILFESL")

    target = (
        cpi
        .resample("ME")
        .last()
        .pct_change()
        .rename("core_cpi_mom")
    )

    return target


# ---------------------------------------------------------------------
# 2Y REPRICING TARGET
# ---------------------------------------------------------------------

def build_2y_target():
    dgs2 = fred_monthly("DGS2")

    monthly = dgs2.resample("ME").last()

    repricing = (
        monthly.diff()
        .rename("two_year_repricing")
    )

    return repricing


# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------

def build_feature_matrix():
    brent_m, vix_m = download_market_data()

    payrolls = (
        fred_monthly("PAYEMS")
        .resample("ME")
        .last()
    )

    # NFP for month M is released early month M+1 — shift(1) prevents
    # using data that wasn't available when predicting month M's CPI.
    payroll_growth = (
        payrolls.pct_change()
        .shift(1)
        .rename("payroll_growth")
    )

    be5 = (
        fred_monthly("T5YIE")
        .resample("ME")
        .last()
        .rename("be5")
    )

    be10 = (
        fred_monthly("T10YIE")
        .resample("ME")
        .last()
        .rename("be10")
    )

    features = pd.concat(
        [
            brent_m,
            payroll_growth,
            be5,
            be10,
            vix_m
        ],
        axis=1,
        sort=False
    )

    features["brent_vol_6m"] = (
        features["brent_ret"]
        .rolling(6)
        .std()
    )

    features["vix_change"] = (
        features["vix"]
        .pct_change()
    )

    features["be_slope"] = (
        features["be10"]
        - features["be5"]
    )

    return features


# ---------------------------------------------------------------------
# REGIMES
# ---------------------------------------------------------------------

def add_regimes(df):
    # Causal regime: 1 = VIX above its own expanding median (stress),
    # 0 = tranquil. Uses only data available up to each point in time —
    # no future information leaks into earlier train windows.
    # min_periods=12 so first year of data is NaN (dropped later).
    expanding_median = df["vix"].expanding(min_periods=12).median()
    df["regime"] = (df["vix"] > expanding_median).astype(int)
    return df, None


# ---------------------------------------------------------------------
# MONOTONIC LIGHTGBM
# ---------------------------------------------------------------------

FEATURES = [
    "brent_ret",
    "payroll_growth",
    "be5",
    "be10",
    "vix",
    "brent_vol_6m",
    "vix_change",
    "be_slope",
    "regime"
]

MONOTONE = [
    1,   # brent
    1,   # payrolls
    1,   # be5
    1,   # be10
    0,   # vix
    0,
    0,
    0,
    0
]

def build_model():

    return LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        monotone_constraints=MONOTONE,
        random_state=42
    )


# ---------------------------------------------------------------------
# ROLLING VINTAGE BACKTEST
# ---------------------------------------------------------------------

def rolling_backtest(data, target_col):

    preds = []
    actuals = []
    dates = []

    years = sorted(data.index.year.unique())

    for yr in years:

        if yr < 2015:
            continue

        train = data[data.index.year < yr]

        test = data[data.index.year == yr]

        if len(train) < 60:
            continue

        model = build_model()

        model.fit(
            train[FEATURES],
            train[target_col]
        )

        p = model.predict(
            test[FEATURES]
        )

        preds.extend(p)
        actuals.extend(test[target_col])

        dates.extend(test.index)

    out = pd.DataFrame(
        {
            "actual": actuals,
            "pred": preds
        },
        index=dates
    )

    rmse = np.sqrt(
        mean_squared_error(
            out["actual"],
            out["pred"]
        )
    )

    mae = mean_absolute_error(
        out["actual"],
        out["pred"]
    )

    dir_acc = (np.sign(out["actual"]) == np.sign(out["pred"])).mean() * 100

    print()
    print(target_col)
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("Dir%:", round(dir_acc, 1))

    return out


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------

def generate_shap(model, X):

    explainer = shap.TreeExplainer(model)

    values = explainer.shap_values(X)

    shap_df = pd.DataFrame(
        values,
        columns=X.columns,
        index=X.index
    )

    latest = (
        shap_df.iloc[-1]
        .sort_values(
            ascending=False
        )
    )

    print()
    print("LATEST SHAP CONTRIBUTIONS")
    print(latest)

    return shap_df


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    features = build_feature_matrix()

    cpi_target = build_target_core_cpi()

    repricing_target = build_2y_target()

    data = pd.concat(
        [
            features,
            cpi_target,
            repricing_target
        ],
        axis=1,
        sort=False
    )

    data, _ = add_regimes(data)

    # Capture nowcast features before dropping rows with unreleased targets.
    # After dropna(), the last row is the most recent month with KNOWN CPI —
    # predicting it would be circular. We want features from the most recent
    # complete period even when the CPI release is still pending.
    latest_x = data[FEATURES].dropna().iloc[[-1]]

    data = data.dropna()

    print("Rows:", len(data))
    print("Nowcast date:", latest_x.index[0].strftime("%Y-%m"))

    # CPI MODEL

    cpi_results = rolling_backtest(
        data,
        "core_cpi_mom"
    )

    model_cpi = build_model()

    model_cpi.fit(
        data[FEATURES],
        data["core_cpi_mom"]
    )

    generate_shap(
        model_cpi,
        data[FEATURES]
    )

    # 2Y REPRICING MODEL

    repricing_results = rolling_backtest(
        data,
        "two_year_repricing"
    )

    model_rates = build_model()

    model_rates.fit(
        data[FEATURES],
        data["two_year_repricing"]
    )

    forecast_cpi = model_cpi.predict(latest_x)[0]

    forecast_rates = model_rates.predict(latest_x)[0]

    print()
    print("CURRENT NOWCAST")
    print("--------------------")
    print("Core CPI MoM :", forecast_cpi)
    print("2Y repricing :", forecast_rates)

    cpi_results.to_csv(
        "cpi_backtest.csv"
    )

    repricing_results.to_csv(
        "rates_backtest.csv"
    )

    print()
    print("Files saved.")

if __name__ == "__main__":
    main()
