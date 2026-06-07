import pandas as pd
import numpy as np
import yfinance as yf
from dbnomics import fetch_series
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

class UKInflationNowcaster:
    def __init__(self, start_date="2015-01-01"):
        self.start = start_date
        self.mu = None
        self.sigma = None

    def get_ons_cpi(self):
        cpi_series = fetch_series("ONS", "MM23", "D7G7.M")
        cpi = cpi_series[["original_period", "value"]].copy()
        cpi["original_period"] = pd.to_datetime(cpi["original_period"])
        cpi = (
            cpi.set_index("original_period")
               .rename(columns={"value": "CPI_YoY"})
               .sort_index()
        )
        # keep monthly frequency explicit
        cpi = cpi.resample("M").last()
        return cpi

    def fetch_data(self):
        tickers = {"Oil": "BZ=F", "GBPUSD": "GBPUSD=X"}

        raw = yf.download(
            list(tickers.values()),
            start=self.start,
            auto_adjust=True,
            progress=False
        )

        prices = raw["Close"].rename(columns={v: k for k, v in tickers.items()}).sort_index()

        # monthly features: choose one convention and keep it consistent
        monthly_prices = prices.resample("M").last()
        monthly_rets = np.log(monthly_prices).diff()

        cpi = self.get_ons_cpi()

        df = monthly_rets.join(cpi, how="outer").sort_index()

        self.mu = df.mean()
        self.sigma = df.std()
        z = (df - self.mu) / self.sigma

        return z

nowcaster = UKInflationNowcaster()
df_final = nowcaster.fetch_data()

# drop rows that are completely empty
df_final = df_final.dropna(how="all")

model = DynamicFactor(
    df_final,
    k_factors=1,
    factor_order=1,
    error_order=1
)

res = model.fit(maxiter=200, disp=False)

# standardized fitted values
fitted_z = res.predict()

# unscale CPI nowcast
cpi_nowcast = fitted_z["CPI_YoY"] * nowcaster.sigma["CPI_YoY"] + nowcaster.mu["CPI_YoY"]

comparison = pd.DataFrame({
    "Model_Nowcast": cpi_nowcast,
    "Official_ONS": nowcaster.get_ons_cpi()["CPI_YoY"]
}).dropna(subset=["Official_ONS"])

print(comparison.tail(12))
print(res.params.index)  # inspect names before extracting loadings