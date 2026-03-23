import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import matplotlib.pyplot as plt
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

# 1. Prepare Data for PyMC
# We use the standardized df_final from your previous step
data = df_final[['Oil', 'GBPUSD', 'CPI_YoY']].values
n_obs, n_vars = data.shape

# Create a mask for the NaNs in CPI so PyMC handles missing data
mask = np.isnan(data)
masked_data = np.ma.masked_array(data, mask)

with pm.Model() as ms_dfm:
    # --- 1. TRANSITION PROBABILITIES (The HMM 'Brain') ---
    # p11: Prob of staying in Regime 0 | p22: Prob of staying in Regime 1
    p11 = pm.Beta("p11", alpha=95, beta=5) # Prior: 95% persistence
    p22 = pm.Beta("p22", alpha=95, beta=5)
    
    # Construct the transition matrix
    # P = [[p11, 1-p11], [1-p22, p22]]
    P = pt.stack([pt.stack([p11, 1-p11]), pt.stack([1-p22, p22])])

    # --- 2. HIDDEN STATES (The Markov Chain) ---
    # This latent variable 's' is 0 or 1 for every day
    s = pm.Categorical("s", p=pt.ones(2)/2, shape=n_obs)

    # --- 3. STATE-DEPENDENT LOADINGS (The 'Switch') ---
    # Regime 0 (Normal): Low sensitivity to Oil/FX
    # Regime 1 (Volatile): High sensitivity
    loading_low = pm.Normal("loading_low", mu=0, sigma=0.5, shape=n_vars)
    loading_high = pm.Normal("loading_high", mu=0, sigma=1.0, shape=n_vars)
    
    # Select the loading based on the current state 's'
    # loadings_t is the matrix of betas for each day
    loadings_t = pt.stack([loading_low, loading_high])[s]

    # --- 4. THE LATENT FACTOR (The 'Inflation Pulse') ---
    # f_t = phi * f_{t-1} + epsilon
    phi = pm.Normal("phi", mu=0.5, sigma=0.2)
    factor = pm.GaussianRandomWalk("factor", sigma=1, shape=n_obs)

    # --- 5. THE LIKELIHOOD (Observation Equation) ---
    # data_t = Loadings_t * Factor_t + Noise
    # This is where the Kalman-like logic happens
    mu_obs = loadings_t * factor[:, None]
    
    sigma_obs = pm.Exponential("sigma_obs", lam=1.0, shape=n_vars)
    
    # PyMC automatically handles the NaNs in masked_data
    obs = pm.Normal("obs", mu=mu_obs, sigma=sigma_obs, observed=masked_data)

    # --- 6. SAMPLING ---
    # We use CategoricalGibbs for the discrete states
    trace = pm.sample(1000, tune=1000, target_accept=0.9, cores=1)

print("Bayesian MS-DFM Sampling Complete.")


# Extract the probability of being in Regime 1 (Volatile) over time
regime_probs = trace.posterior["s"].mean(dim=["chain", "draw"])

plt.figure(figsize=(12, 4))
plt.plot(df_final.index, regime_probs)
plt.title("Probability of 'Volatile' Inflation Regime (UK)")
plt.ylabel("Probability")
plt.show()