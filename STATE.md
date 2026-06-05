# STATE — UK CPI Nowcast Project

*Last updated: 2026-06-05*

## What is Running

**Primary entry point:** `compare_uk.py`
```bash
FRED_API_KEY=<key> .venv/bin/python -W ignore compare_uk.py --start 2015
# with regime-model-combine (~5 min extra):
FRED_API_KEY=<key> .venv/bin/python -W ignore compare_uk.py --start 2015 --rmc
```

**10 models** in `uk_model_zoo.py` (MIDAS removed 2026-06-05):
DFM, RAMM-LGBM, UCM, TVP, HMM, MS-DFM, LSTAR, BVAR, HiddenRF, GBM

**4 combined ensembles:**
Combined-Static, Combined-Dynamic, Combined-Superstar, Combined-Absolute

**Regime-model-combine metamodels (new, --rmc flag):**
RMC-hmm, RMC-lstar, RMC-dfm, RMC-manual_vix

---

## Live Factor Matrix

9 factors currently loading (as of last run):

| Factor        | pub_lag | Method  | Notes                                  |
|---------------|---------|---------|----------------------------------------|
| oil_brent     | 0       | FRED    | Brent log-return                       |
| gbpusd        | 0       | FRED    | USD/GBP log-return                     |
| uk_be5        | 0       | BoE ZIP | 5Y gilt breakeven (nominal−real)       |
| vix           | 0       | yfinance| CBOE VIX level                         |
| gas_hh        | 0       | FRED    | Henry Hub log-return (US proxy)        |
| oil_vol_6m    | 0       | FRED    | 6m rolling Brent log-return std        |
| gbpusd_vol_6m | 0       | FRED    | 6m rolling GBP/USD log-return std      |
| uk_rents      | 1       | dbnomics| ONS L522 YoY (pub_lag=1, leakage fixed)|
| uk_vacancies  | 1       | dbnomics| ONS AP2Y log-return                    |

**Always unavailable (dbnomics failures):** uk_house_prices, uk_paye

**Mixed-frequency fix (2026-06-05):** `apply_publication_lags()` called in `compare_uk.py`
before all model runs. pub_lag=0 factors used contemporaneously; pub_lag=1 factors
shifted by 1 month. Fixes uk_rents leakage (+0.209pp RMSE improvement at lag=0).

---

## Last Backtest Results (pre pub-lag fix, 9-factor run)

| Model             | RMSE  | DM vs AR(1) | Status  |
|-------------------|-------|-------------|---------|
| UCM               | 0.092 | +4.21**     | Tier 1  |
| Combined-Superstar| 0.120 | +4.15**     | Tier 3  |
| TVP               | 0.182 | +3.85**     | Tier 1  |
| Combined-Dynamic  | 0.276 | +3.41**     | Tier 3  |
| DFM               | 0.473 | +1.29       | Tier 2  |
| AR(1)             | 0.480 | —           | baseline|
| LSTAR             | 0.492 | +0.08       | Tier 2  |
| BVAR              | 0.591 | −3.46**     | dropped |
| HMM               | 2.581 | −4.96**     | regime  |
| MS-DFM            | 2.814 | −5.55**     | regime  |
| GBM               | 1.383 | −3.20**     | tree    |
| HiddenRF          | 1.556 | −3.35**     | tree    |
| MIDAS             | 3.243 | −12.46**    | REMOVED |

**Production hierarchy:**
- Tier 1 (beat AR(1) p<0.05): UCM, TVP → Combined-Superstar
- Tier 2 (borderline): DFM, LSTAR
- Tier 3 (combined): Combined-Dynamic (inverse-RMSE-weighted, converges to UCM+TVP)

**Regime consensus (June 2026):**
- VIX-based: stress regime (40% of sample historical frequency)
- HMM on CPI: low-inflation state (84% historical frequency)
- LSTAR: lower regime (79% historical frequency)
- Divergence: financial stress but CPI-level models say benign

---

## Pending Rewrites / Gaps

| Item | Status | File |
|------|--------|------|
| Re-run backtest with pub-lag fix | PENDING | compare_uk.py |
| uk_rents_lag1 as live factor (replace uk_rents) | PENDING | factors.py |
| ISM PMI and US PPI free factors | PENDING | factors.py (NAPM, PPIACO) |
| CSV drop-ins for PMI, core CPI, services CPI, PPI | PENDING | data/*.csv |
| Regime-model-combine RMC results | PENDING | compare_uk.py --rmc |
| Push to adambutlin/nowcast | PENDING | git |
| Current nowcasts for June 2026 | PENDING | compare_uk.py |

---

## Key Files

| File | Purpose |
|------|---------|
| `factors.py` | Factor registry with pub_lag, apply_publication_lags() |
| `uk_model_zoo.py` | 10 models + dm_test() + score_backtest() |
| `compare_uk.py` | Full comparison driver + RMC framework |
| `backtest_2025.py` | Older 3-model backtest (DFM/RAMM-LGBM/2Y); pushed to GitHub |
| `data/` | CSV drop-ins for unavailable/gated factors |

---

## Environment

```bash
cd /Users/Adam/Documents/home/quant/ramm-lgbm
source .venv/bin/activate  # or .venv/bin/python directly
export FRED_API_KEY=<your-key>  # never hardcode
```

Required packages: statsmodels, lightgbm, xgboost, scikit-learn, shap,
fredapi, yfinance, dbnomics, pandas, numpy, scipy, requests, openpyxl
