# STATE — UK CPI Nowcast Project

*Last updated: 2026-06-07*

## What is Running

**Primary directory:** `/Users/Adam/Documents/home/quant/nowcast/`

**Primary entry point:** `code/nowcast_cpi.py`
```bash
# Full 21-model backtest (blind test: --end 2024; 2025+ reserved)
FRED_API_KEY=<key> .venv/bin/python -W ignore code/nowcast_cpi.py --start 2015 --end 2024 --train-from 1992

# With regime-model-combine (top-8 models, dfm_k2 regime):
FRED_API_KEY=<key> .venv/bin/python -W ignore code/nowcast_cpi.py --start 2015 --end 2024 --train-from 1992 --rmc --rmc-methods dfm_k2 --rmc-top-k 8
```

**Folder structure:**
- `code/` — all Python source files
- `data/` — CSV outputs (backtest, metrics, nowcast, RMC perf)
- `plots/` — PNG outputs
- `logs/` — run logs

**21 models** in `code/uk_model_zoo.py`:
DFM, RAMM-LGBM, UCM, TVP, HMM, MS-DFM, BVAR, HiddenRF, GBM, MIDAS, BridgeEq,
CopulaReg, DFM-k2, ElasticNet, MedianElasticNet, HuberNet, PCR, RegimeEns,
SARIMAX, VAR, AutoARIMA

**4 combined ensembles:**
Combined-Static, Combined-Dynamic, Combined-Superstar, Combined-Absolute

**Regime-model-combine metamodels (--rmc flag):**
RMC-dfm_k2 (k=2 DFM KMeans regime labelling)

---

## Live Factor Matrix (30 factors)

| Factor              | pub_lag | Transform | Source       | Notes                                    |
|---------------------|---------|-----------|--------------|------------------------------------------|
| oil_brent           | 0       | logret    | FRED         | Brent crude spot                         |
| gbpusd              | 0       | logret    | FRED         | USD/GBP spot                             |
| uk_be5              | 0       | level     | BoE ZIP      | 5Y gilt breakeven                        |
| vix                 | 0       | level     | yfinance     | CBOE VIX                                 |
| gas_eu              | 0       | logret    | yfinance+IMF | TTF front-month (daily→monthly)          |
| uk_gilt_10y         | 0       | diff      | FRED         | UK 10Y gilt yield change                 |
| oil_vol_6m          | 0       | level     | derived      | 6m rolling Brent log-return std          |
| gbpusd_vol_6m       | 0       | level     | derived      | 6m rolling GBP/USD log-return std        |
| oil_brent_3m        | 0       | level     | derived      | Brent 3m log-return                      |
| gbpusd_3m           | 0       | level     | derived      | GBP/USD 3m log-return                    |
| gbp_eur             | 0       | logret    | FRED         | GBP/EUR cross-rate (DEXUSUK/DEXUSEU)     |
| gbp_eer             | 0       | diff      | FRED         | UK real broad EER, BIS (RBGBBIS)         |
| semiconductors_ppi  | 0       | logret    | FRED         | US BLS PPI semis (PCU334413334413)       |
| deep_sea_freight    | 0       | logret    | FRED         | US BLS PPI deep sea freight              |
| metals_index        | 0       | level     | FRED         | Equal-weight: Cu/Al/Ni/Zn/Fe log-ret avg |
| copper_price        | 0       | logret    | FRED         | PCOPPUSDM                                |
| nickel_price        | 0       | logret    | FRED         | PNICKUSDM                                |
| iron_ore_price      | 0       | logret    | FRED         | PIORECRUSDM                              |
| timber_price        | 0       | logret    | FRED         | WPU081                                   |
| chemicals_ppi       | 0       | logret    | FRED         | WPU061                                   |
| uk_ftse250          | 0       | logret    | yfinance     | ^FTMC — UK domestic profit proxy         |
| uk_ftse100          | 0       | logret    | yfinance     | ^FTSE — international earnings proxy     |
| food_price_index    | 0       | logret    | FRED         | IMF PFOODINDEXM                          |
| wheat_price         | 0       | logret    | FRED         | PWHEAMTUSDM                              |
| vegetable_oil_price | 0       | logret    | FRED         | PSOYBUSDM (soybean proxy)                |
| uk_rents_lag1       | 0       | level     | dbnomics/ONS | L522 lagged 1m (pub_lag=0 after shift)   |
| uk_monthly_gdp      | 1       | yoy       | FRED         | GBRPROINDMISMEI (OECD industrial prod)   |
| uk_awg              | 1       | yoy       | ONS KAB9     | AWE whole economy weekly pay YoY%        |
| uk_vacancies        | 1       | logret    | dbnomics     | ONS AP2Y vacancies                       |
| uk_house_prices     | 2       | yoy       | FRED         | QGBR628BIS (BIS quarterly, ffill→monthly)|

**Excluded from runs:**
- `gas_eu_3m` — ablation shows Δ RMSE +0.024 (noise); `gas_eu` alone better
- `uk_paye` — identical to `uk_awg` (both KAB9 from ONS)
- `uk_rents` — collinear with `uk_rents_lag1` after pub-lag applied
- `uk_cpih`, `uk_services_cpi` — CPI sub-measures predicting CPI (circular)
- `gas_hh`, `us_ppi_all` — region=US, excluded from UK-only runs
- `battery_metals_proxy`, `global_supply_chain_pressure` — CSV-only, no live data

---

## Last Backtest Results (2015–2024, SHAP-screened 20 factors, 2026-06-06)

*`--shap-screen` is now default. Dropped 11 factors (see below). Results significantly better for UCM/TVP/SARIMAX.*

| Model             | RMSE  | MAE   | Dir%  | beats_AR1 |
|-------------------|-------|-------|-------|-----------|
| Combined-Static   | **0.277** | 0.209 | 89.2% | ✓      |
| Combined-Dynamic  | 0.288 | 0.217 | 89.2% | ✓         |
| Combined-Absolute | 0.338 | 0.245 | 91.9% | ✓         |
| ElasticNet        | 0.338 | 0.245 | 91.9% | ✓         |
| MedianElasticNet  | 0.364 | 0.242 | 91.9% | ✓         |
| UCM               | 0.370 | 0.286 | 89.2% | ✓         |
| TVP               | 0.413 | 0.308 | 89.2% | ✓         |
| RegimeEns         | 0.490 | 0.340 | 89.2% | ✓         |
| SARIMAX           | 0.491 | 0.363 | 89.2% | ✓         |
| **AR(1)**         | 0.495 | 0.322 | 93.3% | baseline  |
| RMC-dfm_k2        | 0.545 | 0.345 | 89.2% | ✗         |
| HuberNet          | 0.501 | 0.370 | 89.2% | ✗         |
| BridgeEq          | 0.658 | 0.522 | 100%  | ✗ (n=25)  |
| MIDAS             | 0.758 | 0.547 | 100%  | ✗ (n=25)  |
| PCR               | 0.787 | 0.587 | 91.9% | ✗         |
| BVAR              | 1.093 | 0.728 | 89.2% | ✗         |
| DFM               | 1.112 | 0.767 | 91.9% | ✗         |
| DFM-k2            | 1.176 | 0.790 | 91.9% | ✗         |
| CopulaReg         | 1.486 | 0.822 | 86.5% | ✗         |
| GBM               | 1.657 | 0.892 | 91.9% | ✗         |
| VAR               | 2.008 | 1.147 | 91.9% | ✗         |
| AutoARIMA         | 2.097 | 1.270 | 91.9% | ✗         |
| RAMM-LGBM         | 2.167 | 1.345 | 91.9% | ✗         |
| MS-DFM            | 2.627 | 1.791 | 91.9% | ✗         |
| HiddenRF          | 2.746 | 1.731 | 91.9% | ✗         |
| HMM               | 2.820 | 1.833 | 91.9% | ✗         |

*RMC-dfm_k2: k=2 DFM latent factor KMeans regime labels. r1 survivors (8): UCM, TVP, BridgeEq, ElasticNet, MedianElasticNet, HuberNet, RegimeEns, SARIMAX. r0 has no AR(1) baseline (too few obs).*

**SHAP dropped** (mean |SHAP| < 0.001): gas_eu, oil_vol_6m, gas_eu_3m, gbpusd_3m,
metals_index, copper_price, nickel_price, iron_ore_price, food_price_index,
vegetable_oil_price, uk_vacancies → **20 factors kept**

**Improvement vs 30-factor (no-screen) run**: UCM 0.605→0.370 (-39%), TVP 0.529→0.413 (-22%),
SARIMAX 0.857→0.491 (-43%), Combined-Static 0.310→0.277 (-11%), 9 models beat AR(1) (was 6)

**RMC Metamodel RMSE (regime-model-combine, --rmc flag):**

| RMC Method    | RMSE  | vs AR(1) | Survivors (r1)                              |
|---------------|-------|----------|--------------------------------------------|
| RMC-dfm_k2    | 0.545 | worse    | UCM, TVP, BridgeEq, ElasticNet, MedianElasticNet, HuberNet, RegimeEns, SARIMAX |

No RMC method beats Combined-Static (0.277) or Combined-Dynamic (0.288). Regime-aware selection is useful for understanding which models work in each macro environment.

**Note:** DM test shows p<0.10 for none vs AR(1) — low power at n=37 (quarterly step).
Combined models reduce RMSE 37% vs AR(1) (0.310 vs 0.495).

**Top factor importances (SHAP-screened 20-factor run):**
- ElasticNet: uk_rents_lag1=2.015, cpi_3m_chg=0.149, **gbp_eur=0.111**, **chemicals_ppi=0.093**, vix=0.059
- TVP: uk_rents_lag1=1.162, vix=0.126, **uk_ftse250=0.121**, oil_brent_3m=0.120, **gbp_eur=0.119**
- UCM: uk_rents_lag1=2.224, vix=0.201, uk_be5=0.168, **uk_house_prices=0.167**, **uk_ftse250=0.156**
- DFM: oil_brent_3m=0.886, oil_brent=0.849, **chemicals_ppi=0.616**, **uk_ftse250=0.453**, **uk_ftse100=0.410**
- BVAR: uk_rents_lag1=0.241, **uk_monthly_gdp=0.104**, uk_be5=0.097, uk_gilt_10y=0.052

---

## Current Nowcast (May 2026)

*(2026-06-07, SHAP-screened 20 factors)*

| Model             | Nowcast (%) |
|-------------------|-------------|
| AutoARIMA         | 2.04        |
| PCR               | 2.29        |
| RegimeEns         | 2.31        |
| UCM               | 2.60        |
| MIDAS             | 2.60        |
| BridgeEq          | 2.61        |
| TVP               | 2.63        |
| VAR               | 2.66        |
| MedianElasticNet  | 2.82        |
| HuberNet          | 2.86        |
| GBM               | 2.99        |
| CopulaReg         | 3.10        |
| SARIMAX           | 3.16        |
| DFM               | 3.17        |
| ElasticNet        | 3.31        |
| DFM-k2            | 3.63        |
| BVAR              | 3.68        |
| HiddenRF          | 1.88        |
| MS-DFM            | 4.40        |
| HMM               | 4.57        |
| RAMM-LGBM         | 4.19        |

**Reliable consensus (UCM/TVP/MedianElasticNet/BridgeEq): ~2.6–2.7% YoY for May 2026**
*(April 2026 actual: 3.5%; high-RMSE models cluster 4–4.6%, unreliable)*

---

## Key Files

| File                               | Purpose                                                |
|------------------------------------|--------------------------------------------------------|
| `code/factors.py`                  | 30-factor registry, `apply_publication_lags()`, fetchers |
| `code/uk_model_zoo.py`             | 21 models + score_backtest() + nowcast()               |
| `code/nowcast_cpi.py`              | Main runner: backtest, ensembles, RMC, nowcast output  |
| `code/nowcast_plot.py`             | UCM/TVP 6-month forward forecast plot                  |
| `code/plot_nowcast_history.py`     | Regenerates `plots/nowcast_history_3.png` from CSVs   |
| `code/test_nowcast_cpi.py`         | 15 tests (factors, models, pub-lag discipline)         |
| `data/nowcast_cpi_backtest.csv`    | Backtest predictions (all models, all periods)         |
| `data/nowcast_cpi_metrics.csv`     | RMSE/MAE/dir_acc/beats_ar1 per model                   |
| `data/nowcast_cpi_nowcast.csv`     | Latest nowcast per model                               |
| `data/rmc_dfm_k2_perf.csv`         | RMC per-regime per-model RMSE                          |
| `plots/nowcast_history_3.png`      | Backtest predictions vs actual plot                    |

---

## Environment

```bash
cd /Users/Adam/Documents/home/quant/nowcast
FRED_API_KEY=<key> .venv/bin/python -W ignore code/nowcast_cpi.py --start 2015 --end 2024 --train-from 1992
```

Required packages: statsmodels, lightgbm, xgboost, scikit-learn, shap,
fredapi, yfinance, dbnomics, pandas, numpy, scipy, requests, openpyxl

**FRED_API_KEY:** env-only; never hardcode in any file.
