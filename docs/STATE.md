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

## Last Backtest Results (2015–2024, SHAP-screened 7 factors, 2026-06-07) — CORRECTED

**⚠ Previous results (2026-06-06) used full-sample SHAP screen (lookahead bias). These results use pre-2015 SHAP screen (correct OOS discipline). All prior metrics are invalid.**

*7 live factors available (data fetch issues reduced set from 30). SHAP pre-2015 screen kept all 7. Effective factor rank: 3.4 (FTSE250/FTSE100 ρ=0.80 nearly redundant).*

| Model             | RMSE  | MAE   | Dir%  | mz_pval | beats_AR1 |
|-------------------|-------|-------|-------|---------|-----------|
| Combined-Absolute | 0.455 | 0.309 | 52.5% | 0.093   | ✓         |
| RegimeEns         | 0.455 | 0.309 | 52.5% | 0.093   | ✓ (=Abs)  |
| Combined-Static   | 0.457 | 0.307 | 52.5% | 0.525   | ✓         |
| Combined-Dynamic  | 0.457 | 0.308 | 52.5% | 0.493   | ✓         |
| HuberNet          | 0.464 | 0.307 | 52.5% | 0.923   | ✓         |
| UCM               | 0.468 | 0.325 | 44.2% | 0.057   | ✓         |
| AutoARIMA         | 0.469 | 0.321 | 48.3% | 0.051   | ✓         |
| TVP               | 0.472 | 0.325 | 53.3% | 0.001   | ✓         |
| ElasticNet        | 0.476 | 0.309 | 47.5% | 0.860   | ✓         |
| SARIMAX           | 0.481 | 0.327 | 40.0% | 0.314   | ✓         |
| PCR               | 0.487 | 0.314 | 50.0% | 0.642   | ✓         |
| DFM               | 0.491 | 0.317 | 46.7% | 0.848   | ✓         |
| **AR(1)**         | **0.495** | 0.322 | 45.8% | 0.415 | baseline |
| DFM-k2            | 0.499 | 0.329 | 44.2% | 0.618   | ✗         |
| MedianElasticNet  | 0.556 | 0.374 | 45.8% | 0.000   | ✗         |
| BVAR              | 0.613 | 0.426 | 47.5% | 0.000   | ✗         |
| CopulaReg         | 1.445 | 0.732 | 38.3% | 0.000   | ✗         |
| GBM               | 1.594 | 0.853 | 40.0% | 0.306   | ✗         |
| HiddenRF          | 1.666 | 0.890 | 45.0% | 0.000   | ✗         |
| LSTAR             | 1.731 | 1.120 | 42.7% | 0.000   | ✗         |
| RAMM-LGBM         | 1.737 | 0.966 | 42.5% | 0.024   | ✗         |
| HMM               | 2.699 | 1.718 | 40.8% | 0.111   | ✗         |
| MS-DFM            | 2.763 | 1.854 | 42.5% | 0.349   | ✗         |
| Combined-Superstar | NaN  |  —    |  —    |  —      | ✗ (BH: none) |

*Combined-Superstar = empty after BH FDR correction at 10% — prior "UCM+TVP" result was multiple-testing noise.*
*Combined-Absolute = RegimeEns only (degenerate: greedy subset finds single model).*
*MIDAS/BridgeEq/VAR = 0 rows (data unavailable in this run).*

**DM test** (HLN-corrected, n=120 monthly obs): only HuberNet significant at p<0.05. No model passes BH-corrected FDR at 10%.

**Key factor importances (7-factor run):**
- RAMM-LGBM (mean |SHAP|): cpi_lag1=1.114, uk_rents_lag1=0.754, cpi_3m_chg=0.140, vix=0.112, uk_awg=0.097
- UCM (std |coef|): uk_rents_lag1=0.261, vix=0.084, uk_vacancies=0.067, uk_awg=0.026
- TVP (mean |β·x|): uk_rents_lag1=0.194, uk_ftse100=0.068, uk_awg=0.056, uk_ftse250=0.053
- ElasticNet (coef): cpi_lag1=1.999, cpi_3m_chg=0.114, uk_awg=0.036, uk_vacancies=0.027
- DFM (loading): uk_ftse250=0.962, uk_ftse100=0.841, vix=0.285

**Subsample RMSE (corrected, pre-2015 SHAP):**
```
             2015-19  2020-21  2022-23  2024+   (CI: ±0.18/0.28/0.28/0.40 ×RMSE)
MIDAS          0.159    0.527    0.709  0.390   [n=84 partial]
BridgeEq       0.180    0.523    0.718  0.391   [n=84 partial]
PCR            0.182    0.548    0.836  0.456
ElasticNet     0.186    0.525    0.829  0.411
Combined-S/D   0.187    0.530    0.775  0.389
AR(1)          0.195    0.555    0.859  0.405
UCM            0.202    0.535    0.783  0.437
BVAR           0.263    0.631    1.072  0.557
CopulaReg      0.314    0.541    3.135  0.396   ← blows up 2022-23
```
Differences within 2020-21 and 2022-23 columns are not statistically distinguishable (CI ±28% of RMSE).

**2025 blind test (true OOS, n=12, run 2026-06-07):**

| Model | 2025 RMSE | beats AR(1)=0.349 |
|---|---|---|
| Combined-Absolute | 0.295 | ✓ |
| LSTAR | 0.295 | ✓ (degenerate) |
| Combined-Static | 0.326 | ✓ |
| ElasticNet | 0.330 | ✓ |
| AR(1) | 0.349 | baseline |
| UCM | 0.361 | ✗ |
| TVP | 0.351 | ✗ |

All 2025 MZ slopes ~0.5-0.6 (should be 1.0): models predict compressed range, over-predicting as CPI falls from 2022-23 peak. Systematic bias from energy-shock training era.

**RMC-hmm (recursive labels, string-label bug fixed, 2026-06-07):**
```
Method: HMM recursive
  r0 (stable/low-inflation): AR(1)=0.334. No model beats within-regime AR(1) → falls back to full ensemble
  r1 (high-inflation):       AR(1)=0.543. SARIMAX=0.522 survives (only survivor)
  RMC-hmm metamodel RMSE = 0.485 vs AR(1)=0.495 → slightly beats AR(1)
  vs Combined-Dynamic=0.457 → still doesn't beat best ensemble
  Note: previous RMC code had int vs str label bug (silently always fell back to full ensemble)
```

**Gas_eu status:** fetch fails in current environment (TTF yfinance requires live session). `--always-keep ["gas_eu"]` logic correct but untested until live fetch works. Run with FRED_API_KEY in an environment where yfinance TTF=F downloads.

**To reproduce:**
```bash
FRED_API_KEY=<key> .venv/bin/python -W ignore code/main.py --start 2015 --end 2024 --train-from 1992 --shap-screen 2>&1 | tee logs/run.log
```

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
