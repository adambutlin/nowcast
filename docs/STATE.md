# STATE — UK CPI Nowcast Project

*Last updated: 2026-06-07*

## What is Running

**Primary directory:** `/Users/Adam/Documents/home/quant/nowcast/`

**Primary entry point:** `code/main.py`
```bash
# Full 13-model backtest (blind test: --end 2024; 2025+ reserved)
FRED_API_KEY=<key> .venv/bin/python -W ignore code/main.py --start 2015 --end 2024 --train-from 1992 --shap-screen

# With regime-model-combine (HMM recursive labels):
FRED_API_KEY=<key> .venv/bin/python -W ignore code/main.py --start 2015 --end 2024 --train-from 1992 --shap-screen --rmc
```

**Folder structure:**
- `code/` — all Python source files
- `data/` — CSV outputs (backtest, metrics, nowcast, RMC perf)
- `plots/` — PNG outputs
- `logs/` — run logs

**13 operational models** in `code/uk_model_zoo.py` (`all_models()`):
DFM, DFM-k2, UCM, TVP, BVAR, MIDAS, BridgeEq,
ElasticNet, MedianElasticNet, HuberNet, PCR, SARIMAX, AutoARIMA

**9 experimental models** (`experimental_models()`, RMSE > 1.5×AR(1)):
RAMM-LGBM, HMM, MS-DFM, LSTAR, HiddenRF, GBM, CopulaReg, VAR, RegimeEnsemble

**4 combined ensembles:**
Combined-Static, Combined-Dynamic, Combined-Superstar, Combined-Absolute

**Regime-model-combine metamodel (--rmc flag):**
RMC-hmm (recursive HMM labels, fixed params, no refit)

---

## Live Factor Matrix (38 factors)

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
| cpi_3m_chg          | 0       | level     | derived      | 3m diff of cpi_yoy shifted 1m (post-lag) |

Plus 7 SHAP-derived volatility/momentum factors (oil_vol_6m, gbpusd_vol_6m, oil_brent_3m, gbpusd_3m already listed).

**Excluded from runs:**
- `gas_eu_3m` — ablation shows Δ RMSE +0.024 (noise); `gas_eu` alone better
- `uk_paye` — identical to `uk_awg` (both KAB9 from ONS)
- `uk_rents` — collinear with `uk_rents_lag1` after pub-lag applied
- `uk_cpih`, `uk_services_cpi` — CPI sub-measures predicting CPI (circular)
- `gas_hh`, `us_ppi_all` — region=US, excluded from UK-only runs
- `battery_metals_proxy`, `global_supply_chain_pressure` — CSV-only, no live data

---

## Last Backtest Results (2015–2024, 38 live factors, SHAP pre-2015 screen, 2026-06-07) — CORRECTED

**⚠ All results before 2026-06-07 used full-sample SHAP screen (lookahead bias) and only 7 live factors (SSL cert failure). These are the correct results.**

*38 live factors fetched (SSL certifi fix). Pre-2015 SHAP screen kept 24; dropped 7 (semiconductors_ppi, deep_sea_freight, metals_index, nickel_price, chemicals_ppi, uk_ftse250, uk_ftse100); gas_eu force-kept (always_keep). Effective rank: 6.7 of 24 (up from 3.4 on 7 factors). Most models n=112 (8 months lost to uk_house_prices quarterly lag); AR(1) n=120. 2024+ NaN for all factor models (quarterly BIS data not yet covering end-2024).*

| Model              | RMSE  | MAE   | Dir%  | mz_pval | DM p  | n   | beats_AR1 |
|--------------------|-------|-------|-------|---------|-------|-----|-----------|
| Combined-Dynamic   | 0.453 | 0.301 | 51.8% | 0.874   | 0.026**| 112 | ✓        |
| Combined-Static    | 0.454 | 0.302 | 51.8% | 0.886   | 0.027**| 112 | ✓        |
| Combined-Absolute  | 0.465 | 0.317 | 42.9% | 0.818   | 0.007**| 112 | ✓        |
| HuberNet           | 0.465 | 0.317 | 42.9% | 0.818   | 0.007**| 112 | ✓        |
| AutoARIMA          | 0.467 | 0.315 | 50.0% | 0.121   | 0.076* | 112 | ✓        |
| ElasticNet         | 0.470 | 0.312 | 44.6% | 0.759   | 0.001**| 112 | ✓        |
| UCM                | 0.473 | 0.326 | 49.1% | 0.160   | 0.042* | 112 | ✓        |
| TVP                | 0.482 | 0.321 | 49.1% | 0.149   | 0.645  | 112 | ✓        |
| SARIMAX            | 0.488 | 0.339 | 49.1% | 0.812   | 0.051* | 112 | ✓        |
| PCR                | 0.490 | 0.316 | 50.9% | 0.837   | 0.063  | 112 | ✓        |
| **AR(1)**          | **0.495** | 0.322 | 45.8% | 0.415 | —    | 120 | baseline |
| DFM-k2             | 0.496 | 0.327 | 44.6% | 0.762   | 0.000  | 112 | ✗        |
| DFM                | 0.499 | 0.322 | 46.4% | 0.893   | 0.002  | 112 | ✗        |
| MIDAS              | 0.515 | 0.371 | 59.2% | 0.478   | 0.190  |  76 | ✗        |
| BridgeEq           | 0.520 | 0.381 | 59.2% | 0.250   | 0.292  |  76 | ✗        |
| MedianElasticNet   | 0.575 | 0.383 | 43.8% | 0.002   | 0.000  | 112 | ✗        |
| BVAR               | 0.678 | 0.477 | 48.2% | 0.000   | 0.000  | 112 | ✗        |
| RegimeEns ⚠        | 1.202 | 0.549 | 48.2% | 0.000   | 0.003  | 112 | ✗        |
| Combined-Superstar | NaN   | —     | —     | —       | —     | —   | ✗ (BH: none) |

*⚠ RegimeEns RMSE=1.202 (2.4×AR1) driven by 2020-21 blowup (2.429 subsample RMSE during COVID). Moved to experimental_models().*
*Combined-Absolute = HuberNet only (greedy uncorrelated subset, ρ<0.5 gate selects 1 model).*
*Combined-Superstar = empty: BH FDR correction at 10% kills all DM candidates.*
*MIDAS/BridgeEq n=76: MIDAS daily data coverage shorter; RMSE not comparable to n=112 models.*

**DM test significance:** Combined-Static/Dynamic significant at p<0.05. HuberNet/ElasticNet/AutoARIMA/UCM/SARIMAX significant or borderline at p<0.10. No model passes BH-corrected FDR at 10% across all 14 models.

**24 SHAP-selected factors (from 38 live):**
`oil_brent, gbpusd, uk_be5, vix, gas_eu*, uk_gilt_10y, oil_vol_6m, gbpusd_vol_6m, oil_brent_3m, gbpusd_3m, gbp_eur, gbp_eer, copper_price, iron_ore_price, timber_price, uk_monthly_gdp, uk_awg, food_price_index, wheat_price, vegetable_oil_price, uk_rents_lag1, uk_vacancies, uk_house_prices, cpi_3m_chg`
(*gas_eu force-kept via --always-keep; would have been SHAP-dropped as episodic factor)

**Subsample RMSE (38-factor, pre-2015 SHAP):**
```
                    2015-19  2020-21  2022-23  2024+
MIDAS                 0.159    0.522    0.706    NaN   n=76
BridgeEq              0.182    0.518    0.717    NaN   n=76
AutoARIMA             0.192    0.556    0.769    NaN
AR(1)                 0.195    0.555    0.859   0.405   (only model with 2024+ data)
DFM                   0.196    0.555    0.844    NaN
PCR                   0.200    0.485    0.837    NaN
Combined-Dynamic      0.201    0.491    0.756    NaN
Combined-Static       0.202    0.491    0.757    NaN
ElasticNet            0.211    0.473    0.789    NaN
DFM-k2                0.214    0.555    0.823    NaN
HuberNet              0.228    0.480    0.755    NaN
UCM                   0.231    0.530    0.767    NaN
SARIMAX               0.236    0.528    0.795    NaN
TVP                   0.246    0.562    0.765    NaN
BVAR                  0.312    0.455    1.260    NaN
RegimeEns             0.215    2.429    0.828    NaN   ← 2020-21 COVID blowup
```
*2024+ NaN for factor models: uk_house_prices (BIS quarterly) doesn't cover end-2024 → late-2024 rows dropped from factor matrix.*
*CI: ±RMSE × 1.96/√(2n) per period (2015-19: ±18%, 2020-21: ±28%, 2022-23: ±28%).*
*Differences within shock periods (2020-21, 2022-23) not statistically distinguishable.*

**2025 blind test (true OOS, n=12, run 2026-06-07, 7-factor run):**

| Model | 2025 RMSE | beats AR(1)=0.349 |
|---|---|---|
| Combined-Absolute | 0.295 | ✓ |
| Combined-Static | 0.326 | ✓ |
| ElasticNet | 0.330 | ✓ |
| AR(1) | 0.349 | baseline |
| TVP | 0.351 | ✗ |
| UCM | 0.361 | ✗ |

All 2025 MZ slopes ~0.5-0.6 (should be 1.0): models predict compressed range, over-predicting as CPI falls from 2022-23 peak. Systematic bias from energy-shock training era. Priority 3 (deferred): rolling mean-error bias correction.

**RMC-hmm (recursive labels, string-label bug fixed, 7-factor run, 2026-06-07):**
```
r0 (stable/low-inflation): AR(1)=0.334. No model beats within-regime AR(1) → falls back to full ensemble
r1 (high-inflation):       AR(1)=0.543. SARIMAX=0.522 survives (only survivor)
RMC-hmm metamodel RMSE = 0.485 vs AR(1)=0.495 → slightly beats AR(1)
vs Combined-Dynamic=0.453/0.457 → still doesn't beat best ensemble
```

---

## Current Nowcast (May 2026)

*(Stale — from 7-factor run 2026-06-07. Rerun with 38 factors for current values.)*

**Reliable consensus (UCM/TVP/ElasticNet/BridgeEq): ~2.6–2.8% YoY for May 2026**
*(April 2026 actual: 3.5%)*

---

## Key Files

| File                               | Purpose                                                |
|------------------------------------|--------------------------------------------------------|
| `code/factors.py`                  | 38-factor registry, `apply_publication_lags()`, fetchers |
| `code/uk_model_zoo.py`             | 13 operational + 9 experimental models + scoring       |
| `code/main.py`                     | Main runner: backtest, ensembles, RMC, nowcast output  |
| `code/tests/test_main.py`          | 16 unit tests (factors, models, pub-lag discipline)    |
| `data/nowcast_cpi_backtest.csv`    | Backtest predictions (all models, all periods)         |
| `data/nowcast_cpi_metrics.csv`     | RMSE/MAE/dir_acc/beats_ar1 per model                   |
| `data/nowcast_cpi_nowcast.csv`     | Latest nowcast per model                               |
| `plots/nowcast_history_3.png`      | Backtest predictions vs actual plot                    |
| `logs/exp_38fac.log`               | Full 38-factor backtest run log (authoritative)        |

---

## Environment

```bash
cd /Users/Adam/Documents/home/quant/nowcast
FRED_API_KEY=<key> .venv/bin/python -W ignore code/main.py --start 2015 --end 2024 --train-from 1992 --shap-screen 2>&1 | tee logs/run.log
```

Required packages: statsmodels, lightgbm, xgboost, scikit-learn, shap,
fredapi, yfinance, dbnomics, pandas, numpy, scipy, requests, openpyxl, certifi

**FRED_API_KEY:** env-only; never hardcode in any file.

---

## Deferred Work

- **Priority 3:** Bias correction — rolling 12-month mean-error correction for post-energy-shock compression bias (all MZ slopes ~0.5-0.6 in 2025)
- **Priority 4:** Regularize SHAP threshold via cross-validation (currently fixed at default)
- **Priority 5:** Extend training data as 2025 data accumulates
- **Option B OOS:** Pseudo-OOS with 5 vintage cutoffs (implementable; see HANDOFF for design)
- **RegimeEns investigation:** Why 2020-21 blowup with 38 factors but not 7? Regime misclassification during COVID novel shock. Worth diagnosing before re-admitting to all_models().
- **MIDAS/BridgeEq n=76:** Investigate why MIDAS coverage shorter than other models in 38-factor run; daily cache issue?
