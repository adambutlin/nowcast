# PROCESS — Build Log

Chronological record of design decisions and implementation changes.
Each entry: date, what changed, why.

---

## 2026-06-05 — Session 3 (regime framework + mixed frequency)

### Changes
- **Removed MIDAS** from `uk_model_zoo.all_models()`. Monthly Almon distributed-lag
  model is not genuine mixed-frequency (no intra-month data). RMSE=3.243, bias=+2.9pp,
  DirAcc=51.5%. Left class in zoo for reference but excluded from runs.

- **Added `pub_lag` field to all `factors.py` REGISTRY entries.** Encodes how many
  months after the reference month each factor becomes available before CPI publication.
  pub_lag=0: financial data (oil, FX, VIX, PMI flash, ISM, BoE yields).
  pub_lag=1: ONS releases (uk_rents, uk_vacancies, CPI components).

- **Added `apply_publication_lags()` to `factors.py`.** Shifts each factor by its
  pub_lag. Called in `compare_uk.py` before all model runs. Enforces real-time
  information set discipline.

- **Added new factors to `factors.py` registry:**
  - `us_ism_pmi`: FRED NAPM (ISM Manufacturing PMI). pub_lag=0.
  - `us_ppi_all`: FRED PPIACO pct_change×12. pub_lag=0. US cost-push proxy.
  - `uk_cpih`: dbnomics ONS MM23 L55O YoY. pub_lag=1.
  - `uk_services_cpi`: dbnomics ONS MM23 D7G9. pub_lag=1.
  - `uk_core_cpi`: CSV drop-in only. pub_lag=1.
  - `uk_ppi_output`: CSV drop-in only. pub_lag=1.
  - `uk_trimmed_mean_cpi`: CSV drop-in only (BoE/ONS experimental). pub_lag=1.
  - `uk_pmi_composite/manufacturing/services/input_prices`: CSV drop-ins. pub_lag=0.
  - `ibes_revisions_12m`: CSV drop-in (FactSet/IBES). pub_lag=0.
  - `uk_hciall`: CSV drop-in (ONS HCIS experimental). pub_lag=1.
  - `vimes_boots_index`: CSV drop-in (construct from ONS microdata/JRF). pub_lag=1.

- **Added regime-model-combine (RMC) framework to `compare_uk.py`:**
  - 4 regime methods: HMM, LSTAR, DFM-factor, manual VIX
  - Per-regime expanding-window backtest for each model
  - AR(1) per-regime gate: keeps models beating AR(1) within regime
  - Metamodel: current regime → surviving models → equal-weight
  - Run with `--rmc` flag (off by default, adds ~5 min)
  - Output: RMC-hmm, RMC-lstar, RMC-dfm, RMC-manual_vix backtests + perf tables

- **Updated `compare_uk.py` main():**
  - Applies publication lags before all model runs
  - Prints pub_lag=0 vs pub_lag≥1 factor breakdown
  - Leakage probe now uses raw df (pre-lag) to show the underlying leakage
  - Saves rmc_<method>_perf.csv for each RMC method

- **Wrote STATE.md, SPEC.md, PROCESS.md, updated README.md.**

### Why

Mixed-frequency discipline: with pub_lag=0 factors, models correctly use
contemporaneous financial data for month T. With pub_lag=1, ONS leakage is
eliminated. The +0.209pp RMSE "improvement" from uk_rents at lag=0 was artificial.

Regime-model-combine tests the hypothesis that regime-specific training is better
than model-internal regime handling. Previous data showed monotonic RMSE degradation
with regime-layer count (0-layer=0.092, 1-layer=0.492, 2-layer=1.556), but that
was for explicit regime-switching inside a model. RMC tests whether selecting
WHICH model to use per-regime can outperform letting the model handle it internally.

---

## 2026-06-05 — Session 2 (11-model zoo)

### Changes
- Fixed `MS_DFM._dfm_smooth()` undefined-method bug. Added method using
  `DynamicFactor(...).smooth(params)` returning results with `filtered_state`.

- Added 4 models to `uk_model_zoo.py`:
  - `BVAR` (Minnesota ridge prior, p=3, lambda0=0.3, diagonal penalty ∝ lag²)
  - `MIDAS` (Almon polynomial DL, K=6, degree=2) — later removed
  - `HiddenRF` (K-means + per-regime RF + soft distance weights)
  - `GBM` (XGBoost preferred, sklearn fallback; AR-augmented)

- Added `dm_test(e1, e2, h=1)` and `score_backtest(bt, name)` helpers.

- Built `compare_uk.py` from scratch:
  - ar1_backtest() (expanding OLS, not O(n²) per-obs refitting)
  - combine_static(), combine_dynamic()
  - probe_leakage() — confirmed uk_rents leakage +0.209pp RMSE
  - error_corr_matrix(), greedy_uncorrelated_subset()
  - spa_table() (Diebold-Mariano vs AR(1))
  - Combined-Superstar (DM>0, p<0.10) = UCM + TVP
  - Combined-Absolute (greedy uncorrelated) = UCM + DFM + LSTAR

- Added `oil_vol_6m` and `gbpusd_vol_6m` to `factors.py`.
  With 9 factors, gbpusd_vol_6m=0.277 and vix=0.282 are top DFM loadings.
  Adding vol factors cost DFM its DM significance (p: 0.002 → 0.197).

- Added `uk_rents_lag1` to `factors.py` (leakage-free version of uk_rents).

### Errors Fixed
- MIDAS `importance()`: `X.columns[ok]` used boolean mask as column selector → fixed
- BVAR/MIDAS `importance()` dimension mismatch → inline permutation on full matrix
- GBM sklearn NaN: test features with NaN → `.ffill().fillna(train mean)`
- Greedy subset selected MIDAS (ρ≈0 passes correlation gate despite RMSE=3.243) →
  added AR(1) RMSE gate: only models with RMSE < 1.5×AR(1) included

---

## 2026-06-04 — Session 1 (DFM + backtest)

### Changes
- Built `backtest_2025.py` with 3 models: RAMM-LGBM CPI MoM, RAMM-LGBM 2Y
  repricing, DFM UK CPI YoY.
- Added BoE gilt-implied 5Y breakeven benchmark for UK DFM (replacing Cleveland
  Fed EXPINF1YR which is US-only).
  Formula: nom_spot_5y − real_spot_5y, adjusted −1pp for RPI-CPI wedge.
  `get_boe_spot_5y()` downloads BoE yield curve ZIPs, parses sheet "4. spot curve".
- Pushed to adambutlin/nowcast (repo named nowcast, not nowcaster).
- DFM results: RMSE=1.94 (full) vs BoE BE 2.65 vs AR(1) 3.15.

---

## Pending / Future Work

- [ ] Re-run full backtest with publication-lag-corrected factor matrix
- [ ] Drop CSV data for uk_pmi_composite, uk_services_cpi, uk_core_cpi, uk_ppi_output
- [ ] Run `--rmc` and report regime-model-combine results
- [ ] Evaluate k=2 DFM (two latent factors: global-risk + domestic-services)
      to distinguish Regime A (1995-2007 globalization) from Regime E (2024+ disinflation)
- [ ] ElasticNet model (L1+L2 mixed; likely between BVAR and LSTAR performance)
- [ ] Post-1992 vs post-2005 training window expansion test
- [ ] Shapley factor screening in factors.py (keep candidates above threshold)
- [ ] Push updated files to adambutlin/nowcast
