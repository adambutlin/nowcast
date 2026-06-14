# SPEC — UK CPI Nowcast System Specification

*Version: 2.1 — 2026-06-13 (added §10 Rates Repricing subsystem; §9 vintage note updated by H6/C3 remediation)*

---

## 1. Problem Statement

**Target:** UK CPI All Items YoY (ONS D7G7.M), monthly, in percent.
**Task:** 1-step-ahead nowcast — forecast CPI(T) using information available
before the ONS release of CPI(T) (~16th of month T+1).
**Evaluation:** Expanding-window backtest 2015–present; all models use 1-step-ahead
discipline (no multi-step horizon compression).

---

## 2. Mixed-Frequency Information Set

CPI(T) is published ~16th of month T+1. The real-time information set at that
date contains:

| Data type                  | pub_lag | Available at forecast time?  |
|----------------------------|---------|------------------------------|
| Financial spot (oil, FX)   | 0       | Full month T value — YES      |
| VIX, yields, breakevens    | 0       | Full month T value — YES      |
| PMI flash estimates        | 0       | Released last week of T — YES |
| ISM (US, released 1st T+1) | 0       | YES before UK CPI             |
| US PPI (~2nd week T+1)     | 0       | YES (same week as UK CPI)     |
| ONS CPI components         | 1       | Published same day as CPI(T)  |
| ONS vacancies              | 1       | Published with or after CPI   |
| ONS PAYE/house prices      | 1-2     | Published after CPI           |

**Implementation:** `F.apply_publication_lags(df, live_facs)` shifts each factor
by its `pub_lag` before any model sees the data. Row T of the lagged df contains
the real-time information set for forecasting CPI(T).

**BVAR caveat:** BVAR's internal lag loop starts from lag=1, so a pub_lag=1 factor
in the pre-lagged df effectively appears at lag=2 in BVAR's regression. This is
conservative (loses one month of ONS signal) but correct (eliminates lag=0 leakage).

---

## 3. Factor Registry Design

`factors.py` REGISTRY entries have:
```
fetch     : callable → raw pd.Series, or None for CSV-only
transform : "level"|"yoy"|"mom"|"logret"|"diff"
pub_lag   : int — 0=contemporaneous, 1=one month, 2=two months
candidate : bool — True=screened by Shapley, False=always included
csv       : filename in data/ (CSV drop-in overrides live fetch)
note      : provenance/source string
```

**Core factors (candidate=False, always included):**
oil_brent, gbpusd, uk_be5, uk_rents_lag1

**Candidate factors (screened):**
vix, gas_hh, oil_vol_6m, gbpusd_vol_6m, us_ism_pmi, us_ppi_all,
uk_rents (deprecated—use uk_rents_lag1), uk_vacancies, uk_house_prices,
uk_paye, uk_cpih, uk_services_cpi, uk_core_cpi, uk_ppi_output,
uk_trimmed_mean_cpi, uk_pmi_*, ibes_revisions_12m, uk_hciall,
vimes_boots_index

**Adding a new factor:** add entry to REGISTRY with all fields; drop CSV to
`data/<name>.csv` if live fetch is unavailable.

---

## 4. Model Zoo

### 4.1 Model Interface

All models inherit `BaseModel` and implement:
- `_fit_predict_year(train, test, factors, target) → np.array`
- `importance(df, factors, target) → (pd.Series, type_str)`
- `regimes(df, factors, target) → (pd.Series|None, meta_dict)`

### 4.2 Models (10, as of 2026-06-05)

| Model     | Class    | Regime handling     | 1-step method                    |
|-----------|----------|---------------------|----------------------------------|
| DFM       | DFM      | Implicit (factor)   | .append(refit=False) monthly     |
| RAMM-LGBM | RAMM_LGBM| VIX monotone regs   | LightGBM + SHAP                  |
| UCM       | UCM      | Kalman adaptive     | .append(refit=False) monthly     |
| TVP       | TVP      | Kalman RW coefs     | Random-walk Kalman filter        |
| HMM       | HMM      | Explicit 2-state    | Filtered probs → regime mean     |
| MS-DFM    | MS_DFM   | DFM + Markov        | Causal filter pass               |
| LSTAR     | LSTAR    | Logistic transition | scipy least_squares              |
| BVAR      | BVAR     | None                | Minnesota ridge, p=3 lags        |
| HiddenRF  | HiddenRF | K-means soft weight | RF per cluster, distance weights |
| GBM       | GBM      | None                | XGBoost (sklearn fallback)       |

**Removed (2026-06-05):** MIDAS — monthly-only Almon DL is not genuinely
mixed-frequency; RMSE=3.243, bias=+2.9pp.

### 4.3 AR-Augmentation

RAMM-LGBM, GBM, HiddenRF add `cpi_lag1 = target.shift(1)` internally.
TVP, LSTAR include lagged target in their regression designs.
BVAR explicitly includes lagged target in the VAR block.

---

## 5. Backtest Discipline

**Expanding window:** training = all data up to year Y-1; test = year Y.
**Minimum training:** 60 months before first test year.
**1-step-ahead:** for state-space models, `.append(refit=False)` after each
month's realization; for regression models, test prediction uses only
lagged features (no leakage via same-month target).
**Publication lags:** applied to factor matrix before backtest — models see
the true real-time information set.

---

## 6. Combined Ensembles

| Name              | Method                                    | Notes                     |
|-------------------|-------------------------------------------|---------------------------|
| Combined-Static   | Equal-weight all 10 models               | Baseline combination      |
| Combined-Dynamic  | Inverse-RMSE, 12-month rolling window    | Adapts to model performance|
| Combined-Superstar| Equal-weight DM>0 p<0.10 models          | Currently: UCM+TVP        |
| Combined-Absolute | Greedy uncorrelated errors (ρ<0.5)       | UCM+DFM+LSTAR             |

---

## 7. Regime-Model-Combine (RMC) Framework

**Design philosophy:** regime-first (identify regime from multi-dim signal),
then apply regime-specific model subset. Contrast with model-regime-combine
(each model infers regime internally).

**Algorithm:**
1. Compute causal regime labels via HMM/LSTAR/DFM-factor/VIX-manual
2. For each regime r and model m:
   - Training data = data[year < Y AND regime == r] (min 30 obs, else full window)
   - Test data = data[year == Y AND regime == r]
   - Fit model, evaluate RMSE vs AR(1) within regime r
   - Keep model if RMSE < AR(1)_RMSE in regime r
3. Metamodel prediction: current_regime → surviving models → equal-weight average
4. Fall back to full ensemble if no survivors in current regime

**Regime methods:**
- `hmm`: 2-state MarkovRegression; fit on full data, forward-filter for causal labels
- `lstar`: G function from LSTAR fit; G>0.5 = upper regime
- `dfm`: filtered state sign from DFM(k=1); positive = r1
- `manual_vix`: VIX above expanding median = stress (r1)

**Run with:** `python compare_uk.py --rmc [--rmc-methods hmm lstar]`

---

## 8. Architecture Decisions and Trade-offs

### 8.1 UCM/TVP dominance
Kalman-filter models outperform all regime-explicit models. The Kalman filter
is the optimal implicit regime-switcher for a local-linear trend process: it
adapts O(1) parameters continuously vs O(k²) discrete-regime transition matrices.

### 8.2 uk_rents leakage
ONS L522 (uk_rents) is published same day as headline CPI. Using it contemporaneously
gives leakage lift of +0.209pp RMSE. Fixed by pub_lag=1. uk_rents_lag1 (pre-lagged
in registry) has pub_lag=0 and provides 0.0003pp of independent signal at lag=1.

### 8.3 oil_vol_6m / gbpusd_vol_6m
Derived factors from FRED Brent and GBP/USD. With 9 factors, these become
#1/#2 DFM loadings (gbpusd_vol_6m=0.277, vix=0.282). Adding them reduced
DFM DM significance (lost from p=0.002 with 7 factors to p=0.197 with 9 factors)
because they add noise to the k=1 latent factor.

### 8.4 Regime-model-combine data constraint
With ~400 monthly obs and 2+ regimes, each regime has ~100-200 obs for training
after the initial 60-month warmup. This is the minimum viable for regression models
but tight for state-space models (HMM, MS-DFM). The min_regime_train=30 guard
falls back to full-window training.

### 8.5 Post-1992 training start
Avoids ERM regime (Sept 1992 exchange rate crisis) which distorts UK inflation
and GBP dynamics. FRED Brent data from 1987 and GBP from 1971 are used for
factor history; model training from 1992.

---

## 9. Known Limitations

1. **No ALFRED vintage for UK data:** FRED's ALFRED (archival FRED) covers US series
   (CPILFESL, PAYEMS). UK ONS data has no real-time vintage database. The backtest
   uses revised data throughout — a minor look-ahead bias for UK factors.

2. **dbnomics series stability:** uk_house_prices (HPSSA/HPI.M) and uk_paye
   (RTI/median_pay.M) consistently fail to fetch. CSV drop-ins required.

3. **BVAR pub_lag conservatism:** pub_lag=1 factors appear 1 extra lag deep in BVAR's
   regression matrix (see §2 note). Effect is minor (ρ=0.92 between adjacent lags).

4. **PMI licensed:** UK PMI composite/services/manufacturing require S&P Global
   subscription. CSV drop-ins are the only route.

5. **Vimes Boots Index construction:** No standard free source. Must be constructed
   from ONS Consumer Price Microdata or JRF Minimum Income Standard data.

---

## 10. Rates Repricing Subsystem (`code/rates/`)

*Branch `alpha-gen`, tag `rates-alpha`. Downstream of the CPI nowcast.*

### 10.1 Hypothesis
Does the CPI nowcast contain information about future UK rates repricing **not
already embedded in consensus / market pricing?** Tradeable only via the
*surprise* (forecast − consensus), never the level.

### 10.2 Pipeline
`config.MODEL` forecast → `event_panel` (one row per CPI release; predictors known
at release eve T−1, outcome = release-day signed rate move in bp) → forecast gap →
**Stage 1** (`stage1`: gap predicts realized surprise?) → **Stage 2**
(`gates.gate2_incremental`: gap reprices rates, HAC, walk-forward) → regime /
confidence / risk → 2Y gilt position.

### 10.3 Causality contract
Predictor sources (nowcast, consensus, market-implied) must be knowable at T−1.
The only release-day read is the signed move = level(release) − level(prev
business day). All standardization is expanding with `shift(1)`. Walk-forward
estimation throughout. LDI window (2022-09-19…10-31) and budget months excluded.

### 10.4 Mechanical-identity guard (Stage 1)
A benchmark whose horizon/units mismatch the monthly print collapses the gap into
the CPI level and the test degenerates to forecast accuracy. The guard runs a
constant-anchor placebo (if it reproduces the slope, the anchor adds nothing) plus
a gap-vs-forecast-level correlation check, and returns `INVALID_MECHANICAL`. This
rejected the BoE 2.5Y-RPI breakeven benchmark.

### 10.5 Benchmark precedence (`event_panel._anchor`)
`economist_consensus` (incl. `consensus_cpi.csv`) → `market_implied` → `ucl` →
`naive_rw`. Real survey consensus is licensed; the shipped proxy is univariate
(AutoARIMA). Note: a univariate proxy is a **lower bar** than a professional survey
— a Stage-1 PASS against it is necessary, not sufficient.

### 10.6 Production layer
`regime` (causal policy×inflation regime + `regime_trust∈[0,1]`), `prod_signal`
(forecast_gap_z, revision_z, confidence = trust×strength), `risk` (LDI/budget
exclusion, vol kill switch, low-confidence suppression), `production`
(confidence-weighted vol-targeted position, backtest, regime attribution),
`run_production` (8-step workflow, `config.MODEL` switch).

### 10.7 Current verdict
No deployable edge on current free data. Stage 1 vs univariate consensus passes but
is tiny (OOS R²≈0.05) and fails pre-2020; Stage 2 / production repricing OOS R² is
negative (≈−0.28); the risk layer suppresses trading (latest live rec FLAT).
Decisive next test requires point-in-time survey consensus (`data/consensus_cpi.csv`).

---

## 11. Intramonth regime-dependent nowcasting (`code/intramonth/`)

**Goal:** nowcast UK CPI at forecast origins T-30…T-1 (calendar days before the
reference month-end), letting daily high-frequency data accrue intramonth.

**Layers:** (1) AutoARIMA baseline; (2) BVAR factor residual; (3) TVP; (4) MIDAS on
HF as-of features; (5) HMM 3-state regime detector. Residual framework: layers 2-4
predict `CPI − AutoARIMA`.

**Causality:** HF features at origin T-k use only daily rows ≤ (month_end − k days)
(`hf_data.asof_features`); monthly factors are pub-lagged; standardization is
walk-forward. Future-injection-invariance is unit-tested.

**Weights:** `w ∝ Σ_r post[r]·softmax(−RMSE_{m,r}/τ)·horizon_prior(k)` — regime- and
horizon-conditional, half-life decayed. Sums to 1.

**Scenario tree:** base / normalisation / shock (+ surprise tails) from the regime
posterior + HF momentum; points demeaned under the posterior so the probability-
weighted mean equals the model nowcast (coherence, not a fan chart).

### 11.1 Verdict (hostile ensemble review)
Regime weighting does **not** improve OOS RMSE vs flat-equal or AutoARIMA alone
(`ensemble_review.py`: regime 0.5185 > flat 0.5080 ≈ AutoARIMA 0.5086; Diebold–Mariano
rejects nothing, all p>0.29). The HF→regime mapping has near-zero OOS skill outside the
2022/23 energy shock. **The regime + scenario apparatus is an interpretation /
communication layer, not an alpha source.** The deployable point forecast is
`AutoARIMA + factor residual`; the scenario tree explains it but does not beat it.
