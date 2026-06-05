# nowcast

UK CPI YoY and US macroeconomic nowcasting. 10-model zoo with mixed-frequency
factor discipline, regime identification, ensemble combination, and
regime-model-combine framework.

**Documentation:**
- [STATE.md](STATE.md) — current results, factor matrix, pending work
- [SPEC.md](SPEC.md) — system specification and design decisions
- [PROCESS.md](PROCESS.md) — chronological build log

---

## Models

### US Models (`ramm_lgbm_v1.py`)

#### Model 1 — RAMM-LGBM: US Core CPI MoM

**Target:** US Core CPI MoM (CPILFESL, decimal)  
**Features:** Brent returns, payroll growth, 5Y/10Y TIPS breakevens, VIX, volatility features, VIX-regime  
**Monotone:** Oil ↑→CPI↑, payrolls ↑→CPI↑, breakevens ↑→CPI↑  
**Data:** FRED (CPILFESL, PAYEMS, T5YIE, T10YIE); yfinance (Brent, VIX)

| Period | RMSE | MAE | Dir% |
|--------|------|-----|------|
| Full 2015–now | 0.00166 | 0.00108 | 97.0% |
| 2025 holdout | 0.00152 | 0.00121 | 100.0% |

Benchmarks (full sample): AR(1) 0.00144 · T5YIE/12 0.00145 · Cleveland Fed 0.00147

#### Model 2 — RAMM-LGBM: 2Y Treasury Repricing

| Period | RMSE | MAE | Dir% |
|--------|------|-----|------|
| Full 2015–now | 0.245 | 0.179 | 54.5% |
| 2025 holdout | 0.158 | 0.144 | 50.0% |

Benchmarks: AR(1) 0.245 · Zero-forecast 0.241

---

## UK CPI YoY Model Zoo

**Target:** UK CPI YoY (ONS D7G7.M via dbnomics, %)  
**Factors:** 9 live (oil_brent, gbpusd, uk_be5, vix, gas_hh, oil_vol_6m, gbpusd_vol_6m, uk_rents, uk_vacancies)  
**Mixed-frequency:** pub_lag applied per factor — financial data contemporaneous (pub_lag=0), ONS data lagged 1 month (pub_lag=1)  
**Backtest:** Expanding window 2015–present, 1-step-ahead for all models  
**Training start:** 1992 (post-ERM)

### Last Results (9-factor, pre pub-lag fix — re-run pending)

| Model | RMSE | DM vs AR(1) | Notes |
|-------|------|-------------|-------|
| UCM | **0.092** | +4.21 ** | Tier 1: Kalman local-level |
| Combined-Superstar | 0.120 | +4.15 ** | Tier 3: UCM+TVP equal-weight |
| TVP | 0.182 | +3.85 ** | Tier 1: Kalman time-varying params |
| Combined-Dynamic | 0.276 | +3.41 ** | Tier 3: inverse-RMSE rolling-12m |
| DFM | 0.473 | +1.29 | Tier 2: dynamic factor (k=1) |
| **AR(1) benchmark** | **0.480** | — | |
| LSTAR | 0.492 | +0.08 | Tier 2: logistic smooth transition |
| BVAR | 0.591 | −3.46 ** | Minnesota ridge, p=3 |
| GBM | 1.383 | −3.20 ** | XGBoost AR-augmented |
| HiddenRF | 1.556 | −3.35 ** | K-means + per-regime RF |
| RAMM-LGBM | 1.681 | −3.92 ** | Monotone LGBM |
| HMM | 2.581 | −4.96 ** | Markov 2-state (CPI-only) |
| MS-DFM | 2.814 | −5.55 ** | DFM factor + Markov |
| ~~MIDAS~~ | ~~3.243~~ | — | **Removed** — not genuine MF |

DM > 0 = model beats AR(1); ** p<0.05.

### Factor Importance (cross-model consensus, 9-factor run)

| Rank | Factor | pub_lag | Signal |
|------|--------|---------|--------|
| 1 | `uk_rents` | 1 | ONS private rents YoY (large basket share, persistent) |
| 2 | `cpi_lag1` | — | AR(1) persistence (auto-added by tree models) |
| 3 | `vix` | 0 | Global risk — #1 DFM loading (0.282) |
| 4 | `gbpusd_vol_6m` | 0 | Import uncertainty — #2 DFM loading (0.277) |
| 5 | `uk_be5` | 0 | BoE gilt 5Y breakeven — inflation expectations |
| 6 | `uk_vacancies` | 1 | Labour market tightness → services inflation |
| 7 | `oil_brent` | 0 | Energy cost-push |

### Regime-Model-Combine Framework

Run with `--rmc` flag. Tests whether regime-first model selection outperforms
models that handle regimes internally (UCM/TVP Kalman adaptation).

Four regime methods: HMM (Markov filtered), LSTAR (G function), DFM (factor sign),
manual VIX (expanding-median threshold). For each method: trains every model on
regime-specific data, keeps models beating AR(1) within regime, builds metamodel.

Prior result: UCM/TVP dominate because Kalman filter is the optimal implicit
regime-switcher with O(1) adaptive parameters vs O(k²) discrete-regime HMM.
RMC expected to improve at regime transition months (~15% of backtest horizon).

---

## Files

| File | Description |
|------|-------------|
| `ramm_lgbm_v1.py` | US RAMM-LGBM: Core CPI MoM + 2Y repricing |
| `factors.py` | Factor registry with pub_lag; apply_publication_lags() |
| `uk_model_zoo.py` | 10-model zoo (MIDAS removed); dm_test(); score_backtest() |
| `compare_uk.py` | Full comparison driver + regime-model-combine framework |
| `backtest_2025.py` | 3-model backtest with BoE breakeven benchmark |
| `STATE.md` | Current system state, last results, pending work |
| `SPEC.md` | System specification and design decisions |
| `PROCESS.md` | Chronological build log |

---

## Adding New Factors

Drop a CSV in `data/<name>.csv` with columns `[date, value]`. Then register in `factors.py`:

```python
"my_factor": dict(
    fetch=None,               # None = CSV-only, or lambda: _fred("SERIES_ID")
    transform="level",        # "level" | "yoy" | "mom" | "logret" | "diff"
    candidate=True,           # True = screened; False = always included
    csv="my_factor.csv",
    note="Source description"),
```

Every model + the factor importance table picks it up automatically on next run.

Gated/paywalled sources (Bloomberg, ICE, Smart Data Foundry): export once to CSV and drop it. No code changes needed.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy yfinance lightgbm shap scikit-learn statsmodels \
            fredapi requests openpyxl dbnomics pytest scipy xgboost
```

**UK model zoo (FRED optional for backtest):**
```bash
export FRED_API_KEY=your_key_here     # only needed for US models
python compare_uk.py --start 2015     # full comparison
python compare_uk.py --start 2015 --train-from 1992  # post-ERM long history
```

**US models:**
```bash
FRED_API_KEY=your_key_here python ramm_lgbm_v1.py
FRED_API_KEY=your_key_here python backtest_2025.py
```

**Tests:**
```bash
FRED_API_KEY=x python -m pytest test_ramm_lgbm_v1.py -v
```

---

## Backtesting Notes

**Evaluation discipline:** All 11 models use 1-step-ahead evaluation (expanding window). State-space models (DFM, UCM, HMM, MS-DFM) use `.append(refit=False)` for per-month 1-step; lag-aware models (LGBM, TVP, LSTAR, BVAR, MIDAS) use causal lagged features. Apples-to-apples.

**ALFRED real-time discipline (partial):** ALFRED vintage helpers are implemented for CPILFESL and PAYEMS (the two US series with meaningful revisions). Market data (Brent, VIX, breakevens, DGS2) has no meaningful vintage. UK data has no ALFRED equivalent.

**Breakeven convergence caveat:** 5Y breakevens at time t already incorporate last month's CPI print. The BoE BE5 benchmark partially proxies AR(1) persistence. A strictly real-time test would require ALFRED vintages of breakevens.

**Training window:** 1992–present for UK models (post-ERM; avoids regime incompatible with modern inflation targeting). FRED provides Brent from 1987, GBP/USD from 1971, enabling true post-1992 start.

### Current nowcasts (June 2026)

| Target | Model | Nowcast |
|--------|-------|---------|
| US Core CPI MoM | RAMM-LGBM | +0.38% |
| 2Y Treasury repricing | RAMM-LGBM | +0.044 pp |
| UK CPI YoY | UCM | — |
| UK CPI YoY | TVP | — |
| UK CPI YoY | Combined-Dynamic | — |

Cleveland Fed 1Y inflation expectation (2026-05-31): 3.54%  
US 5Y TIPS breakeven (2026-06-30): 2.48%  
BoE gilt-implied 5Y breakeven (2026-05-31): 3.55% (RPI-linked, ~1pp above CPI)
