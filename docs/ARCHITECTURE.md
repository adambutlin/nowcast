# Architecture — code & data flow

## Production path
```
factors.py (REGISTRY, pub-lags, build_matrix)
        │   PINNED = oil_brent, gas_eu, uk_quarterly_gdp, imf_all_commodity,
        │            mpc_rate_change, ofgem_cap_delta, uk_ppi_input, deep_sea_freight
        ▼
new_factors/two_stage.py  (load_matrix; AutoARIMA backtest; member backtests)
        ▼
production/model.py        FROZEN: Forecast = AA + 0.25·TVP + 0.25·LGBM
   ├─ AA      = uk_model_zoo.AutoARIMA   (anchor; univariate CPI YoY)
   ├─ TVP     = uk_model_zoo.TVP         (residual model on PINNED)
   └─ LGBM    = lightgbm on AA residual  (PINNED features)
        ▼
production/update_live_scorecard.py → data/live_scorecard.csv
production/generate_live_report.py  → docs/live_report.md
```

## Information boundary (governance: NOT release-day, NOT post-month-end)
- Each factor: `resample('ME').last()` then `shift(pub_lag≥0)` → month-T row uses ≤ month-T-end.
- LGBM/TVP residual features are the same pub-lagged monthly matrix. AA uses CPI through the
  last released month. No post-month-end / post-release data enters (leakage audit: 0 violations).
- The model is a **reference-month nowcast that completes at month-end T**; release is T+15…T+21.

## Layers (what each does)
| layer | role | not |
|---|---|---|
| AutoARIMA | level: persistence, seasonality, base effects (~96%) | — |
| TVP overlay | shock pass-through; diversifier | a standalone forecaster |
| LGBM overlay | nonlinear PPI/cost-push residual map | a multi-factor learner (it's ~PPI) |
| λ = 0.5 | magnitude shrinkage of the overlay | a regime switch |

## Removed / rejected (see final_model.md §9–10)
BVAR, MIDAS (redundant); HMM/regime/scenario/detector/switching (non-predictive OOS);
intramonth point-forecast stack (different model). All retained as research only.

## Data
- Inputs: FRED (Brent, GBP, gilts), yfinance (VIX, TTF, MOVE), ONS JSON API (CPI, PPI),
  dbnomics (ONS/OECD), BoE (gilt curves). CSV drop-ins under `data/` take priority over live fetch.
- `data/` is gitignored (regenerable); narrative/spec live in `docs/` (tracked).
