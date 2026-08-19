# Live scorecard — UK CPI nowcast (frozen production model)

Production model: **AA + 0.25·TVP + 0.25·LGBM + RegEvent** (λ=0.5; RegEvent is the announced Ofgem cap step, headline only, zero outside reset months). Genesis: May 2026.

**Rows are AS-RUN.** May–July 2026 were produced by the pre-2026-08-19 pipeline, before the TVP prediction-month fix, the RegEvent layer and the data-freshness guards. They are kept unaltered as the prospective record; `code/production/restate.py` shows what the current pipeline makes of the same months (docs/final_model.md §11).

Releases scored: **3** | forecasts logged: 3

## Cumulative accuracy

| forecaster | n | RMSE | MAE | beats AA | beats consensus |
|---|---|---|---|---|---|
| aa | 3 | 0.239 | 0.200 |  | 33% |
| current_production | 2 | 0.250 | 0.225 | 0% | 33% |
| final_production | 3 | 0.211 | 0.177 | 67% | 67% |
| consensus | 2 | 0.158 | 0.150 | 33% |  |
| ucl | 1 | 0.250 | 0.250 | 0% | 0% |
| experimental_overlay | 3 | 0.247 | 0.214 | 67% | 33% |

## Final production — hit rate

- beats AutoARIMA : 67%
- beats consensus : 67%
- beats UCL       : 33%
- beats current-prod: 67%

## Rolling 6-release RMSE

| forecaster | rolling-6 RMSE |
|---|---|
| aa | 0.239 |
| current_production | 0.250 |
| final_production | 0.211 |
| consensus | 0.158 |
| ucl | 0.250 |
| experimental_overlay | 0.247 |

## Per-release (signed error)

| month | AA | curr-prod | **final** | consensus | UCL | exp(λ=1) | actual |
|---|---|---|---|---|---|---|---|
| 2026-05 | 2.71 | 2.92 | **2.91** | 3.00 | 3.05 | 3.11 | 2.80 |
| 2026-06 | 2.73 | 2.93 | **2.68** | 2.70 | nan | 2.64 | 2.60 |
| 2026-07 | 2.52 | nan | **2.56** | nan | nan | 2.61 | 2.90 |

## May 2026 — GENESIS (permanent record, not reinterpreted)

First true prospective observation. Final production 2.91 vs actual 2.80 (err +0.11). AutoARIMA 2.71 (err −0.09) was best; the λ=1 experimental overlay (3.11) was worst — a calm/base-effect month where the cost-push overlay overshot. λ=0.5 halved that error vs λ=1. One adverse point; the forward record decides.

---

## Core CPI nowcast (ex energy, food, alcohol & tobacco)

Core model: **AA + 0.25·TVP + 0.25·LGBM** on 5 factors (services-PMI, output-PPI, 1y inflation-swap, corporate-profits, GBP-EER; λ=0.5); validation rel-RMSE **0.934** (DM p=0.006, n=120). See docs/core_factor_race.md.

Releases scored: **2**

| forecaster | n | RMSE | MAE |
|---|---|---|---|
| core_aa | 2 | 0.107 | 0.107 |
| core_final | 2 | 0.128 | 0.120 |
| core_consensus | 1 | 0.100 | 0.100 |

| month | core AA | **core final** | consensus | actual |
|---|---|---|---|---|
| 2026-06 | 2.49 | **2.44** | 2.50 | 2.60 |
| 2026-07 | 2.50 | **2.53** | nan | 2.60 |
