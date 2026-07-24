# Live scorecard — UK CPI nowcast (frozen production model)

Production model: **AA + 0.25·TVP + 0.25·LGBM** (λ=0.5). Genesis: May 2026.

Releases scored: **2** | forecasts logged: 2

## Cumulative accuracy

| forecaster | n | RMSE | MAE | beats AA | beats consensus |
|---|---|---|---|---|---|
| aa | 2 | 0.109 | 0.108 |  | 50% |
| current_production | 2 | 0.250 | 0.225 | 0% | 50% |
| final_production | 2 | 0.097 | 0.097 | 50% | 100% |
| consensus | 2 | 0.158 | 0.150 | 50% |  |
| ucl | 1 | 0.250 | 0.250 | 0% | 0% |
| experimental_overlay | 2 | 0.222 | 0.175 | 50% | 50% |

## Final production — hit rate

- beats AutoARIMA : 50%
- beats consensus : 100%
- beats UCL       : 50%
- beats current-prod: 100%

## Rolling 6-release RMSE

| forecaster | rolling-6 RMSE |
|---|---|
| aa | 0.109 |
| current_production | 0.250 |
| final_production | 0.097 |
| consensus | 0.158 |
| ucl | 0.250 |
| experimental_overlay | 0.222 |

## Per-release (signed error)

| month | AA | curr-prod | **final** | consensus | UCL | exp(λ=1) | actual |
|---|---|---|---|---|---|---|---|
| 2026-05 | 2.71 | 2.92 | **2.91** | 3.00 | 3.05 | 3.11 | 2.80 |
| 2026-06 | 2.73 | 2.93 | **2.68** | 2.70 | nan | 2.64 | 2.60 |

## May 2026 — GENESIS (permanent record, not reinterpreted)

First true prospective observation. Final production 2.91 vs actual 2.80 (err +0.11). AutoARIMA 2.71 (err −0.09) was best; the λ=1 experimental overlay (3.11) was worst — a calm/base-effect month where the cost-push overlay overshot. λ=0.5 halved that error vs λ=1. One adverse point; the forward record decides.

---

## Core CPI nowcast (ex energy, food, alcohol & tobacco)

Core model: **AA + 0.25·TVP + 0.25·LGBM** on 5 factors (services-PMI, output-PPI, 1y inflation-swap, corporate-profits, GBP-EER; λ=0.5); validation rel-RMSE **0.934** (DM p=0.006, n=120). See docs/core_factor_race.md.

Releases scored: **1**

| forecaster | n | RMSE | MAE |
|---|---|---|---|
| core_aa | 1 | 0.112 | 0.112 |
| core_final | 1 | 0.165 | 0.165 |
| core_consensus | 1 | 0.100 | 0.100 |

| month | core AA | **core final** | consensus | actual |
|---|---|---|---|---|
| 2026-06 | 2.49 | **2.44** | 2.50 | 2.60 |
