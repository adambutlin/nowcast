# Live scorecard — UK CPI nowcast (frozen production model)

Production model: **AA + 0.25·TVP + 0.25·LGBM** (λ=0.5). Genesis: May 2026.

Releases scored: **1** | forecasts logged: 1

## Cumulative accuracy

| forecaster | n | RMSE | MAE | beats AA | beats consensus |
|---|---|---|---|---|---|
| aa | 1 | 0.090 | 0.090 |  | 100% |
| current_production | 1 | 0.117 | 0.117 | 0% | 100% |
| final_production | 1 | 0.110 | 0.110 | 0% | 100% |
| consensus | 1 | 0.200 | 0.200 | 0% |  |
| ucl | 1 | 0.250 | 0.250 | 0% | 0% |
| experimental_overlay | 1 | 0.311 | 0.311 | 0% | 0% |

## Final production — hit rate

- beats AutoARIMA : 0%
- beats consensus : 100%
- beats UCL       : 100%
- beats current-prod: 100%

## Rolling 6-release RMSE

| forecaster | rolling-6 RMSE |
|---|---|
| aa | 0.090 |
| current_production | 0.117 |
| final_production | 0.110 |
| consensus | 0.200 |
| ucl | 0.250 |
| experimental_overlay | 0.311 |

## Per-release (signed error)

| month | AA | curr-prod | **final** | consensus | UCL | exp(λ=1) | actual |
|---|---|---|---|---|---|---|---|
| 2026-05 | 2.71 | 2.92 | **2.91** | 3.00 | 3.05 | 3.11 | 2.80 |

## May 2026 — GENESIS (permanent record, not reinterpreted)

First true prospective observation. Final production 2.91 vs actual 2.80 (err +0.11). AutoARIMA 2.71 (err −0.09) was best; the λ=1 experimental overlay (3.11) was worst — a calm/base-effect month where the cost-push overlay overshot. λ=0.5 halved that error vs λ=1. One adverse point; the forward record decides.
