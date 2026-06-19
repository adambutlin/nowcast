# Model Card — UK CPI YoY Nowcast (frozen production)

| field | value |
|---|---|
| **Name** | UK CPI YoY two-stage nowcast |
| **Version** | 1.0 (frozen 2026-06-19) |
| **Spec** | `Forecast = AA + 0.25·TVP + 0.25·LGBM` (AA + λ·overlay, λ=0.5, overlay = 0.5 TVP + 0.5 LGBM) |
| **Entry point** | `code/production/model.py` (`nowcast()` / `python -m`) |
| **Target** | UK CPI all-items YoY (ONS D7G7), reference-month nowcast |
| **Members** | AutoARIMA (anchor) + TVP + LightGBM, all on `code/factors.py` PINNED set |
| **Standpoint** | month-end T (info ≤ T); release is T+15…T+21. NOT release-day / NOT T-30 / NOT post-month-end updating |
| **Training** | walk-forward, expanding; AA from 2001, overlay residual from vintage; eval 2015-2024 |
| **Weights** | fixed; no regime switching, no detector, no latent state |

## Intended use
Monthly point nowcast of UK CPI YoY for the first unreleased reference month, as a **research/
decision-support** estimate reported alongside AutoARIMA. Not a standalone trading signal
(rates pipeline found no deployable edge; the CPI edge is modest and shock-concentrated).

## Performance (walk-forward 2015-2024, vs AutoARIMA)
- rel-RMSE ≈ **0.89** full sample (λ=0.5); edge concentrated in 2022/23, ~neutral in calm.
- **Statistically insignificant overall (DM p≈0.17)** — treat as modest, not proven.
- Live record (genesis May-2026): final 2.91 vs actual 2.80 (err +0.11); beat consensus
  (3.00) and UCL (3.05) and current-prod (2.92); **lost to AutoARIMA (2.71)**.

## Limitations / failure modes
- Calm/base-effect months (food/services-driven) → cost-push overlay overshoots up.
- LGBM ≈ nonlinear `uk_ppi_input` wrapper; PPI dominance is post-2021 era-dependent.
- Overlay ~79% noise (R²≈0.21); magnitude unreliable in calm → λ shrunk to 0.5.
- ~74% of the AA residual is unexplained (drivers outside the factor set).

## Ethics / governance
Accredited ONS inputs; financial inputs are market prices (no revision/leakage). No
post-month-end or post-release leakage (audited). Frozen by governance; see `docs/final_model.md`.

## Maintenance / monitoring
Append each release via `update_live_scorecard.py`; fill the actual; regenerate `live_report.md`.
Decision gate: after ~12 prospective releases, judge final-production vs **AutoARIMA-only**.
If it does not beat AA live, demote the overlay (ship AA alone).
