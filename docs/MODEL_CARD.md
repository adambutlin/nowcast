# Model Card — UK CPI YoY Nowcast (frozen production)

| field | value |
|---|---|
| **Name** | UK CPI YoY two-stage nowcast |
| **Version** | 1.1 (frozen 2026-06-19; TVP month fix + regulatory-event layer 2026-08-19) |
| **Spec** | `Forecast = AA + 0.25·TVP + 0.25·LGBM + RegEvent` (AA + λ·overlay + regulatory-event step; λ=0.5, overlay = 0.5 TVP + 0.5 LGBM; `RegEvent = 0.75·β·Ofgem-cap-%`, headline only, zero outside cap-reset months) |
| **Entry point** | `code/production/model.py` (`nowcast()` / `python -m`) |
| **Target** | UK CPI all-items YoY (ONS D7G7), reference-month nowcast |
| **Members** | AutoARIMA (anchor) + TVP + LightGBM, all on `code/factors.py` PINNED set |
| **Standpoint** | month-end T (info ≤ T); release is T+15…T+21. NOT release-day / NOT T-30 / NOT post-month-end updating |
| **Training** | walk-forward, expanding; AA from 2001, overlay residual from vintage; eval 2015-2024 |
| **Weights** | fixed; no regime switching, no detector, no latent state |
| **Known gaps** | (1) motor-fuel channel runs off crude (`oil_brent`), not pump prices. `uk_pump_price` was raced 2026-08-20 and **not pinned** — real at factor level (resid corr +0.245 vs −0.031) but undetectable at headline (`docs/energy_fx_race.md`). **Resolved as a diagnosis 2026-08-24**: the factor set was the wrong container. As a *component* the same signal is the largest single contributor to a naive component nowcast (−0.0272 RMSE, `docs/components.md`). Production is unchanged; the fix is architectural, not a factor swap. (2) **no FX factor** in the headline pinned set — dropped by accident at `d25f185`, raced and null. (3) `global_supply_chain_pressure` is **dead in production** since ≥2026-07: NY Fed URL returns HTML, no CSV fallback exists, degradation is silent — live runs carry 8 pinned factors, not 9. (4) `uk_be5` is mis-constructed (`nominal − 2·real`); fix sits on `fix/uk-be5-breakeven`, unmerged because it touches a FROZEN_PATHS file. Not pinned, so no live forecast is affected — see `docs/fx_expectations.md`. |

## Intended use
Monthly point nowcast of UK CPI YoY for the first unreleased reference month, as a **research/
decision-support** estimate reported alongside AutoARIMA. Not a standalone trading signal
(rates pipeline found no deployable edge; the CPI edge is modest and shock-concentrated).

## Core CPI (dual output, added 2026-07-23)
`model.py` also prints a **Core CPI YoY** nowcast (ex energy, food, alcohol & tobacco; ONS DKO8),
same architecture on a **5-factor** core pinned set — services PMI, output PPI, 1y inflation swap,
corporate profits (GOS YoY, the profit-led/"greedflation" channel), GBP-EER (λ=0.5). Headline output
is unchanged/byte-identical. Core overlay passed a headline-grade purged+embargoed gate (rel-RMSE
**0.934**, DM p=0.006, 2015-2024); else it would print AA-only. Genesis June-2026: core 2.435 vs
actual 2.6 (overlay overshot down this calm month). The PMIs, inflation swap, and corporate profits
are sourced via **LSEG Workspace** (needs the desktop app + `LSEG_APP_KEY`); without it these factors
degrade to unavailable. Details: `docs/core_factor_race.md`, figures `plots/core/`. Tracked in
`data/live_scorecard_core.csv`.

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
