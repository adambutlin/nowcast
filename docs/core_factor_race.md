# Core CPI factor race + validation (updated 2026-07-24, with LSEG factors)

Target `uk_core_cpi` (ONS DKO8, CPI ex energy/food/alcohol/tobacco). AA residual 2015-01 → 2024-12
(n=120), walk-forward window 2015–2024. Decision metric **`univ_rel_rmse`** — walk-forward
single-factor ridge on the core AA residual, reconstructed RMSE relative to AA-only (<1 = the factor
helps core alone). Same methodology as the headline factor race. (Auto-summary of each run is written
to `core_factor_race_auto.md`; this file is the curated record.)

## Race results (with LSEG-sourced PMIs / inflation swap / corporate profits)

```
factor                 group    n  abs_corr  univ_rel_rmse   channel
uk_cpih           cpi_family  120    0.3558         0.9378   (circular — excluded)
uk_pmi_composite   exogenous  120    0.2849         0.9588   activity (redundant w/ services)
uk_pmi_services    exogenous  120    0.2908         0.9589   demand / services      *PIN
uk_ppi_output      exogenous  120    0.3351         0.9660   cost-push              *PIN
uk_monthly_gdp     exogenous  112    0.2184         0.9879   (data dead — excluded)
uk_corp_profits    exogenous  120    0.2267         0.9908   profit-led/greedflation *PIN
uk_infl_swap_1y    exogenous  120    0.3685         0.9915   inflation expectations *PIN
gbp_eer            exogenous  120    0.1141         0.9956   FX / import prices     *PIN
uk_retail_sales    exogenous  120    0.1536         1.0107   (does not beat AA)
uk_quarterly_gdp   exogenous  120    0.2977         1.0137   (does not beat AA)
uk_vacancies       exogenous  120    0.0464         1.0139   (does not beat AA)
uk_services_cpi   cpi_family  120    0.1377         1.0168   (circular — excluded)
uk_rents           exogenous  120    0.1297         1.0314   (does not beat AA)
uk_awg             exogenous  120    0.2032         1.0403   (does not beat AA)
```

## Data sourcing (2026-07-24)

LSEG Workspace access was granted, unblocking the licensed drivers via the Data Library
(`code/factors._lseg_history`, desktop session, needs Workspace running + `LSEG_APP_KEY`):

- **Services PMI** `uk_pmi_services` → RIC `aGBBUSA/A` (long history from 1996; the `GBPMIS=ECI` poll
  RIC is only 3y). **Composite PMI** `uk_pmi_composite` → `aGBPMIOUTCE/A` (from 1998).
- **1y inflation swap** `uk_infl_swap_1y` → `GBRPIZ1Y=` (RPI-linked, MID_PRICE; UK's liquid
  inflation-expectations market).
- **Corporate profits** `uk_corp_profits` (NEW) → `aGBIAGDPGOS/A` = ONS gross operating surplus of
  corporations (£, quarterly), YoY = nominal corporate-profit growth — the profit-led / "greedflation"
  channel (margin/pricing-power pressure the cost-push factors miss).

Still unavailable: `uk_pmi_input_prices` (no working long-history RIC found — cost-push is already
covered by `uk_ppi_output`); ONS ASHE wages (annual, low value; monthly AWE `uk_awg` is free and
already tested at rel-RMSE 1.040 — wages don't help core).

## Pinning decision — 5 factors, one per economic channel

**Pinned:** `uk_pmi_services` (0.959, demand), `uk_ppi_output` (0.966, cost-push),
`uk_infl_swap_1y` (0.992, expectations), `uk_corp_profits` (0.991, profit-led), `gbp_eer` (0.996, FX).
All beat AA alone, are live-timely, and are non-redundant (distinct channels).

**Dropped though they beat AA:** `uk_pmi_composite` (0.959) — redundant with services PMI (both are
UK activity, highly correlated); `uk_monthly_gdp` (0.988) — dbnomics series dead (ends 2024-03);
`uk_cpih` (0.938) — circular (CPIH ⊃ core items).

## Calibration

Equal TVP/LGBM 0.5/0.5 + λ=0.5 governance haircut (same as headline). Validated below: the equal
combo (rel 0.934) beats either member alone, so no weight re-optimization (which overfits OOS).

## Validation gate → **SHIP**

Purged+embargoed walk-forward (PURGE_HORIZON=12, EMBARGO=1), 2015-2024, n=120:

| model | RMSE | rel-RMSE | note |
|---|---|---|---|
| AA-only (core) | 0.3213 | 1.000 | anchor baseline |
| **AA + 0.5·(0.5·TVP+0.5·LGBM)** | **0.3000** | **0.934** | **shipped (5-factor)** |
| — TVP member recon | 0.3031 | 0.944 | workhorse |
| — LGBM member recon | 0.3227 | 1.004 | flat alone; diversifies |

Diebold–Mariano overlay-vs-AA: **DM = 2.498, one-sided p = 0.006**. The LSEG factors + corporate
profits improved the overlay from the earlier free-only 2-factor version (rel 0.952 → **0.934**).

**Live June-2026 core nowcast: 2.435** (AA 2.488 + TVP −0.023 + LGBM −0.030; overlay pulls down on
soft mid-2026 PMIs). Actual core = 2.6, so the overlay hurt this one calm month (err −0.165 vs AA's
−0.112) — the expected noisy-but-net-positive behaviour. Headline unchanged at 2.683.

## Live dependency & governance

The three LSEG factors (services PMI, inflation swap, corporate profits) need the Workspace desktop
app running for the live nowcast; without it they degrade to unavailable and the core overlay trains
on the reduced set (or falls back toward AA). `uk_corp_profits` is quarterly (pub_lag 2) — within the
ffill budget for a month-end nowcast, but monitor for staleness at quarter boundaries.
