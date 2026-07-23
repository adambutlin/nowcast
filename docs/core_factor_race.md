# Core CPI factor race + screen (2026-07-23)

Target `uk_core_cpi` (ONS DKO8, CPI ex energy/food/alcohol/tobacco). AA residual 2015-01 → 2024-12
(n=120), walk-forward window 2015–2024. Decision metric **`univ_rel_rmse`** — walk-forward
single-factor ridge on the core AA residual, reconstructed RMSE relative to AA-only
(<1 = the factor helps core when added alone). Same methodology as the headline factor race.

## Results (available candidates, sorted by univ_rel_rmse)

```
factor                 group    n  abs_corr  univ_rel_rmse  shap_mean_abs
uk_cpih           cpi_family  120    0.3558         0.9378            n/a
uk_ppi_output      exogenous  120    0.3351         0.9660         0.2027
uk_monthly_gdp     exogenous  120    0.2184         0.9879         0.1352
gbp_eer            exogenous  120    0.1141         0.9956         0.0994
uk_retail_sales    exogenous  120    0.1536         1.0107         0.1668
uk_quarterly_gdp   exogenous  120    0.2977         1.0137         0.1032
uk_vacancies       exogenous  120    0.0464         1.0139         0.1255
uk_services_cpi   cpi_family  120    0.1377         1.0168            n/a
uk_rents           exogenous  120    0.1297         1.0314         0.0753
uk_awg             exogenous  120    0.2032         1.0403         0.0918
```

## Unavailable candidates (no free data source)

`uk_ashe_pay`, `uk_paye` (wages), `uk_pmi_services`, `uk_pmi_composite`, `uk_pmi_input_prices`
(Markit — paid), `uk_infl_swap_1y` (Bloomberg), `uk_trimmed_mean_cpi`, `uk_hciall`. These are the
theoretically strongest core drivers (wages, expectations, survey prices); their absence caps how
much the core overlay can add. Recorded as a known data limitation.

## Pinning decision

**Pinned (exogenous, univ_rel_rmse < 1):** `uk_ppi_output` (0.966), `uk_monthly_gdp` (0.988),
`gbp_eer` (0.996) — output PPI (factory-gate goods prices), monthly GDP (demand slack), and the
sterling effective rate (import-price channel into core goods).

**Excluded despite beating AA:** `uk_cpih` (0.938) — CPIH contains the same items as core plus
owner-occupier housing, so pinning it would make the overlay partly track a contemporaneous CPI
aggregate (near-circular). Same "genuine exogenous driver" philosophy as the headline model, which
also dropped CPI-family and redundant clones.

**Caveat:** all edges are marginal (best 0.966). Core CPI is highly persistent, so the AA anchor
already captures most of it. Whether the *combined* 3-factor overlay actually beats AA-only
out-of-sample is decided by the purged+embargoed validation gate (T7 below).

## T6 — calibration

Priors adopted (equal TVP/LGBM 0.5/0.5 + λ=0.5 governance haircut), the same as headline. Validated
by the gate below: the equal combo (rel 0.969) beats either member alone (LGBM 0.972, TVP 1.016), so
weight-optimization is unnecessary — and the headline research established it overfits OOS. λ=0.5 is
the deliberate calm-month haircut (statistical optimum is higher), kept for governance consistency.

## T7 — validation gate result → **SHIP**

Purged+embargoed walk-forward (PURGE_HORIZON=12, EMBARGO=1), 2015-2024, n=112 months:

| model | RMSE | rel-RMSE | note |
|---|---|---|---|
| AA-only (core) | 0.3239 | 1.000 | anchor baseline |
| **AA + 0.5·(0.5·TVP+0.5·LGBM)** | **0.3138** | **0.969** | **shipped** |
| — LGBM member recon | 0.3150 | 0.972 | workhorse |
| — TVP member recon | 0.3290 | 1.016 | flat alone; diversifies |

Diebold–Mariano overlay-vs-AA: **DM = 1.884, one-sided p = 0.030** (overlay significantly beats AA at
5%). **Decision: SHIP** the core overlay (`CORE.pinned` = output-PPI, monthly-GDP, GBP-EER; λ=0.5).
The edge is modest (core is highly persistent, and the strongest core drivers — wages, PMIs,
inflation expectations — are unavailable), but it is real and leakage-honest.
