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

## Candidate availability (corrected 2026-07-24)

Not a flat "no free data" — the missing candidates split into three tiers:

- **Free & already tested — wages.** Monthly wages via ONS **AWE** (`uk_awg`, KAB9 whole-economy
  pay, live to May-2026) ARE in the race and scored **rel-RMSE 1.040** — they do NOT beat AA. So
  wages are not a missing free lever; the free monthly wage signal is present and simply doesn't help
  core. (Earlier "wages have no free source" wording was wrong.)
- **Free but low-value / unused.** `uk_ashe_pay` is free (ONS) but **annual** (ASHE — one release a
  year), a poor fit for a monthly nowcast and dominated by the monthly AWE that already failed. ONS
  **PAYE RTI median monthly pay** is free + monthly and currently UNUSED (`uk_paye` is only the
  payroll *count*, not pay); a cheap future test, though AWE's failure tempers expectations.
- **Licensed — available via LSEG Workspace, not wired here.** The S&P Global (Markit) PMIs
  (`uk_pmi_services` / `uk_pmi_composite` / `uk_pmi_input_prices`) and the 1y inflation swap
  (`uk_infl_swap_1y`) are the genuinely missing drivers (cost-push + expectations). They are carried
  by LSEG Workspace under licence, but this env has **no LSEG SDK** (`lseg-data` / `refinitiv-data` /
  `eikon` all absent) and no reachable Workspace session — so they cannot be fetched here yet.

CPI-family (`uk_trimmed_mean_cpi`, `uk_hciall`) are near-circular with core and excluded on principle.

## Pinning decision

**Pinned (exogenous, univ_rel_rmse < 1, AND reliably live-timely):** `uk_ppi_output` (0.966) and
`gbp_eer` (0.996) — output PPI (factory-gate goods prices) and the sterling effective rate
(import-price channel into core goods).

**Dropped for staleness:** `uk_monthly_gdp` (0.988) beat AA historically but its dbnomics series is
**dead — ends 2024-03** (2+ years stale), so it cannot feed a live nowcast. A factor must be
live-timely, not merely backtest-valid. Removing it also *improved* the overlay (rel-RMSE 0.969 →
0.952), since its stale/ffilled values had been adding noise.

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

Purged+embargoed walk-forward (PURGE_HORIZON=12, EMBARGO=1), 2015-2024, n=120 months, final
2-factor overlay (`uk_ppi_output`, `gbp_eer`):

| model | RMSE | rel-RMSE | note |
|---|---|---|---|
| AA-only (core) | 0.3213 | 1.000 | anchor baseline |
| **AA + 0.5·(0.5·TVP+0.5·LGBM)** | **0.3059** | **0.952** | **shipped** |
| — LGBM member recon | 0.3015 | 0.939 | workhorse |
| — TVP member recon | 0.3154 | 0.982 | diversifies |

Diebold–Mariano overlay-vs-AA: **DM = 2.726, one-sided p = 0.003** (overlay beats AA well beyond the
5% bar). **Decision: SHIP** the core overlay (`CORE.pinned` = output-PPI, GBP-EER; λ=0.5). The edge
is modest (core is highly persistent, and the strongest missing driver class — PMIs and inflation
expectations — is LSEG-gated, not yet wired; free monthly wages are present but don't help), but it
is real, leakage-honest, and live-robust.

Live June-2026 core nowcast: **2.494** (AA 2.488 + overlay +0.006). Headline unchanged at 2.683.

## Future work — pending LSEG Workspace access (branch `core` shelved 2026-07-24)

Awaiting LSEG Workspace approval. The 2-factor overlay above is **complete and shipping**; the
licensed drivers are a potential enhancement, not a blocker. When access lands:

1. `pip install lseg-data` into the `quant` env.
2. With Workspace running (the SDK attaches to the desktop app) or RDP machine-creds in `.env`, wire
   fetches in `code/factors.py` for the PMI economic-indicator RICs (`uk_pmi_services` /
   `uk_pmi_composite` / `uk_pmi_input_prices`) and the GBP 1y inflation-swap RIC (`uk_infl_swap_1y`).
3. Re-run `code/new_factors/core_race.py` then `core_validate.py` to see whether PMI-input-prices,
   services PMI, or the inflation swap earn a pin and lift the core overlay beyond rel-RMSE 0.952.
   Update `CORE.pinned` in `code/targets.py`, this doc, and the validation gate accordingly.
4. Optionally also add the free ONS PAYE **RTI median monthly pay** as a second wage test (cheap;
   low expectations given AWE `uk_awg` already failed at rel-RMSE 1.040).
