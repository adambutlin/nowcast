# The valid experiment: PRODUCTION model under as-of truncation (T-30 → R-1)

`code/timing/production_asof.py` + `may2026_prod_path.py`. Runs the **canonical production
model** (`two_stage`: AutoARIMA + 0.375 BVAR + 0.25 TVP + 0.375 Z.MIDAS, PINNED incl
uk_ppi_input/deep_sea_freight, weights frozen, sample 2015-2024) and truncates only the
within-reference-month varying inputs as-of each horizon (daily oil_brent/gas_eu + MIDAS
daily mean). Monthly cost factors don't change intra-month; AA is the common baseline.
This is the ONLY valid horizon experiment (the earlier `horizon_backtest.py` used the
intramonth stack — a different model — so its negative-edge result is void).

## PART E/F — edge by horizon (production, as-of)
| horizon | rmse_full | edge_full | rel | %of final edge | edge ex-shock | edge 2022/23 |
|---|---|---|---|---|---|---|
| T-30 | 0.4678 | +0.0009 | 0.998 | 3% | +0.0015 | 0.000 |
| T-21 | 0.4558 | +0.0129 | 0.973 | 39% | **−0.0279** | +0.098 |
| T-14 | 0.4462 | +0.0224 | 0.952 | 68% | −0.0120 | +0.095 |
| T-10 | 0.4460 | +0.0227 | 0.952 | 68% | −0.0130 | +0.098 |
| T-7 | 0.4482 | +0.0205 | 0.956 | 62% | −0.0157 | +0.096 |
| T-5 | 0.4460 | +0.0226 | 0.952 | 68% | −0.0118 | +0.095 |
| T-2 | 0.4420 | +0.0266 | 0.943 | 80% | −0.0050 | +0.094 |
| T-1 | 0.4427 | +0.0260 | 0.945 | 78% | −0.0045 | +0.090 |
| **T** | 0.4355 | **+0.0332** | 0.929 | 100% | +0.0100 | +0.084 |
| T+5 | 0.4355 | +0.0332 | 0.929 | 100% | +0.0100 | +0.084 |
| T+10 | 0.4355 | +0.0332 | 0.929 | 100% | +0.0100 | +0.084 |
| R-1 | 0.4355 | +0.0332 | 0.929 | 100% | +0.0100 | +0.084 |

- Edge first appears at **T-21** (≈0 at T-30 → +0.013). ~**two-thirds earned by T-14**, ~80%
  by T-2, last ~20% from completing the month's data at **T**.
- **Flat T → R-1** (0.0332 constant): no improvement after month-end (confirms the earlier
  timing audit — the information set closes at month-end).
- **The edge is ENTIRELY the 2022/23 energy shock** (+0.09 at every pre-T horizon). **Ex-shock
  it is NEGATIVE until month-end** (−0.005 to −0.028) and only +0.010 at T. In calm months the
  intramonth overlay HURTS.

## PART G — attribution (mean |overlay|; AA+single-member edge)
No single member beats AA standalone at any pre-T horizon (AA+BVAR −0.05…−0.22; AA+TVP
−0.02…−0.21; AA+MIDAS negative except T-30 +0.13). The positive full-sample edge is a
**diversification effect** of averaging the three (uncorrelated errors), realised only in
shock months — not one model "creating" skill. TVP throws the largest overlay (0.32→0.42)
and shrinks at T (0.29) as full-month data sharpens it.

## PART H — May 2026 (production, as-of); actual = 2.8
| horizon | AA | BVAR ov | TVP ov | MIDAS ov | full |
|---|---|---|---|---|---|
| T-30 | 2.710 | – | – | – | 2.710 |
| T-21 | 2.710 | +0.09 | +0.631 | +0.018 | 2.909 |
| T-14 | 2.710 | +0.103 | +0.582 | +0.053 | 2.914 |
| T-7 | 2.710 | +0.109 | +0.591 | +0.065 | 2.923 |
| T-2 | 2.710 | +0.126 | +0.647 | +0.075 | 2.948 |
| T-1 | 2.710 | +0.103 | +0.626 | +0.077 | 2.934 |
| T..R-1 | 2.710 | +0.058 | +0.641 | +0.066 | 2.917 |

**AutoARIMA = 2.71 (err 0.09) at every horizon — closest to actual.** The overlay ramps the
forecast UP from 2.71 to ~2.92 (driven by TVP +0.64) as within-month energy data accrues,
**making it worse** (full err 0.12 vs AA 0.09). May 2026 was a calm base-effect month, so
the cost-push overlay pushed the wrong way — exactly the ex-shock negative-edge case.
Charts: `plots/timing/{may2026_production_path,production_edge_by_horizon}.png`.

## PART I — hostile review
1. Edge at T-30? ~zero (rel 0.998).
2. Only near month-end? No — builds T-21→T-14 (two-thirds by T-14), plateaus, completes at T.
3. Continue after month-end? NO — flat T→R-1.
4. TVP responsible? TVP moves most, but AA+TVP is negative; the edge is shock-month
   diversification, not TVP skill. In May, TVP drove the forecast the WRONG way.
5. Tradeable? The edge is real only in 2022/23; ex-shock it's negative/zero; full-sample
   +0.033 was insignificant (DM p=0.17, IC audit). A PM gains nothing in normal months.

## Final deliverable
1. **Canonical production model:** `code/new_factors/two_stage.py` (AA + 0.375 BVAR + 0.25
   TVP + 0.375 Z.MIDAS, PINNED incl uk_ppi_input/deep_sea_freight). rel 0.93.
2. **Legacy stacks:** `code/intramonth/*` (different model), `code/main.py`,
   `sweep_*`, `resid_target_compare`, `retrain_pinned`; separate product `code/rates/*`.
3. **Deprecate:** the intramonth point-forecast layer + `timing/horizon_backtest.py`
   (invalid model comparison); rename intramonth MIDAS→ElasticNet alias. See purge_candidates.md.
4. **Edge first appears:** T-21.
5. **Edge:** T-30 ≈ +0.001 (3%); T = +0.033 (100%); R-1 = +0.033 (no gain after T).
6. **Improve after month-end?** NO — flat T→R-1.
7. **Forecast / month-end nowcast / release-day nowcast?** A **reference-month nowcast that
   completes at month-end T**: it earns its edge DURING the month (T-21→T) as financials
   accrue, then freezes. Not a forecast (no edge at T-30), not a release-day nowcast (flat
   after T).

### Where the model earns its edge
Between **T-21 and T** (within the reference month, as energy/financial data accrues), and
**only in energy-shock months**. At T-30 it's just AutoARIMA; after month-end it adds
nothing. Ex-shock the within-month overlay is value-destructive until T. So the production
model is a genuine *shock-month* intramonth nowcaster and a *no-op-to-harmful* overlay in
normal months — and the headline +0.033/rel-0.93 is the shock-weighted average of those two
regimes (and statistically insignificant overall).
