# Disaggregated UK CPI nowcast — Phase 0 and Phase 1

`code/components/`. Started 2026-08-21. A parallel research track: rebuild the rejected DFM
extension in the spirit of the Cleveland Fed inflation nowcast (Knotek & Zaman 2017; FRB
Cleveland WP 24-06) — nowcast components, aggregate to headline — rather than as another
aggregate model. **Never writes to the production path.**

The hypothesis under test, stated so it is falsifiable:

> The component architecture separates mechanically known CPI aggregation and index
> arithmetic from the statistical problem of forecasting current-month component price
> changes, while allowing heterogeneous information-arrival processes across components.

## Design decisions

| decision | choice |
|---|---|
| aggregation transform | component **MoM**, weight-aggregate, chain onto published index, derive YoY |
| seasonality | month effects fitted **inside** the walk-forward (ONS publishes no SA components) |
| end state | parallel product, repo's standing gate; promotion is a separate later decision |
| gate | two-level — each component beats its own naive benchmark before entering the aggregate |

Because `Index(T-1)` and `Index(T-12)` are both published, the only quantity forecast is
this month's component movements; base effects are arithmetic.

## Phase 0 — basket reconciliation (`basket.py`)

The gate: no component is modelled until the aggregation identity reconciles.

**A correction is recorded in the module.** The first pass concluded that MM23 carries no
COICOP division weights. That was wrong: the search matched the title prefix `CPI wts:` and
MM23 spells the COICOP hierarchy `CPI WEIGHTS`. There are **322** of the latter, running the
full tree down to 4-digit classes (`L8BK` is *CPI WEIGHTS 07.2.2.2 Petrol*).

Partitions verified over 2005-2026, all exact on weights (max deviation 1e-4):

| partition | reconstruction floor | worst month | lag-1 autocorr |
|---|---|---|---|
| 2-way `efat + core` | 0.0552 pp | — | −0.54 |
| 4-way `hh_energy + motor_fuel + food_alc_tob + core` | 0.0576 pp | 0.319 pp | −0.42 |
| **12-way COICOP divisions 01…12** | **0.0456 pp** | **0.115 pp** | −0.47 |

The 12-division partition is both finer *and* more accurate: averaging twelve independently
rounded indices beats carrying four coarse aggregates that each round once.

### The reconstruction floor is rounding, and it is irreducible

Reproducing headline MoM from published components cannot close perfectly. MM23 publishes
indices to **one decimal place**, so an index near 143 carries ±0.035% quantisation. Two
diagnostics say rounding rather than a structural error:

- the residual's lag-1 autocorrelation is −0.42 to −0.54, against a theoretical **−0.5** for
  differenced white noise, which is exactly what rounding a level and then differencing
  produces;
- the error barely moves with component count (2-way 0.0552, 4-way 0.0576), so it is
  dominated by rounding in the **headline** index, not by compounding across components.

**Chain-linking is not the cause.** The correct December-based chained form
`I_t = I_dec · Σᵢ wᵢ · (I_i,t / I_i,dec)` gives 0.0570 against 0.0576 for the naive
weighted-MoM sum. Worth knowing before building the aggregator on the wrong formula.

The floor is pre-registered. A perfect component nowcast still carries ~0.0576 pp against
production's 0.431 pp headline RMSE — about 13% of it. All Phase 1 reporting includes
floor-adjusted RMSE, `sqrt(RMSE² − floor²)`.

### What remains unavailable

The ECOICOP goods/services split still cannot partition core. The 49 `CPI wts:` analytical
aggregates **overlap** rather than tile: the eight goods subgroups sum to 333-409 per 1000
against a *Non-energy industrial goods* weight of 276-347, and goods plus the four services
groups leave 61-75 per 1000 unaccounted. So `core_goods` vs `core_services` cannot be
sourced from published data; any services split must be built from COICOP classes instead.

## Phase 1 — does the architecture pay before any modelling? (`naive.py`)

Nothing is fitted beyond two single-regressor pass-throughs estimated on prior months only.
A single benchmark number would be uninterpretable, so the run is a **ladder** in which each
rung differs from the one above by exactly one change.

| | benchmark | n | RMSE | rel | floor-adj | DM | p |
|---|---|---|---|---|---|---|---|
| A | aggregate AutoARIMA | 120 | 0.4687 | 1.0000 | 0.4651 | — | — |
| B0 | headline MoM naive, **no components** | 120 | 0.4285 | 0.9143 | 0.4246 | 1.064 | 0.144 |
| B1 | components, all naive | 120 | 0.4291 | 0.9157 | 0.4253 | 1.029 | 0.152 |
| B3 | B1 + deterministic Ofgem energy | 120 | 0.4163 | 0.8883 | 0.4123 | 1.388 | 0.083 |
| **B2** | B3 + observed pump prices | 120 | **0.3891** | **0.8303** | 0.3849 | 2.071 | **0.019** |

Purged walk-forward 2015-2024. Production reference: 0.4312, rel 0.9201.

```
MoM + seasonality + chaining   A  → B0   -0.0402
decomposition itself           B0 → B1   +0.0006
deterministic Ofgem energy     B1 → B3   -0.0128
observed pump prices           B3 → B2   -0.0272
```

### Decomposition on its own contributes nothing

**+0.0006.** Splitting the basket four ways and aggregating with exact weights is
indistinguishable from running the same seasonal-naive rule on headline MoM directly. B0 —
which needs no components at all — carries more than half the total gain and already matches
production.

The component architecture earns its keep **only through the observable mechanisms**: Ofgem
−0.0128 and pump prices −0.0272, together −0.0400 from B1. The value is in components whose
current-month movement is *known* at month-end, not in disaggregation per se.

Two consequences:

- **The energy thread closes.** The same pump signal failed three times as a pinned FACTOR
  (`docs/energy_fx_race.md`, `docs/energy_passthrough.md`) and is now the single largest
  contributor as a COMPONENT. The factor set was the wrong container for it.
- **κ is gone and was not missed.** Household energy carries no haircut. The 0.75 in
  `production/reg_events.py` exists because the step is applied at aggregate level with
  nothing to tell it what share of the basket it moves; the basket weight now carries that
  explicitly. B3 vs B1 is that construction earning −0.0128 unaided.

### Pre-registered rules

| component | rule |
|---|---|
| `household_energy` | 0 outside Ofgem reset months; on reset, `β·cap_pct` with β through the origin on **prior** resets only. No κ. |
| `motor_fuel` | `a + b·pump_MoM(T)` — the DESNZ monthly-mean pump price for month T is known at month-end T. |
| `food_alc_tob`, `core` | naive: seasonal (same calendar month, prior 5 years) or MA12 (the Cleveland food rule). Both reported. |

Standpoint is month-end T with information ≤ T, matching production. CPI(T-1) publishes
mid-month T, so component MoM history through T-1 and `Index(T-1)` are legitimately
available.

## Where this points next

Phase 1 says finer decomposition is worth building **only where a component has an
observable or deterministic current-month driver**:

| candidate | driver | expected value |
|---|---|---|
| administered/indexed services (rail RPI each January, water in April, telecoms CPI+3.9%) | rules known in advance, same shape as Ofgem | **high** |
| alcohol & tobacco | Budget duty steps, known in advance; needs `CHZS`/`D7BV`, both published | **high** |
| core goods vs market services | no observable driver — and unsourceable from published weights | **low, and blocked** |

The two hardest splits to source are also the two the evidence says will pay least.

## Open decisions

1. Re-base the partition on the 12 COICOP divisions now (finer, exact, floor 0.0456) rather
   than later.
2. Whether a 7-way that separates administered services and alcohol/tobacco, leaving core
   goods and market services merged, is the right stopping point.

## Status

Phase 0 **passes**. Phase 1 **recorded**. No sophisticated component models exist yet, and
none should be built until the taxonomy is fixed. Nothing here touches production.

Artifacts: `data/components/naive_results.csv`.
