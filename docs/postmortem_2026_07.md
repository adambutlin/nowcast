# Post-mortem — July 2026 nowcast miss (headline −0.34)

| | AA | overlay | **final** | actual | error |
|---|---|---|---|---|---|
| Headline CPI YoY | 2.516 | +0.046 | **2.562** | 2.90 | **−0.338** |
| Core CPI YoY | 2.498 | +0.028 | **2.526** | 2.60 | −0.074 |

Largest live error since genesis (May-2026: +0.11, June-2026: +0.08). The overlay was directionally
right and reduced the error by 0.046 — but for the wrong reason (see cause 3). The anchor carried
the whole miss: AA error −0.384.

---

## 1. What actually drove the July print

July CPI rose 2.6 → 2.9 (official D7G7; the rounded index D7BT implies 2.59 → 2.81, +0.21).
Monthly index change was **+0.281%** against a July-2025 base of only **+0.072%**.

The driver was **regulated household energy**, not motor fuel. Ofgem raised the default-tariff cap
by **13% on 1 July 2026** (electricity unit rate 24.67 → 26.11 p/kWh, gas 5.74 → 7.33 p/kWh;
standing charges flat).

| CPI component (2026 weight /1000) | MoM Jul-26 | MoM Jul-25 | contrib to MoM | contrib to YoY change |
|---|---|---|---|---|
| 04.5.2 Gas (10.3) | **+14.72%** | −7.18% | +0.152pp | **+0.226pp** |
| 04.5.1 Electricity (20.6) | **+3.59%** | −3.76% | +0.074pp | **+0.151pp** |
| 07.2.2 Motor fuels (26.4) | −3.16% | +1.67% | −0.083pp | −0.127pp |
| 07.3 Transport services (27.8) | +4.52% | +8.63% | +0.126pp | −0.114pp |
| **All items** | **+0.281%** | +0.072% | | **+0.209pp** |

Household energy alone contributed **+0.38pp** to the YoY change — nearly twice the realised
+0.21pp rise, with motor fuels and travel services offsetting. Absent the cap step, July would
have printed close to AA's 2.5. **The anchor was right about everything except the policy step.**

## 2. What the model saw (real-time matrix, post-publication-lag)

```
             oil_brent  gas_eu  uk_qtr_gdp  imf_comm  mpc_rate  ofgem_cap_delta  uk_ppi_input  freight
2026-06-30     -0.2763 -0.0679         NaN   -0.0971       0.0              0.0        8.7179   0.0598
2026-07-31     +0.3192 +0.1723         NaN   -0.0073       0.0            >0.0<           NaN   0.0438
```

Two entries carry the whole story: `ofgem_cap_delta = 0` in the month of a +13% cap step, and
`uk_ppi_input = NaN` (the LGBM's top feature).

---

## 3. Causes, ranked

### Cause 1 — the regulated-energy channel is pinned but carries zero weight (dominant)

`ofgem_cap_delta` was **0 for July**: `data/ofgem_cap.csv` ends 2026-06 (`fetch=None`, manual
source), and `two_stage.load_matrix` fills regulatory factors with `0` — so *missing* is silently
encoded as *no cap change*. The true value was ≈ **+£213** (1641 → ~1854 on the CSV's TDCV basis).

But patching the true value in changes the forecast by **nothing**:

```
cap_delta =    0 -> tvp +0.230  lgbm -0.046  fc 2.562
cap_delta = +213 -> tvp +0.230  lgbm -0.046  fc 2.562
cap_delta = +2000 -> tvp +0.230 lgbm -0.046  fc 2.562
```

LGBM gain importance for `ofgem_cap_delta` is **0.0%** (the column is zero in all but 23 months of
the sample, so no split ever pays); TVP cannot see it at all (cause 3). Yet across those 23
cap-change months, **corr(AA residual, cap delta) = +0.54** — a real signal the production overlay
discards entirely.

### Cause 2 — the only live energy factor pointed at the wrong fuel

`oil_brent` (1-month log return, pub_lag=0) was **+0.319** in July — Brent 70.5 → 97.0, a rebound
from a −27% June crash. Motor-fuel CPI *fell* 3.2% in the same month, because pump prices track
crude with a multi-week lag and are damped by duty/VAT. The model's one "energy shock" input was a
signal about the component that dropped, and LGBM read it bearishly (−0.046 vs +0.118 at
`oil_brent = 0`).

### Cause 3 — BUG: the TVP overlay nowcasts a 13-month-stale month

`production/model.py` embargoes the residual series (`V.embargo_series`, PURGE 12 + EMBARGO 1)
before calling `Z.TVP().nowcast(d, live, "resid")`. `BaseModel._nowcast_row` derives its own target
month as *"first month after the last non-NaN target"* — which, after embargoing, is **2025-06-30**.

Verified:

```
resid last known (unembargoed): 2026-06-30
resid last known (embargoed)  : 2025-05-31
TVP nowcast value 0.2304 for month: 2025-06-30      <-- not 2026-07-31
TVP output is invariant to every 2026-07 factor value (oil, gas, cap, ...)
perturbing 2025-06 oil_brent moves it: 0.230 -> -0.088
```

So the "shock pass-through diversifier" contributed a June-2025 number to a July-2026 forecast, and
has done so on every run since the embargo was introduced. LGBM is unaffected — it predicts the row
explicitly (`df.loc[[nd], live]`).

Fixing the month makes July slightly *worse*: the shipped fix (embargoed training, prediction month
pinned to `nd`) gives TVP +0.119 → **2.534**, err −0.366; core 2.526 → 2.536. A variant that also
drops the embargo from TVP training gives +0.005 → 2.505. Either way the +0.058 "help" was luck.
The point is not the sign this month — as wired, the TVP leg was not a forecast of the target month
at all. **Fixed 2026-08-19** (`_tvp_resid` + optional `nowcast_date` in the zoo); see
`docs/final_model.md` §11 for the governance entry and the scoring caveat.

### Cause 4 — the anchor's July seasonal prior

AA is univariate: persistence + seasonality + base effects. July is historically a *soft* month
(MoM: 2023 −0.46, 2024 −0.22, 2025 +0.07; negative in 5 of the last 12). July-2026's +0.281% was
the strongest July MoM since 2022. AA predicted 2.516 — a further fall from June's 2.6 — and owns
−0.384 of the −0.338 total error. No univariate anchor prices a discretionary cap step.

### Cause 5 — stale CSV shadowed a live fetch (minor here, systemic risk)

`data/uk_ppi_input.csv` (written 17 Jun) ends 2026-05, and `load_factor` gives **CSV priority over
live fetch with no freshness check**. After `pub_lag=1`, the July row was NaN — LGBM's top feature
(31.5% gain) was missing. Refreshing from the live ONS GHIP series: LGBM −0.001 → forecast 2.573
(+0.011). Small this month; silent data rot in general.

---

## 4. Counterfactuals

| scenario | headline | error |
|---|---|---|
| as-run | 2.562 | −0.338 |
| + true Ofgem cap delta (+£213) | 2.562 | −0.338 |
| + refreshed PPI (live June value) | 2.573 | −0.327 |
| + both | 2.573 | −0.327 |
| TVP predicting the correct month (shipped fix) | 2.534 | −0.366 |
| TVP correct month + no training embargo | 2.505 | −0.395 |
| λ = 1 (experimental overlay, as-run) | 2.608 | −0.292 |
| AA + ex-ante cap-step arithmetic (see below) | ≈2.75 | ≈−0.15 |

**Ex-ante cap arithmetic.** The July cap rates were public in late May — legitimately available at
a 31-July nowcast. Applying the announced unit-rate changes at the observed index pass-through
(elec 3.59/5.84 ≈ 0.61; gas 14.72/27.7 ≈ 0.53) with 2026 CPI weights gives ≈ +0.23pp — about
two-thirds of the gap, from public information, with no model retraining.

---

## 5. Recommended actions

**Bug fix — DONE 2026-08-19 (commit on `main`):**

1. ~~`production/model.py` — pass the prediction row to TVP explicitly.~~ Shipped:
   `BaseModel._nowcast_row` / `BaseModel.nowcast` / `TVP.nowcast` take an optional `nowcast_date`
   (default behaviour unchanged for every other caller); `production.model._tvp_resid` pins it to
   `nd`. Training rows stay embargoed. Covered by `tests/test_production_overlay.py::
   TestProductionTVPMonth`; `test_headline_regression.py` re-based off unreproducible goldens onto
   spec + arithmetic invariants. Governance entry: `docs/final_model.md` §11.

**Data-integrity fixes:**

2. `two_stage.load_matrix` — stop `fillna(0)` on regulatory factors *beyond the source's last
   observation*. Fill zeros inside the observed span; leave months after the CSV ends as NaN and
   surface them like any other stale factor.
3. `factors.load_factor` — when a CSV drop-in ends earlier than the live fetch, prefer the live
   series (or at minimum warn). Add a freshness assertion to the production run.
4. Automate `ofgem_cap.csv` (Ofgem publishes the cap ~6 weeks before it takes effect, so pub_lag=0
   is honest) or add it to a pre-run checklist.

**Model-design questions (require governance):**

5. The regulated-energy channel needs a representation the learners can use — a level diff in £/yr
   that is zero 3 months in 4 is invisible to a regularised GBM. A direct **CPI-contribution
   feature** (announced cap % × per-fuel weight × historical pass-through, in pp) is the natural
   form, and is known ex-ante.
6. Replace or complement contemporaneous `oil_brent` with a lagged/3-month form (`oil_brent_3m`
   already exists as a candidate) or with weekly UK pump prices, which is the actual transmission
   channel for the motor-fuel component.

---

## Sources

- [Ofgem — Changes to energy price cap between 1 July and 30 September 2026](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-july-and-30-september-2026)
- [Ofgem — Changes to energy price cap between 1 April and 30 June 2026](https://www.ofgem.gov.uk/news/changes-energy-price-cap-between-1-april-and-30-june-2026)
- [Ofgem — Energy price cap unit rates and standing charges](https://www.ofgem.gov.uk/information-consumers/energy-advice-households/energy-price-cap-unit-rates-and-standing-charges)
- ONS MM23 series: D7BT (all items index), D7DT (04.5.1 electricity), D7DU (04.5.2 gas),
  D7EC (07.2.2 fuels & lubricants), D7CQ (07.3 transport services); CPI weights CJXA, CJXB, CJXR, CJVO.
