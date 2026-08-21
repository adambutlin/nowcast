# DMS for UK petrol pump prices — it works, and the gain is state-dependent

`code/new_factors/pump_dms.py`, run 2026-08-21. Follow-up to the peer comparison against
the Cleveland Fed inflation nowcast (Knotek & Zaman 2017; FRB Cleveland WP 24-06).

## What DMS actually is

Deterministic Model Switching is not a regime-switching model in the usual sense — nothing
is estimated about which regime you are in. It is a **family of models indexed by the
information set**, with the switch made by the release calendar:

```
A_s(t) · Z_t  =  B_s(t)  +  C_s(t) · X_t  +  Σ_j D_{j,s(t)} · Z_{t-j}  +  ε_s(t)
```

- `Z` — the aggregates being nowcast (Cleveland: headline & core CPI and PCE).
- `X` — the disaggregates that inform them (Cleveland: core, food, **gasoline**).
- `s(t)` — the information state: *which series have actually published by now.*
- `A, B, C, D` all carry the `s(t)` subscript, so the coefficients — including the
  weights on disaggregates vs lagged aggregates — are different objects in different
  states. Switching is deterministic because the calendar is known.

Three design commitments make it work:

1. **Parsimony.** Ten series total: eight monthly, one weekly, one daily. Their paper
   notes a 15-daily-variable MF-DFM and a 150-variable random forest do *not* beat it.
2. **Disaggregate then aggregate.** Nowcast the components, combine them — rather than
   regressing the aggregate on everything.
3. **Dumb rules where data is thin.** Food inflation is a recursive 12-month moving
   average, on the Atkeson-Ohanian principle. No learner.

The gasoline leg is where the accuracy comes from: monthly retail gasoline inflation is
nowcast from *current weekly gasoline prices plus current daily oil*, so they routinely
hold one or two more months of gasoline than of CPI. (The exact functional form is in the
2017 JMCB paper; the Cleveland Fed site blocks programmatic access and it was not
retrieved here.)

## Can it forecast UK petrol prices? Yes.

DESNZ publishes weekly ULSP pump prices — and, in the same CSV, the **duty rate and VAT
rate**, so the tax wedge is observed rather than assumed. The natural UK information state
is how many of the month's weekly readings have published: `s = 1…5`. Target is the
month's *mean* pump price, which is what CPI 07.2.2 tracks (corr 0.993 on MoM).

Walk-forward, expanding window, one Ridge per state, 2014–2026 (n=647–659):

| model | all | s=1 | s=2 | s=3 | s=4 | s=5 |
|---|---|---|---|---|---|---|
| RW (last published week) | 1.378 | 1.855 | 0.850 | 0.655 | 1.372 | 2.274 |
| partial mean, no oil | 1.156 | 1.855 | 1.290 | 0.759 | 0.334 | **0.000** |
| **DMS, pump space** | **0.910** | **1.645** | **0.761** | 0.402 | **0.272** | 0.044 |
| DMS, wholesale space | 0.908 | 1.614 | 0.761 | **0.393** | 0.414 | 0.066 |

RMSE in pence/litre. **DMS beats the random walk by 34%** overall (0.910 vs 1.378), and
0.91 p/l is 0.56% of the current 161 p/l pump price.

The state profile is the whole argument for switching:

- **s=1** — one week in, oil is the only forward information. DMS gains 11% over RW.
- **s=3** — DMS 0.402 vs RW 0.655; oil still projects the unobserved weeks.
- **s=4–5** — the partial mean *is* the answer (0.334, then 0.000 once the month closes),
  while the RW gets **worse** (1.372, 2.274) because the last week diverges from the mean.

No single pooled model expresses both ends of that. A model that trusts oil in week 1 and
trusts the partial mean in week 5 is exactly what deterministic switching buys.

### The switching, made visible

Fitted coefficients, one Ridge per state (`partial_mean` and `last_week` are summed — at
low `s` they are near-identical series, at `s=1` literally the same number, so Ridge splits
the weight between them arbitrarily; the sum is the meaningful quantity and sits near 1.0
throughout, as it must for a level):

| s | pump-level terms (summed) | Brent momentum |
|---|---|---|
| 1 | 1.248 | **4.170** |
| 2 | 0.918 | 2.293 |
| 3 | 0.968 | 0.945 |
| 4 | 0.991 | 0.583 |
| 5 | 0.993 | **−0.051** |

The oil coefficient decays 4.17 → 0. In week 1 oil carries most of the forward information
about the three unpublished weeks; by week 5 there are no unpublished weeks and oil is
worth nothing. A pooled model must pick one number for that coefficient and is wrong at
both ends.

### Why the RW degrades as the month advances

The target is the month's **mean** — a *central* statistic. The RW reports the **last
published week** — an *endpoint* statistic.

| s | \|partial mean − target\| | \|last week − target\| |
|---|---|---|
| 1 | 1.121 | 1.121 |
| 2 | 0.798 | 0.530 |
| 3 | 0.452 | 0.375 |
| 4 | **0.121** | **0.907** |
| 5 | **0.000** | **1.392** |

Early in the month the endpoint sits near the centre, so the RW looks fine — at `s=3` it
beats the partial mean (0.375 vs 0.452). As the month advances the last week moves toward
the *end* of the window while the mean stays at the centre, so in any trending month the
gap widens mechanically.

The RW's error is therefore **specification error, not estimation error**: it answers "what
is the price now?" when the question is "what was the average price this month?" More weeks
cannot shrink it, because it only ever reads one number. The partial mean has the opposite
property — it is a consistent estimator of the target, and at `s=5` it is not approximately
right but *exactly* right, since the mean of all five weeks IS the month's mean.

Note the terminal state: DMS scores 0.044 at `s=5` where the plain partial mean scores
0.000, i.e. the fitted model is slightly WORSE than doing nothing, because Ridge
regularisation pulls the coefficient off exactly 1.0. A real implementation should
hard-code the terminal state rather than fit it — the same "dumb rules where data is thin"
instinct that has Cleveland Fed forecasting food with a moving average.

## The tax-wedge transform did not help — and that prediction was wrong

The UK pump price is `pump = (wholesale + duty) × (1 + VAT)`, with duty a fixed
pence/litre (52.95p; **rising 1p on 2026-09-01**, then 2p in December and 2p in March
2027). The wedge is currently **49.5%** of the pump price. The expectation was that
modelling wholesale and reinstating the wedge would restore a stable elasticity and win.

It did not: 0.908 vs 0.910 is a wash. The reason is that duty and VAT are *constant within
a month*, so for interpolating a monthly mean the wedge is a fixed offset that cancels.

The wedge is real, but it acts on the cross-time elasticity, not the within-month one:

| era | pump-on-Brent | wholesale-on-Brent |
|---|---|---|
| 2011-2014 | 0.184 | 0.413 |
| 2015-2018 | 0.106 | 0.272 |
| 2019-2022 | 0.106 | 0.227 |
| 2023-2026 | 0.181 | 0.324 |

Removing the wedge roughly **doubles** the elasticity, confirming the damping mechanism.
It does not stabilise it — the wholesale elasticity still ranges 0.227–0.413, so refining
margins and crack spreads are moving too, not just tax. Use wholesale space when the
question is pass-through across time; it is unnecessary for nowcasting the monthly mean.

## Where this matters most is now

Within-month spread of weekly pump prices, by year:

| year | mean spread | max |
|---|---|---|
| 2021 | 2.23 | 5.71 |
| **2022** | **6.53** | 15.94 |
| 2024 | 2.01 | 5.79 |
| 2025 | 1.45 | 4.70 |
| **2026** | **5.20** | **16.64** |

2026 is as dispersed within-month as 2022 was. When weekly prices move 16p inside a month,
which weeks you have changes the answer materially — precisely the condition under which
state-dependent weighting pays and a month-end snapshot misleads. August 2026 is a live
example: the monthly mean has gone 152.2 → 161.1 p/l, +5.8%.

## Verdict

- **DMS transfers to UK pump prices and beats a random walk by 34%.** The structure earns
  its keep through state-dependence, not through the regression being clever.
- **Nowcasting works far better than forecasting.** At s=1 — the closest thing here to a
  genuine one-month-ahead forecast — DMS gains only 11% over RW. The value is in
  interpolating a partially-observed month, not in predicting an unobserved one.
- **Skip the wholesale transform for this purpose**, and keep it for pass-through work.
- **Next step**, if this is worth pursuing: chain it the Cleveland way — nowcast the pump
  price, map to CPI 07.2.2 (corr 0.993), and enter headline as a weighted disaggregate
  rather than as `oil_brent` in the pinned factor set. That is the architecture
  `docs/energy_passthrough.md` concluded was needed and `docs/energy_fx_race.md` showed the
  factor-set route could not deliver.
- **Watch the 2026-09-01 duty rise.** +1p on duty is +1.2p at the pump after VAT, ~0.75% on
  the motor-fuel index, arriving as a step no learner will price. It is the same problem
  shape as the Ofgem cap, which production already handles arithmetically via `RegEvent`.

Artifacts: `data/new_factors/pump_dms.csv`.
