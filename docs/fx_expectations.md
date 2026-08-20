# `gbpusd` and `uk_be5`: one is a dud, the other is measuring the wrong thing

`code/new_factors/fx_expectations.py`, run 2026-08-20. Follow-up to `docs/energy_fx_race.md`,
which found that both series are registered `candidate=False` — documented in `factors.py` as
"core factor, always kept" — while neither is in `targets.HEADLINE.pinned`, and that the flag
is dead metadata read only by the legacy `main.py` path. Both nonetheless out-rank most
pinned factors in `data/new_factors/factor_race.csv` (`uk_be5` 8/50 at rel 0.982, `gbpusd`
12/50 at 0.987).

## 1. `uk_be5` is not a breakeven

`factors._uk_breakeven` computed:

```python
nom  = _boe_spot_5y(BOE_BASE + "glcinflationmonthedata.zip")
real = _boe_spot_5y(BOE_BASE + "glcrealmonthedata.zip")
be   = nom - real
```

`glcinflationmonthedata.zip` is the Bank of England's **implied inflation** curve, which
already *is* nominal minus real. Verified against `glcnominalmonthedata.zip`:

```
BoE inflation curve == nominal - real : max abs diff 0.000000  (1985-2026, n=498)
```

So the variable named `nom` is the breakeven, and the factor evaluates to
`nominal - 2 × real` — the real yield subtracted twice.

| | value |
|---|---|
| corr(registry `uk_be5`, **true** 5Y breakeven) | **+0.208** |
| corr(registry `uk_be5`, **negated real yield**) | **+0.746** |
| mean level: registry vs true | 2.52 vs 3.70 |

`uk_be5` is a real-rate factor wearing an inflation-expectations label. Whatever rank it holds
in the factor race, it earned as one.

## 2. Fixed — and shortened — it is the strongest factor in this repo's research so far

Shorter tenors predict near-term CPI monotonically better. Walk-forward univariate ridge on
the AA residual (the `factor_race.py` decision metric):

| spec | rel-RMSE | corr with residual | n |
|---|---|---|---|
| `be_slope_10y_2_5y` | 0.8892 | −0.469 | 66 |
| `be2_5` (2.5Y breakeven) | **0.8960** | **+0.466** | 66 |
| `be3` (3Y) | 0.9206 | +0.428 | 89 |
| `be5` (5Y, **fixed**) | 0.9471 | +0.390 | 96 |
| `be10` (10Y) | 0.9548 | +0.370 | 96 |
| `uk_be5` (**as registered, broken**) | 0.9822 | +0.291 | 96 |

The monotonicity in tenor is the point: a nowcast wants short-horizon expectations, and a 5Y
breakeven is the most anchored point on the curve. Merely fixing the construction moves 5Y
from 0.982 to 0.947; going short as well takes it to 0.896, which would rank **#2 of 50** in
`factor_race.csv`.

Lead-lag against headline CPI YoY agrees (HAC lag 12+k; all profiles `hump`, not `spike`):

| series | peak lag | corr | HAC p |
|---|---|---|---|
| `be2_5` | 7 | 0.573 | 0.057 |
| `be3` | 7 | 0.530 | 0.067 |
| `be10` | 6 | 0.497 | 0.034 |
| `be5` | 6 | 0.442 | 0.093 |
| `be5_registry_bug` | 11 | 0.433 | 0.107 |
| `real5` | 16 | −0.379 | 0.146 |

## 3. In the pinned set — best result yet, still not significant

Purged+embargoed walk-forward, **matched window** (n=97, the months every arm can score — the
breakeven series are ragged, so the unmatched comparison in `fx_exp_arms.csv` is not
like-for-like and should not be quoted):

| arm | RMSE | rel-RMSE | DM vs base | p |
|---|---|---|---|---|
| AA-only | 0.4932 | 1.0000 | — | — |
| base (incumbent) | 0.4481 | 0.9086 | — | — |
| **+be3** | **0.4108** | **0.8331** | **+1.010** | **0.156** |
| +be2_5 | 0.4156 | 0.8426 | +0.760 | 0.224 |
| +be5 (fixed) | 0.4476 | 0.9076 | +0.147 | 0.442 |
| +be5 (registry, broken) | 0.4321 | 0.9219 | −0.604 | 0.727 |
| +`gbpusd_12m_l12` | 0.4337 | 0.9253 | −1.155 | 0.876 |

Sub-periods show where it comes from:

| window | n | base | +be2_5 | +be3 | +be5 |
|---|---|---|---|---|---|
| 2015-2019 | 43 | 0.1699 | 0.1680 | 0.1690 | 0.1714 |
| 2020 | 11 | 0.4034 | 0.4084 | 0.4100 | 0.4145 |
| **2021-2023** | 33 | 0.6643 | 0.6034 | **0.5961** | 0.6660 |
| **2024** | 10 | 0.4339 | 0.4110 | **0.3963** | 0.4055 |

Flat in calm, −10.3% RMSE through the inflation shock, −8.7% in 2024. That is the same
shape the overlay has always had (`docs/final_model.md`): value concentrated in shock months.
p=0.156 still does not clear the gate, and n=33 in the window doing the work.

**`be3` over `be2_5` on operational grounds.** The 2.5Y point is missing in **25 of 140**
months since 2015 against **9** for 3Y (5Y: 2). A stale pinned factor blocks `_nowcast_row`
and kills the TVP leg outright, so 18% raggedness is disqualifying for a marginally better
number.

## 4. `gbpusd` is a dud

Same mis-specification as the energy factors — exchange-rate pass-through is slow, and the
registered contemporaneous log return points the wrong way:

| spec | rel-RMSE | corr with residual |
|---|---|---|
| `gbpusd_12m_l9` (12m change, lagged 9) | 0.9888 | +0.262 |
| `gbpusd_12m_l12` | 0.9943 | +0.204 |
| `gbpusd_1m_l0` (**as registered**) | 0.9979 | **−0.173** |

So the pass-through is real and slow (9-12 month lag), confirming the specification is wrong.
But the effect is small, and in the pinned set FX is actively harmful: `+fx_slow` gives rel
0.9253 against base 0.9086, DM **−1.155**, p=0.876. `gbp_eer` (broad effective rate, the
better-weighted measure for import costs) does no better — its best spec is 0.9986.

This is consistent with the plain `+fx` null in `docs/energy_fx_race.md`. **FX has been tested
three ways now and fails every time. Stop proposing it.** The accidental drop at `d25f185`
cost nothing.

## RPI caveat

UK index-linked gilts are RPI-linked, so every breakeven here prices **RPI**, not CPI. The
RPI-CPI wedge (formula effect plus housing) averages roughly 0.8-1.0pp and is persistent, so
as a LEVEL regressor it is absorbed by the intercept — but these series must never be read as
a market forecast of CPI. `uk_infl_swap_1y` (LSEG, used by CORE) covers the short end properly
but needs a Workspace session; the free BoE curve stops at 2.5 years.

## Verdict

- **`_uk_breakeven` is a defect and should be fixed regardless of any race** — it is not a
  judgement call about factor selection, the function does not compute what its name says.
  Fixing it rescales the series by ~1.2pp.
- **Fixing it is not free downstream.** `uk_model_zoo.RegimeConditional.THRESHOLD` is a
  hardcoded 3.0 calibrated against the broken scale (mean 2.52); against the true breakeven
  (mean 3.70) it splits the sample somewhere else entirely and needs re-calibration.
  RegimeConditional is research-only — production is AutoARIMA + TVP + LGBM — so no live
  forecast moves, and `test_headline_regression` passes unchanged.
- **`uk_be3` is the strongest pinning candidate this research has produced** (rel 0.8331 vs
  base 0.9086, DM +1.010). It still does not clear p<0.05, so by the standing gate it does not
  ship. Unlike the energy factors it is *not* redundant with `uk_ppi_input` — it is an
  expectations channel the headline model has never had.
- **Retire `gbpusd` as a candidate for headline.** Three independent tests, three failures.
- **Delete the `candidate=False` flag or wire it up.** It has now caused two rounds of
  confusion about what is and is not in the model.

Artifacts: `data/new_factors/fx_exp_{surface,univ,arms}.csv`.
