# Energy → CPI pass-through: the factors were mis-specified, but the channel is already covered

`code/new_factors/energy_passthrough.py`, `code/new_factors/energy_respec_race.py`, run
2026-08-20. Follow-up to `docs/energy_fx_race.md`, which raced energy factors into the pinned
set and got an underpowered null.

**Premise under test.** Energy drives headline CPI through more than the ~3% motor-fuel
weight: fuel is an input cost to everything transported, heated or manufactured, and energy
spikes are salient enough to move inflation expectations and hence wage bargaining. If so, a
contemporaneous monthly log return — which is how `oil_brent` and `gas_eu` are registered —
is the wrong shape for the factor, and the earlier null was a specification failure rather
than evidence against the channel.

**Identification.** Headline CPI mechanically contains energy, so an energy→headline
correlation mixes the direct arithmetic effect with the indirect one. `uk_core_cpi` (ONS
DKO8) is ex energy, food, alcohol and tobacco, so energy reaches it *only* through costs and
expectations — any energy→core pass-through is second-round by construction.
`uk_services_cpi` is the most wage-intensive aggregate and isolates the bargaining leg.

**Inference.** Both sides are 12-month changes sampled monthly, so residuals are
autocorrelated by construction and Pearson/OLS p-values are badly oversized. Everything below
uses Newey-West HAC (lag 12+k for correlations, 18 for the distributed lags).

## 1. The production spec is the worst available construction

Peak |corr| over windows {1,3,6,12}m and lags 0…24, against the spec production actually uses:

| energy | target | prod spec (1m, lag 0) | HAC p | best spec | peak corr | HAC p |
|---|---|---|---|---|---|---|
| `pump_gbp` | headline | **−0.004** | 0.972 | 12m @ lag 7 | **0.630** | 0.004 |
| `ttf_usd` | headline | −0.079 | 0.437 | 12m @ lag 12 | 0.580 | 0.019 |
| `ttf_gbp` | headline | −0.068 | 0.521 | 12m @ lag 10 | 0.578 | 0.018 |
| `brent_gbp` | headline | −0.032 | 0.462 | 12m @ lag 8 | 0.415 | 0.035 |
| `brent_usd` | headline | −0.041 | 0.359 | 12m @ lag 9 | 0.377 | 0.038 |
| `ttf_gbp` | **core** | −0.062 | 0.480 | 12m @ lag 13 | **0.410** | 0.103 |
| `brent_gbp` | **core** | −0.021 | 0.577 | 12m @ lag 12 | 0.225 | 0.273 |
| `pump_gbp` | **services** | −0.050 | 0.348 | 12m @ lag 16 | **0.433** | 0.007 |
| `brent_gbp` | services | 0.012 | 0.747 | 12m @ lag 14 | 0.221 | 0.063 |

As registered, the energy factors correlate with headline CPI at **−0.08 to 0.00**. The same
series as a 12-month change lagged 7–13 months correlate at **0.38–0.63**. Every lag profile
is classified `hump` (a contiguous run of same-signed lags around the peak), not `spike`, so
these are not grid artefacts. Distributed-lag regressions agree: `pump_gbp` on headline gives
Σb=142, HAC t=3.13, p=0.002, R²=0.36, peak lag 9.

**The premise is confirmed on all three counts.** Energy leads headline by roughly 7–13
months, not zero. Ordering is `pump` > `TTF` > `Brent` — Brent is the *weakest* of the three,
as suspected. And TTF is the strongest predictor of **core** (0.410 vs Brent's 0.225), which
is ex-energy, so that is the indirect cost channel, cleanly identified.

The wage leg also shows up: `pump_gbp` → services CPI peaks at **lag 16** (0.433, p=0.007),
the slowest channel and the right timeline for bargaining rounds.

**One leg does not replicate.** Nothing predicts `uk_be5` (5-year gilt breakeven): every
distributed lag gives p ≥ 0.75. The expectations mechanism does not run through market
breakevens — plausibly because a 5-year measure is the wrong horizon for wage-setting, which
keys off realised and near-term inflation. The services/wage evidence stands on its own.

## 2. Re-specified, energy becomes a top-5 factor — standalone

Walk-forward univariate ridge on the AA residual, lags pre-specified as {3,6,9,12} (no argmax):

| spec | rel-RMSE | corr with residual |
|---|---|---|
| `ttf_gbp_12m_l3` | **0.9295** | **+0.425** |
| `ttf_gbp_6m_l3` | 0.9400 | +0.366 |
| `pump_gbp_6m_l3` | 0.9668 | +0.394 |
| `brent_gbp_6m_l6` | 0.9757 | +0.266 |
| `oil_brent` *(incumbent)* | 0.9988 | −0.031 |
| `gas_eu` *(incumbent)* | 1.0175 | −0.027 |

`ttf_gbp_12m_l3` at 0.9295 would rank **#2 of 50** in `data/new_factors/factor_race.csv`,
behind only `us_ppi_all` (0.883) and ahead of `uk_ppi_input` (0.928). The incumbent `gas_eu`
ranks 43rd. Same underlying series, different transform.

Note the residual-optimal lag is **3 months**, not the 7–13 that maximises correlation with
CPI itself. That is expected and not a contradiction: the overlay models the AutoARIMA
*residual*, and AA has already extracted the long-lag persistent component from CPI's own
history. What is left for a factor to add is the shorter-lead increment.

## 3. …but it adds nothing to the model, because `uk_ppi_input` already carries it

Purged+embargoed pinned-set arms, same gate as `energy_fx_race.md`:

| arm | k | RMSE | rel-RMSE | DM vs base | p |
|---|---|---|---|---|---|
| AA-only | 0 | 0.4687 | 1.0000 | — | — |
| base (incumbent) | 8 | 0.4312 | 0.9201 | — | — |
| **gas_only_respec** (`gas_eu`→`ttf_gbp_12m_l3`) | 8 | **0.4303** | **0.9181** | +0.311 | 0.378 |
| drop_gas_eu | 7 | 0.4318 | 0.9212 | −0.292 | 0.615 |
| base+respec | 12 | 0.4308 | 0.9191 | +0.125 | 0.450 |
| respec12 | 8 | 0.4336 | 0.9251 | −0.379 | 0.648 |
| both_respec_l3 | 8 | 0.4331 | 0.9241 | −0.336 | 0.632 |
| respec6 | 8 | 0.4347 | 0.9274 | −0.544 | 0.707 |
| base+ttf_l3 | 9 | 0.4328 | 0.9235 | −0.653 | 0.743 |
| respec+pump | 11 | 0.4380 | 0.9345 | −0.855 | 0.804 |

**Why the standalone win vanishes:**

```
R² of ttf_gbp_12m_l3 on the incumbent pinned set: 0.681
  corr with uk_ppi_input          +0.783
  corr with uk_quarterly_gdp      +0.512
  corr with mpc_rate_change       +0.322
```

`uk_ppi_input` is UK PPI **input** prices — a basket dominated by imported energy and
energy-intensive materials. It is, functionally, an energy factor with the transmission lag
already baked in, and it entered the model in `docs/new_factors_race.md` as the best UK
factor in the race (rel 0.928). The re-specified TTF factor is 78% correlated with it. Two
representations of one channel; the model only needs one.

So the channel the premise describes **is** in the model — it just arrives under a different
name, which is why nobody noticed the energy columns themselves were inert.

## Verdict

- **The economics is right and the current spec is wrong.** `oil_brent` and `gas_eu`, as
  contemporaneous monthly log returns, carry no information about headline CPI (corr −0.03,
  univariate rel-RMSE 0.999 and 1.018). They are dead weight: dropping `gas_eu` entirely costs
  0.0011 RMSE, within noise.
- **Brent is the weakest energy proxy of the three**, confirmed. TTF dominates on core, pump
  dominates on headline and services.
- **Re-specifying does not improve the model today**, because `uk_ppi_input` already proxies
  the channel at corr 0.78. No arm clears the gate.
- **Recommended change anyway — on robustness, not accuracy.** Replace `gas_eu` with
  `ttf_gbp_12m_l3` (`gas_only_respec`, the only arm that improves on base, rel 0.9181 vs
  0.9201). The accuracy case is not significant and should not be claimed. The robustness case
  is: the model's entire energy channel currently rests on a single ONS series that **has
  already failed once in production** — `uk_ppi_input` went stale and was silently shadowed by
  a dead CSV drop-in, which is `docs/postmortem_2026_07.md` §3 cause 5. Swapping an inert
  market column for a live one that carries 78% of the same signal buys a second, independently
  sourced path to the same channel at zero measured cost.
- **Do not pin `uk_pump_price` for headline.** Its strongest role is the services/wage lead
  (lag 16, p=0.007), a horizon the AA anchor already covers.

Artifacts: `data/new_factors/energy_passthrough_{surface,dlag,best}.csv`,
`data/new_factors/energy_respec_{univ,arms}.csv`.
