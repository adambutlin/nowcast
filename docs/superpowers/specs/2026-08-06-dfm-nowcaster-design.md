# Design — structured mixed-frequency DFM nowcaster (alternative UK CPI model)

**Status:** approved design, pre-implementation. **Branch:** `dfm` (off `core`). **Date:** 2026-08-06.

## Goal

Build an **alternative** UK CPI nowcaster — a structured mixed-frequency Bayesian dynamic factor
model inspired by the NY Fed Staff Nowcast redesign — and answer one question honestly:

> Can a structured mixed-frequency DFM with stochastic volatility, outlier adjustment, factor-lag
> interactions, and a revised loading structure beat the frozen production model on UK CPI
> nowcasting?

A **null result is a valid, publishable outcome.** The deliverable is the evidence, not a win.

### Non-goal — production is frozen

The frozen production model is **not touched**. No edits to `code/production/`, `code/factors.py`,
`code/targets.py`, `code/new_factors/two_stage.py`, or `code/uk_model_zoo.py`. This branch adds
`code/dfm/` and its own docs/plots/data outputs. Nothing here ships to production without a separate
governance decision.

**Proof of non-interference (hard gate):** `code/tests/test_headline_regression.py` must pass
unchanged on this branch, and `git diff main --stat -- code/production code/factors.py
code/targets.py code/new_factors/two_stage.py code/uk_model_zoo.py` must be **empty**.

## The bar

Walk-forward 2015–2024, purged + embargoed, scored against the same windows production uses.

| Benchmark | RMSE | Source |
|---|---|---|
| AR(1) | 0.495 | `README.md` model table |
| AutoARIMA (anchor) | 0.467 | `README.md` |
| TVP overlay member | 0.482 | `README.md` |
| LGBM overlay member | 0.443 | `README.md` |
| **Frozen production** (AA + 0.25·TVP + 0.25·LGBM) | **≈0.416** (rel-RMSE **0.89** vs AA) | `docs/final_model.md` §3 |
| *Prior DFM attempts in this repo* | *1.97 (DFM), 1.96 (DFM-k2), 2.62 (MS-DFM)* | `data/rmc_dfm_perf.csv` |

Prior DFMs lost to AA by a factor of four. **The reason is diagnosed and addressed** (§3.2): they
modelled CPI **YoY**, whose lag-1 autocorrelation is 0.983 — 11/12 of its variance is mechanical
carry-over the factors were forced to re-learn. This design models **MoM** (lag-1 autocorr **0.048**)
and makes the carry-over arithmetic. That reformulation is what makes this a fair fight rather than
a rerun.

## Locked decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| **DFM target** | **Both** — standalone CPI *and* factors→AA-residual overlay | Only way to answer the final question fairly; hedges the real risk that the DFM loses on level/base-effects while still adding value on the residual |
| **Frequency** | Monthly latent state + **as-of-day-`d`** panel | Keeps the month-end standpoint comparable to production; ragged edge handled natively by the Kalman filter; gives the forecast-evolution deliverable without a daily state |
| **Refit cadence** | **Monthly** refit, ~2000 draws (warm-started) | Matches the TVP/LGBM backtest convention; annual refit would not be apples-to-apples |
| **Panel width** | **32 series**, block-balanced, all availability-verified | Each factor identified by 3–8 series, so factors are economically interpretable and the by-block ablation has bite. No LSEG/Workspace dependency (§1.6) |
| **Reg/event block** | **Observed exogenous** shifter, not latent | 3 sparse, mostly-zero administered-event series cannot identify a latent factor; it would just track `ofgem_cap_delta` |
| **CPI space** | **Model MoM, aggregate to YoY** | See §3.2 — the single most important choice in this design |
| **Eval windows** | **Match production exactly** | Headline claims directly comparable to 0.89 rel-RMSE with no re-derivation |
| **May/Jun-2026** | **Held out from all estimation**, retained as blind scoring points | Extends the existing 2025+ blind hold-out to the two live months |
| **Estimation** | Hand-written **Gibbs sampler** (numpy + numba) | PyMC cannot handle a sampled 350×N latent state with SV; the Gibbs blocks are the standard Kim-Nelson / Stock-Watson lineage and make every ablation *exact* rather than approximate |

Environment: `quant` conda only (`/opt/anaconda3/envs/quant/bin/python`). PyMC 5.28, numba 0.64,
statsmodels 0.14 available; **no JAX / numpyro / Stan**. Never run parallel Python processes here
(`docs/CLAUDE.md`).

---

## 1. Model specification

### 1.1 Measurement equation

`N = 32` standardized series in 6 economic blocks, plus CPI entering as MoM log-change (33
measurement equations total). **Every series was empirically verified available on 2026-08-06** — see
§1.6:

```
x_it = λ_i' f_t  +  γ_i' z_t  +  e_it            i = 1..N
π_t  = λ_π' f_t  +  γ_π' z_t  +  e_πt            π_t = 100 · Δ log(D7BT_t)
```

- `f_t` — **`K = 7`** latent factors: one **pervasive `global`** factor plus **six block factors**
  (`activity`, `nominal`, `energy`, `supply`, `fx`, `food`). The global factor is a distinct object
  from the activity block — every series loads on `global`; only its own members load on a block factor.
- `z_t` — observed exogenous reg/event block: `mpc_rate_change`, `ofgem_cap_delta`, `budget_event`.
- `e_it` — idiosyncratic AR(1), carried **in the state vector** (not quasi-differenced, because
  idiosyncratic SV makes quasi-differencing ill-defined).

### 1.2 State equation — factor VAR with SV

```
f_t  = Φ₁ f_{t-1} + Φ₂ f_{t-2} + η_t        η_t ~ N(0, diag(exp h_t))
h_kt = h_{k,t-1} + σ_k ζ_kt                 ζ ~ N(0,1)      factor SV (log-vol random walk)
g_it = g_{i,t-1} + τ_i ξ_it                                 idiosyncratic SV
```

**Factor-lag interactions are the off-diagonal elements of `Φ`.** E1–E3 restrict `Φ` diagonal (each
factor an independent AR process); E4 frees the off-diagonals under a Minnesota prior. So "do
VAR-style lead-lag spillovers between factors help?" becomes a **shrinkage decision with a prior**,
not an unconstrained parameter grab — which is what keeps it honest at `T ≈ 290`.

### 1.3 Outlier adjustment (Stock-Watson mixture)

```
e_it = exp(g_it / 2) · o_it · ε_it
o_it = 1              with prob (1 − p)
o_it ~ Uniform(2, 10) with prob p                p ~ Beta(a, b), small
```

Chosen over a Student-t scale mixture because it is **directly interpretable**: the posterior gives
`P(o_it > 1)` — a per-series, per-month outlier probability. Expected to light up on Ofgem cap
resets, the 2020 VAT cut, and COVID collection gaps. That series is itself a deliverable.

### 1.4 Loading structure

- **E1–E4 (conventional):** unrestricted `Λ` — every series loads on all 7 factors — identified by the
  standard normalization (upper `K×K` block lower-triangular with positive diagonal). Factors here are
  statistical, not nameable.
- **E5 (revised):** **block-restricted** — every series loads on `global` **plus its own block factor
  only**; all other loadings pinned to zero. This is the "revised factor loading structure"
  requirement, and the restriction is what makes each factor economically nameable.

**`K = 7` is held fixed across all five arms**, so E5 vs E4 isolates the *loading restriction* alone
and does not confound it with a change in factor count. A `K ∈ {5, 6, 7, 8}` sensitivity check is run
on the best arm and reported separately.

### 1.5 Blocks

All 32 series load on the pervasive `global` factor; the block column below is the *second* factor
each series loads on under E5.

| Block | n | Series | Economic role |
|---|---|---|---|
| `activity` | 7 | `uk_quarterly_gdp`, `uk_retail_sales`, `uk_vacancies`, `uk_awg`, `uk_house_prices`, `vix`, `move_index` | real activity, labour, demand; `vix`/`move_index` as risk-appetite proxies |
| `nominal` | 8 | `uk_ppi_output`, `uk_ppi_input`, `uk_core_cpi`, `uk_services_cpi`, `uk_rents`, `uk_be5`, `uk_gilt_10y`, `slope_2s10s` | price level, inflation expectations, rates (`uk_be5` = BoE gilt-implied 5y breakeven) |
| `energy` | 5 | `oil_brent`, `gas_eu`, `gas_hh`, `imf_all_commodity`, `oil_vol_6m` | cost-push |
| `supply` | 6 | `deep_sea_freight`, `semiconductors_ppi`, `chemicals_ppi`, `metals_index`, `copper_price`, `timber_price` | freight / industrial materials / bottleneck |
| `fx` | 3 | `gbp_eer`, `gbpusd`, `gbp_eur` | import pass-through |
| `food` | 3 | `food_price_index`, `wheat_price`, `vegetable_oil_price` | **new channel** |
| **total** | **32** | | |
| `reg` *(exogenous `z_t`)* | 3 | `mpc_rate_change`, `ofgem_cap_delta`, `budget_event` | administered / regulatory shocks |

Block sizes 3–8, so every factor is identified by at least three series.

**The `food` block is the only genuinely new information channel vs. production.** `docs/final_model.md`
§4 attributes the ~74% unexplained residual to "food/services/idiosyncratic", and the documented
calm-month failure mode is precisely food/services-driven prints. If any block earns its place, this
is the prior favourite — and if it does not, that is a substantive finding about the residual.

`mpc_rate_change` (2015+) and `ofgem_cap_delta` (2015+) start mid-sample and are zero-filled before
their start, following the existing production convention (`Target.reg` / `two_stage.load_matrix`).

### 1.6 Data availability — verified, not assumed

A full `factors.build_matrix` probe was run on 2026-08-06 before finalizing the panel. **Seven
initially-proposed series were unavailable and have been removed**, which is why the block structure
above differs from the first draft:

| Excluded | Reason |
|---|---|
| `uk_pmi_composite`, `uk_pmi_services`, `uk_infl_swap_1y` | LSEG — need the Workspace desktop app running (`localhost:9000`); fetch fails otherwise |
| `uk_pmi_manufacturing`, `uk_paye` | CSV-only (licensed / no free API) and **no CSV present** in `data/` |
| `freightos_fbx` | no free source at all; registry note says use `deep_sea_freight` as the proxy — done |
| `global_supply_chain_pressure` | known expected-drop (xlsx engine error), already `OPTIONAL_PINNED` in production |

Also excluded by construction: `uk_monthly_gdp` (dies 2024-03), `uk_cpih` and `uk_trimmed_mean_cpi`
(circular with the target / unavailable), `battery_metals_proxy`, `uk_hciall`, `vimes_boots_index`,
`uk_ashe_pay`, `ons_econ_activity` (all unavailable), `slope_5s30s` (starts 2016 — too short).

**Design consequence — no LSEG dependency.** The panel is built entirely from unconditionally-fetchable
series. A multi-hour walk-forward must not depend on a desktop application being open, and a result
that cannot be reproduced without one is not a result. The three LSEG series are therefore treated as
an **optional enrichment tested separately** (an E5+LSEG variant, run only if Workspace is up), never
as part of the headline panel.

**Staleness guard.** `uk_house_prices` last prints 2026-01 and `uk_awg` 2026-05. Ragged edges are
handled natively by the filter, but the panel builder logs any series whose last observation is >6
months stale at run time so a silently-dead series is visible rather than quietly contributing nothing.

---

## 2. Mixed frequency and the as-of-day panel

### 2.1 Aggregation

- **Daily series** (market/financial, `pub_lag=0`): at origin day `d` in reference month `T`, the
  month-`T` observation is the aggregate of days `≤ d` only. Level series use last-value, return
  series use the partial-month sum. Days `> d` are **not** observable and enter as missing.
- **Monthly series** (`pub_lag=1`): month `T` is missing at every origin in `T`; the last observed
  month is `T-1`, exactly as `apply_publication_lags` enforces.
- **Quarterly series** (`pub_lag=2`, e.g. `uk_quarterly_gdp`): loaded through a 3-month aggregator
  `y_q = ⅓(f_t + f_{t-1} + f_{t-2})`, with the two intra-quarter months observed as missing. This is
  the Mariano-Murasawa aggregation applied at monthly rather than daily resolution.

Missingness is handled natively by the **sequential (univariate) Kalman treatment** — a missing
observation is simply a skipped update step. No interpolation, no EM fill.

### 2.2 Forecast evolution

Re-run the filter at origins `d = T-28, T-21, T-14, T-7, T` over the as-of-day panel (5 weekly
origins; daily resolution adds cost without adding information, since most blocks update at most
weekly).

**This is cheap, and the reason matters for the runtime budget:** MCMC parameters are drawn **once
per month**, then each within-month origin re-runs only the **Kalman filter** on the as-of-day panel,
holding the posterior parameter draws fixed. Filtering is ~3 orders of magnitude cheaper than
sampling, so evolution costs ≈5 extra filter passes per month, not 5 extra fits. This is also the
*correct* conditioning: parameters estimated on data through `T-1` are valid at every origin inside
month `T`.

Evolution is produced for the **selected best arm only** — not all 10 configs.

**Production has no intramonth product**, so it appears on the evolution chart as a flat line at its
month-end value, clearly labelled as such. **The headline comparison against production is at `d = T`
(month end) only**, because that is production's information standpoint. Earlier origins are reported
as a descriptive curve — how the DFM's view firms up as data arrives — and are **never used to claim a
win over production**, which would be comparing against a model that simply has not run yet.

---

## 3. The target

### 3.1 Data

CPI index **D7BT** (ONS MM23), verified available via dbnomics: `n=949`, 1947-06 → 2026-06, last
`142.5`. The registry carries only YoY (`D7G7`), so `code/dfm/panel.py` defines `cpi_index` locally
reusing `factors._dbnomics` — **zero diff to `factors.py`.**

### 3.2 MoM → YoY reconstruction

The base-effect carry is fully known at `T` (every term published):

```
c_T   = 100 · (I_{T-1} / I_{T-12} − 1)
YoY_T = 100 · ((1 + c_T/100) · exp(π_T / 100) − 1)
```

The DFM forecasts **`π_T` only**. All model risk concentrates on the single unknown monthly
increment — where the factors actually have signal — instead of being spread over 12 months of
arithmetic.

**Measured justification** (computed 2026-08-06 on live ONS data):

| Quantity | Value |
|---|---|
| CPI YoY lag-1 autocorrelation (2000+) | **0.983** |
| CPI MoM lag-1 autocorrelation (2000+) | **0.048** |
| MoM std, 2015–2024 | 0.456 pp |
| MoM std, 2022–23 shock | **0.688 pp** |
| MoM std, ex-shock | **0.336 pp** |

The 2× volatility ratio between shock and calm months is direct evidence that **stochastic volatility
has something real to model** — it is not decoration.

### 3.3 Known accuracy floor

ONS publishes D7BT to 1 d.p. Reconstructing YoY purely from the index differs from official D7G7 by
up to **0.135 pp** (2000+). Mitigation: anchor on the **official** published index/YoY and add only
the modelled increment, which confines rounding to a single index ratio (≈0.07 pp). This is ~6× below
AA's 0.467 RMSE, so it does not threaten the comparison, but it **is a floor** and is reported as
such in the model card. The target `D7G7` is itself 1 d.p., so this floor binds every model equally.

---

## 4. Estimation

Gibbs sampler, numba-compiled, single process:

1. **Factors `f_{1:T}`** — Durbin-Koopman simulation smoother, sequential/univariate treatment
   (`O(N·k²)`, missing-data-native).
2. **Loadings `Λ`, `γ`, idio AR coefficients** — series-by-series Bayesian regression with
   Normal-Gamma shrinkage; block restrictions imposed by zeroing (E5).
3. **`Φ`** — VAR posterior under a **Minnesota prior** (own-lag loose, cross-lag tight).
4. **SV states `h`, `g`** — Kim-Shephard-Chib 7-component mixture approximation + FFBS.
5. **Outlier indicators `o`** — conjugate discrete/uniform mixture draw; `p` from its Beta posterior.
6. **Hyperparameters** — `σ_k`, `τ_i`, shrinkage scales.

**Draws:** 2000 (1000 burn-in) for the first walk-forward fit; **800 (200 burn-in) thereafter**,
warm-started from the previous month's posterior mean. Warm-starting uses **no future information** —
it is initialization only, and the previous month's posterior conditions on a strict subset of the
current month's data. Convergence checked by `arviz` R-hat / ESS on a sampled subset of fits, not
every fit.

**Runtime budget:** (10 experiment configs + 7 ablations) × 120 test months ≈ **2040 fits**. The
runner **checkpoints to disk per (config, month)** and is resumable, so a multi-hour sweep survives
interruption. Runs sequentially in one background process (`docs/CLAUDE.md` forbids parallel Python
here — processes serialize on the macOS security scanner).

**Cost control is a first-class implementation task, not an afterthought.** Before the full sweep, a
timing probe measures seconds-per-fit on the real panel. If the projected sweep exceeds ~8 hours, the
documented fallbacks are applied **in this order**: (1) drop ablations to the best arm only, (2) halve
post-warm-start draws to 400, (3) fall back to quarterly parameter refit with monthly filtering. Any
fallback used is recorded in the model card, because it changes what the numbers mean.

---

## 5. Leakage safety

Reuses the repo's existing discipline:

- Panel built through `factors.build_matrix` + `apply_publication_lags` — same `pub_lag` tiers.
- As-of-day aggregation never reads past day `d` (§2.1).
- Expanding-window walk-forward; **2025+ blind**; May/Jun-2026 excluded from every estimation sample.
- Leakage audit run over the DFM panel, mirroring `code/new_factors/leakage_audit.py`; target 0 violations.

### 5.1 The purge-horizon asymmetry — stated, not buried

| Mode | Label | Purge + embargo |
|---|---|---|
| Overlay (factors → AA residual) | `cpi_yoy` residual — a genuine 12-month difference | **12 + 1** (identical to production) |
| Standalone (factors → `π_T`) | MoM — a **1-month** difference | **1 + 1** |

Forcing 12+1 on a 1-month label would discard real information rather than prevent leakage — that is
the stated rationale in `code/validation.py`. But the consequence is that the standalone arm trains
on more recent data than production does. **Any standalone win must be reported with this caveat
attached**, and the model card will additionally report the standalone arm re-run under 12+1 as a
sensitivity check, so a reader can see how much of the edge is the reformulation vs. the horizon.

---

## 6. Experiments

### 6.1 Required arms

| # | Arm | `Φ` | SV | Outlier | Loadings |
|---|---|---|---|---|---|
| **E1** | DFM baseline | diagonal | — | — | unrestricted |
| **E2** | + stochastic volatility | diagonal | factor + idio | — | unrestricted |
| **E3** | + outlier adjustment | diagonal | factor + idio | SW mixture | unrestricted |
| **E4** | + factor-lag interactions | **full VAR** | factor + idio | SW mixture | unrestricted |
| **E5** | revised loading structure | full VAR | factor + idio | SW mixture | **block-restricted** |

Each in **both modes** (standalone, overlay) → 10 configs. SV is bundled (factor + idiosyncratic) in
E2; a factor-only-SV sub-variant is tested inside E2 and reported, not promoted to its own arm.

### 6.2 Ablations

Leave-one-block-out on the best-performing arm: 6 latent blocks + the exogenous reg block = **7
ablations**, each reporting ΔRMSE by window.

### 6.3 Benchmarks

AutoARIMA · AR(1) · **replication of frozen production** (re-run on this branch's harness so the
comparison is same-code, same-windows) · the live scorecard values for May/Jun-2026.

### 6.4 Comparison dimensions

- Full-sample RMSE 2015–2024
- **Shock** window 2022–23
- **Calm** windows: ex-shock, and pre-2020
- **May-2026 genesis** observation (AA 2.71 / production 2.91 / actual 2.80) and Jun-2026
  (AA 2.726 / production 2.683 / actual 2.6)
- Forecast evolution `T-30 → T` (RMSE by origin day)
- Diebold-Mariano vs AA and vs production
- **Calm-month overshoot**: mean `|forecast − AA|` in calm months — production's documented failure mode
- Error correlation vs TVP/LGBM — the redundancy test

---

## 7. Model-selection rule

Applied exactly as specified:

1. Keep a component **only** if it improves out-of-sample performance **without** obvious calm-month
   overshoot. Overshoot is **measured, not eyeballed**, with the threshold fixed *before* results are
   seen: a component is rejected for overshoot if it raises mean `|forecast − AA|` in calm months
   (ex-2022/23 and pre-2020) **more than 10% above the production replication's** value on the same
   window. Both numbers are reported regardless of the verdict.
2. If a component helps **only in shock periods**, keep it but **document that explicitly** — the repo
   already has precedent (`docs/final_model.md` §3: production's own edge is shock-concentrated).
3. If a component is **redundant with production**, say so and **do not keep it**. Redundancy test is
   the repo's existing one: error correlation with the TVP+LGBM ensemble. BVAR was dropped at 0.91
   corr with LGBM; **that 0.9 threshold is the precedent this branch will apply.**

A component that improves in-sample fit but not OOS RMSE is **dropped**, regardless of how good the
economics look.

---

## 8. Deliverables

| Deliverable | Path |
|---|---|
| DFM package | `code/dfm/{__init__,panel,state_space,sampler,model,backtest,experiments,benchmarks,diagram,report}.py` |
| Tests | `code/tests/test_dfm_*.py` |
| **Factor loading diagram** | `plots/dfm/loading_diagram.png` |
| Experiment figures | `plots/dfm/{arm_race,ablation,evolution,outlier_prob,factor_paths,sv_paths}.png` |
| **Model card** (what each factor does economically) | `docs/dfm_model_card.md` |
| Backtest write-up | `docs/dfm_backtest.md` |
| Result tables | `data/dfm/*.csv` |

Figures follow the repo `plots/` convention and are in **percentage (YoY) terms, never index levels**
(established in the core-CPI spec).

### Module boundaries

| Module | Responsibility | Depends on |
|---|---|---|
| `panel.py` | series specs, block map, as-of-day panel, standardization, `cpi_index` | `factors` (read-only) |
| `state_space.py` | Kalman filter/smoother + simulation smoother (numba) | numpy/numba |
| `sampler.py` | Gibbs blocks: loadings, VAR, SV, outliers | `state_space` |
| `model.py` | `DFMConfig` + `DFMNowcaster.fit/nowcast/forecast_path` | `panel`, `sampler` |
| `backtest.py` | walk-forward harness, purge/embargo, metric windows | `model`, `validation` |
| `benchmarks.py` | AA / AR(1) / production-replication wrappers | `uk_model_zoo` (read-only) |
| `experiments.py` | E1–E5 + ablation runner, checkpointing | `backtest`, `benchmarks` |
| `diagram.py`, `report.py` | loading diagram; markdown tables | results CSVs |

`state_space.py` and `sampler.py` are pure numeric and testable without any data fetch.

---

## 9. Testing

- **`state_space`**: simulation smoother recovers known factors from synthetic data with known `Λ, Φ, Σ`;
  filter log-likelihood matches a statsmodels `KalmanFilter` on a model both can express.
- **`sampler`**: each Gibbs block recovers its parameters on synthetic data generated from the same DGP
  (loadings, `Φ`, SV path, outlier indicators — the last checked by outlier-detection AUC on planted outliers).
- **`panel`**: as-of-day panel at `d = T` equals the month-end `apply_publication_lags` matrix; at `d < T`
  strictly fewer non-missing cells; **no cell ever populated from data dated after `d`**.
- **YoY reconstruction**: identity reproduces official D7G7 within the documented 0.07 pp rounding floor.
- **Leakage**: audit reports 0 violations; May/Jun-2026 absent from every estimation sample.
- **Production isolation**: `test_headline_regression.py` passes; production-path `git diff` vs `main` empty.

---

## 10. Honest prior

The bar is 0.416 RMSE. Prior DFMs here scored 1.97. The MoM reformulation, Bayesian shrinkage, and a
6-block panel are substantive changes, but:

- Production's own edge over AA is **statistically insignificant** (DM p≈0.17) and shock-concentrated.
  A DFM that ties production is therefore a **legitimate** outcome, and distinguishing a tie from a win
  at `n=120` may not be possible. The write-up will report DM p-values, not just point RMSE.
- The most likely *good* outcome is that the DFM wins as an **overlay member** — a genuine diversifier
  against TVP/LGBM — rather than as a standalone forecaster. The standalone arm must beat AA's
  base-effect arithmetic outright, which is a high bar even with the MoM reformulation.
- **A null result will be reported as a null result**, in the same register the repo already uses for
  the falsified regime-detection work (`docs/final_model.md` §10).
