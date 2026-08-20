# Design — mixed-frequency Bayesian DFM research model

**Status:** approved design, pre-implementation. **Branch:** `dfm-research` (off `main`). **Date:** 2026-08-19.

Supersedes [2026-08-06-dfm-nowcaster-design.md](2026-08-06-dfm-nowcaster-design.md), which was
approved but never implemented. Differences from that spec are listed in §12.

## Goal

Build a **mixed-frequency Bayesian dynamic factor model** for UK CPI as a research alternative to the
frozen production model, and answer one question:

> Which structural features of a Bayesian DFM — stochastic volatility, robust innovations, factor-lag
> interactions, structured loadings, hierarchical shrinkage — add **economically meaningful
> information** beyond the frozen `AA + 0.25·TVP + 0.25·LGBM + RegEvent` benchmark?

The objective is **not** to minimise RMSE by tuning. It is to attribute performance to structure. A
null result is a valid, publishable outcome and is reported in the same register as the repo's
falsified regime-detection work (`docs/final_model.md` §10).

Each layer is built and validated **sequentially**, so the contribution of SV, robust shocks, factor
interactions, structured loadings and hierarchical shrinkage is **separately identified**. The full
model is never the starting point.

### Non-goal — production is frozen

No edits to `code/production/`, `code/factors.py`, `code/targets.py`, `code/new_factors/two_stage.py`,
or `code/uk_model_zoo.py`. This branch adds `code/dfm/` plus its own `data/dfm/`, `plots/dfm/`,
`docs/dfm_*.md`. Nothing ships to production without a separate governance decision.

**Proof of non-interference (hard gate, enforced as a test).** `code/tests/test_dfm_isolation.py`
asserts that

```
git diff main --stat -- code/production code/factors.py code/targets.py \
    code/new_factors/two_stage.py code/uk_model_zoo.py
```

is empty, and `code/tests/test_headline_regression.py` must pass unchanged on this branch. The
research model imports production code read-only, for benchmarking.

## The bar

Walk-forward 2015–2024, purged + embargoed, scored on the windows production uses.

| Benchmark | RMSE | Source |
|---|---|---|
| AR(1) | 0.495 | `README.md` model table |
| AutoARIMA (anchor) | 0.467 | `README.md` |
| TVP overlay member | 0.482 | `README.md` |
| LGBM overlay member | 0.443 | `README.md` |
| **Frozen production** (AA + 0.25·TVP + 0.25·LGBM) | **≈0.416** (rel-RMSE **0.89** vs AA) | `docs/final_model.md` §3 |
| *Prior DFM attempts in this repo* | *1.97 (DFM), 1.96 (DFM-k2), 2.62 (MS-DFM)* | `data/rmc_dfm_perf.csv` |

Prior DFMs lost to AA by a factor of four. The cause is diagnosed and addressed in §3.2: they modelled
CPI **YoY**, whose lag-1 autocorrelation is 0.983, forcing the factors to re-learn 11/12 of the
variance as mechanical carry-over. This design models **MoM** (lag-1 autocorr **0.048**) and makes the
carry-over arithmetic. That reformulation is what makes this a fair fight rather than a rerun.

## Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Publication calendar | **Dated release calendar, revised values** | Real release *timing* per series; today's (revised) values. True vintages rejected: UK vintage coverage is patchy, so parts of the panel would silently fall back to revised data anyway — a half-vintage panel is worse than an honestly-labelled revised one |
| Stage ladder | **6 arms** (E1–E6) | Splits structured loadings (E5) from hierarchical shrinkage (E6) so Stage 4 and Stage 5 are separately identified |
| Modes | **Both** — standalone and overlay | The overlay mode answers the incremental-information question directly; prior evidence says a DFM is likelier to win as a diversifier than as a standalone level forecaster |
| Compute ceiling | **~8h overnight**, checkpointed and resumable | Full draws with documented ordered fallbacks (§5.3) |
| CPI space | **Model MoM, aggregate to YoY** | §3.2 — the single most important choice in this design |
| Latent state | **Monthly**; daily/weekly enter as partial-month observations | Keeps the state small; the Kalman filter handles the ragged edge natively |
| Refit cadence | **Monthly** parameter refit, warm-started | Matches the TVP/LGBM backtest convention; annual refit would not be apples-to-apples |
| Panel width | **32 series**, 6 blocks, availability-verified, no LSEG dependency | Each factor identified by 3–8 series; a multi-hour sweep must not depend on a desktop app being open |
| Reg/event block | **Observed exogenous** shifter, not latent | 3 sparse, mostly-zero administered-event series cannot identify a latent factor |
| Estimation | Hand-written **Gibbs sampler** (numpy + numba) | PyMC cannot handle a sampled ~350-dimensional latent state with SV; hand-rolled Gibbs makes every ablation exact rather than approximate |
| Eval windows | **Match production exactly** | Headline claims directly comparable to 0.89 rel-RMSE with no re-derivation |
| 2025+ and May–Jul 2026 | **Held out from all estimation** | Extends the existing blind hold-out to the three live scored months |

Environment: `quant` conda only (`/opt/anaconda3/envs/quant/bin/python`). Verified 2026-08-19: numba
0.64.0, numpy 2.4.2, pandas 2.3.3, statsmodels 0.14.6, scipy 1.17.0, arviz 0.23.4, pymc 5.28.2. No
JAX / numpyro / Stan. Never run parallel Python processes here (`docs/CLAUDE.md`) — they serialise on
the macOS security scanner.

---

## 1. Model specification

### 1.1 Measurement equation

`N = 32` standardized series in 6 economic blocks, plus CPI entering as MoM log-change (33 measurement
equations):

```
x_it = λ_i' f_t  +  γ_i' z_t  +  e_it            i = 1..N
π_t  = λ_π' f_t  +  γ_π' z_t  +  e_πt            π_t = 100 · Δ log(D7BT_t)
```

- `f_t` — **`K = 7`** latent factors: one pervasive `global` factor plus six block factors
  (`activity`, `nominal`, `energy`, `supply`, `fx`, `food`). The global factor is a distinct object
  from the activity block: every series loads on `global`; only its own members load on a block factor.
- `z_t` — observed exogenous reg/event block: `mpc_rate_change`, `ofgem_cap_delta`, `budget_event`.
- `e_it` — idiosyncratic AR(1), carried **in the state vector** (not quasi-differenced, because
  idiosyncratic SV makes quasi-differencing ill-defined).

`K = 7` is held fixed across all six arms so E5 vs E4 isolates the loading restriction alone and does
not confound it with a change in factor count. A `K ∈ {5, 6, 7, 8}` sensitivity check runs on the best
arm and is reported separately.

### 1.2 State equation — factor VAR with SV

```
f_t  = Φ₁ f_{t-1} + Φ₂ f_{t-2} + η_t        η_t ~ N(0, diag(exp h_t))
h_kt = h_{k,t-1} + σ_k ζ_kt                 ζ ~ N(0,1)      factor SV (log-vol random walk)
g_it = g_{i,t-1} + τ_i ξ_it                                 idiosyncratic SV
```

**Factor-lag interactions are the off-diagonal elements of `Φ`.** E1–E3 restrict `Φ` diagonal (each
factor an independent AR process); E4 frees the off-diagonals under a Minnesota prior (own-lag loose,
cross-lag tight). "Do VAR-style lead-lag spillovers between factors help?" therefore becomes a
**shrinkage decision with a prior**, not an unconstrained parameter grab — which is what keeps it
honest at `T ≈ 290`.

### 1.3 Robust innovations (Stock-Watson outlier mixture)

```
e_it = exp(g_it / 2) · o_it · ε_it
o_it = 1              with prob (1 − p)
o_it ~ Uniform(2, 10) with prob p                p ~ Beta(a, b), small
```

Chosen over a Student-t scale mixture because it is **directly interpretable**: the posterior yields
`P(o_it > 1)`, a per-series, per-month outlier probability. Expected to light up on Ofgem cap resets,
the 2020 VAT cut and COVID collection gaps. That series is itself a deliverable
(`plots/dfm/outlier_prob.png`).

### 1.4 Loading structure

- **E1–E4 (conventional):** unrestricted `Λ` — every series loads on all 7 factors — identified by the
  standard normalization (upper `K×K` block lower-triangular with positive diagonal). Factors here are
  statistical, not nameable.
- **E5–E6 (structured):** **block-restricted** — every series loads on `global` plus its own block
  factor only; all other loadings pinned to zero. The restriction is what makes each factor
  economically nameable. Within the free entries, E5 additionally applies a spike-and-slab sparsity
  prior so a series can drop off its own block factor if the data say so; the posterior inclusion
  probabilities are reported in the loading diagram.

### 1.5 Hierarchical shrinkage (E6 only)

E1–E5 use fixed, weakly-informative priors with hand-set scales. E6 puts hyperpriors on those scales
so they are learned and pooled:

```
λ_ij | ψ_j  ~ N(0, ψ_j)          ψ_j  ~ InvGamma(a_λ, b_λ)      per-factor loading scale
Φ    | δ    ~ Minnesota(δ)        δ    ~ Gamma(a_δ, b_δ)          learned overall tightness
σ²_i | ν, s ~ InvGamma(ν/2, νs/2) ν, s ~ hyperprior               pooled idio variances
```

This is the layer that decides *how much* to shrink rather than being told. Its contribution is
measured as E6 − E5 on identical everything else.

### 1.6 Blocks

All 32 series load on the pervasive `global` factor; the block column is the *second* factor each
series loads on under E5–E6.

| Block | n | Series | Economic role |
|---|---|---|---|
| `activity` | 7 | `uk_quarterly_gdp`, `uk_retail_sales`, `uk_vacancies`, `uk_awg`, `uk_house_prices`, `vix`, `move_index` | real activity, labour, demand; `vix`/`move_index` as risk-appetite proxies |
| `nominal` | 8 | `uk_ppi_output`, `uk_ppi_input`, `uk_core_cpi`, `uk_services_cpi`, `uk_rents`, `uk_be5`, `uk_gilt_10y`, `slope_2s10s` | price level, inflation expectations, rates |
| `energy` | 5 | `oil_brent`, `gas_eu`, `gas_hh`, `imf_all_commodity`, `oil_vol_6m` | cost-push |
| `supply` | 6 | `deep_sea_freight`, `semiconductors_ppi`, `chemicals_ppi`, `metals_index`, `copper_price`, `timber_price` | freight / industrial materials / bottleneck |
| `fx` | 3 | `gbp_eer`, `gbpusd`, `gbp_eur` | import pass-through |
| `food` | 3 | `food_price_index`, `wheat_price`, `vegetable_oil_price` | **new channel** |
| **total** | **32** | | |
| `reg` *(exogenous `z_t`)* | 3 | `mpc_rate_change`, `ofgem_cap_delta`, `budget_event` | administered / regulatory shocks |

Block sizes 3–8, so every factor is identified by at least three series.

**The `food` block is the only genuinely new information channel versus production.**
`docs/final_model.md` §4 attributes the ~74% unexplained residual to "food/services/idiosyncratic",
and the documented calm-month failure mode is precisely food/services-driven prints. If any block
earns its place, this is the prior favourite — and if it does not, that is a substantive finding about
the residual.

`mpc_rate_change` and `ofgem_cap_delta` (both 2015+) are zero-filled before their start, following the
existing production convention, and only inside their observed span (the 2026-08-19 freshness-guard
rule — a missing month is not "no cap reset").

### 1.7 Panel availability

The 2026-08-06 availability probe removed seven initially-proposed series; that exclusion list stands
and is re-verified by `code/dfm/panel.py` at build time:

| Excluded | Reason |
|---|---|
| `uk_pmi_composite`, `uk_pmi_services`, `uk_infl_swap_1y` | LSEG — need the Workspace desktop app running (`localhost:9000`) |
| `uk_pmi_manufacturing`, `uk_paye` | CSV-only (licensed) and no CSV present in `data/` |
| `freightos_fbx` | no free source; registry note says use `deep_sea_freight` as the proxy |
| `global_supply_chain_pressure` | known expected-drop (xlsx engine error), already `OPTIONAL_PINNED` in production |

Also excluded by construction: `uk_monthly_gdp` (dies 2024-03), `uk_cpih` / `uk_trimmed_mean_cpi`
(circular with the target or unavailable), `battery_metals_proxy`, `uk_hciall`, `vimes_boots_index`,
`uk_ashe_pay`, `ons_econ_activity` (unavailable), `slope_5s30s` (starts 2016 — too short).

**No LSEG dependency in the headline panel.** A result that cannot be reproduced without a desktop
application running is not a result. The three LSEG series are an optional enrichment tested
separately (an E6+LSEG variant, run only if Workspace is up), never part of the headline panel.

**Staleness guard.** `uk_house_prices` last prints 2026-01 and `uk_awg` 2026-05. Ragged edges are
handled natively by the filter, but the panel builder logs any series whose last observation is
>6 months stale at run time, so a silently-dead series is visible rather than quietly contributing
nothing.

---

## 2. Information sets — publication calendar and the as-of panel

This is the layer that differs most from the 2026-08-06 spec, which used month-tier `pub_lag` shifts.

### 2.1 Dated release calendar

`code/dfm/release_calendar.py` builds `data/dfm/release_calendar.csv`, one row per
`(series, ref_period)`:

| column | meaning |
|---|---|
| `series` | registry name |
| `ref_period` | reference month (or quarter) the value describes |
| `release_ts` | the timestamp at which the value first became public |
| `source` | `ons_api` \| `provider_doc` \| `rule` |

Sources, in precedence order:

1. **`ons_api`** — actual ONS release dates from the ONS releases API for CPI (D7BT/D7G7), PPI, GDP,
   retail sales, vacancies, AWE. Fetched once and committed as a CSV so the sweep is reproducible
   offline.
2. **`provider_doc`** — documented publication offsets for non-ONS series: IMF primary commodity
   prices (+7 days), US BLS PPI series used for `deep_sea_freight` / `semiconductors_ppi` /
   `chemicals_ppi` (+13 days), BoE yield-curve files (+1 business day), FRED/market daily series
   (same day). These reuse and extend the offsets already encoded in
   `code/timing/nowcast_window.py`.
3. **`rule`** — fallback only: ONS CPI publishes on the 3rd Wednesday of the following month
   (`code/rates/event_panel.py:37`). **The existing `data/timing/release_calendar.csv` is
   rule-generated, not actual dates**, and is not reused as-is.

**Fallback accounting is a first-class output.** The model card reports the share of
`(series, ref_period)` cells filled by `rule` rather than a real date, overall and for the 2015–2024
headline window. If the ONS API does not expose pre-2015 release dates, pre-2015 falls back to the
rule and the headline window stays exact; that split is stated rather than glossed.

### 2.2 As-of panel

`panel.asof(ref_month, d)` returns the observation matrix visible at timestamp `d`:

- A month-`T` cell for series *i* is populated **only if** `release_ts(i, T) ≤ d`. Otherwise it is
  missing, and the filter skips that update step. No interpolation, no EM fill.
- **Daily / weekly series** aggregate over observations dated `≤ d` within the month — levels take the
  last value, returns take the partial-month sum — and enter the monthly state as **partial-month
  observations**. There is no daily state vector: high-frequency data informs the monthly state
  through a partially-observed monthly aggregate, which is the mixed-frequency mechanism.
- **Quarterly series** (`uk_quarterly_gdp`) enter through the Mariano–Murasawa aggregator
  `y_q = ⅓(f_t + f_{t-1} + f_{t-2})`, with the two intra-quarter months observed as missing.

Missingness is handled by the **sequential (univariate) Kalman treatment**: a missing observation is a
skipped update, which is `O(N·k²)` and missing-data-native.

This reuses the causal machinery already proven in `code/intramonth/hf_data.py` (as-of daily
truncation, leakage-tested) rather than reinventing it.

### 2.3 Forecast path T-30 → R-1

Origins: `d ∈ {T-30, T-21, T-14, T-7, T, T+5, T+10, R-1}`, where `R` is the actual release date from
§2.1 and `R-1` is one day before it. The path therefore runs from a month before month-end through to
the eve of publication — past month-end, unlike the 2026-08-06 spec, which stopped at `T`.

**Cost.** MCMC parameters are drawn **once per month**; each within-month origin re-runs only the
**Kalman filter** on the as-of panel, holding the posterior draws fixed. Filtering is ~3 orders of
magnitude cheaper than sampling, so the path costs ≈8 extra filter passes per month, not 8 extra fits.
This is also the correct conditioning: parameters estimated on data through `T-1` are valid at every
origin inside month `T` and inside the release window.

**Production has no intramonth product**, so it appears on evolution charts as a flat line at its
month-end value, clearly labelled. **The headline comparison against production is at `d = T` only**,
because that is production's information standpoint. Origins after `T` are reported as a descriptive
curve — how the DFM's view firms up as data arrives — and are **never** used to claim a win over
production, which would be comparing against a model that has not run yet.

The forecast path is produced for the **selected best arm only**, not all 12 configs.

### 2.4 Leakage tests

- Injecting fabricated observations dated after `d` must not change `panel.asof(·, d)` by a single
  cell.
- `panel.asof(T, month_end(T))` must equal the month-end `apply_publication_lags` matrix for the
  series both can express.
- At `d < month_end(T)`, strictly fewer non-missing cells than at `d = month_end(T)`.
- Leakage audit over the DFM panel mirroring `code/new_factors/leakage_audit.py`; target 0 violations.
- 2025+ and May–Jul 2026 absent from every estimation sample.

---

## 3. The target

### 3.1 Data

CPI index **D7BT** (ONS MM23), verified available via dbnomics: `n=949`, 1947-06 → 2026-06. The
registry carries only YoY (`D7G7`), so `code/dfm/panel.py` defines `cpi_index` locally, reusing
`factors._dbnomics` — **zero diff to `factors.py`**.

### 3.2 MoM → YoY reconstruction

The base-effect carry is fully known at `T` (every term published):

```
c_T   = 100 · (I_{T-1} / I_{T-12} − 1)
YoY_T = 100 · ((1 + c_T/100) · exp(π_T / 100) − 1)
```

The DFM forecasts **`π_T` only**. All model risk concentrates on the single unknown monthly increment
— where the factors actually have signal — instead of being spread over 12 months of arithmetic.

Measured justification (computed 2026-08-06 on live ONS data):

| Quantity | Value |
|---|---|
| CPI YoY lag-1 autocorrelation (2000+) | **0.983** |
| CPI MoM lag-1 autocorrelation (2000+) | **0.048** |
| MoM std, 2015–2024 | 0.456 pp |
| MoM std, 2022–23 shock | **0.688 pp** |
| MoM std, ex-shock | **0.336 pp** |

The 2× volatility ratio between shock and calm months is direct evidence that stochastic volatility
has something real to model — it is not decoration.

### 3.3 Known accuracy floor

ONS publishes D7BT to 1 d.p. Reconstructing YoY purely from the index differs from official D7G7 by up
to **0.135 pp** (2000+). Mitigation: anchor on the **official** published index/YoY and add only the
modelled increment, confining rounding to a single index ratio (≈0.07 pp). That is ~6× below AA's
0.467 RMSE so it does not threaten the comparison, but it **is a floor** and is reported as such in
the model card. `D7G7` is itself 1 d.p., so the floor binds every model equally.

---

## 4. The stage ladder

### 4.1 Arms

Each stage changes **exactly one** thing relative to the stage below it.

| # | Arm | `Φ` | SV | Innovations | Loadings | Priors |
|---|---|---|---|---|---|---|
| **E1** | classical Stock-Watson DFM | diagonal | — | Gaussian | unrestricted | fixed, diffuse |
| **E2** | + stochastic volatility | diagonal | factor + idio | Gaussian | unrestricted | fixed |
| **E3** | + robust / heavy-tailed | diagonal | factor + idio | SW outlier mixture | unrestricted | fixed |
| **E4** | + sparse factor VAR | **full, Minnesota** | ✓ | ✓ | unrestricted | fixed |
| **E5** | + structured loadings | full | ✓ | ✓ | **block-restricted + spike-slab** | fixed |
| **E6** | full Bayesian hierarchical | full | ✓ | ✓ | block-restricted | **hierarchical (§1.5)** |

E1 doubles as eval benchmark #3 (classical Stock-Watson DFM). A factor-only-SV sub-variant is tested
inside E2 and reported, not promoted to its own arm.

**Stage mapping to the request:** Stage 1 = E1 · Stage 2 = E2 + E3 · Stage 3 = E4 · Stage 4 = E5 ·
Stage 5 = E6.

### 4.2 Modes

Every arm runs in **both** modes:

- **standalone** — the DFM forecasts `π_T` directly; YoY rebuilt per §3.2.
- **overlay** — the DFM's factors predict the residual left by **AA + RegEvent**, i.e.
  `actual − (AA + RegEvent)`, so the arm can be scored as a candidate overlay member against TVP
  and LGBM.

  **The RegEvent term must be in the target, not just the composition** (corrected 2026-08-20).
  Production is `AA + λ·Overlay + RegEvent`. Training a candidate overlay on the AA-only residual
  asks it to explain the Ofgem cap step that RegEvent already prices arithmetically — the overlay
  would earn credit for predicting reset months, and RegEvent would then add the same step again.
  That measures redundancy as skill and double-counts it in the composed forecast. The correct
  target for a *new* member is what production does not already explain.

  Note this differs from frozen production's own internal convention: `production.model.
  _residual_history` returns the AA-only residual and adds `reg_events.reg_event_step` separately,
  so production's TVP and LGBM legs do carry that double-count. Production is frozen and not
  changed here; the point is that a new member must not inherit the flaw. `reg_event_step(nd)` is
  causal by construction — `passthrough(before=nd, …)` estimates β from resets strictly before
  `nd` — so it is walk-forwardable month by month.

  RegEvent is headline-only and nonzero in 23 of 306 months, so for most months the two residual
  definitions coincide; the difference concentrates in exactly the cap-reset months where the
  overlay's apparent skill would otherwise be spurious.

6 arms × 2 modes = **12 configs**.

### 4.3 Sequential validation gate

Arms are built and validated **in order**. E*n+1* is not implemented until E*n* passes:

1. its own synthetic-recovery tests (§9),
2. a completed walk-forward producing finite predictions for ≥95% of test months,
3. a written one-paragraph result in `docs/dfm_backtest.md` with its RMSE against AA.

This is what makes the contribution of each layer separately identifiable, and it means an early
failure surfaces before the expensive layers are built.

### 4.4 Ablations

Leave-one-block-out on the best-performing arm: 6 latent blocks + the exogenous reg block =
**7 ablations**, each reporting ΔRMSE by window.

---

## 5. Estimation and runtime

### 5.1 Gibbs blocks

Single process, numba-compiled:

1. **Factors `f_{1:T}`** — Durbin-Koopman simulation smoother, sequential/univariate treatment,
   missing-data-native.
2. **Loadings `Λ`, `γ`, idio AR coefficients** — series-by-series Bayesian regression; block
   restrictions imposed by zeroing (E5–E6); spike-and-slab inclusion drawn per free entry (E5–E6).
3. **`Φ`** — VAR posterior under a Minnesota prior (diagonal-restricted in E1–E3).
4. **SV states `h`, `g`** — Kim-Shephard-Chib 7-component mixture approximation + FFBS.
5. **Outlier indicators `o`** — conjugate discrete/uniform mixture draw; `p` from its Beta posterior.
6. **Hyperparameters** — `σ_k`, `τ_i`, shrinkage scales; in E6 also the hyperpriors of §1.5.

### 5.2 Draws

2000 (1000 burn-in) for the first walk-forward fit; **800 (200 burn-in) thereafter**, warm-started
from the previous month's posterior mean. Warm-starting uses **no future information** — it is
initialization only, and the previous month's posterior conditions on a strict subset of the current
month's data. Convergence checked by `arviz` R-hat / ESS on a sampled subset of fits, not every fit.

### 5.3 Budget and fallbacks

12 configs × ~120 test months ≈ **1440 fits**, plus 7 ablations on the best arm. The runner
**checkpoints to disk per (config, month)** and is resumable, so a multi-hour sweep survives
interruption. One background process only.

**A timing probe runs on the real panel before the sweep.** If it projects over **8 hours**, the
fallbacks apply in this fixed order, and whichever was used is recorded in the model card because it
changes what the numbers mean:

1. drop ablations to the best arm only,
2. halve post-warm-start draws to 400,
3. quarterly parameter refit with monthly filtering.

---

## 6. Evaluation

### 6.1 Models compared

1. AutoARIMA
2. Frozen production replication (`AA + 0.25·TVP + 0.25·LGBM + RegEvent`, re-run on this branch's
   harness so the comparison is same-code, same-windows). **Two variants, and they are not
   interchangeable:** RegEvent-*excluded* reproduces the documented 2015–2024 figures (README's
   AA 0.467 / rel-RMSE 0.89 predate the 2026-08-19 RegEvent layer) and is the headline baseline;
   RegEvent-*included* is the only fair opponent for an overlay arm, since an overlay trained on
   the `AA + RegEvent` residual must be scored against a production that also prices the step.
   Reporting an overlay arm against the RegEvent-excluded baseline would flatter it.
3. Classical Stock-Watson DFM (= E1)
4. DFM + SV (= E2)
5. DFM + SV + robust outliers (= E3)
6. DFM + SV + outliers + factor VAR (= E4)
7. Full structured Bayesian DFM (= E6; E5 reported as the intermediate)

Plus AR(1) as the floor benchmark and the live scorecard values for May–Jul 2026.

### 6.2 Outputs per model

| Output | Definition |
|---|---|
| **RMSE** | full-sample 2015–2024, purged walk-forward |
| **MAE** | same windows — reported alongside RMSE because the shock months dominate a squared loss |
| **Forecast path** | T-30 → R-1 at the 8 origins of §2.3; RMSE by origin. Produced for the **best arm plus the benchmarks** (AA, production replication), not for all 12 configs — see §2.3 |
| **Shock / calm subsamples** | shock = 2022–23; calm = ex-shock and pre-2020, reported separately |
| **Factor / news decomposition** | Bańbura-Modugno news weights: which release moved the forecast and by how much, decomposed to block level |
| **Parameter stability** | posterior mean of `Λ`, `Φ`, idio `σ` across the walk-forward; rolling-window loading dispersion; flag any loading whose sign flips |
| **Computational cost** | wall-clock per fit, per config total, and seconds-per-fit on the timing probe |

### 6.3 Additional comparison dimensions

- Diebold-Mariano vs AA and vs production replication
- **Calm-month overshoot**: mean `|forecast − AA|` in calm months — production's documented failure
  mode
- Error correlation vs TVP/LGBM — the redundancy test
- May-2026 (AA 2.71 / production 2.91 / actual 2.80), Jun-2026 (2.726 / 2.683 / 2.60), Jul-2026
  (2.52 / 2.56 / 2.90) as blind scored points

### 6.4 The purge-horizon asymmetry — stated, not buried

| Mode | Label | Purge + embargo |
|---|---|---|
| Overlay (factors → AA + RegEvent residual) | `cpi_yoy` residual — a genuine 12-month difference | **12 + 1** (identical to production) |
| Standalone (factors → `π_T`) | MoM — a **1-month** difference | **1 + 1** |

Forcing 12+1 on a 1-month label would discard real information rather than prevent leakage — the
stated rationale in `code/validation.py`. The consequence is that the standalone arm trains on more
recent data than production does. **Any standalone win is reported with this caveat attached**, and
the standalone arm is additionally re-run under 12+1 as a sensitivity check, so a reader can separate
the reformulation from the horizon.

---

## 7. Model-selection rule

Thresholds fixed **before** results are seen:

1. Keep a component **only** if it improves out-of-sample performance **without** calm-month
   overshoot. Overshoot is measured, not eyeballed: a component is rejected if it raises mean
   `|forecast − AA|` in calm months (ex-2022/23 and pre-2020) **more than 10% above the production
   replication's** value on the same window. Both numbers are reported regardless of the verdict.
2. If a component helps **only in shock periods**, keep it but **document that explicitly** — the repo
   already has precedent (`docs/final_model.md` §3: production's own edge is shock-concentrated).
3. If a component is **redundant with production**, say so and **do not keep it**. Redundancy is error
   correlation with the TVP+LGBM ensemble; BVAR was dropped at 0.91 corr with LGBM, and **that 0.9
   threshold is the precedent this branch applies**.
4. A component that improves in-sample fit but not OOS RMSE is **dropped**, regardless of how good the
   economics look.

---

## 8. Deliverables and module boundaries

| Deliverable | Path |
|---|---|
| DFM package | `code/dfm/` |
| Tests | `code/tests/test_dfm_*.py` |
| Release calendar | `data/dfm/release_calendar.csv` |
| Factor loading diagram | `plots/dfm/loading_diagram.png` |
| Experiment figures | `plots/dfm/{arm_race,ablation,evolution,outlier_prob,factor_paths,sv_paths,stability}.png` |
| Model card | `docs/dfm_model_card.md` |
| Backtest write-up | `docs/dfm_backtest.md` |
| Result tables | `data/dfm/*.csv` |

Figures follow the repo `plots/` convention and are in **percentage (YoY) terms, never index levels**.

| Module | Responsibility | Depends on |
|---|---|---|
| `panel.py` | series specs, block map, as-of panel, standardization, `cpi_index` | `factors` (read-only), `release_calendar` |
| `release_calendar.py` | dated release table, fallback accounting | ONS API, `code/timing` offsets (read-only) |
| `state_space.py` | Kalman filter/smoother + simulation smoother (numba) | numpy/numba |
| `sampler.py` | Gibbs blocks: loadings, VAR, SV, outliers, hyperpriors | `state_space` |
| `model.py` | `DFMConfig` + `DFMNowcaster.fit/nowcast/forecast_path` | `panel`, `sampler` |
| `backtest.py` | walk-forward harness, purge/embargo, metric windows | `model`, `validation` |
| `benchmarks.py` | AA / AR(1) / production-replication wrappers | `uk_model_zoo`, `code/production` (read-only) |
| `experiments.py` | E1–E6 + ablation runner, checkpointing, resumability | `backtest`, `benchmarks` |
| `news.py` | Bańbura-Modugno news decomposition | `state_space`, `model` |
| `stability.py` | parameter-path extraction and dispersion metrics | `experiments` outputs |
| `report.py`, `diagram.py` | markdown tables; loading diagram | results CSVs |

`state_space.py` and `sampler.py` are pure numeric and testable without any data fetch.

---

## 9. Testing

- **`state_space`**: simulation smoother recovers known factors from synthetic data with known
  `Λ, Φ, Σ`; filter log-likelihood matches a statsmodels `KalmanFilter` on a model both can express.
- **`sampler`**: each Gibbs block recovers its parameters on synthetic data from the same DGP —
  loadings, `Φ`, SV path, outlier indicators (the last scored by detection AUC on planted outliers),
  and the E6 hyperpriors by recovering a known shrinkage scale.
- **`release_calendar`**: every `(series, ref_period)` has a `release_ts` and a `source`; no
  `release_ts` precedes the end of its `ref_period`; fallback share is computed and asserted to be
  reported.
- **`panel`**: the four leakage tests of §2.4.
- **YoY reconstruction**: the identity reproduces official D7G7 within the documented 0.07 pp floor.
- **Production isolation**: `test_dfm_isolation.py` (§Non-goal) and `test_headline_regression.py`.

---

## 10. Honest prior

The bar is 0.416 RMSE. Prior DFMs here scored 1.97. The MoM reformulation, hierarchical shrinkage and
a 6-block panel are substantive changes, but:

- Production's own edge over AA is **statistically insignificant** (DM p≈0.17) and shock-concentrated.
  A DFM that ties production is a **legitimate** outcome, and distinguishing a tie from a win at
  `n=120` may not be possible. The write-up reports DM p-values, not just point RMSE.
- The most likely *good* outcome is that E4–E6 win as an **overlay member** — a genuine diversifier
  against TVP/LGBM — rather than as a standalone forecaster. The standalone arm must beat AA's
  base-effect arithmetic outright, which is a high bar even with the MoM reformulation.
- **A null result will be reported as a null result.**

---

## 11. Out of scope

- Any change to the frozen production model, or any promotion of a DFM arm into it. That is a separate
  governance decision with its own written entry.
- True real-time vintages (revision-aware values). Release *timing* only — see the locked decision.
- A daily or weekly latent state. High-frequency data enters as partial-month observations.
- Core CPI and services CPI as targets. Headline only in this phase.
- LSEG-dependent series in the headline panel.

## 12. Differences from the 2026-08-06 spec

| Area | 2026-08-06 | This spec |
|---|---|---|
| Arms | 5 (E1–E5), shrinkage baked in | **6** (E1–E6); hierarchical shrinkage separately identified as E6 |
| Loadings | block restriction only | block restriction **+ spike-and-slab sparsity** within free entries |
| Information set | month-tier `pub_lag` + as-of-day aggregation | **dated release calendar** per (series, ref_period) with source provenance and fallback accounting |
| Forecast path | T-28 → T (5 weekly origins) | **T-30 → R-1** (8 origins, into the release window) |
| Deliverables | RMSE, DM, evolution, ablation | adds **MAE**, **parameter stability**, **computational cost**, **Bańbura-Modugno news decomposition** |
| Sequencing | 10 configs run as a sweep | **sequential validation gate** (§4.3) — E*n+1* not built until E*n* passes |
| Branch | `dfm` (off `core`) | `dfm-research` (off `main`) |
