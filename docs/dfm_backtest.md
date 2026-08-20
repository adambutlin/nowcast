# DFM research backtest — results log

Branch `dfm-research`. Spec:
[docs/superpowers/specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md](superpowers/specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md).
Plan: [2026-08-19-dfm-plan1-foundation-stage1.md](superpowers/plans/2026-08-19-dfm-plan1-foundation-stage1.md).

Each stage appends one section. A stage is not complete until its section is written.
**Production is untouched** — `git diff main -- code/production code/factors.py code/targets.py
code/new_factors/two_stage.py code/uk_model_zoo.py` is empty, enforced by
`code/tests/test_dfm_isolation.py`.

## Status and verdict so far

**Stage 1 complete (E1, classical Stock-Watson DFM). Stages 2–5 not built.**

| Question | Answer |
|---|---|
| Did the MoM reformulation fix the prior DFM failure? | **Yes.** Prior attempts scored 1.97 / 1.96 / 2.62 against AA's 0.467. E1 scores **0.4462** — a 4.4× improvement in the same model family. |
| Does E1 beat AutoARIMA standalone? | **Yes, but not significantly.** 0.4462 vs 0.4687 (rel 0.952), DM p = 0.703. |
| Does E1 beat the frozen production model? | **No.** 0.4462 vs 0.4142. |
| Does E1 pass the pre-registered selection rule? | **No.** Calm-month overshoot 0.2983 against a 0.0651 threshold — 4.6× production's. |
| Does E1 work as an overlay member? | **No, decisively.** 0.6571 — worse than predicting zero (0.4621), DM p = 0.014 *against*. |
| Should anything ship to production? | **No.** Nothing here recommends a change, and production is untouched. |

**The one thing Stage 1 does establish** is that the reformulated DFM is a live research candidate
rather than the four-times-worse also-ran the repo previously recorded, and that its failure is
precisely diagnosable rather than diffuse: it is *confidently wrong* in a handful of shock months
(overlay leg sd 1.32× the residual it predicts; 2022-11 predicted +2.856pp against a realised
−0.215pp). That is the specific target
[Plan 2](superpowers/plans/2026-08-20-dfm-plan2-stage2-sv-outliers.md) is written to hit, with an
explicit instruction to abandon the overlay route if it misses.

---

## Baselines (this harness, 2015–2024, purged 12+1)

| Model | RMSE | MAE | rel vs AA | Source |
|---|---|---|---|---|
| AR(1) | 0.5459 | 0.3790 | 1.165 | `dfm.benchmarks.ar1_backtest` |
| **AutoARIMA** | **0.4687** | 0.3205 | 1.000 | `dfm.benchmarks.autoarima_backtest` |
| **Production replication** | **0.4142** | 0.2864 | **0.884** | `dfm.benchmarks.production_backtest` |

README documents AA **0.467** and production rel-RMSE **0.89**. This harness reproduces both to
within 1% (0.4687 / 0.8837) from an independently-built code path, so the baseline every claim below
is measured against is trustworthy. It also reproduces the documented *shock concentration* of
production's edge without being asked to: rel-RMSE **0.847** in 2022–23 against **0.925** ex-shock.

RegEvent is excluded from the replication, because the documented figures predate the 2026-08-19
layer. See §Appendix for what including it would do.

## Release calendar coverage

| source | rows | share |
|---|---|---|
| `ons_api` | 381 | 7.6% |
| `provider_doc` | 4587 | 91.7% |
| `rule` | 36 | 0.7% |

CPI bulletin: **127 of 139 reference months carry a real ONS release date**; the 12 fallbacks are
exactly the 2015 months, which predate ONS release pages entirely (verified: every slug pattern 404s
before 2016-01). Actual release lags run **11 to 26 days** after month-end, so the 3rd-Wednesday rule
the repo previously used is wrong by up to 7 days — 2022-03 published 13 Apr against a rule date of
20 Apr, and 2026-06 published 22 Jul against 15 Jul.

## Known accuracy floor

Reconstructing YoY from the index rather than reading D7G7 costs, by window:

| window | n | max | mean |
|---|---|---|---|
| 2000+ | 319 | 0.1348 | 0.0455 |
| **2015–2024 (eval)** | 120 | **0.1151** | **0.0379** |
| 2025+ (blind) | 19 | 0.0942 | 0.0398 |

Matches the spec's documented 0.135 figure for 2000+. The mean floor in the eval window is 0.038pp
against AA's 0.4687 RMSE — about 12× below — so it does not threaten the comparison, but it binds
every model equally and is reported as a floor.

## Compute

E1 timing probe on the real 336×36 panel: **44.5 s/fit** at 800 draws. The realised walk-forward ran
**37.7 s/fit, 75 minutes total** for 120 months. No fallback from spec §5.3 was applied.

A performance pass on `state_space.py` was required to get here: the original Kronecker Lyapunov
initialisation was O(n⁶) and 75% of filter time. Replacing it with a bilinear solve, dropping a
discarded covariance recursion, swapping a per-period `pinv` for a Cholesky solve, and exploiting the
structural sparsity of the design matrix cut the simulation smoother from **273 ms to 52 ms** — an
11× filter speedup — with the log-likelihood unchanged to 3.3e-16 relative and the statsmodels
cross-check unmoved at 5.5e-13.

The eventual 12-config sweep still projects to **18.1 h** against the 8 h ceiling. That is a Plan 2–4
problem, not Plan 1's, and the honest options there are batch parallelism or fewer draws.

## Stage 1 — E1, classical Stock-Watson DFM, standalone

Mode: standalone (the DFM forecasts CPI MoM directly; YoY rebuilt from the known carry). Purge 1+1.
K=7, factor_order=2, 800 draws / 200 burn, warm-started. Coverage **120/120 = 100%**.

| Window | n | RMSE | MAE | AA RMSE | rel vs AA |
|---|---|---|---|---|---|
| full 2015–2024 | 120 | **0.4462** | 0.3010 | 0.4687 | **0.952** |
| shock 2022–23 | 24 | 0.7811 | — | 0.7715 | 1.012 |
| ex-shock | 96 | 0.3103 | — | 0.3546 | 0.875 |
| pre-2020 | 60 | 0.3049 | — | 0.1908 | 1.598 |

Diebold-Mariano: vs AA **stat −0.382, p = 0.703**; vs production **stat +0.553, p = 0.582**.
Bias +0.038. Error correlation with production **0.405**.

### Read

**The MoM reformulation is vindicated.** The repo's prior DFM attempts scored 1.97, 1.96 and 2.62
against AA's 0.467 — worse by a factor of four. E1 scores **0.4462**, a 4.4× improvement, and it is
the same model family. The diagnosis in spec §3.2 was correct: those models were forced to re-learn
11/12 of CPI YoY's variance as mechanical carry-over (lag-1 autocorr 0.983, measured here at 0.9827),
and making the carry arithmetic removes that burden entirely. This alone settles the question the
2026-08-06 spec set out to answer about why prior DFMs failed.

**E1 beats AutoARIMA, but not significantly, and not production.** rel-RMSE 0.952 is a real 4.8%
improvement on the anchor, but DM p = 0.703 — nowhere near significance, and a weaker result than
production's own p ≈ 0.17. Production remains clearly ahead at 0.4142 vs 0.4462. A classical DFM with
no stochastic volatility, no outlier handling, no factor VAR and unrestricted loadings does not
displace the frozen model.

**E1 fails the pre-registered calm-month overshoot rule, decisively.** Spec §7 fixed the threshold
before results were seen: reject a component if it raises mean `|forecast − AA|` in calm months more
than 10% above the production replication's value. Measured:

| | mean \|forecast − AA\|, calm months |
|---|---|
| Production replication | 0.0592 |
| Threshold (production × 1.1) | 0.0651 |
| **E1** | **0.2983** |

E1 is **4.6× production's** calm-month deviation from the anchor — 360% above the rejection
threshold. It is a far more aggressive model than production, and the pre-2020 window shows the
cost: AA scores 0.1908 there and E1 scores 0.3049, a 60% deterioration in the calmest stretch of the
sample. E1 wanders away from the anchor when the anchor is at its best.

**But E1 is not redundant.** Error correlation with production is **0.405**, far below the 0.9
threshold that killed BVAR. The errors are genuinely different, and the window pattern is
*complementary* to production's rather than duplicative: production's edge is shock-concentrated
(rel 0.847 in 2022–23), whereas E1's is calm-concentrated (rel 0.875 ex-shock, 1.012 in shock — it is
marginally *worse* than AA when inflation is moving).

That combination — beats the anchor, loses to production, fails the overshoot rule, uncorrelated
errors — is precisely the profile spec §10 anticipated:

> The most likely *good* outcome is that the DFM wins as an **overlay member** — a genuine
> diversifier against TVP/LGBM — rather than as a standalone forecaster.

**Nothing here recommends E1 for production**, and by the selection rule as written it would be
rejected on overshoot alone. What it does establish is that the reformulated DFM is a live candidate
worth carrying into Stage 2, and that the interesting test is the overlay framing, which Plan 1 does
not run.

**Gate: PASS.** Spec §4.3 requires synthetic-recovery tests passing, ≥95% finite predictions, and a
written result. All three hold (100% coverage). Stage 2 is cleared to build stochastic volatility
(E2) and the Stock-Watson outlier mixture (E3) on this foundation. The measured MoM volatility ratio
— 0.688 in shock months against 0.336 ex-shock — says SV has something real to model, and E1's
shock-window weakness (rel 1.012) is exactly the deficiency SV and robust innovations exist to
address.

## Appendix — the RegEvent overlay-residual question

Production trains TVP and LGBM on the AA-only residual and then adds RegEvent, so the Ofgem cap step
appears in both the overlay's training target and the arithmetic layer — a double count in principle.
Measured, on an identical walk-forward with only the residual definition changed:

| window | n | AA residual | AA+RegEvent residual | delta |
|---|---|---|---|---|
| full 2015–2024 | 120 | 0.4073 | 0.4070 | −0.0002 |
| shock 2022–23 | 24 | 0.6395 | 0.6394 | −0.0001 |
| cap-reset months | 5 | 0.8059 | 0.8048 | −0.0011 |

Max forecast difference across all 120 months: **0.011pp**; mean 0.0005pp. The double count is
**empirically nil**, confirming README §4 — the overlay cannot see the cap at all (`ofgem_cap_delta`
carries 0.0% LGBM gain), which is why RegEvent exists as arithmetic in the first place. Note also
that RegEvent is nonzero in only **5 months** of the scored window, not 23: the cap regime starts in
2019 and β requires prior resets before it fires.

No production change was made or is recommended. The spec's overlay target for *DFM* arms is
nonetheless defined as `actual − (AA + RegEvent)` (§4.2), because a new member should not inherit a
known double count even when its measured cost is zero.

## Data-quality notes carried forward

- **The panel build needs a retry loop.** dbnomics/IMF fetches failed transiently on two separate
  occasions, each time on a different series (`imf_all_commodity`, then `timber_price`), before
  succeeding on the third attempt. `panel._monthly_matrix` fails loudly rather than dropping a
  series, which is correct, but a multi-hour sweep should not die on a transient.
- **Two series are stale by design.** `uk_quarterly_gdp` and `uk_house_prices` both last print
  2026-01. The filter handles this natively as a ragged edge and `panel.staleness_report` surfaces
  it; no series exceeded the 6-month DEAD threshold at the 2026-06 as-of date.
- `global_supply_chain_pressure` remains an expected drop (xlsx engine error), leaving 8 of 9
  production-pinned factors live — matching production's own `OPTIONAL_PINNED` handling.

---

## Overlay test — (AA + RegEvent) + DFM vs the frozen production model

Requested 2026-08-20, outside Plan 1's scope. E1 in **overlay mode**: the DFM's factors predict the
`actual − (AA + RegEvent)` residual, purge 12+1 (production's own discipline, since the residual is a
genuine 12-month difference). 120/120 months, 35.5 s/fit, 71 min.

| Model | RMSE | MAE | rel vs AA | shock 22–23 | calm | pre-2020 |
|---|---|---|---|---|---|---|
| AutoARIMA | 0.4687 | 0.3205 | 1.000 | 0.7715 | 0.3546 | 0.1908 |
| AA + RegEvent | 0.4621 | 0.3172 | 0.986 | 0.7589 | 0.3507 | 0.1908 |
| Production (no RegEvent) | 0.4142 | 0.2864 | 0.884 | 0.6536 | 0.3281 | 0.1734 |
| **Production + RegEvent [FROZEN]** | **0.4073** | **0.2831** | **0.869** | 0.6395 | 0.3242 | 0.1734 |
| (AA+RegEvent) + DFM, λ=1 | 0.6571 | 0.4136 | 1.402 | **1.3049** | 0.3378 | 0.2280 |
| (AA+RegEvent) + 0.5·DFM | 0.4710 | 0.3243 | 1.005 | 0.8300 | 0.3241 | 0.1812 |
| AA+RegEvent + 0.5·(TVP,LGBM,DFM) | 0.4119 | 0.2887 | 0.879 | 0.6587 | 0.3220 | 0.1729 |
| AA+RegEvent + 0.5·(TVP,LGBM) | 0.4073 | 0.2831 | 0.869 | 0.6395 | 0.3242 | 0.1734 |

Diebold-Mariano vs the frozen model: λ=1 **stat +2.493, p = 0.014** (significantly *worse*);
λ=0.5 stat +1.587, p = 0.115; three-leg ensemble stat +0.400, p = 0.690.

### Verdict: rejected

**The DFM overlay is worse than no overlay at all.** Predicting zero for the residual scores 0.4621.
The DFM's residual forecast scores **0.6571**. It is not merely unhelpful, it is actively harmful,
and the DM test says so at p = 0.014 — the only statistically significant result in this whole
exercise, and it is significant in the wrong direction.

**Adding it to the existing legs does not help either.** The three-leg ensemble scores 0.4119 against
the frozen model's 0.4073 — slightly worse, and indistinguishable (p = 0.690). By spec §7 rule 4, a
component that does not improve OOS RMSE is dropped regardless of how good the economics look. This
one is dropped.

### Why it fails — the diagnosis is clean

| leg | corr with true residual | forecast sd | RMSE as residual forecast |
|---|---|---|---|
| TVP | **+0.505** | 0.403 | 0.4276 |
| LGBM | +0.315 | 0.171 | 0.4319 |
| **DFM** | **+0.269** | **0.600** | **0.6571** |
| *(zero overlay)* | — | 0.000 | 0.4621 |

True residual sd is 0.455. The DFM leg is **the least informative and the most over-scaled**: worst
correlation with the truth, yet the largest amplitude — 32% wider than the thing it is predicting,
where LGBM is a cautious 0.171. It is confidently wrong.

The over-confidence is concentrated, not diffuse:

- Shock window: DFM leg sd **0.784** against true 0.750 — right magnitude, wrong months. Shock RMSE
  blows out to **1.3049** against AA's 0.7715.
- Calm window: sd 0.234 against true 0.346 — under-dispersed, and here it is harmless (calm RMSE
  0.3378, slightly better than AA's 0.3546).
- Worst single month **2022-11**: the DFM predicted a residual of **+2.856pp** when the realised
  residual was **−0.215pp**. Wrong sign, 3pp of error, in one month.

Ex-post optimal shrinkage is **λ\* = 0.204**, not production's 0.5. Even at that hindsight-chosen
value the composed forecast reaches only **0.4388** — better than AA's 0.4621, still well short of
production's 0.4073. So there is *some* signal in the DFM leg, but not enough to matter, and
production's λ=0.5 haircut is far too generous for a leg this noisy.

Error correlations with the incumbent legs are 0.578 (TVP) and 0.587 (LGBM) — below the 0.9 that
killed BVAR, so the DFM is not formally redundant. But non-redundancy is worthless when the leg
itself is bad: diversifying into a worse forecaster does not help.

### What this does and does not settle

It settles that **E1 is not a viable overlay member**, and it kills the most optimistic reading of
Stage 1 — that the low error correlation (0.405) implied latent ensemble value. It did not.

It does **not** settle the overlay question for the DFM family, because the failure mode is exactly
what the later arms exist to fix:

- **Over-confidence that is catastrophic in a few months** is what the Stock-Watson outlier mixture
  (**E3**) targets. A single month contributing a 3pp error with the wrong sign is the textbook case
  for fat-tailed innovations.
- **Uniform confidence across a sample whose volatility doubles** is what stochastic volatility
  (**E2**) targets. E1 has no mechanism to know that 2022–23 is a different regime from 2016; the
  measured MoM volatility ratio is 0.688 vs 0.336.

So the honest position is: the minimal arm fails as an overlay, decisively and significantly, and its
failure is precisely diagnosable as the absence of the two features Stage 2 adds. That is a useful
negative result, not a dead end — but it is a negative result, and if E2 and E3 do not fix the
over-confidence, the overlay route should be abandoned rather than tuned.

**RegEvent, separately, earns its place.** AA + RegEvent beats AA (0.4621 vs 0.4687) and frozen
production beats the RegEvent-free version (0.4073 vs 0.4142) — a 1.7% gain from pure arithmetic on
5 reset months in the window, consistent with README §4.
