# FINAL PRODUCTION MODEL — UK CPI YoY nowcast (FROZEN 2026-06-19)

> Research phase closed. This document is the governance specification. Changes to
> members/weights/λ require a governance decision, not a code edit.

## 1. Final architecture
```
Forecast = AA + λ · Overlay + RegEvent        λ = 0.5
Overlay  = 0.5·TVP + 0.5·LGBM
RegEvent = κ · β · Ofgem-cap-%                κ = 0.75   (headline only, 0 outside cap resets)
=> Forecast = AutoARIMA + 0.25·TVP_resid + 0.25·LGBM_resid + RegEvent
```
- **AA** — AutoARIMA on CPI YoY (univariate). The anchor: persistence, seasonality, mean
  reversion, base-effect arithmetic. ≈96% of the level.
- **TVP** — time-varying-parameter regression on the PINNED factors, predicts the AA residual.
- **LGBM** — LightGBM on the AA residual over the PINNED factors.
- **RegEvent** — the announced Ofgem price-cap reset priced into CPI arithmetically
  (`code/production/reg_events.py`, added 2026-08-19). Zero in the ~3 months in 4 with no reset.
- Members from `code/uk_model_zoo.py`; factors from `code/factors.py` (PINNED). Entry point:
  `code/production/model.py`.

## 2. Economic interpretation
- AA carries the deterministic core (seasonal + base effects).
- **TVP = shock pass-through overlay** — leans into energy/cost-push within the reference month;
  the genuine diversifier (error-corr ~0.69 with the others).
- **LGBM = cost-pressure / PPI overlay** — a stable nonlinear map from input-PPI (and freight,
  commodities) to the residual. Residual decomposition: **PPI cost-push dominates calm; the
  administered Ofgem cap dominates shock; spot energy is minor; ~74% of the residual is
  unexplained** (food/services/idiosyncratic, outside the factor set).

## 3. Historical performance (walk-forward 2015-2024, vs AA)
- Overlay at λ=1: rel-RMSE ≈ 0.87–0.88 full; concentrated in 2022/23 (rel ~0.84), ~neutral
  ex-shock, slight hurt pre-2020.
- Overlay at λ=0.5 (production): rel-RMSE ≈ 0.89 full — ~most of the edge, less magnitude risk.
- **The full-sample edge is statistically insignificant (DM p≈0.17 at the month-end standpoint)
  and shock-concentrated.** Treat as modest.

## 4. Known failure modes
- **Calm / base-effect months** — when food/services (not in the factor set) drive the print,
  the cost-push overlay pushes the wrong way. *Genesis May-2026: AA 2.71 / actual 2.80; λ=1
  overlay → 3.11 (worst); λ=0.5 → 2.91.*
- **LGBM is a narrow PPI wrapper** — ~90% of its edge is `uk_ppi_input`; PPI dominance is a
  post-2021 (energy-era) phenomenon and may fade.
- **TVP is shock-overfit-ish** — loses to AA standalone outside shock windows.
- **No post-month-end information** — the model freezes at month-end T (release is T+15…T+21);
  it is a *reference-month* nowcast, not a release-day or T-30 product.
- **Overlay is ~79% noise** (predictive R²≈0.21); its magnitude is unreliable in calm.

## 5. Governance decisions
- Frozen members: AA, TVP, LGBM. Dropped: **BVAR** (redundant, 0.91 err-corr with LGBM, no
  insurance value), **MIDAS** (redundant, worst member). No regime-switch / detector / HMM /
  scenario tree / release-day updating / latent-state forecasting.
- Equal overlay split (TVP=LGBM=0.5): BVAR/MIDAS sweeps showed the split is flat and
  in-sample weight optimisation (min-variance) overfits catastrophically OOS.
- Production λ = 0.5 (see §6).

## 6. Why λ = 0.5
- **Statistical optimum λ ≈ 0.8** (overlay-shrinkage grid argmin; Bayesian reliability
  Cov(o,r)/Var(o) ≈ 0.81; MAE optimum ≈ 1.0).
- **Production λ = 0.5 is a deliberate governance haircut below the statistical optimum**
  because: (a) the overlay is ~79% noise (R²≈0.21); (b) the reliability ratio is unstable
  (walk-forward 0.58–0.85); (c) OOS/calm reliability is worse than in-sample (May-2026
  overshoot). λ=0.5 keeps ~all the full-sample edge (rel 0.89 vs 0.87) while **halving the
  calm-month magnitude risk** (May error 0.31 at λ=1 → 0.11 at λ=0.5). Robust-Bayesian range
  0.5–0.65; 0.5 is the conservative floor. Revisit upward only if the live record confirms
  OOS reliability.

## 7. Why TVP survives
Only low-correlation member (err-corr ~0.69 vs ~0.9 among BVAR/MIDAS/LGBM) → the genuine
diversifier. Carries the shock pass-through and rescues the ensemble when LGBM's PPI signal
fails (TVP+LGBM_noPPI ≈ TVP+LGBM). It moves the most; its value is diversification, not
standalone skill.

## 8. Why LGBM survives
Lowest-RMSE member (0.443) and the most *stable* (beats AA in 6/6 rolling 5y windows). A
genuine, well-behaved nonlinear PPI→residual map — accurate, if narrow. Dominates BVAR/MIDAS
as the cost-push representative.

## 9. Why BVAR and MIDAS were removed
- **BVAR**: error 0.86 corr with the TVP+LGBM ensemble, 0.91 with LGBM; adds no information
  (Tier-1 ≥ Tier-2 in every window); no model-risk insurance (fails in the same PPI scenarios).
- **MIDAS**: worst standalone member (RMSE 0.554), 0.89–0.93 correlated with BVAR/LGBM —
  a redundant cost-push clone.

## 10. Why regime detection failed
Every timing/switching/gating attempt was falsified out-of-sample:
- HMM regimes, HelpfulStage2 detector, ObservableShock label/switch, TVP-vs-LGBM winner
  detector — all at/below chance OOS (AUC 0.37–0.58, DM-insignificant).
- A real ex-post regime exists (TVP-in-shock / LGBM-in-calm) but is **not predictable
  ex-ante**; switching on predictions loses to fixed averaging (oracle gap unrealizable).
- Conclusion: **the fixed average is the answer; magnitude-shrinkage (λ), not regime-switching,
  is the only defensible "regime" adjustment.**

## 11. Governance changelog (post-freeze)
Changes ratified after the 2026-06-19 freeze. Each is a governance decision, not a silent edit.

- **2026-06-24 — Purged + embargoed overlay training.** The overlays now train on the
  AutoARIMA-residual history with a López-de-Prado-style purge + embargo: residuals within
  `PURGE_HORIZON=12 + EMBARGO=1 = 13` months of the nowcast month are excluded (the residual
  target `cpi_yoy` is a 12-month difference, so the trailing months share the target's YoY
  window → autocorrelation/regime-shift leakage). Helpers: `code/validation.py`
  (`purge_embargo`, `embargo_series`). Backtest evaluation also purged via
  `two_stage.backtest`; the model-zoo default stays off (`purge_horizon=0`).
  - **Freeze lifted:** `_residual_history` previously capped residuals at `END=2024`, so no
    post-2024 data could train and the embargo was inert. It now runs to the latest actual,
    with the **embargo** (not a calendar freeze) bounding recency.
  - **Boundary (chosen):** keep the 12+1 window. For the June-2026 nowcast the last training
    month is **2025-04**; Jan–Apr 2025 newly enter training (residual history reaches 2026-05).
  - **Forecast impact:** **2.871 → 2.686** (−0.185). TVP resid 0.641 → −0.067 (contribution
    +0.160 → −0.017); LGBM resid −0.061 → −0.095. Driven by TVP: 2025-Q1 residuals pulled its
    time-varying coefficients down from the positive 2024 regime. AA anchor unchanged (2.726).
  - **Unchanged by design:** λ=0.5, members, weights. Diagnostic walk-forwards in `timing/`
    (`residual_lgbm.py`, `may2026_reconstruction.py`) and the in-sample SHAP screen in
    `retrain_pinned.py` were not modified.
  - **Caveat added 2026-08-19:** the impact figures above were measured through the TVP
    prediction-month defect below, so they conflate the embargo's effect on TVP training with
    a 13-month slip in the month TVP was forecasting. Treat "TVP resid 0.641 → −0.067" as a
    historical record of what the code did, not as the embargo's isolated effect.

- **2026-08-19 — BUG FIX: the TVP leg was forecasting a 13-month-stale month.** `nowcast()`
  embargoes the residual column before handing it to the TVP, so that column's last non-NaN
  month is `PURGE_HORIZON + EMBARGO = 13` months before `nd`. `BaseModel._nowcast_row` infers
  its target month from that column, so the TVP was predicting the first *embargoed* month and
  reading THAT month's factor row — the July-2026 overlay was computed from June-2025 factors.
  The leg was therefore blind to current-month shocks (verified: its output was invariant to
  every 2026-07 factor value, including a +£213 Ofgem cap step). LGBM was never affected — it
  pins its prediction row explicitly.
  - **Fix:** `BaseModel._nowcast_row` / `BaseModel.nowcast` / `TVP.nowcast` take an optional
    `nowcast_date`; `production.model._tvp_resid` pins it to `nd`, mirroring `_lgbm_resid`.
    Default behaviour (month inferred from the target column) is unchanged for every other
    caller. Training rows stay embargoed — only the predicted month moved.
  - **Forecast impact:** July-2026 (live) **2.562 → 2.534**, TVP resid 0.230 → 0.119; core
    2.526 → 2.536. June-2026 re-run with the target frozen at May-2026 on today's factor data:
    **2.711 → 2.796**, TVP resid −0.067 → 0.273.
  - **Honest scoring note:** the defect *flattered* both scored months. Against actuals, the
    fixed model would have logged July err −0.366 (vs −0.338 as-run) and June err +0.196 (vs
    +0.083 as-run). The live scorecard keeps the as-run values; the fixed path is scored from
    August-2026 onward. n=2 — this is not evidence about the fix, only a caveat on the record.
  - **Regression test re-based:** `tests/test_headline_regression.py` no longer pins June-2026
    goldens (2.726 / 2.683). They were not reproducible even before this fix — re-running the
    pre-fix code against a May-2026-frozen target on current factor data gives 2.711, because
    the factor matrix is revised and extended continuously. The test now locks the frozen spec
    (λ, weights, 9 pinned / 8 live) and the forecast arithmetic instead.
  - **Not fixed here (see `docs/postmortem_2026_07.md`):** `ofgem_cap_delta` carries no weight
    in either leg, reg-factor `fillna(0)` masks a missing source, and CSV drop-ins shadow
    fresher live fetches. Those are model-design/data-integrity decisions, not defects in the
    frozen spec.

- **2026-08-19 — Live path no longer purges/embargoes recent residuals.** The purge/embargo
  window is a backtest device: it stops a *test* month's YoY window leaking into training.
  Applied to a live nowcast it only discarded residuals a real-time analyst can legitimately
  see (every month up to the last CPI release), and `two_stage.py` had always documented the
  purge as backtest-only. `_lgbm_resid` now takes `purge_horizon`/`embargo` arguments that
  default to OFF; walk-forward backtests pass `TS.PURGE_HORIZON`/`TS.EMBARGO` as before.
  - **Trade-off (accepted):** the overlay now sees the last 12 months of residuals, which are
    heavily autocorrelated with the target month's. That is information, not leakage — but it
    means the overlay extrapolates recent residual momentum. July-2026: LGBM resid −0.046 →
    −0.195 before the other changes below, i.e. it leaned *against* the cap step.

- **2026-08-19 — NEW LAYER: regulatory events (headline only).**
  `Forecast = AA + λ·Overlay + RegEvent`, where `RegEvent = κ·β·cap_pct_change` and
  `code/production/reg_events.py` owns the arithmetic. Rationale: the AA residual has RMSE
  **0.83 in Ofgem cap-reset months against 0.32 in all other months** — the model's error is
  concentrated in the ~1 month in 4 this layer covers — while the `ofgem_cap_delta` factor
  that was supposed to carry it has **0.0% LightGBM gain** (nonzero in 23 of 306 months) and
  moved the forecast by 0.000 for *any* value fed to it. A regulated quarterly step announced
  six weeks ahead is arithmetic, not something to learn.
  - **β (pass-through):** least squares through the origin of the realised electricity + gas
    CPI contribution on the announced cap % change; ≈0.0176pp of CPI per 1% of cap, corr
    0.853, R²(origin) 0.74, n=20. Below 1:1 because the cap governs standard variable tariffs
    while CPI prices all tariffs, and standing charges move separately. Re-estimated from
    PRIOR resets only at every evaluation.
  - **κ = 0.75 (chosen):** governance haircut in the spirit of λ. Walk-forward over reset
    months, β from prior resets only: RMSE **0.730 → 0.671 (rel 0.918)** at MIN_PRIOR=8;
    on the longer window (MIN_PRIOR=4) **0.796 → 0.655 (rel 0.824)**. Better in 7/11 and
    10/15 reset months. κ=1.0 wins marginally on the strict window and loses on the long one
    (the 2022 crisis reset overshoots by −1.23), so the haircut buys crisis-month safety.
  - **Scope:** HEADLINE only (`Target.reg_event_layer`). Core CPI is ex-energy by construction.
  - **Data:** `data/ofgem_cap.csv` is now maintained by `reg_events.refresh_cap_csv()`,
    which rebuilds each quarter's bill from Ofgem's published unit rates at a FIXED
    consumption basis. Ofgem re-based its TDCV in July 2026 (£1,862 old basis vs £1,663 new);
    a level series that switches basis mid-stream invents a step, so only fixed-basis ratios
    are ever used.
  - **July-2026 effect:** cap +13.5% → step **+0.178pp**.

- **2026-08-19 — Data-freshness guards.** (a) A CSV drop-in is now authoritative only for the
  span it covers; months after its last observation are spliced from the live fetch (raw,
  before the transform), status `csv+live`. `csv_frozen: True` opts the target series out,
  since their deliberate truncation is what pins the nowcast month. (b) Regulatory factors are
  zero-filled only INSIDE their source's observed span — past it they stay NaN instead of
  asserting "no event". (c) `nowcast()` reports `stale_factors`: event series must cover the
  target month exactly, everything else gets `factor_health`'s pub_lag + grace budget.
  All three failed silently in July-2026.

- **2026-08-19 — Run manifests.** Every run writes `data/production/runs/<target>_<YYYY-MM>.json`
  with the nowcast, the factor row the overlays saw, per-factor source status and last
  observation, and the git SHA. The July post-mortem could not reconstruct the inputs exactly,
  because the factor matrix is fetched live and moves continuously.

- **2026-08-19 — Renamed AdminStep → RegEvent.** The layer covers regulatory/administered
  events generally (today: the Ofgem cap; the same arithmetic would carry a VAT change or a
  fuel-duty move), so it is named for the class of event rather than for prices.
  `production/admin_prices.py` → `production/reg_events.py`, `admin_step()` →
  `reg_event_step()`, `Target.admin_layer` → `Target.reg_event_layer`, nowcast keys
  `admin_step`/`admin_detail` → `reg_event`/`reg_event_detail`. Behaviour unchanged.

- **2026-08-19 — BUG FIX: a dead overlay leg produced a NaN forecast.** `BaseModel._nowcast_row`
  blocks a model outright when any PINNED factor is past its ffill budget (`pub_lag + 2`), and
  the TVP then returns NaN. `nowcast()` weighted that NaN into the overlay sum, so the whole
  headline forecast came back NaN — hit on the August-2026 roll, where the quarterly
  `uk_quarterly_gdp` (last observation 2026-Q1) exceeded its budget. Dead legs are now dropped
  and the overlay renormalised over the survivors (the convention `two_stage._weighted_combo`
  already used for backtests); with both legs dead the nowcast falls back to AA + RegEvent.
  Reported as `dead_legs` and printed loudly. August-2026 headline: NaN → **2.843**, TVP dead,
  LGBM at full overlay weight.
  - **Follow-up needed:** a quarterly factor in a monthly PINNED set will keep killing the TVP
    late in every quarter. Either widen the ffill budget for `uk_quarterly_gdp` or reconsider
    its place in the pinned set — a governance question, not a code fix.

- **2026-08-19 — Restatement of Apr–Jul 2026 through the current pipeline.**
  `code/production/restate.py <YYYY-MM> …` re-runs a past reference month by truncating the
  target series to the month before and calling `nowcast()` unchanged. NOT a vintage replay:
  factor values are today's, so revisions are baked in. The RegEvent layer is honest by
  construction (β fitted on resets strictly before the target month) and the overlays only
  train on residuals up to the truncation, but a restated number is "what the current code
  makes of today's data", not what would have printed at the time. The live scorecard keeps
  its as-run values.

  | month | AA | TVP | LGBM | RegEvent | restated | as-run | actual | err new | err old |
  |---|---|---|---|---|---|---|---|---|---|
  | 2026-04 | 3.270 | −0.006 | +0.015 | **−0.088** | 3.191 | — | 2.80 | +0.391 | — |
  | 2026-05 | 2.710 | +0.064 | −0.007 | 0 | 2.767 | 2.910 | 2.80 | −0.033 | +0.110 |
  | 2026-06 | 2.726 | +0.025 | −0.055 | 0 | 2.697 | 2.683 | 2.60 | +0.097 | +0.083 |
  | 2026-07 | 2.516 | +0.058 | +0.017 | **+0.178** | 2.768 | 2.562 | 2.90 | −0.132 | −0.338 |

  Headline RMSE over the three months production actually ran: **0.096 restated vs 0.211
  as-run**. Core: 0.137 vs 0.128 over its two scored months — the layer never touches core, so
  that difference is the TVP month fix plus the unpurged live path, and it is marginally worse.
  April is the outlier for both (+0.39 / +0.63): the print collapsed 3.3 → 2.8 while AA sat at
  3.27, and a −6.7% cap reset only justified −0.088 of the gap. **n=4. This is corroboration,
  not evidence** — the layer's case rests on the 15-reset walk-forward above.

- **2026-08-19 — New candidate factor: `uk_pump_price`.** DESNZ weekly road-fuel prices
  (mean of ULSP/ULSD), monthly MEAN, pub_lag=0. The direct CPI motor-fuel input: its monthly
  mean tracks CPI 07.2.2 MoM at **corr 0.993** (vs 0.831 for a month-end close, n=271), while
  `oil_brent` is crude — which led pump prices by weeks and pointed the wrong way in July-2026
  (Brent +32% MoM, CPI motor fuels −3.2%). Candidate only: replacing or joining `oil_brent`
  in the PINNED set requires the factor race + validation gate.

## 12. Core CPI nowcast (dual output, added 2026-07-23)

`model.py` now prints a **second** nowcast alongside headline: **Core CPI YoY** (ex energy, food,
alcohol & tobacco; ONS MM23 **DKO8**). The pipeline is parameterized by `targets.Target`
(`HEADLINE` / `CORE`); the headline path is byte-identical (regression-locked: June-2026 = 2.683).

- **Same architecture**, core-specific factors: `Core = AA + 0.5·(0.5·TVP + 0.5·LGBM)` (λ=0.5) on a
  **5-factor core pinned set**, one per channel — `uk_pmi_services` (demand), `uk_ppi_output`
  (cost-push), `uk_infl_swap_1y` (expectations), `uk_corp_profits` (profit-led / "greedflation"),
  `gbp_eer` (FX). All beat AA-only alone and are non-redundant. Dropped though they beat AA:
  `uk_pmi_composite` (redundant with services PMI), `uk_monthly_gdp` (dbnomics series dead, ends
  2024-03), `uk_cpih` (near-circular). The PMIs, inflation swap, and corporate-profits (ONS GOS YoY)
  are sourced via **LSEG Workspace** (`factors._lseg_history`, needs Workspace + `LSEG_APP_KEY`);
  free monthly wages (AWE) were tested but don't help (rel-RMSE 1.040).
- **Validation gate (headline-grade):** purged+embargoed walk-forward 2015-2024 (n=120), core overlay
  **rel-RMSE 0.934** vs AA-only, **Diebold-Mariano p = 0.006** → the overlay **ships** (else core would
  print AA-only). Edge is modest — core is highly persistent, so the AA anchor already does most of it.
- **First observation (June-2026):** core nowcast **2.435** vs actual **2.6** (err −0.165); the overlay
  pulled down on soft mid-2026 PMIs and overshot this one calm month (AA 2.488 was closer) — the
  expected noisy-but-net-positive behaviour. Tracked in `data/live_scorecard_core.csv`.
- **Evidence:** `docs/core_factor_race.md` (race + calibration + gate); figures `plots/core/*.png`;
  research code `code/new_factors/core_race.py`, `core_validate.py`; config `code/targets.py`.

---
Entry point: `code/production/model.py` (prints headline + core). Live evaluation: `code/production/
{update_live_scorecard,generate_live_report,core_scorecard}.py`, `data/live_scorecard{,_core}.csv`,
`docs/live_report.md`. Full evidence trail: `docs/timing_*.md`, `docs/new_factors_*.md`,
`docs/core_factor_race.md`, `docs/reg_detect_*.md`.
