# FINAL PRODUCTION MODEL — UK CPI YoY nowcast (FROZEN 2026-06-19)

> Research phase closed. This document is the governance specification. Changes to
> members/weights/λ require a governance decision, not a code edit.

## 1. Final architecture
```
Forecast = AA + λ · Overlay        λ = 0.5
Overlay  = 0.5·TVP + 0.5·LGBM
=> Forecast = AutoARIMA + 0.25·TVP_resid + 0.25·LGBM_resid
```
- **AA** — AutoARIMA on CPI YoY (univariate). The anchor: persistence, seasonality, mean
  reversion, base-effect arithmetic. ≈96% of the level.
- **TVP** — time-varying-parameter regression on the PINNED factors, predicts the AA residual.
- **LGBM** — LightGBM on the AA residual over the PINNED factors.
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
