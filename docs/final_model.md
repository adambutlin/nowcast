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

---
Entry point: `code/production/model.py`. Live evaluation: `code/production/
{update_live_scorecard,generate_live_report}.py`, `data/live_scorecard.csv`, `docs/live_report.md`.
Full evidence trail: `docs/timing_*.md`, `docs/new_factors_*.md`, `docs/reg_detect_*.md`.
