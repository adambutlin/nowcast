# Research notes — chronological findings (branches reg-detect → new-factors → timing)

Evidence trail for the frozen model. Each bullet links to its full doc.

## reg-detect (regime/detector hunt — all NULL)
- `reg_detect_FINDINGS.md` — **HelpfulStage2**: can we predict when Stage-2 beats AA? Base
  rate 0.48; best OOS AUC ~0.58 (insignificant); gating ⊀ Stage-2-always. Kill detector.
- `reg_detect_observable_shock.md` — **ObservableShock** label + switched AA+BVAR/AA+MIDAS:
  significantly WORSE than fixed combo (DM p=0.02–0.05). Latent HMM/TVP don't detect shocks.

## new-factors (factor expansion + ensemble design)
- `new_factors_race.md` — factor race on AA residual: `us_ppi_all`, **`uk_ppi_input`**,
  **`deep_sea_freight`** top; MOVE/2s10s/5s30s don't help CPI. Added ppi_input + freight to PINNED.
- `new_factors_alloc.md` — BVAR/MIDAS weight sweep (TVP fixed): RMSE surface flat; error
  corr BVAR-MIDAS 0.93; equal split robust; DM rejects nothing.

## timing (information boundary, members, shrinkage)
- `timing_nowcast_window.md` — model is **frozen month-end → release** (no post-T info gain).
- `timing_production_asof.md` — as-of horizons T-30→R-1: edge builds T-21→T then flat; entirely
  2022/23; ex-shock negative until T.
- `timing_reconciliation.md` — **intramonth stack ≠ production**; cross-comparison invalid (retraction).
- `timing_residual_lgbm.md` — LGBM not superior to ensemble; complementary calm specialist.
- `timing_residual_decomp.md` — residual = PPI(calm) + Ofgem-cap(shock); spot energy minor; ~74% unexplained.
- `timing_diversification.md` — drop MIDAS; TVP+LGBM best; min-variance weights overfit (reject).
- `timing_stability.md` — LGBM = stable but narrow **PPI wrapper** (~90% of edge is ppi_input).
- `timing_bvar_necessity.md` — **drop BVAR** (0.91 err-corr with LGBM; no info; no insurance).
- `timing_regime_tvp_lgbm.md` — TVP-vs-LGBM winner **not predictable** OOS; switching loses to averaging.
- `timing_overlay_shrinkage.md` — grid optimum λ≈0.8; recommend 0.5 for calm-risk.
- `timing_bayesian_shrinkage.md` — reliability λ≈0.8 but overlay ~79% noise (R²≈0.21);
  robust range 0.5–0.65.
- `leakage_audit.md` — no post-month-end / post-release leakage.

## Net
AutoARIMA + a half-shrunk TVP/LGBM cost-push overlay. Regime-switching, extra members, and
unshrunk magnitude all falsified. See `final_model.md` (spec) and `final_research_summary.md`.
