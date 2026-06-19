# Final research summary — UK CPI nowcast

## Original hypothesis
A multi-factor / regime-aware residual stack on top of AutoARIMA can materially and reliably
improve UK CPI YoY nowcasts — i.e. factor information and regime detection beat a univariate
benchmark.

## What survived
- **AutoARIMA anchor** — does ~96% of the level work; the robust baseline and the thing every
  benchmark is measured against.
- **A small fixed residual overlay**: TVP (shock pass-through diversifier) + LGBM (stable
  nonlinear PPI cost-push), equal weight, **shrunk by λ=0.5**.
- **Two factors earned their place**: `uk_ppi_input` (input PPI) and `deep_sea_freight`
  (added to PINNED; factor-race winners; LGBM's edge is ~entirely PPI).
- **Magnitude shrinkage** (λ) as the only defensible "regime" adjustment.

## What failed (falsified, documented, removed)
- **Regime detection / switching** — HMM, HelpfulStage2, ObservableShock, TVP-vs-LGBM winner
  detector: all at/below chance OOS; switching loses to averaging. A real ex-post
  shock/calm regime exists but is **not predictable ex-ante**.
- **BVAR and MIDAS** — redundant cost-push clones (0.89–0.93 error-corr); no information,
  no insurance; removed.
- **The intramonth stack** as a comparator — a different model; its negative-edge "result"
  was an invalid comparison (reconciliation §`timing_reconciliation`).
- **Nonlinear ML as a replacement** — LGBM is a stable but narrow PPI wrapper, not a richer
  learner; complements but does not replace the linear overlay, and only via diversification.

## Key empirical findings
- The overlay edge is **modest, shock-concentrated (2022/23), and DM-insignificant** (p≈0.17).
- The model is a **month-end reference-month nowcast** — it does NOT improve between month-end
  and release (no post-T information); the edge accrues T-21→T as within-month financials land.
- AA residual decomposition: **PPI cost-push (calm) + Ofgem-cap (shock)** explain a minority;
  spot energy minor; **~74% unexplained** (food/services not modelled).
- Overlay is **~79% noise** (R²≈0.21) but its ~21% of signal is correctly scaled (reliability
  λ≈0.8); production λ=0.5 trades ~2.5% backtest RMSE for halved calm-month magnitude risk.
- Genesis live month (May-2026) was **adverse**: a calm/base-effect print where the overlay
  overshot and AutoARIMA alone was best — exactly the documented failure mode.

## Remaining uncertainty
- Is the overlay edge real out-of-sample, or a 2022/23 (energy-era) artefact? Unresolved —
  in-sample insignificant; one adverse live point.
- Will `uk_ppi_input`'s dominance persist outside the energy era? Unknown.
- Is λ=0.5 right, or should it be 0.6–0.8 (statistical optimum)? The forward record decides.
- We never tested against a true point-in-time *survey* consensus (only a proxy); the
  consensus comparison is provisional.

## Conclusion
The architecture is frozen and converged: **AutoARIMA + a half-shrunk TVP/LGBM cost-push
overlay**, with regime-switching, extra members, and unshrunk magnitude all falsified and
removed. The honest edge is small and unproven out-of-sample.

**The model should now be evaluated prospectively rather than modified.**
