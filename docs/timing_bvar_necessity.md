# BVAR necessity audit — Tier 1 (AA+TVP+LGBM) vs Tier 2 (+BVAR)

`code/timing/bvar_necessity.py`. Tier1 = AA + 0.5 TVP + 0.5 LGBM; Tier2 = AA +
(TVP+LGBM+BVAR)/3. eval 2015-2024, walk-forward.

## 1/4. Necessity by window (rel vs AA; T2−T1<0 ⇒ BVAR helps)
| window | rel T1 | rel T2 | T2−T1 | DM(T1 vs T2) p |
|---|---|---|---|---|
| full | **0.874** | 0.889 | +0.008 (BVAR hurts) | 0.41 |
| 2022/23 | **0.838** | 0.868 | +0.023 (hurts) | 0.31 |
| ex_shock | 0.917 | 0.915 | −0.001 (tie) | 0.92 |
| pre_2020 | **0.918** | 0.937 | +0.004 (hurts) | 0.52 |

Tier 1 is better or tied in **every** window; BVAR adds nothing positive and slightly hurts
full/shock/pre-2020. All DM p>0.30 (differences not significant either way).

## 2. Rolling 5y (T2−T1<0 ⇒ BVAR helps)
BVAR helps in **3/6** windows (the shock-spanning 2016-20/2017-21/2018-22), hurts in 3/6,
magnitudes tiny (±0.01). No consistent benefit.

## 3. Error correlation
corr(BVAR_err, Tier1_err) = **0.863**; corr(BVAR, LGBM) = **0.912**; corr(BVAR, TVP) = 0.686;
corr(BVAR, mean(TVP,LGBM)) = 0.863. **BVAR is largely redundant with LGBM** (both cost-push on
the same factors incl uk_ppi_input).

## 5. PPI-ablation — does BVAR help when LGBM has no PPI? (model-risk insurance test)
| window | RMSE T1_noPPI | RMSE T2_noPPI | T2n−T1n |
|---|---|---|---|
| full | 0.4533 | 0.4632 | +0.0099 (hurts) |
| 2022/23 | 0.6289 | 0.6644 | +0.0355 (hurts) |
| ex_shock | 0.3770 | 0.3727 | −0.0043 (tiny help) |
| pre_2020 | 0.1695 | 0.1768 | +0.0074 (hurts) |

Even when LGBM is stripped of uk_ppi_input, **adding BVAR does NOT rescue the ensemble** — it
hurts full/shock/pre-2020. BVAR also leans on PPI/cost-push, so it **fails in the same
scenarios** as LGBM (correlated failure) → it is **not orthogonal model-risk insurance**.

## Answers
1. **Is BVAR genuinely adding information?** No — 0.86 error-corr with Tier 1, 0.91 with LGBM;
   no positive edge in any window; full-sample it slightly hurts (rel 0.874→0.889).
2. **Just another cost-pressure model?** Yes — 0.91 correlated with LGBM; redundant.
3. **Diversification benefit remaining after LGBM?** ~None. LGBM already supplies the
   cost-push signal (more accurately); TVP is already the diversifier in Tier 1. A third
   cost-push clone dilutes the better TVP+LGBM blend.
4. **Retain as model-risk insurance?** No — the PPI-ablation shows BVAR fails in the same
   scenarios it would need to insure (correlated, PPI-based). Not orthogonal protection.

## Conclusion — Tier 1 or Tier 2?
**TIER 1.** BVAR is redundant (0.86–0.91 error correlation), adds no information, slightly
hurts on average, and provides no genuine model-risk insurance (it fails when LGBM fails).
Parsimony and full-sample RMSE both favour dropping it. (Caveat: differences are small and
DM-insignificant — this is "BVAR is unnecessary," not "Tier 1 significantly better.")

## Recommended production model
**AA + 0.5·TVP + 0.5·LGBM** (equal-weight Tier 1).
- AutoARIMA = the anchor (does most of the work; the robust fallback).
- TVP = the genuine diversifier (shock-leaning; only low-correlation member).
- LGBM = stable nonlinear cost-push / PPI model (calm-leaning; most stable member).
- Drop BVAR (redundant with LGBM) and MIDAS (redundant, worst member). No switching / regime
  gate / detector (all shown null).
For live prospective testing: deploy Tier 1 equal-weight; report against AutoARIMA-only as the
honest baseline. Edge is real but modest (rel ~0.87) and energy-era-weighted, so treat the
12-month live record as the decisive test before committing capital.
