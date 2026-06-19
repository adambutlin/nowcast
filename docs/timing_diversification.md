# Diversification audit — residual ensemble (base = AutoARIMA)

`code/timing/diversification_audit.py`. Members reconstructed AA+resid, eval 2015-2024.

## Member full-sample RMSE
LGBM **0.4431** (best) < TVP 0.4882 < BVAR 0.5206 < MIDAS 0.5543 (worst). (AA = 0.4687.)

## Error correlation matrix
|       | tvp | bvar | midas | lgbm |
|---|---|---|---|---|
| tvp | 1.00 | 0.69 | 0.70 | 0.71 |
| bvar | 0.69 | 1.00 | **0.93** | **0.91** |
| midas | 0.70 | 0.93 | 1.00 | 0.89 |
| lgbm | 0.71 | 0.91 | 0.89 | 1.00 |

BVAR/MIDAS/LGBM are 0.89–0.93 correlated — one cost-push signal. **TVP (0.69–0.71) is the
only diversifier.**

## Ensemble metrics (rel vs AA by window; <1 = beats AA)
| ensemble | full | 2022/23 | ex_shock | pre_2020 | DM vs prod (p) |
|---|---|---|---|---|---|
| A production (TVP/BVAR/MIDAS) | 0.929 | 0.891 | 0.972 | **1.022** | — |
| B equal-4 | 0.909 | 0.883 | 0.938 | 0.989 | 0.088 |
| C inverse-RMSE | 0.906 | 0.881 | 0.934 | 0.984 | 0.083 |
| D min-variance (error-cov optimal) | **2.20** | 1.57 | 2.77 | 4.80 | <0.001 (worse) |
| **E TVP+LGBM 50/50** | **0.879** | **0.838** | 0.926 | 0.967 | 0.145 |
| F TVP+LGBM+BVAR | 0.894 | 0.868 | **0.924** | 0.978 | 0.102 |

## Answers
1. **Does LGBM add information or repackage PPI?** Mostly **repackages** — its error is
   **0.91 correlated with BVAR**, 0.89 with MIDAS (same cost-push/PPI signal). But it is the
   **lowest-RMSE member** (0.443): it repackages the linear signal *more accurately*
   (nonlinear PPI mapping). It adds **accuracy, not orthogonal information**.
2. **Does TVP+LGBM beat the current ensemble?** **Yes on RMSE in every window** (full rel
   0.879 vs 0.929; beats A at 2022/23, ex-shock, and pre-2020 where A *hurts*). But **DM
   p=0.145 — not statistically significant**.
3. **Most robust ensemble?** D (min-variance) **catastrophically overfits** (weights lgbm
   −4.05/bvar +2.43 → rel 2.2; reject). E has the best point estimate but only 2 members.
   **C (inverse-RMSE, all 4) and F (TVP+LGBM+BVAR) are the robust middle** — beat AA in all
   windows incl pre-2020, no in-sample weight optimization beyond RMSE. F drops the redundant
   MIDAS.
4. **Diversification vs individual skill?** AA 0.469 → best member LGBM 0.443 (skill −0.026)
   → E 0.412 (diversification −0.031). **Roughly half the edge over AA is the best single
   member's skill (LGBM), half is diversifying it with TVP.** Averaging beats the mean member
   (0.502→0.41–0.44) and beats the best member (0.443→0.412) — diversification is real, but
   only because TVP is genuinely uncorrelated; adding more cost-push clones (MIDAS) does little.

## Conclusion — best production residual ensemble
**Not the current A (TVP/BVAR/MIDAS).** Findings:
- **MIDAS is redundant** — worst member, 0.89–0.93 correlated with BVAR/LGBM. Drop it.
- **LGBM dominates BVAR/MIDAS** as the cost-push representative (lower RMSE, same signal).
- **TVP is the essential diversifier** (only low-correlation member).
- Best point estimate: **E = TVP + LGBM (50/50)** (rel 0.879, best in every window).
- Most **robust** (more members, no overfit): **F = TVP + LGBM + BVAR equal** (rel 0.894) or
  **C = inverse-RMSE over all four** (rel 0.906).

**Recommendation:** move to **TVP + LGBM (+ BVAR for robustness), drop MIDAS**, inverse-RMSE
or equal weights. NEVER error-covariance min-variance weights (D) — in-sample optimal,
OOS-catastrophic. Caveats: gains over production are DM-insignificant (p 0.08–0.15); LGBM
rests on a single factor (uk_ppi_input, fragile on ~120 months); so this is a *better point
estimate*, not a proven significant upgrade. The honest gain is modest: rel 0.929 → ~0.88–0.91.
