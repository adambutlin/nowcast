# Live scorecard framework — final production model (AA + 0.5·TVP + 0.5·LGBM)

Prospective evaluation only. No retraining, no architecture change.
- `code/timing/may2026_reconstruction.py` — seeds the genesis row (May 2026), pre-release info.
- `code/timing/update_live_scorecard.py` — append forecast / set actual / regenerate report.
- State: `data/live_scorecard.csv`. Report: `data/timing/live_report.md`.

## Genesis row — May 2026 (reconstructed, info ≤ release)
| forecaster | forecast | abs error | signed error |
|---|---|---|---|
| **AutoARIMA** | 2.71 | **0.09** | −0.09 |
| Consensus | 3.00 | 0.20 | +0.20 |
| Old production (AA+0.375BVAR+0.25TVP+0.375MIDAS) | 2.92 | 0.12 | +0.12 |
| UCL | 3.05 | 0.25 | +0.25 |
| **Final model (AA+0.5TVP+0.5LGBM)** | **3.11** | **0.31** | **+0.31** |
| Actual | 2.80 | — | — |

Decomposition of the final model: AA 2.71 + TVP contribution +0.32 (resid +0.64) + LGBM
contribution +0.08 (resid +0.16) = 3.11.

## Reported answers
**1. May-2026 performance ranking (abs error, best→worst):**
AutoARIMA 0.09 < Old-prod 0.12 < Consensus 0.20 < UCL 0.25 < **Final model 0.31**.
The final model was the **worst** of all on its genesis month.

**2. Would the final model have improved upon…**
| vs | final model better? |
|---|---|
| AutoARIMA | **No** (0.31 vs 0.09) |
| Old production model | **No** (0.31 vs 0.12) |
| Consensus | **No** (0.31 vs 0.20) |
| UCL | **No** (0.31 vs 0.25) |

**It lost to every benchmark.** Both overlay members pushed UP on cost-push (PPI/energy
rising) while the realised May print was held down by food base effects (outside the factor
set) — the exact calm/base-effect failure mode flagged throughout the audits.

## Reading
n=1, and it is an **adverse** start: the cost-push overlay overshot a calm base-effect month,
and AutoARIMA alone won. This is consistent with every prior finding — the overlay's edge is
shock-concentrated and modest/insignificant overall; in calm months it can hurt. The live
scorecard therefore opens **0/1 vs AA, 0/1 vs consensus**.

The genesis row is one data point, not a verdict. The decisive test is the forward record:
run `update_live_scorecard.py --add ...` each release, `--actual ...` once the print lands,
and judge the final model over ~12 releases against AutoARIMA-only (the honest baseline) and
consensus. Until then, the prudent live deployment is **AutoARIMA as the headline with the
TVP+LGBM overlay reported alongside**, not as the committed forecast.
