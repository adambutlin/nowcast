# Overlay-shrinkage calibration — Forecast = AA + λ·(0.5 TVP + 0.5 LGBM)

`code/timing/overlay_shrinkage.py`. Walk-forward member resids, eval 2015-2024.

## RMSE by λ (full sample)
| λ | full | 2022/23 | ex_shock | pre_2020 |
|---|---|---|---|---|
| 0.0 (AA) | 0.5158 | 0.7715 | 0.3954 | 0.1950 |
| 0.5 | 0.4603 | 0.6648 | 0.3677 | 0.1713 |
| 0.6 | 0.4544 | 0.6533 | 0.3648 | **0.1702** |
| 0.8 | **0.4485** | 0.6419 | 0.3617 | 0.1721 |
| 1.0 | 0.4506 | 0.6465 | 0.3625 | 0.1791 |

argmin RMSE λ: full **0.8**, 2022/23 0.8, ex_shock 0.9, pre_2020 **0.6**. The curve is **flat
from 0.7–1.0** (0.448–0.451): shrinking 1.0→0.8 gains only ~0.5%.

## MAE by λ
argmin MAE λ: full **1.0**, 2022/23 1.0, ex_shock 0.9, pre_2020 0.5. (MAE wants more overlay
than RMSE — RMSE penalises the calm-month overshoots harder, so it shrinks more.)

## May 2026 (AA=2.71, overlay=0.401, actual=2.8)
| λ | forecast | abs err |
|---|---|---|
| 0.0 | 2.71 | 0.090 |
| **0.2** | **2.79** | **0.010** |
| 0.3 | 2.83 | 0.030 |
| 0.5 | 2.91 | 0.110 |
| 1.0 | 3.11 | 0.311 |

λ that nails May = **0.225**. Any λ < ~0.45 beats AA; λ=1 (current) is the worst.

## Closed-form λ* + directional stats
| window | λ* | sign-hit | corr(overlay, AA-resid) | mean\|overlay\| | mean\|resid\| |
|---|---|---|---|---|---|
| full | 0.848 | 0.635 | 0.455 | 0.212 | 0.363 |
| 2022/23 | 0.843 | 0.708 | 0.526 | 0.423 | 0.576 |
| ex_shock | 0.861 | 0.611 | 0.372 | 0.141 | 0.292 |
| pre_2020 | 0.631 | 0.694 | 0.539 | 0.118 | 0.153 |

## Answers
1. **λ minimising RMSE:** **0.8** full (flat 0.7–1.0; closed-form λ*≈0.85). 2022/23 0.8, ex_shock 0.9, pre_2020 0.6.
2. **λ minimising MAE:** **1.0** full (0.9 ex_shock, 0.5 pre_2020).
3. **λ improving May 2026:** **~0.2** (0.225 nails it); any λ<0.45 beats AA. λ=1 worst.
4. **Real information but excessive magnitude?** **Mildly, full-sample** (λ*≈0.85 ⇒ ~15%
   over-scaled; RMSE gain from shrinking only ~0.5%). The excess is **regime-specific**: the
   overlay is ~correctly scaled in **shock** (λ* 0.84) but **far too large in calm** (pre_2020
   λ* 0.63; May λ* ≈ 0.2). So: real info, magnitude excessive **specifically in calm regimes**.
5. **Directional signal vs level?** Leans **directional** — sign-hit 0.61–0.71 (good direction)
   but corr only 0.46 and λ*<1 (level over-scaled in calm). The **sign is more reliable than the
   magnitude**; the overlay is best read as a damped directional nudge, not a full-confidence
   level — especially in calm months.

## Conclusion — optimal shrinkage coefficient
- **Backtest-optimal: λ ≈ 0.8** (RMSE), but the 0.7–1.0 region is within 0.5% — the backtest
  barely distinguishes them.
- **Recommended for live: λ ≈ 0.5.** It keeps most of the full-sample edge (RMSE 0.460, rel
  0.892 vs AA — within ~2.5% of the 0.8 optimum) while **halving the calm-month overshoot**
  (May: λ=0.5 → 2.91 / err 0.11 vs λ=1 → 3.11 / err 0.31). Given the overlay is a noisy
  *directional* signal whose magnitude misfires in calm regimes (and the live May genesis
  miss), trading ~2.5% backtest RMSE for roughly halved calm-regime tail risk is prudent.
- Net production overlay: **Forecast = AA + 0.5·(0.5 TVP + 0.5 LGBM) = AA + 0.25 TVP + 0.25 LGBM.**
  Report AA alongside. Revisit λ upward toward 0.8 only if the forward live record shows the
  overlay's magnitude is reliable out-of-sample.
