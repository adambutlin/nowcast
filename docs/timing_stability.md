# Stability audit — TVP / LGBM / BVAR; is LGBM genuine or a nonlinear PPI wrapper?

`code/timing/stability_audit.py`. Members reconstructed AA+resid; LGBM walk-forward (full +
no-PPI). eval 2015-2024.

## 1. Rolling 5-year rel-rmse vs AA (<1 beats AA)
| window | TVP | BVAR | LGBM | LGBM_noPPI | E(TVP+LGBM) | E_noPPI |
|---|---|---|---|---|---|---|
| 2015-19 | 1.121 | 1.091 | 0.949 | 1.020 | 0.938 | 0.888 |
| 2016-20 | 1.253 | 1.084 | 0.928 | 0.955 | 1.029 | 1.005 |
| 2017-21 | 1.024 | 0.966 | 0.865 | 0.987 | 0.902 | 0.948 |
| 2018-22 | 0.960 | 0.906 | 0.878 | 0.978 | 0.837 | 0.861 |
| 2019-23 | 0.934 | 0.999 | 0.935 | 0.988 | 0.858 | 0.867 |
| 2020-24 | 0.937 | 1.006 | 0.945 | 0.993 | 0.871 | 0.879 |

**LGBM is the most stable member** — beats AA in **6/6** windows (mean 0.917, std 0.036).
TVP beats AA 3/6 (vol 0.93–1.25), BVAR 3/6. (Q1: YES, LGBM stable.)

## 2. Rolling SHAP (PPI share + top factor)
| through | PPI share | top factor |
|---|---|---|
| 2019 | 0.27 | imf_all_commodity |
| 2020 | 0.24 | imf_all_commodity |
| 2021 | 0.32 | uk_ppi_input |
| 2022 | 0.41 | uk_ppi_input |
| 2023 | 0.35 | uk_ppi_input |
| 2024 | 0.32 | uk_ppi_input |

**SHAP drifts**: PPI is top factor only **from 2021** (energy era); pre-2021 imf_commodity led.
PPI dominance is **recent, not stable across history**. (Q2: moderately stable but era-dependent.)

## 3/4/5. PPI ablation
| window | rel LGBM | rel LGBM_noPPI | edge lost |
|---|---|---|---|
| full | 0.946 | **0.995** | 0.050 |
| 2022/23 | 0.976 | 0.987 | 0.011 |
| ex_shock | 0.908 | **1.006** | 0.097 |
| pre_2020 | 0.949 | **1.020** | 0.071 |

Removing `uk_ppi_input`: LGBM's edge over AA (0.054 full) **collapses to 0.005 → ~90% gone**;
**hurts in calm (1.006) and pre-2020 (1.020)**. Replacement top SHAP factor = `uk_quarterly_gdp`
(weak). (Q3: highly dependent. Q5: LGBM retains essentially NO edge without PPI.)

## 6. LGBM edge concentration
Wins **63/120 months** (53% — barely > coin flip). Top-3 months = 15% of positive gain,
top-5 = 25%, top-10 = **42%**. Not pathologically concentrated (no 1-2 month memorisation),
but ~8% of months carry ~40% of the gain. (Q6: moderately concentrated.)

## 7. TVP complementarity after PPI removal
corr(TVP_err, LGBM_err) 0.711 → corr(TVP_err, LGBM_noPPI_err) **0.656** (slightly MORE
complementary). Ensemble: E 0.961, **E_noPPI 0.967** — the TVP+LGBM blend barely degrades when
PPI is removed, because TVP carries the edge that LGBM_noPPI loses. (Q7: YES — TVP remains
complementary; the ensemble is robust to PPI removal even though LGBM alone is not.)

## Conclusion — genuine model or nonlinear PPI wrapper?
**LGBM is largely a nonlinear wrapper around `uk_ppi_input`** — ~90% of its edge vanishes
without PPI, and without PPI it is ≈AA (and hurts in calm). It is NOT a rich multi-factor
learner. BUT it is a *well-behaved* wrapper: the most stable member (beats AA 6/6 rolling
windows, lowest variance), edge not memorised in a handful of months. So it is a **genuine,
stable, nonlinear PPI→residual model** — accurate, but narrow.

Two caveats for live use: (1) PPI's dominance is a **post-2021 (energy-era) phenomenon** —
in a future calm regime the wrapper may track a weaker factor and fade; (2) per-month it beats
AA only 53% of the time. Therefore deploy the **ensemble, not LGBM alone**: TVP+LGBM survives
PPI removal (E_noPPI 0.967) because TVP is genuinely complementary. The robustness lives in
the *blend*, not in LGBM being more than a PPI model.
