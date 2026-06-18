# Residual-target LightGBM benchmark vs production ensemble

`code/timing/residual_lgbm.py`. Target resid_t = CPI_yoy − AutoARIMA; features = production
PINNED set; walk-forward expanding (test 2015-2024, train from AA_START vintage), no
lookahead; conservative LGBM (num_leaves 7, depth 3, lr 0.02, reg). SHAP + OOS permutation.

## RMSE comparison (rel<1 beats AA)
| window | n | AA | ENS | AA+LGB | rel_ENS | rel_LGB |
|---|---|---|---|---|---|---|
| full | 120 | 0.4687 | 0.4355 | 0.4431 | 0.929 | 0.946 |
| 2022/23 | 24 | 0.7715 | 0.6876 | 0.7530 | 0.891 | **0.976** |
| ex_shock | 96 | 0.3546 | 0.3447 | 0.3221 | 0.972 | **0.908** |
| pre_2020 | 60 | 0.1908 | 0.1950 | 0.1810 | 1.022 | **0.949** |

DM full: AA vs AA+LGB p=0.290; ENS vs AA+LGB p=0.668 — **neither significant**.

## SHAP (mean|SHAP| on residual, full-fit) & OOS permutation
| factor | SHAP share | OOS perm ΔRMSE |
|---|---|---|
| uk_ppi_input | **31.6%** | **+0.0389** |
| imf_all_commodity | 18.5% | −0.0007 |
| deep_sea_freight | 15.1% | +0.0055 |
| oil_brent | 12.8% | −0.0028 |
| gas_eu | 12.0% | +0.0019 |
| uk_quarterly_gdp | 9.0% | −0.0010 |
| mpc_rate_change | 0.9% | 0 |
| ofgem_cap_delta | 0.0% | 0 |

SHAP spreads across cost-push factors, but **OOS permutation shows only `uk_ppi_input` is
robust** — the LGBM is essentially a nonlinear PPI-input→residual mapping; the rest is
in-sample SHAP noise.

## Answers
1. **Does nonlinear ML improve residual prediction?** Modestly full-sample (rel 0.946,
   −5.4%) but **insignificant (DM p=0.29)**, and **worse than the linear ensemble** full-sample
   (0.946 vs 0.929). YES in calm (ex-shock 0.908, pre-2020 0.949 — beats both AA and ensemble);
   weak in shock.
2. **Which factors dominate SHAP?** `uk_ppi_input` (32%), then imf_commodity, deep_sea_freight,
   oil, gas. OOS only `uk_ppi_input` survives.
3. **Does LGBM replace TVP/BVAR/MIDAS?** **No.** Full-sample slightly worse and statistically
   indistinguishable (DM p=0.67); much worse in 2022/23 (0.976 vs 0.891). It is a *complement*
   (calm specialist), not a replacement.
4. **Does LGBM memorize 2022?** **No — the opposite.** It is WEAKEST in 2022/23 (rel 0.976,
   far below the ensemble's 0.891). Its edge lives in calm/ex-shock, where the linear ensemble
   fails.

## Conclusion — is a nonlinear residual model superior to the current ensemble?
**No.** Full-sample it is slightly worse (rel 0.946 vs 0.929) and statistically
indistinguishable from both AA (p=0.29) and the ensemble (p=0.67). Its only robust driver is
a single factor (`uk_ppi_input`); with 8 features and ~120 eval months the nonlinear gains
are fragile.

**The one genuinely interesting result is COMPLEMENTARITY, not superiority:** LGBM and the
linear ensemble have mirror-image regime profiles —
- ensemble: edge in SHOCK (2022/23 rel 0.891), neutral/hurts in calm.
- LGBM: edge in CALM (ex-shock 0.908, pre-2020 0.949), weak in shock.
A regime blend (LGBM-calm / ensemble-shock) would in principle beat either, but realising it
needs a reliable calm/shock detector — already shown to fail (reg_detect nulls). So the
complementarity is **not actionable** without a working regime gate, and on current evidence
the nonlinear model is **not superior** and should not replace the ensemble. At most, LGBM is
a candidate calm-regime overlay pending a detector that does not yet exist.
