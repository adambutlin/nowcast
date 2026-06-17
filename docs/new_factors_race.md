# Factor race — AA-residual predictability (new-factors branch)

`code/new_factors/factor_race.py`. Target = AutoARIMA residual (walk-forward 2015–24,
n=120). Metrics: **univ_rel_rmse** (walk-forward single-factor ridge added to AA;
<1 ⇒ helps), mean|SHAP| and permutation ΔRMSE from one GBT on all 50 factors, |corr|.

## Top of the race (univ_rel_rmse, lower = better)

| rank | factor | region | rel_rmse | abs_corr | SHAP | perm ΔRMSE |
|---|---|---|---|---|---|---|
| 1 | us_ppi_all | US | 0.883 | 0.481 | 0.052 | 0.051 |
| 2 | **uk_ppi_input** (NEW) | UK | **0.928** | **0.387** | 0.002 | 0.001 |
| 3 | **deep_sea_freight** (FRED) | UK | 0.946 | 0.314 | 0.046 | 0.050 |
| 4 | uk_cpih | UK | 0.951 | 0.326 | 0.007 | 0.004 |
| 5 | ofgem_cap_delta | UK | 0.966 | 0.409 | 0.062 | 0.163 |
| 6 | **uk_ppi_output** (NEW) | UK | 0.970 | 0.309 | 0.002 | 0.002 |
| 7 | chemicals_ppi | UK | 0.982 | 0.194 | 0.020 | 0.015 |
| 8 | uk_be5 | UK | 0.982 | 0.291 | 0.032 | 0.055 |
| 9 | semiconductors_ppi | UK | 0.984 | 0.242 | 0.046 | 0.063 |

23/50 factors beat AA standalone. SHAP/perm leaders: ofgem_cap_delta, us_ppi_all,
semiconductors_ppi, deep_sea_freight, oil_brent, uk_be5.

## New factors verdict

| factor | rel_rmse | corr | verdict |
|---|---|---|---|
| **uk_ppi_input** (ONS GHIP) | **0.928** | 0.387 | **WINNER** — best UK factor by rel_rmse & corr. Pipeline input cost-push leads CPI. ADD to model. |
| **deep_sea_freight** (FRED) | 0.946 | 0.314 | **STRONG** — high SHAP/perm too. Shipping cost channel. ADD. |
| uk_ppi_output (ONS GB7S) | 0.970 | 0.309 | modest — output PPI is closer to CPI, less *leading* info than input. optional. |
| move_index | 1.060 | 0.155 | **NO** — hurts the CPI residual; rates-vol is monetary, not near-term cost-push. |
| slope_2s10s | 1.027 | 0.027 | **NO** — no CPI-residual signal. |
| slope_5s30s | 1.031 | 0.107 | **NO** — no CPI-residual signal. |
| freightos_fbx | n/a | — | unavailable (no free source; CSV drop-in only). |

## SHAP vs univariate discrepancy (important)

`uk_ppi_input` ranks #2 univariate but **low SHAP/perm** (0.002). Cause: collinearity —
in the 50-factor GBT, PPI signal is shared with `us_ppi_all`, `chemicals_ppi`,
`semiconductors_ppi`, `oil_brent`, so SHAP splits the credit. The walk-forward
univariate rel_rmse is the cleaner "does this factor add over AA" metric; SHAP/perm
measure *marginal-given-all-others* and are deflated by collinear cost-push factors.
Trust rel_rmse for the add/drop decision; SHAP for which single proxy survives if you
keep only one.

## Recommendation

- **Add `uk_ppi_input` and `deep_sea_freight`** to the CPI two-stage factor set — the two
  best new UK cost-push channels (rel_rmse 0.93 / 0.95). Keep one of the PPI family
  (input dominates) to avoid collinear dilution.
- **Do not add `move_index` / `slope_2s10s` / `slope_5s30s`** to the CPI model — they do
  not improve the CPI residual (rel_rmse >1). They remain registered for the rates
  pipeline, where curve/vol signals belong.
- Caveat: gains are univariate-standalone and modest (3–7% UK); consistent with prior
  findings that Stage-2's net value is diversification, concentrated in shock months.
  Adding PPI/freight should be validated in the full two-stage backtest before pinning.

Artifacts: `data/new_factors/factor_race.csv`, `plots/new_factors/factor_race.png`.

## End-to-end confirmation (two-stage backtest, +uk_ppi_input +deep_sea_freight)

| window | 2-stage base | +PPI/freight | Δ | rel base→plus |
|---|---|---|---|---|
| full | 0.4375 | 0.4287 | −0.0088 | 0.933→0.915 |
| 2022_23 | 0.6803 | 0.6692 | −0.0111 | 0.882→0.867 |
| ex_shock | 0.3515 | 0.3432 | −0.0083 | 0.991→0.968 |
| pre_2020 | 0.1949 | 0.1954 | +0.0005 | calm-neutral |

Consistent ~2% improvement on the 2-stage (8.5% over AA); ex-shock edge sharpens
(0.991→0.968). **uk_ppi_input + deep_sea_freight added to two_stage PINNED.**

Caveat: live May-2026 nowcast moved 2.84→2.96 (overlay +0.25; TVP member +0.64). The
new cost-push factors pushed the point UP while the realised 2.8 was held down by food
base effects (not in the factor set). Backtest RMSE improves on average; the single live
month flags TVP overreaction to the new factors — candidate for member shrinkage.
