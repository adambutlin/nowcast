# Residual decomposition audit — economic channels of CPI surprises (after AutoARIMA)

`code/timing/residual_decomp.py`. resid = CPI_yoy − AutoARIMA. Three blocks → PC1 composite
each (sign-aligned to +corr with residual); OLS resid ~ Cost + Energy + Reg; LMG
(orderings-averaged incremental R²). Explanatory in-sample. Windows within eval 2015-2024.

Blocks: Cost = uk_ppi_input + uk_ppi_output; Energy = brent + gas + deep_sea_freight +
imf_commodity; Regulatory = ofgem_cap_delta + budget_event + mpc_rate_change + mpc_vote_split.
(Regulatory PC1 ≈ +ofgem_cap − mpc_rate: an "administered-price / policy" axis.)

## Total residual variance explained (R²)
| window | total R² | unexplained |
|---|---|---|
| full | 0.260 | **74%** |
| 2022/23 | 0.408 | 59% |
| ex_shock | 0.157 | 84% |
| pre_2020 | 0.175 | 82% |

## LMG decomposition (absolute R² and share of explained)
| window | Cost R² (share) | Energy R² (share) | Regulatory R² (share) |
|---|---|---|---|
| full | 0.149 (57%) | 0.015 (6%) | 0.096 (37%) |
| 2022/23 | 0.106 (26%) | 0.031 (8%) | **0.271 (66%)** |
| ex_shock | **0.134 (86%)** | 0.022 (14%) | 0.001 (0.6%) |
| pre_2020 | 0.106 (60%) | 0.025 (15%) | 0.044 (25%) |

## Answers
1. **Fraction of residual variance by block (full):** Cost 15%, Regulatory 10%, Energy 1.5%
   — together 26%; **74% unexplained**.
2. **Does PPI dominate calm?** YES — Cost Pressure is 86% of explained ex-shock, 60% pre-2020.
   In normal months the AA residual is a PPI-cost-push story.
3. **Does energy dominate shock?** **NO — surprisingly.** In 2022/23, **Regulatory dominates
   (66%)**, Cost 26%, spot Energy only 8%. The UK energy shock transmits through the
   **administered Ofgem price cap** (in the Regulatory block), not spot Brent/gas — a key UK
   institutional feature. Spot energy explains little of the AA residual anywhere (1.5–15%).
4. **Does regulation matter?** YES in shock/cap-change periods (66% in 2022/23, 37% full,
   25% pre-2020) — essentially the Ofgem-cap channel — but **≈0 in calm** (ex-shock 0.6%).
5. **Can the ensemble edge be decomposed into interpretable channels?** Only partially. The
   explainable 26% splits into a PPI cost-push channel (calm) + an Ofgem-cap regulatory channel
   (shock); spot energy is minor. The majority (74%) is unexplained — consistent with the
   ensemble's edge resting heavily on idiosyncratic/noise and on drivers absent from this set
   (food, services), which is why the edge is small and shock-concentrated.

## Conclusion — economic decomposition of UK inflation surprises (post-AutoARIMA)
After removing AutoARIMA, the residual is **mostly unexplained (~74%)**. The explainable
minority is two clean channels:
- **Cost-pressure (PPI input/output)** — dominates CALM periods (~60–86% of explained);
- **Regulatory / administered (Ofgem cap, + policy)** — dominates SHOCK periods (66% in
  2022/23).
**Spot energy (Brent/gas/freight/commodity) is consistently minor** — the UK energy passthrough
runs through the *regulated cap*, not spot prices. So the ensemble's interpretable edge is a
"PPI in calm, Ofgem-cap in shock" story, but it accounts for only a quarter of inflation
surprises; the rest (food, services, measurement) is outside the current factor set.

Caveats: in-sample explanatory R² (not OOS); small windows (2022/23 n=24) inflate R²;
PC1 composites; Regulatory PC mixes Ofgem cap (+) and MPC (−).
