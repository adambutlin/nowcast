# TVP-vs-LGBM regime audit — is the member-winner predictable?

`code/timing/regime_tvp_lgbm.py`. ShockAdvantage_t = |err_LGBM| − |err_TVP| (>0 ⇒ TVP wins).
Features (causal, pub-lagged): uk_ppi_input, deep_sea_freight, move_index, ofgem_cap_delta,
oil_brent(+|.|), gas_eu(+|.|), imf_all_commodity. Walk-forward, test 2018-2024 (n=96 after
dropping NaN). Not HMM / HelpfulStage2 / ObservableShock — a fresh target.

## A real regime EXISTS ex-post (descriptive)
| window | n | TVP win-rate | mean ShockAdv |
|---|---|---|---|
| full | 96 | 0.385 | −0.029 |
| 2022/23 | 24 | **0.542** | +0.025 |
| ex_shock | 72 | 0.333 | −0.047 |
| pre_2020 | 36 | 0.333 | −0.043 |

TVP is competitive only in the 2022/23 shock (54%); LGBM wins ~⅔ of calm months. So there
IS a genuine shock(TVP) / calm(LGBM) split — consistent with every prior finding.

## But it is NOT predictable ex-ante (detectors, OOS 2018-2024; base rate 0.417)
| detector | AUC | sign-hit | Brier |
|---|---|---|---|
| linear reg (ShockAdv) | 0.369 | 0.444 | 0.339 |
| LGBM reg (ShockAdv) | – | 0.611* | 0.339 |
| logistic (TVP_wins) | 0.456 | 0.528 | 0.339 |

All **at or below chance** (AUC <0.5; Brier ≥ base-rate Brier). *LGBM's 0.611 sign-hit just
reproduces the base rate (predict "LGBM wins" most months). Logistic coefficients are small
and mixed (|coef| ≤ 0.64; abs_oil_brent −0.64, imf_commodity +0.35) — **no robust driver**.

## Switching does NOT beat averaging (RMSE, OOS window)
| model | full | 2022/23 | ex_shock |
|---|---|---|---|
| E equal (TVP+LGBM) | **0.5083** | 0.6465 | 0.4225 |
| F equal (+BVAR) | 0.5176 | 0.6697 | 0.4215 |
| switch_hard (logistic) | 0.5540 | 0.7668 | 0.4080 |
| switch_soft (prob-weighted) | 0.5403 | 0.7284 | 0.4154 |
| **oracle (perfect foresight)** | 0.4476 | 0.5765 | 0.3666 |

Both realizable switches are **worse than equal-weight E** (0.554 / 0.540 vs 0.508), badly so
in the shock. The **oracle** (0.448) shows ~0.06 RMSE of theoretical value in perfect
switching — but the detector cannot capture it (AUC<0.5), so switching **destroys** value.

## Answers
1. **Can TVP-vs-LGBM superiority be predicted?** **No** — AUC 0.37–0.46, Brier ≥ base rate.
   The monthly winner is unpredictable from these factors.
2. **Which factors matter?** None robustly — small, mixed logistic coefficients; no clean
   energy/shock predictor of the winner.
3. **Does switching beat E / F?** **No** — hard/soft switching are worse than equal-weight in
   full and shock windows.
4. **Does diversification still win?** **Yes** — equal-weight averaging beats every realizable
   switch.

## Conclusion — predictable regime, or averaging optimal?
**Averaging remains optimal.** A real ex-post regime exists (TVP-in-shock / LGBM-in-calm),
and an oracle switch would help (~12% RMSE), but the regime is **not predictable ex-ante** —
the detector is at/below chance, so switching on it loses to a fixed blend. This is the SAME
verdict reached (via different, now-forbidden targets) for HMM / HelpfulStage2 /
ObservableShock: **the timing/switching/gating problem is not solvable on this data; the
fixed average is the production answer.** For live testing, deploy the **equal-weight blend**
(E = TVP+LGBM, or F = +BVAR for robustness) — never a switch.
