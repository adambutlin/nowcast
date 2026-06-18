# Forensic reconciliation: production two_stage vs intramonth stack

Verdict up front: **They are NOT the same model.** The "contradiction" (prod rel 0.93 vs
intramonth negative edge) is an artefact of comparing two different models on different
data. **Any conclusion drawn from comparing the two backtests is invalid** until they are
reconciled. This retracts the prior-turn claim that the intramonth result "falsifies" the
production edge — it does not; it measured a different, weaker model.

## PART D — config_diff (code-level, decisive)

| aspect | production (`new_factors/two_stage.py`) | intramonth (`intramonth/stack.py`+`config.py`) |
|---|---|---|
| **MIDAS** | `Z.MIDAS` — U-MIDAS, own yfinance daily Brent/GBP/VIX/TTF, **resample('ME').mean() = FULL month** | `"MIDAS"` → **aliased to `ElasticNet`** on HF as-of cols only (`_zoo_class` alias) |
| **BVAR** | `Z.BVAR` on PINNED monthly (month-end) | `BVAR` on monthly+HF as-of (`factor_columns`) |
| **TVP** | `Z.TVP` on PINNED monthly | `TVP` on monthly+HF as-of |
| **factor set** | oil_brent, gas_eu, gdp, imf, mpc_rate_change, ofgem, **uk_ppi_input, deep_sea_freight** | oil_brent, gas_eu, gdp, imf, mpc_rate_change, ofgem, **mpc_vote_split, budget_event** + HF cols (brent/gas/gbp/vix lvl&ret). **NO uk_ppi_input, NO deep_sea_freight.** |
| **data window** | month-END snapshot (`resample('ME').last()`), FULL month | as-of T-k (HF truncated to days ≤ T-k; monthly ffilled) — at T-1 still partial |
| **sample** | 2015-01..2024-12, **n=120** | 2012-01..2024-12, **n=156** (TRAIN_FROM=2010, start=TRAIN_FROM+4) |
| **residual target** | actual − AA(cpi_yoy) | actual − AA(panel target) — same concept, different baseline |
| **weights** | 0.375/0.25/0.375 (both, I applied production weights in the intra reconstruction) | same weights applied |
| **AA_START** | 2001 | 2001 (same) |

**The single most important difference:** the intramonth factor models do **not have
`uk_ppi_input` or `deep_sea_freight`** — the factors the production factor-race ranked #1
and #2 by SHAP and that delivered production's edge. The intramonth stack instead carries
two factors production *dropped* as SHAP dead weight (`mpc_vote_split`, `budget_event`).
Plus MIDAS is a different class (ElasticNet vs U-MIDAS). The intramonth model is therefore
a **weaker, different** model — it cannot reproduce production by construction.

## PART A/B/C — prediction alignment (existing reconstructions, common 120 months 2015-24)
| | RMSE AA | RMSE full | edge | rel |
|---|---|---|---|---|
| production (same dates) | 0.4687 | 0.4355 | **+0.0332** | 0.929 |
| intramonth (same dates) | 0.4402 | 0.4527 | **−0.0125** | 1.028 |

Even on identical dates the edge flips sign → **not a sample-size effect; it's the model.**
- AA divergence: corr 1.00 but mean |ΔAA| = 0.063 (intra AA is 0.06 RMSE *better* — different
  vintage/target resolution in the panel).
- Full-forecast divergence: mean |Δ| = 0.134.
- Decomposition of the full divergence variance: **72% overlay (factors/MIDAS/as-of), 28% AA**.
Files: `production_preds.csv`, `intramonth_preds.csv`, `convergence_audit.csv`
(= prediction_alignment), `discrepancy_decomposition.csv`, `information_difference.csv`.

## PART E — May 2026
| system | AA | TVP ov | BVAR ov | MIDAS ov | Final |
|---|---|---|---|---|---|
| production (frozen) | 2.710 | (0.188) | (0.135) | (0.066) → weighted +0.13 | 2.84 |
| intramonth T-1 | 2.819 | −0.086 | 0.026 | 0.084 | 2.839 |

Different AA (2.71 vs 2.82, Δ0.11), different overlays. The finals land near each other
(2.84 vs 2.84) by coincidence via different baselines + different overlays — not the same
computation. (Production's latest live June nowcast is 2.93; May was 2.84.)

## PART F — verdict
1. **Same model? NO.** Different MIDAS class, different factor set (production's two best
   factors absent from intramonth; two dropped factors present), different data window
   (full month-end vs as-of), different sample, different AA baseline.
2. **What differs?** MIDAS (U-MIDAS vs ElasticNet), factors (±uk_ppi_input/deep_sea_freight,
   ∓vote_split/budget), full-month vs as-of HF, n=120 vs 156, AA vintage.
3. **RMSE discrepancy explained by implementation:** the full +0.046 edge swing (prod +0.033
   → intra −0.013) on common dates is entirely implementation: 72% overlay (dominated by the
   missing ppi_input/freight factors + ElasticNet-MIDAS + as-of truncation), 28% AA baseline.
4. **Can the negative-edge result be compared to rel 0.93? NO.** They are different models on
   different data. The comparison is invalid. The intramonth result says nothing about whether
   the PRODUCTION model's edge survives intramonth.

## How can rel≈0.93 coexist with negative intramonth edge?
It cannot, for the SAME model — and it doesn't. The 0.93 is the production model (with
uk_ppi_input + deep_sea_freight + U-MIDAS, full month-end, 2015-24). The negative intramonth
edge is a DIFFERENT model (no ppi/freight, ElasticNet-MIDAS, as-of partial month, 2012-24).
No bug in either backtest individually; the **bug was comparing them as if identical**
(my error in the prior turn). 

## Required next step (not done here — would be a new run)
To genuinely test whether the PRODUCTION model's edge survives intramonth, re-run the
production `two_stage` (Z.MIDAS + PINNED incl uk_ppi_input/deep_sea_freight) under as-of
truncation at each horizon — i.e. give the intramonth panel the production factor set and
MIDAS class. Until then, the intramonth horizon conclusions are **suspended**.
