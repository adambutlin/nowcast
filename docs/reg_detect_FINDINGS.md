# reg-detect — Can we predict when Stage-2 adds value? (HelpfulStage2)

Branch `reg-detect`. Walk-forward, no full-sample fitting. Target built from the
production stack: Stage-1 = AutoARIMA; Stage-2 = equal-weight {BVAR, UCM, TVP, MIDAS}
on the AA residual. `HelpfulStage2_t = 1[ |Stage2_err_t| < |AA_err_t| ]`,
`SkillGain_t = |AA_err_t| − |Stage2_err_t|`. n=96 monthly (2017–2024); detector OOS
eval 2018–2024 (n=84).

## Headline verdict: NO.

HelpfulStage2 is **~a coin flip** (base rate 0.476–0.490) and **no detector predicts it
materially better than chance** out-of-sample. Gating Stage-2 on a learned detector
(System C) does **not** beat "Stage-2 always" (System B); no DM test is significant.
**Do not deploy a regime/helpfulness detector.**

## Detector OOS metrics (2018–2024, n=84, base 0.476)

| detector | AUC | Brier | prec | rec | acc |
|---|---|---|---|---|---|
| base_rate | 0.445 | 0.273 | 0.08 | 0.03 | 0.40 |
| persistence (TVP-collapse) | 0.432 | 0.268 | 0.42 | 0.25 | 0.48 |
| hmm_skill (Markov-switching) | **0.582** | 0.265 | 0.49 | 0.88 | 0.50 |
| ucm_skill (local level) | 0.548 | 0.250 | 0.62 | 0.38 | 0.60 |
| logit (observables, unreg.) | 0.572 | 0.401 | 0.52 | 0.35 | 0.54 |
| logit_l2 (regularised) | 0.424 | 0.378 | 0.41 | 0.33 | 0.45 |
| gbt | 0.445 | 0.354 | 0.44 | 0.38 | 0.48 |

Best AUC ≈ 0.58 on ~40 positives → AUC SE ≈ 0.06, i.e. **< 1.5 SE above 0.5: not
significant.** Regularising the logit (`logit_l2` 0.424) and trees (`gbt` 0.445) push it
**below** chance → the apparent logit/hmm signal is fragile noise, not structure.

## AUC by window (hostile — is it just 2022/23?)

| window | n | base | logit | hmm | ucm | logit_l2 | gbt |
|---|---|---|---|---|---|---|---|
| full | 84 | 0.48 | 0.572 | 0.582 | 0.548 | 0.424 | 0.445 |
| 2022_23 | 24 | 0.50 | **0.500** | **0.500** | 0.500 | 0.549 | 0.396 |
| ex_shock | 60 | 0.47 | 0.593 | 0.634 | 0.527 | 0.378 | 0.439 |
| pre_2020 | 24 | 0.29 | 0.445 | 0.752* | 0.248 | 0.235 | 0.361 |

- **Within 2022/23 every detector = exactly 0.50** → the signal is NOT a rediscovery of
  the energy shock (good), but it is also absent there.
- Weak apparent skill lives **ex-shock** (hmm 0.634, logit 0.593) but does not survive
  regularisation (logit_l2 0.378) and is model-dependent.
- pre_2020 hmm 0.752 is a **7-positive small-sample fluke** (base 0.29); ucm/logit_l2/gbt
  are all below chance there. And pre-2020 is exactly where Stage-2 *hurts*
  (helpful rate 0.29, mean SkillGain −0.024), so detecting it buys nothing.

## Architecture test A/B/C (RMSE; DM stat<0 ⇒ system better; gate = logit, thr 0.4)

| window | n | rmseA | rmseB | rmseC | dm C–A (p) | dm C–B (p) | dm B–A (p) |
|---|---|---|---|---|---|---|---|
| full | 84 | 0.544 | **0.513** | 0.525 | −0.81 (0.42) | 0.79 (0.43) | −1.08 (0.28) |
| 2022_23 | 24 | 0.772 | 0.721 | 0.757 | −0.26 (0.79) | 1.13 (0.27) | −0.74 (0.47) |
| ex_shock | 60 | 0.420 | 0.400 | **0.395** | −1.65 (0.10) | −0.43 (0.67) | −1.01 (0.32) |
| ex_covid | 60 | 0.536 | 0.502 | 0.525 | −0.34 (0.74) | 1.23 (0.23) | −0.88 (0.38) |
| pre_2020 | 24 | 0.171 | 0.193 | **0.169** | −0.66 (0.52) | −1.61 (0.12) | 1.33 (0.20) |

- **System B (Stage-2 always) beats A on full RMSE** (0.513 vs 0.544) — but DM p=0.28,
  **not significant**, and B **hurts pre_2020** (0.193 vs 0.171).
- **System C (gated) does not beat B** on full (0.525 vs 0.513). C's only role is to undo
  B's pre-2020 damage (≈A there) — achievable by a hard "off in calm" rule, not a learned
  detector. No DM test reaches p<0.10 except ex_shock C–A at the boundary (0.10).

## Variable importance (perm AUC drop, OOS L2 logit)

Top: `brent_rv` (0.034), `vix_lvl` (0.033), `mpc_rate_change` (0.017). Energy/market
**volatility** is the only economically sensible direction — Stage-2 helps marginally
when energy is moving — but the drops are tiny and several features have **negative**
drop (permuting them improves AUC = noise). Consistent with prior post-mortems: factor
info matters in energy-passthrough/volatile regimes, but the effect is too weak/unstable
to time.

## Answers to the brief

1. **Best detector:** hmm_skill (Markov-switching on SkillGain), AUC 0.582 — and even it is
   not deployable.
2. **ROC AUC:** ~0.58 full / 0.63 ex-shock; SE ≈ 0.06 → indistinguishable from chance.
3. **Calibration:** poor (Brier ≈ 0.25–0.40, ≥ base-rate Brier 0.25); latent models barely
   match the base rate, observable models worse.
4. **Most important variable:** `brent_rv` (Brent realised vol), then `vix_lvl`.
5. **Is HelpfulStage2 predictable?** No — base rate 0.48, no detector materially beats chance OOS.
6. **Should Stage-2 be gated?** Not by a learned detector. C does not beat B. The only robust
   structural rule: keep Stage-2 *off* in quiet pre-2020-style regimes (it hurts there); it
   contributes in shock/volatile periods, but that is not timeable in advance with skill.
7. **Deploy detector?** **No. Kill the helpfulness-detector idea.**
8. **Production architecture:** `Forecast = AutoARIMA + w·Stage2` with **fixed** small w
   (shrinkage combo, BVAR-anchored), NOT `Detector × Stage2`. Optionally hard-gate Stage-2
   weight to zero when realised energy/market vol is below a threshold (observable rule, no
   latent state), since Stage-2 only hurts in calm. No HMM posterior, no shock/disinflation
   labels, no scenario tree.

## Core question

**Can we predict when factor information adds value beyond AutoARIMA?**
**No.** HelpfulStage2 is ~unpredictable (AUC ≈ 0.5–0.58, not significant; coin-flip base
rate; signal vanishes under regularisation and inside the one regime it should matter).
Stage-2's value is real but **diffuse and concentrated in volatile/shock months**; it
cannot be timed month-by-month with skill. The "Stage-2 useful / not useful" regime is
**not a learnable target** on this data. Detector branch result = **negative; do not build
the layer.** Files: `targets.csv`, `detector_results.csv`, `auc_by_window.csv`,
`architecture_comparison.csv`, `robustness.csv`, `importance.csv`; plots in
`plots/reg_detect/`.
