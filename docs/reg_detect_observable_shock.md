# ObservableShock label + switched architecture — negative result

Branch `reg-detect`. Follow-up to `reg_detect_FINDINGS.md`. n=96 (2017–24), eval 2018+.
Module: `code/reg_detect/observable_shock.py`. Artifacts: `data/reg_detect/
observable_shock.csv`, `shock_detection.csv`, `switched_architecture.csv`.

## 1. ObservableShock label

Causal OR-of-exceedances over expanding q85 thresholds (data strictly before t):
- **energy** — |Brent month logret|, |TTF gas month logret|
- **commodity** — |IMF all-commodity logret| (loaded)
- **shipping** — |deep-sea-freight PPI logret| (GSCPI failed Excel load; deep_sea_freight used)
- **regulatory** — Ofgem-cap-change month | MPC rate change | Budget event
- **weather** — **ABSENT**: no UK HDD/temperature series ingested. Documented, not faked.
  `gas_rv` retained only as a cold-snap proxy, kept OUT of the core label.

Base rate (broad ≥1 channel) = **0.72** of months. A broad observable-shock definition
is the *normal* state, not a rare regime. Stricter variants: ≥2 channels → 0.40;
energy-only → 0.37.

## 2. Detection: TVP vs HMM vs simple observable (AUC vs ObservableShock, 2018+)

| detector | AUC | bal-acc | AUC ex-shock |
|---|---|---|---|
| HMM (2-state Markov-switching, switching variance, on AA resid) | **0.423** | 0.500 | 0.517 |
| TVP (stochastic-vol proxy: expanding-standardised rolling resid vol) | 0.494 | 0.577 | 0.505 |
| simple observable (Brent realised-vol expanding percentile) | **0.545** | 0.503 | 0.530 |

- **Latent regime models do NOT detect ObservableShock** — HMM is *below* chance (0.42),
  TVP is chance (0.49). A CPI-residual variance regime is not the same object as an
  exogenous observable-event shock.
- The simple observable is best (0.545) but still weak, because one indicator (Brent RV)
  cannot see the commodity/shipping/regulatory channels. Observable events are best detected
  by **observing the events**, not by latent-state inference — but the edge is small.

## 3. Switched architecture: AA+BVAR (normal) / AA+MIDAS (shock) vs fixed combo

Per-member standalone RMSE (eval 2018+): AA 0.544, **BVAR 0.550, MIDAS 0.554** (worst),
fixed equal-weight Stage-2 combo **0.513** (best — diversification, not selection).

Switched vs fixed combo, across label definitions (DM stat>0 ⇒ switched WORSE):

| shock label | shock rate | RMSE combo | RMSE switched | DM (sw−combo) | p |
|---|---|---|---|---|---|
| broad (≥1) | 0.80 | 0.5127 | 0.5576 | +1.99 | **0.050** |
| strict (≥2) | 0.40 | 0.5127 | 0.5535 | +2.11 | **0.038** |
| energy-only | 0.37 | 0.5127 | 0.5660 | +2.30 | **0.024** |

By window (broad label): switched 0.558 (full) / 0.763 (22-23) / 0.450 (ex-shock) /
0.199 (pre-2020) — worse than the fixed combo (0.513 / 0.721 / 0.400 / 0.193) in every
window, and worse than AA-only pre-2020.

**The switched architecture loses to the fixed equal-weight combo, and the loss is
significant and grows as the shock label tightens** (p: 0.050 → 0.038 → 0.024).

## Why it fails

Routing shock months to **MIDAS** is the error. MIDAS is the *worst* standalone member
(0.554), not the best in shock — it is merely the only daily-distinct channel; distinct
≠ accurate. The fixed combo wins by **averaging** BVAR+MIDAS+UCM+TVP (error
diversification). Switching to a single member in any month throws away that
diversification, so any hard switch — even with a perfect, tight, causal shock label —
underperforms the blend.

## Verdict

1. **ObservableShock can be labelled** from energy/commodity/shipping/regulatory events
   (weather absent), but a broad definition fires ~72% of months — it is not a rare regime.
2. **Detection:** latent TVP/HMM fail (AUC ≤0.5); simple observables weakly win (0.545)
   but only over the channels they see. Detect observable shocks by observing, not by
   latent state — but the signal is weak.
3. **Switched AA+BVAR / AA+MIDAS does NOT beat the fixed-weight combo — it is significantly
   worse (DM p=0.02–0.05) at every label.** Keep the fixed equal-weight (shrinkage) Stage-2
   combo. Do not switch members by regime. Consistent with `reg_detect_FINDINGS.md`: the
   gain from Stage-2 is diversification, not regime-timed model selection.
