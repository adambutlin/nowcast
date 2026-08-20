# Pinned-set race — energy channel (crude vs pump) and the missing FX channel

`code/new_factors/energy_fx_race.py`, run 2026-08-20. Target = Headline CPI YoY.
Production spec `AA + λ·(0.5·TVP + 0.5·LGBM)`, λ=0.5, purged+embargoed walk-forward
2015–2024 (n=120, purge 12 + embargo 1). Method mirrors `new_factors/ablation.py` and the
core validation gate, so the numbers are comparable to prior ship decisions.

Two questions the incumbent pinned set never answered:

1. The motor-fuel channel runs off `oil_brent` (crude), not `uk_pump_price` — the direct
   CPI 07.2.2 input. `docs/MODEL_CARD.md` lists this as a known gap; `docs/postmortem_2026_07.md`
   §6 blocks the swap on "the factor race + validation gate".
2. There is **no FX factor at all**. `gbpusd` was a `candidate=False` "core factor" in the
   pre-`two_stage` `main.py` era and was not carried into `targets.HEADLINE.pinned` when the
   pinned tuple was created at `d25f185` (2026-06-13). No doc records that decision. Core CPI
   pins `gbp_eer` off a real race (`docs/core_factor_race.md`); headline pins nothing.

## Result — nothing ships

| arm | k | RMSE | rel-RMSE | DM vs base | p(arm better) |
|---|---|---|---|---|---|
| AA-only | 0 | 0.4687 | 1.0000 | — | — |
| **base** (incumbent) | 8 | 0.4312 | 0.9201 | — | — |
| +pump | 9 | 0.4297 | 0.9169 | +0.466 | 0.321 |
| swap_pump (`oil_brent`→`uk_pump_price`) | 8 | 0.4337 | 0.9253 | −0.512 | 0.696 |
| +fx (`gbpusd`) | 9 | 0.4305 | 0.9185 | +0.417 | 0.338 |
| +pump+fx | 10 | 0.4278 | 0.9129 | +0.812 | 0.208 |

**No arm clears the gate.** `+pump+fx` is the best at rel 0.9129 vs 0.9201, but p=0.208.
`swap_pump` — retiring crude in favour of pump — is the only arm *worse* than the incumbent
set on the full sample. The incumbent set stands unchanged.

## This is an underpowered null, not a null effect

| arm | ΔRMSE | sd(d) | MDE at n=120 | ratio to MDE |
|---|---|---|---|---|
| +pump | +0.0015 | 0.0302 | 0.0053 | 0.28× |
| swap_pump | −0.0024 | 0.0452 | 0.0080 | 0.31× |
| +fx | +0.0007 | 0.0167 | 0.0029 | 0.25× |
| +pump+fx | +0.0034 | 0.0394 | 0.0069 | 0.49× |

Every effect is a quarter to a half of the minimum detectable effect at one-sided 5%.
Resolving effects this size needs roughly 4–16× the sample — 40 to 160 years of monthly CPI.
**The race cannot distinguish these factors from noise and never could.** Read the table as
"no evidence to repin", not "these factors don't matter".

Sub-periods and the shock window agree in sign but not significance:

| window | base | +pump | swap_pump | +fx | +pump+fx |
|---|---|---|---|---|---|
| 2015–2019 | 0.1734 | 0.1756 | 0.1762 | 0.1725 | 0.1733 |
| 2020–2024 | 0.5847 | 0.5818 | 0.5875 | 0.5839 | 0.5797 |
| 2021–2023 (shock) | 0.6746 | 0.6690 | 0.6672 | 0.6732 | 0.6670 |

Note `swap_pump` flips: worst arm on the full sample, second-best in the shock window
(DM +0.899, p=0.184). Consistent with the standing finding that the overlay's value is
concentrated in shock months — but n=36 and p=0.18, so it is a hint, not a result.

## Univariate leg — the postmortem's claim survives

Walk-forward single-factor ridge on the AA residual (same method as `factor_race.py`):

| factor | rel-RMSE | corr with residual |
|---|---|---|
| `uk_pump_price` | 0.9969 | **+0.245** |
| `gbpusd` | 0.9979 | −0.173 |
| `oil_brent` | 0.9988 | −0.031 |
| `gas_eu` | 1.0175 | −0.027 |

`uk_pump_price` carries ~8× the residual correlation of `oil_brent` and beats it on
rel-RMSE, which is what `docs/postmortem_2026_07.md` §6 predicted from the 0.993-vs-0.831
CPI 07.2.2 MoM correlation. The signal is real at the factor level; it is simply too small
at the **headline** level to detect, because motor fuels are ~3% of the basket. The right
place to cash that in is a motor-fuel component model, not the headline pinned set.

`gas_eu` remains the worst-ranked pinned factor and hurts standalone (rel 1.0175, matching
its 43/50 rank in `factor_race.csv`). Its case has to be complementarity in shock months;
that case has never been made explicitly and is worth a dedicated ablation.

## Verdict

- **Do not pin `uk_pump_price`.** Keep it a registry candidate. Revisit as the input to a
  motor-fuel component model, where its 0.993 correlation is measured against the series it
  actually drives.
- **Do not retire `oil_brent`.** `swap_pump` is worse on the full sample.
- **Do not pin `gbpusd`** on this evidence — but the *reason* headline has no FX factor is
  still an accident of the `d25f185` rewrite, not a finding. This race gives it a proper
  null; the gap is now documented rather than silent.
- **Stop using this gate for factors with sub-1% RMSE effects.** It has no power there.

## Two defects found while racing

1. **`global_supply_chain_pressure` is dead in production.** The NY Fed xlsx URL now returns
   an HTML page (verified unsandboxed, 2026-08-20: HTTP 200, `content-type: text/html`), the
   documented CSV fallback `data/global_supply_chain_pressure.csv` **does not exist**, and
   `load_matrix` drops unavailable pinned factors silently. Both live headline manifests —
   `data/production/runs/cpi_yoy_2026-07.json` and `..._2026-08.json` — show 8 pinned factors,
   not 9. Production has been running without its supply-chain channel for at least two months.
   (The base arm above therefore reflects what production really runs.)
2. **`candidate=False` is stale metadata.** `factors.py` documents it as "core factor, always
   kept", but only the legacy `main.py` / `sweep_factors.py` path reads it. `gbpusd`, `uk_be5`
   and `uk_rents_lag1` still carry the flag while being absent from every pinned tuple — they
   read as pinned and are not.

Artifacts: `data/new_factors/energy_fx_race.csv`.
