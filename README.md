# nowcast — UK CPI YoY two-stage nowcast

A **reference-month nowcast of UK CPI all-items YoY** (ONS D7G7). The model completes at
month-end **T** using only information dated **≤ T**; the official ONS print lands at **T+15…T+21**.
The research phase is **closed and frozen** (2026-06-19) — the model is evaluated prospectively,
and changes to it are governance decisions with a written entry, not code edits (post-freeze
changelog: [docs/final_model.md](docs/final_model.md) §11; latest round 2026-08-19, driven by
[the July post-mortem](docs/postmortem_2026_07.md)). Everything below is the live specification;
there is no hidden sauce, and the honest edge is small
(see [§Performance](#the-models--walk-forward-rmse)).

**Entry point:** [`code/production/model.py`](code/production/model.py) · **Spec:**
[docs/final_model.md](docs/final_model.md) · **Model card:** [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
· **Live record:** [docs/live_report.md](docs/live_report.md)

---

## The multi-layer architecture

The forecast is a univariate anchor, a **shrunk residual overlay**, and an arithmetic term for
announced regulatory events — never a single black box.

```
Forecast = AA + λ · Overlay + RegEvent,     λ = 0.5
Overlay  = 0.5 · TVP + 0.5 · LGBM
RegEvent = κ · β · Ofgem-cap-%              κ = 0.75   (headline only; 0 in non-reset months)
=>  Forecast = AutoARIMA + 0.25 · TVP(resid) + 0.25 · LGBM(resid) + RegEvent
```

| Layer | Model | Role | What it is *not* |
|-------|-------|------|------------------|
| **1 — anchor** | AutoARIMA on CPI YoY (univariate) | Level: persistence, seasonality, mean reversion, base-effect arithmetic. **≈96% of the print.** | — |
| **2a — overlay** | TVP (time-varying-parameter regression on the residual) | Shock pass-through; the genuine **diversifier** (error-corr ≈0.69 vs ≈0.9 among the rejects). | a standalone forecaster — it loses to AA outside shock windows |
| **2b — overlay** | LightGBM on the AA residual | Stable nonlinear **PPI / cost-push** map. Lowest-RMSE member; beats AA in 6/6 rolling 5y windows. | a rich multi-factor learner — ~90% of its edge is `uk_ppi_input` |
| **3 — shrinkage** | λ = 0.5 | Magnitude haircut on the overlay. | a regime switch |
| **4 — regulatory events** | `κ·β·` announced Ofgem cap % ([`reg_events.py`](code/production/reg_events.py)) | Prices a **quarterly administered step** the learners cannot: the AA residual has RMSE **0.83 in cap-reset months vs 0.32 elsewhere**, yet `ofgem_cap_delta` carries **0.0% LGBM gain** (nonzero in 23 of 306 months). Announced ~6 weeks ahead, so it is arithmetic, not a forecast. | a learned factor — it is deliberately outside the ML overlay, and never applies to core CPI (ex-energy) |

**Why RegEvent is a layer, not a factor.** A step that is zero in ~93% of months is invisible to a
regularised GBM and diluted to nothing by a standardised TVP — feeding the true +£213 July-2026 cap
delta into the frozen overlay moved the forecast by **0.000**. As arithmetic it earns its place:
walk-forward across reset months with β re-estimated from prior resets only, RMSE **0.730 → 0.671**
(rel 0.918). κ=0.75 is the same kind of governance haircut as λ.

**Why two stages, equal split, and a haircut.** The AA residual is the only thing the factors
predict, and they predict it badly — the overlay is **~79% noise** (predictive R² ≈ 0.21). The
statistical optimum is λ≈0.8, but production ships **λ = 0.5** as a deliberate governance haircut:
it keeps ~all the full-sample edge (rel-RMSE 0.89 vs 0.87 at λ=1) while **halving the calm-month
magnitude risk**. Equal TVP/LGBM weighting is used because in-sample weight optimisation overfits
catastrophically out-of-sample.

### What was removed or rejected (nothing hidden)

| Dropped | Why |
|---------|-----|
| **BVAR** | 0.91 error-corr with LGBM — redundant cost-push clone, no information, no model-risk insurance. |
| **MIDAS** | Worst standalone member (RMSE 0.554), 0.89–0.93 corr with BVAR/LGBM. |
| **HMM / regime-switch / detector / latent-state / release-day updating** | Every timing/switching/gating variant was **falsified out-of-sample** (AUC 0.37–0.58, DM-insignificant). A real ex-post shock/calm regime exists but is **not predictable ex-ante**; switching on predictions loses to fixed averaging. |

Conclusion: the fixed average is the answer; **magnitude shrinkage (λ), not regime-switching, is
the only defensible adjustment to what the models say.** RegEvent is not a counter-example — it
adds no state and predicts no regime; it prices an event that has already been announced.
Rejected code is retained as research context only.

---

## The factor hierarchy

Factors flow through a hierarchy: a broad registry → SHAP-screened candidates → a **pinned
production set of 8**, each placed in a **publication-lag tier** that fixes what information is
legitimately available at month-end T.

```
factors.py REGISTRY (38 live)
        │  candidate=True flag
        ▼
SHAP pre-2015 screen (look-ahead-free: screens on pre-backtest data only)
        ▼
PINNED = 8 production factors  →  build_matrix(): resample('ME').last() then shift(pub_lag)
```

**Publication-lag tiers** — `pub_lag` is the number of months a series is shifted so a month-T row
never uses information published after T:

| Tier | pub_lag | Meaning | Pinned factors |
|------|---------|---------|----------------|
| **0 — contemporaneous** | 0 | Market / financial prices, available the day they print, weeks before the CPI release | `oil_brent` (logret), `gas_eu` (logret), `imf_all_commodity` (logret), `deep_sea_freight` (logret), `mpc_rate_change` (level), `ofgem_cap_delta` (diff) |
| **1 — monthly ONS** | 1 | ONS monthly statistics, ~1 month behind | `uk_ppi_input` (yoy) — input PPI, the LGBM workhorse |
| **2 — quarterly** | 2 | First preliminary quarterly estimate, ~6 weeks behind | `uk_quarterly_gdp` (yoy) |

The two factors that **earned** their pin in the factor race were `uk_ppi_input` and
`deep_sea_freight` (top-2 SHAP of the pinned set; univariate rel-RMSE 0.93 / 0.95). The same
pub-lagged monthly matrix feeds both overlay members; AA uses CPI through the last released month.

**Under review (2026-08-19).** `uk_quarterly_gdp` is the only pinned factor that can block a
feature row outright and kill the TVP leg — its FRED source has slipped to a 2026-01 last
observation. [`code/new_factors/ablation.py`](code/new_factors/ablation.py) puts its accuracy
value at nothing: rel-RMSE **0.9201 with it vs 0.9162 without** (ΔRMSE −0.0018, DM p=0.13).
Drop / re-budget / re-source is an open governance choice. A new candidate, `uk_pump_price`
(DESNZ weekly road-fuel prices, monthly mean), tracks CPI motor fuels at **corr 0.993** where
the pinned `oil_brent` is crude — it is a candidate only until the factor race and gate run.

**Residual decomposition** (what the factors actually explain): PPI cost-push dominates calm
months, the administered **Ofgem price cap** dominates shock months, spot energy is minor — and
**~74% of the residual is unexplained** (food/services/idiosyncratic, outside the factor set).

---

## Robustness: walk-forward, purge & embargo

The model is evaluated and trained out-of-sample with explicit leakage controls.

- **Expanding-window walk-forward.** AutoARIMA expands from 2001; the overlay residual history
  expands from its vintage start. Evaluation window 2015–2024; **2025+ is a blind hold-out, never
  used to fit or tune.**
- **López-de-Prado purge + embargo — in backtests** ([`code/validation.py`](code/validation.py),
  `purge_embargo` / `embargo_series`). The residual target `cpi_yoy` is a **12-month** difference,
  so training rows within 12 months of a *test* month share its YoY window. Walk-forwards therefore
  **purge `PURGE_HORIZON = 12` months and embargo a further `EMBARGO = 1`** (13 months total),
  eliminating autocorrelation / regime-shift leakage a naïve cutoff would admit.
- **The live path does not purge (2026-08-19).** Every residual up to the last released CPI month
  is observable in real time, so purging them in a live nowcast discards real information rather
  than preventing leakage. `_lgbm_resid` takes explicit `purge_horizon`/`embargo` arguments;
  backtests pass them, production does not. The trade-off is stated in the changelog: the overlay
  now sees 12 heavily autocorrelated months and can extrapolate recent residual momentum.
- **Information boundary.** Every factor is `resample('ME').last()` then `shift(pub_lag)`, so a
  month-T row uses only data ≤ T-end. **No post-month-end and no post-release data enters; the
  leakage audit reports 0 violations.** This is a reference-month nowcast, **not** a release-day or
  T-30 product.
- **Freshness guards (2026-08-19).** A stale CSV drop-in used to shadow a fresher live fetch, and a
  regulatory factor's missing months were filled with `0` — i.e. "no cap reset" in the month of a
  +13% reset. Now: drop-ins are authoritative only for the span they cover and are spliced forward
  from the live source; reg factors zero-fill only inside their observed span; and every run
  reports `stale_factors` and `dead_legs`. Each run also writes a manifest to
  `data/production/runs/<target>_<YYYY-MM>.json` — the factor row the overlays saw, per-factor
  source status and last observation, and the git SHA — because the live matrix moves continuously
  and a past run cannot otherwise be reconstructed.

---

## The models — walk-forward RMSE

Two-stage members and the wider comparison set, walk-forward 2015–2024 (corrected 2026-06-07:
look-ahead-free SHAP screen, 38 factors; `n≈112`, AR(1) baseline `n=120`).

| Model | RMSE | Role | In production |
|-------|------|------|---------------|
| **LGBM** (cost-push overlay) | **0.443** | Lowest-RMSE member; stable PPI map | ✅ |
| AutoARIMA (anchor) | 0.467 | ~96% of the level | ✅ |
| TVP (shock overlay) | 0.482 | Diversifier (low corr) | ✅ |
| **AR(1) baseline** | **0.495** | benchmark | — |
| MIDAS | 0.554 | redundant cost-push clone | ❌ dropped |
| BVAR | 0.678 | redundant (0.91 corr w/ LGBM) | ❌ dropped |
| RegimeEns ⚠ | 1.202 | 2020-21 COVID blow-up | ❌ rejected |

**Combined forecast (production):** rel-RMSE ≈ **0.89** vs AutoARIMA on the full 2015–2024 sample
(λ=0.5; ≈0.87 at λ=1). The full-sample improvement is **shock-concentrated (2022/23) and
statistically insignificant — Diebold–Mariano p ≈ 0.17.** Treat the edge as modest and unproven
out-of-sample, not established.

---

## Live nowcasts (prospective record)

Scored from the genesis month forward in [`data/live_scorecard.csv`](data/live_scorecard.csv) →
[docs/live_report.md](docs/live_report.md). Each release logs the anchor, the production model, the
λ=1 experimental overlay, external consensus and a UCL comparison, then the realised actual.

| release | AA | **final (prod)** | exp overlay (λ=1) | consensus | UCL | **actual** | prod error |
|---------|----|----|----|----|----|----|----|
| 2026-05 (genesis) | 2.71 | **2.91** | 3.11 | 3.00 | 3.05 | **2.80** | **+0.11** |
| 2026-06 | 2.73 | **2.68** | 2.64 | 2.70 | — | **2.60** | **+0.08** |
| 2026-07 | 2.52 | **2.56** | 2.61 | — | — | **2.90** | **−0.34** |

Cumulative over three releases: final production RMSE **0.211** vs AutoARIMA **0.239**.

**Honest genesis read.** May-2026 was a calm / base-effect month — exactly the documented failure
mode. The production model (2.91) beat consensus (3.00), UCL (3.05) and the prior production build,
and the λ=0.5 haircut halved the λ=1 overlay's error (3.11 → 2.91)... but **AutoARIMA alone (2.71)
was the single best forecast.**

**July-2026 — the miss that drove the 2026-08-19 changes.** A 13% Ofgem cap rise on 1 July put CPI
gas **+14.7%** and electricity **+3.6%** on the month; the model had no channel for it and printed
2.56 against 2.90. The [post-mortem](docs/postmortem_2026_07.md) found three separate causes: the
cap factor carried no weight, its CSV had gone stale and was silently read as "no change", and the
TVP leg had been forecasting a 13-month-stale month since the embargo landed.

**These rows are AS-RUN and stay that way.** [`restate.py`](code/production/restate.py) re-runs past
months through the current pipeline for comparison — headline RMSE **0.096 restated vs 0.211 as-run**
over the three scored months (July: 2.768 vs 2.56) — but a restatement uses today's revised factor
data, so it judges changes, it does not rewrite the record.

**Decision gate:** after ~12 prospective releases, judge final-production vs **AutoARIMA-only**.
If it does not beat AA live, demote the overlay and ship AA alone. Note the record now has a
**pipeline discontinuity**: May–Jul 2026 were produced by the pre-2026-08-19 build, and Aug-2026
onward scores the current one (TVP fix + RegEvent + unpurged live path). The gate counts releases
from genesis regardless — a changed model does not reset the clock, it just means the early rows
and the later rows are not the same estimator.

---

## Run

> ### ⚠️ ONLY use the `quant` conda env — never `base` or the project `.venv`
>
> On this Mac the project `.venv` (and conda `base`) hit **pathologically slow Python imports**
> (`import pandas` takes 10–16 min; runs appear to hang). The `quant` conda env
> (`/opt/anaconda3/envs/quant`) imports the full stack in ~3.5 s. **Always run with `quant`.**
> It has every dependency (`lightgbm` + `fredapi` were added 2026-07-22; `pmdarima` is not needed).

```bash
Q=/opt/anaconda3/envs/quant/bin/python          # the ONLY interpreter to use here

export FRED_API_KEY=your_key_here               # or: set -a; . ./.env; set +a
# LSEG_APP_KEY + the Workspace desktop app are needed for the CORE overlay's PMI / swap /
# profits factors; without them core degrades to a reduced factor set (headline is unaffected).

# Production nowcast for the first unreleased reference month
PYTHONPATH=code $Q -u code/production/model.py

# Append the latest release to the live scorecard, then regenerate the report
$Q code/production/update_live_scorecard.py
$Q code/production/generate_live_report.py

# Re-run past reference months through the current pipeline (comparison, not the record)
PYTHONPATH=code $Q -u code/production/restate.py 2026-05 2026-06 2026-07

# Ablate a PINNED factor: purged walk-forward with and without it
PYTHONPATH=code $Q -u code/new_factors/ablation.py uk_quarterly_gdp

# Tests
$Q -m pytest code/tests/ -q
```

| Path | What |
|------|------|
| [`code/production/model.py`](code/production/model.py) | `Forecast = AA + 0.25·TVP + 0.25·LGBM + RegEvent` |
| [`code/production/reg_events.py`](code/production/reg_events.py) | Ofgem cap arithmetic: pass-through β, κ haircut, self-updating cap series |
| [`code/production/restate.py`](code/production/restate.py) · [`code/new_factors/ablation.py`](code/new_factors/ablation.py) | Restate past months · ablate a pinned factor |
| [`code/validation.py`](code/validation.py) | `purge_embargo`, `embargo_series` (López-de-Prado controls) |
| [`code/factors.py`](code/factors.py) | Factor registry, `PINNED` set, pub-lag application |
| [`code/production/update_live_scorecard.py`](code/production/update_live_scorecard.py) · [`generate_live_report.py`](code/production/generate_live_report.py) | Prospective scoring |

---

## Governance

Frozen 2026-06-19. Members (AA, TVP, LGBM), weights (equal overlay), λ=0.5 and the RegEvent
parameters (κ, β estimator, scope) change only by a **governance decision**, not a code edit; the
post-freeze changelog lives in [docs/final_model.md](docs/final_model.md) §11 — purge+embargo
(2026-06-24), then the 2026-08-19 round: TVP prediction-month fix, live path unpurged, RegEvent
layer, freshness guards, run manifests, and the NaN-leg fix. Inputs are accredited ONS
series and market prices (no revision/leakage); the model is research / decision-support reported
**alongside** AutoARIMA, not a standalone trading signal.
