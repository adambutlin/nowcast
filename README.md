# nowcast — UK CPI YoY two-stage nowcast

A **reference-month nowcast of UK CPI all-items YoY** (ONS D7G7). The model completes at
month-end **T** using only information dated **≤ T**; the official ONS print lands at **T+15…T+21**.
The research phase is **closed and frozen** (2026-06-19) — the model is now evaluated
prospectively, not modified. Everything below is the live specification; there is no hidden
sauce, and the honest edge is small (see [§Performance](#the-models--walk-forward-rmse)).

**Entry point:** [`code/production/model.py`](code/production/model.py) · **Spec:**
[docs/final_model.md](docs/final_model.md) · **Model card:** [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
· **Live record:** [docs/live_report.md](docs/live_report.md)

---

## The multi-layer architecture

The forecast is a univariate anchor plus a **shrunk residual overlay** — three models stacked
in two stages, never a single black box.

```
Forecast = AA + λ · Overlay,        λ = 0.5
Overlay  = 0.5 · TVP + 0.5 · LGBM
=>  Forecast = AutoARIMA + 0.25 · TVP(resid) + 0.25 · LGBM(resid)
```

| Layer | Model | Role | What it is *not* |
|-------|-------|------|------------------|
| **1 — anchor** | AutoARIMA on CPI YoY (univariate) | Level: persistence, seasonality, mean reversion, base-effect arithmetic. **≈96% of the print.** | — |
| **2a — overlay** | TVP (time-varying-parameter regression on the residual) | Shock pass-through; the genuine **diversifier** (error-corr ≈0.69 vs ≈0.9 among the rejects). | a standalone forecaster — it loses to AA outside shock windows |
| **2b — overlay** | LightGBM on the AA residual | Stable nonlinear **PPI / cost-push** map. Lowest-RMSE member; beats AA in 6/6 rolling 5y windows. | a rich multi-factor learner — ~90% of its edge is `uk_ppi_input` |
| **3 — shrinkage** | λ = 0.5 | Magnitude haircut on the overlay. | a regime switch |

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

Conclusion: the fixed average is the answer; **magnitude shrinkage (λ), not regime-switching,
is the only defensible "regime" adjustment.** Rejected code is retained as research context only.

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

**Residual decomposition** (what the factors actually explain): PPI cost-push dominates calm
months, the administered **Ofgem price cap** dominates shock months, spot energy is minor — and
**~74% of the residual is unexplained** (food/services/idiosyncratic, outside the factor set).

---

## Robustness: walk-forward, purge & embargo

The model is evaluated and trained out-of-sample with explicit leakage controls.

- **Expanding-window walk-forward.** AutoARIMA expands from 2001; the overlay residual history
  expands from its vintage start. Evaluation window 2015–2024; **2025+ is a blind hold-out, never
  used to fit or tune.**
- **López-de-Prado purge + embargo** ([`code/validation.py`](code/validation.py),
  `purge_embargo` / `embargo_series`). The residual target `cpi_yoy` is a **12-month** difference,
  so the trailing months around the nowcast share its YoY window. Training therefore **purges
  `PURGE_HORIZON = 12` months and embargoes a further `EMBARGO = 1`** (13 months total) before the
  nowcast month, eliminating the autocorrelation / regime-shift leakage that a naïve cutoff would
  admit. Adopted 2026-06-24; it moved the June-2026 nowcast **2.871 → 2.686** as 2025-Q1 residuals
  legitimately pulled the TVP coefficients down.
- **Information boundary.** Every factor is `resample('ME').last()` then `shift(pub_lag)`, so a
  month-T row uses only data ≤ T-end. **No post-month-end and no post-release data enters; the
  leakage audit reports 0 violations.** This is a reference-month nowcast, **not** a release-day or
  T-30 product.

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

**Honest genesis read.** May-2026 was a calm / base-effect month — exactly the documented failure
mode. The production model (2.91) beat consensus (3.00), UCL (3.05) and the prior production build,
and the λ=0.5 haircut halved the λ=1 overlay's error (3.11 → 2.91)... but **AutoARIMA alone (2.71)
was the single best forecast.** One adverse point; the forward record decides.

**Decision gate:** after ~12 prospective releases, judge final-production vs **AutoARIMA-only**.
If it does not beat AA live, demote the overlay and ship AA alone.

---

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy yfinance lightgbm shap scikit-learn statsmodels \
            fredapi requests openpyxl dbnomics pytest scipy xgboost
export FRED_API_KEY=your_key_here

# Production nowcast for the first unreleased reference month
python -m code.production.model

# Append the latest release to the live scorecard, then regenerate the report
python code/production/update_live_scorecard.py
python code/production/generate_live_report.py

# Tests
.venv/bin/python -m pytest code/tests/ -q
```

| Path | What |
|------|------|
| [`code/production/model.py`](code/production/model.py) | Frozen `Forecast = AA + 0.25·TVP + 0.25·LGBM` |
| [`code/validation.py`](code/validation.py) | `purge_embargo`, `embargo_series` (López-de-Prado controls) |
| [`code/factors.py`](code/factors.py) | Factor registry, `PINNED` set, pub-lag application |
| [`code/production/update_live_scorecard.py`](code/production/update_live_scorecard.py) · [`generate_live_report.py`](code/production/generate_live_report.py) | Prospective scoring |

---

## Governance

Frozen 2026-06-19. Members (AA, TVP, LGBM), weights (equal overlay) and λ=0.5 change only by a
**governance decision**, not a code edit; the post-freeze changelog (incl. the purge+embargo
adoption) lives in [docs/final_model.md](docs/final_model.md) §11. Inputs are accredited ONS
series and market prices (no revision/leakage); the model is research / decision-support reported
**alongside** AutoARIMA, not a standalone trading signal.
