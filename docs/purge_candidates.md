# Purge candidates (RECOMMENDATIONS ONLY — nothing deleted)

## Classification key: production / research-keep / deprecate / orphaned

### KEEP — production
- `code/new_factors/two_stage.py` (canonical), `code/uk_model_zoo.py`, `code/factors.py`.

### KEEP — research record (don't run in prod, but evidentially valuable)
- `code/new_factors/{factor_race,alloc_sweep,weight_sweep,compare_factors,leakage_audit,shap_pinned}.py`
- `code/timing/{production_asof,nowcast_window,reconcile}.py`
- `code/reg_detect/*` (documented null results).

### DEPRECATE — superseded / duplicate functionality
- `code/sweep_residual_regime.py` — residual+regime sweep superseded by factor_race + alloc_sweep.
- `code/sweep_factors.py` — pre-freeze factor screen, superseded by factor_race.
- `code/resid_target_compare.py` — ad-hoc, **untracked**, superseded by factor_race.
- `code/retrain_pinned.py` — ad-hoc, **untracked**.
- `code/main.py` — 13-model zoo backtest; useful as a blind-test harness but NOT the
  production path. Keep behind a clear "legacy/research" label or move to `code/legacy/`.
- `code/timing/horizon_backtest.py` + `may2026_path.py` — built on the INTRAMONTH stack
  (different model). Superseded by `production_asof.py`. Deprecate to avoid the exact
  invalid comparison that caused the contradiction.

### DUPLICATE forecasting-stack risk (the root cause of the confusion)
- **MIDAS**: production uses `Z.MIDAS` (U-MIDAS); intramonth aliases `MIDAS`→`ElasticNet`
  in `stack.py` (`_zoo_class` alias). Two different "MIDAS" under one name. RECOMMEND:
  rename the intramonth alias to `ElasticNetHF` so it can never be mistaken for production MIDAS.
- **Factor set**: intramonth `config.MONTHLY_FACTORS` diverged from production `PINNED`
  (missing uk_ppi_input/deep_sea_freight; still has mpc_vote_split/budget_event that
  production dropped). RECOMMEND: either align intramonth to PINNED, or clearly mark the
  intramonth stack as a frozen legacy experiment.
- **AutoARIMA**: one class, but production and intramonth feed it different samples/vintages
  → different baselines (RMSE 0.4687 vs 0.4402). RECOMMEND: a single shared backtest
  harness so AA is computed identically everywhere.

### ORPHANED / SHELVED
- `code/run_dashboard.py`, `code/dashboard/*` — shelved Streamlit; move out of `code/` root.
- `refs/` (untracked) — stray.

### NOT a CPI forecaster (leave; separate product)
- `code/rates/*` — rates-repricing pipeline (2Y gilt). Independent objective.

## Highest-value single action
Make `code/new_factors/two_stage.py` the **only** thing called "the model," and demote the
intramonth point-forecast stack to clearly-labelled legacy. The intramonth regime/scenario/
HMM layers were already shown decorative (reg_detect nulls) — keep for commentary only, not
as an alpha or as a second "production" forecaster.
