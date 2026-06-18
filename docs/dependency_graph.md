# Dependency graph — every CPI-forecast code path

## Shared canonical libraries (single source of truth)
- **`code/uk_model_zoo.py`** — ALL model classes: `AutoARIMA`, `BVAR`, `TVP`, `MIDAS`
  (U-MIDAS), + 18 others. No duplicate model implementations anywhere; every stack imports
  these. The intramonth "different MIDAS" is an *alias* (`MIDAS`→`ElasticNet`) in stack.py,
  not a second class.
- **`code/factors.py`** — single factor REGISTRY + `build_matrix` / `apply_publication_lags`.

## Diagram
```
factors.py (REGISTRY) ──┐
uk_model_zoo.py ────────┤
                        ▼
   ┌──────────────── PRODUCTION ────────────────┐
   │ code/new_factors/two_stage.py              │  <-- CANONICAL. rel_rmse 0.93
   │   AA + 0.375 BVAR + 0.25 TVP + 0.375 MIDAS │      out: data/new_factors/{backtest,
   │   PINNED incl uk_ppi_input, deep_sea_freight│      metrics,nowcast}.csv
   └────────────────────────────────────────────┘
        ▲ reused by (RESEARCH, new_factors/):
          alloc_sweep, compare_factors, factor_race, leakage_audit, shap_pinned, weight_sweep
        ▲ reused by (AUDIT, timing/):
          horizon_backtest*, nowcast_window, reconcile, production_asof, may2026_path
          (*horizon_backtest uses the INTRAMONTH stack, not production — see reconciliation)

   ┌──────────── SEPARATE STACK: intramonth/ ────────────┐
   │ run.py→stack.py (baseline=AutoARIMA, factor=BVAR,    │  DIFFERENT model:
   │ regime_tvp=TVP, intramonth=MIDAS→ElasticNet ALIAS)   │  - ElasticNet not U-MIDAS
   │ panel.py (as-of HF) hf_data.py weights.py regime.py  │  - config factors (no ppi/freight;
   │ scenarios.py attribution.py targets.py config.py     │    has vote_split/budget)
   │ ensemble_review.py evolution.py                      │  - sample 2012-24, as-of HF
   └──────────────────────────────────────────────────────┘  out: data/intramonth/*, scenarios/*

   ┌──────────── reg_detect/ (NULL-RESULT RESEARCH) ─────┐
   │ build_targets, detectors, observable_shock,         │  HelpfulStage2 / ObservableShock
   │ architecture, plots  (reuse two_stage + factors)    │  -> all negative; kept as record
   └──────────────────────────────────────────────────────┘

   ┌──────────── LEGACY (code/ root) ────────────────────┐
   │ main.py (13-model zoo backtest), sweep_factors.py,  │  superseded by two_stage +
   │ sweep_residual_regime.py, resid_target_compare.py,  │  factor_race; pre-freeze research
   │ retrain_pinned.py, plot_aa_residuals.py,            │
   │ plot_nowcast_history.py                             │
   └──────────────────────────────────────────────────────┘

   ┌──────────── SEPARATE PRODUCT: rates/ ───────────────┐
   │ run_production.py→production.py, signal.py, gates,  │  rates-repricing (2Y gilt), NOT a
   │ stage1, model_sweep, regime, risk, event_panel ...  │  CPI forecaster. Independent.
   └──────────────────────────────────────────────────────┘

   ┌──────────── SHELVED: dashboard ─────────────────────┐
   │ run_dashboard.py, code/dashboard/*                  │  Streamlit; not in pipeline.
   └──────────────────────────────────────────────────────┘
```

## Entry points that PRODUCE a CPI forecast
- Production nowcast: `python code/new_factors/two_stage.py` (CANONICAL).
- Legacy multi-model: `python code/main.py` (13-model backtest, blind-test harness).
- Intramonth path: `python -m intramonth.run` (different model — research).
- Audits: `code/timing/*`, `code/new_factors/{leakage_audit,factor_race,...}` (reuse production).
