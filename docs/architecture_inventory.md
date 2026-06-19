# Architecture inventory — all forecasting code (classified)

Status: **production** | **research** (keep as evidence) | **deprecated** (superseded) |
**orphaned** (ad-hoc/untracked). Single model source: `code/uk_model_zoo.py`. Single factor
registry: `code/factors.py`. (No duplicate model classes — the intramonth "MIDAS" is an alias.)

## Production
| path | role |
|---|---|
| `code/production/model.py` | **CANONICAL forecaster** — AA + 0.25 TVP + 0.25 LGBM (frozen) |
| `code/production/update_live_scorecard.py` | append/seed/actual the live scorecard |
| `code/production/generate_live_report.py` | render `docs/live_report.md` |
| `code/uk_model_zoo.py` | model classes (AutoARIMA, TVP, BVAR, MIDAS, LightGBM via sklearn/lgbm) |
| `code/factors.py` | factor registry + pub-lags + matrix builder |
| `code/new_factors/two_stage.py` | factor matrix + AA/member backtest helpers (imported by production) |

## Research (keep — evidence trail, not run in prod)
| path | role |
|---|---|
| `code/new_factors/{factor_race,alloc_sweep,weight_sweep,compare_factors,leakage_audit,shap_pinned}.py` | factor/weight/leakage audits |
| `code/timing/{production_asof,nowcast_window,reconcile,residual_lgbm,residual_decomp,diversification_audit,stability_audit,bvar_necessity,overlay_shrinkage,bayesian_shrinkage,regime_tvp_lgbm}.py` | horizon/shrinkage/decomp audits |
| `code/reg_detect/*` | HelpfulStage2 / ObservableShock null-result detectors |

## Deprecated (superseded by the frozen model / audits)
| path | reason |
|---|---|
| `code/intramonth/*` (run, stack, evolution, panel, hf_data, weights, regime, scenarios, attribution, targets, ensemble_review, config) | **different model** (ElasticNet-MIDAS alias, no ppi/freight, regime/scenario decoration shown non-predictive). Reconciliation proved it ≠ production. |
| `code/timing/horizon_backtest.py`, `code/timing/may2026_path.py` | built on the intramonth stack → invalid cross-comparison; superseded by `production_asof.py` |
| `code/main.py` | legacy 13-model zoo backtest (blind-test harness); not the production path |
| `code/sweep_residual_regime.py`, `code/sweep_factors.py` | pre-freeze sweeps; superseded by factor_race/alloc_sweep |
| `code/plot_aa_residuals.py`, `code/plot_nowcast_history.py` | legacy diagnostic plots |

## Orphaned (ad-hoc / untracked)
| path | note |
|---|---|
| `code/resid_target_compare.py` | ad-hoc residual compare (untracked) — superseded by factor_race |
| `code/retrain_pinned.py` | ad-hoc (untracked) |
| `code/run_dashboard.py`, `code/dashboard/*` | shelved Streamlit workstation |
| `refs/` | stray untracked dir |

## Separate product (not a CPI forecaster — leave intact)
| path | role |
|---|---|
| `code/rates/*` | rates-repricing pipeline (2Y gilt signal). Independent objective; no deployable edge found. |

See `docs/purge_candidates.md` for recommended (non-destructive) cleanup actions.
