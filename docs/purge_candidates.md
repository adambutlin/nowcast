# Purge candidates (RECOMMENDATIONS ONLY — nothing deleted)

Final governance review (2026-06-19). Classification in `architecture_inventory.md`.
No code is deleted; this records status + recommended action.

## Keep — production
`code/production/*`, `code/uk_model_zoo.py`, `code/factors.py`,
`code/new_factors/two_stage.py` (imported by production).

## Keep — research evidence (don't run in prod)
`code/new_factors/{factor_race,alloc_sweep,weight_sweep,compare_factors,leakage_audit,shap_pinned}.py`,
`code/timing/*` (all audits), `code/reg_detect/*` (null results).

## Recommend DEPRECATE (mark legacy; optionally move to `code/legacy/`)
- `code/intramonth/*` — different model (ElasticNet-MIDAS alias, no ppi/freight; regime/scenario
  decoration shown non-predictive). Root cause of the earlier invalid comparison.
  **Action:** rename its MIDAS alias to `ElasticNetHF`; clearly label the package "legacy/research".
- `code/timing/horizon_backtest.py`, `code/timing/may2026_path.py` — built on the intramonth
  stack (invalid vs production). Superseded by `production_asof.py` / `production/model.py`.
- `code/main.py`, `code/sweep_residual_regime.py`, `code/sweep_factors.py` — pre-freeze
  harnesses; keep as blind-test/legacy, not on the production path.
- `code/plot_aa_residuals.py`, `code/plot_nowcast_history.py` — legacy diagnostics.

## Recommend ARCHIVE (orphaned / untracked)
- `code/resid_target_compare.py`, `code/retrain_pinned.py` — ad-hoc, untracked; superseded.
- `code/run_dashboard.py`, `code/dashboard/*` — shelved Streamlit; move out of `code/` root.
- `refs/` — stray untracked dir.

## Duplicate-stack hygiene (the confusion source — fix labelling, not behaviour)
- "MIDAS" means two things: production `Z.MIDAS` (U-MIDAS) vs intramonth alias→`ElasticNet`.
  **Action:** rename the alias so the names can't be confused.
- intramonth `config.MONTHLY_FACTORS` diverged from production `PINNED`. **Action:** mark
  intramonth frozen-legacy; single source of truth for factors = `factors.py` PINNED via
  `production/model.py`.

## Leave — separate product
`code/rates/*` — rates-repricing pipeline (independent objective).

## Highest-value single action
There is now exactly one thing called "the model": `code/production/model.py`. Everything else
is research or legacy. Keep it that way.
