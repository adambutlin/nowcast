# Design — dual headline + Core CPI nowcast

**Status:** approved design, pre-implementation. **Branch:** `core`. **Date:** 2026-07-23.

## Goal

`code/production/model.py` prints **two** nowcasts every run:

1. **Headline CPI YoY** (ONS D7G7) — the existing frozen model, unchanged.
2. **Core CPI YoY** (ex energy, food, alcohol & tobacco) — new.

Core uses the same two-stage shape (AutoARIMA anchor + shrunk residual overlay) but with a
**core-specific** pinned factor set and calibration. A core overlay ships into the printed number
**only if it clears headline-grade validation**; otherwise core prints **AA-only** (still fulfils
"print both", honestly).

## Locked decisions (from brainstorming)

- **Architecture:** anchor + *core-specific* overlay — NOT a reuse of the headline energy/cost-push
  overlay (those factors — oil, gas, PPI-input, freight — are exactly what core excludes).
- **Success bar:** full headline-grade validation — look-ahead-free factor screen, purged+embargoed
  walk-forward, Diebold–Mariano test. Overlay ships only if it beats an AA-only core baseline OOS;
  else core degrades to AA-only (explicitly approved).
- **Code structure:** generalize the pipeline by *target*; the headline path stays byte-identical.
- **Scorecard:** a **parallel** `data/live_scorecard_core.csv` (mirrors the headline scorecard) —
  keeps the frozen headline genesis record clean; the report renders both.
- **Visualizations:** a few matplotlib figures (repo `plots/` convention) to inspect the screen,
  the race, and headline-vs-core history — all in **percentage (YoY) terms**, never index levels.

## Architecture — parameterize by target

A `Target` config carries everything target-specific:

```
Target(name, pinned, overlay_weights, lambda_, aa_start, start, end)
```

Two instances: `HEADLINE` (`cpi_yoy` + today's frozen params: PINNED-8, {tvp:.5, lgbm:.5}, λ=0.5,
AA_START=2001) and `CORE` (`uk_core_cpi` + params the research phases below produce).

- **`two_stage.py`:** replace module-level `TARGET`/`PINNED` with functions accepting a `Target`
  (`load_matrix(t)`, `backtest(df, live, t)`, `nowcast(t)`), `HEADLINE` as default so existing
  behaviour is unchanged.
- **`production/model.py`:** `nowcast(target=HEADLINE)`; `main()` runs both and prints two blocks.
- **Hard regression guard:** headline output stays byte-identical — golden June nowcast = **2.683**,
  unchanged backtest RMSE. A test locks this before any core work merges.

## Phase 0 — Data (gating dependency)

`uk_core_cpi` today has `fetch=None` and no CSV. Give it a real ONS MM23 fetch (dbnomics),
identifying the correct series code by matching known prints (**Jun-2026 = 2.6, May-2026 = 2.6**,
recent 2.6–2.8 path); `data/uk_core_cpi.csv` drop-in fallback. Need history to ~2001 for the AA
expanding-window backtest; if core history is shorter, set core-specific `aa_start`/`start` and
record the reduced window. Verify core candidate factors reach back far enough for the pre-2015
screen.

## Phase 1 — Pipeline refactor

Implement the `Target` parameterization above with the headline regression test green. No behaviour
change to headline. This phase is pure refactor + test.

## Phase 2 — Factor screen + race (core)

Run the existing look-ahead-free tools with `uk_core_cpi` as target:

- `new_factors/shap_pinned.py` — pre-2015 SHAP screen over the core-relevant universe: wages
  (`uk_ashe_pay`, `uk_awg`, `uk_paye`), slack (`uk_vacancies`, `uk_monthly_gdp`, `uk_pmi_*`),
  `uk_services_cpi`, `uk_rents`, expectations (`uk_infl_swap_1y`), `gbp_eer`, `uk_ppi_output`,
  `uk_trimmed_mean_cpi`, `uk_retail_sales`.
- `new_factors/factor_race.py` — univariate rel-RMSE race on the core AA-residual → pin the core
  set. Count is race-decided, not fixed at 8.

## Phase 3 — Overlay calibration (core)

Fit TVP + LGBM on the core AA-residual. Priors = headline's equal 0.5/0.5 split + λ=0.5 governance
haircut; re-check with `weight_sweep.py` / `alloc_sweep.py` + the overlay-shrinkage reliability
logic. Keep equal-weight + a haircut unless core evidence clearly says otherwise. Document the
choice and its rationale.

## Phase 4 — Validation gate (ship / fallback)

- Purged+embargoed walk-forward (same `PURGE_HORIZON=12`, `EMBARGO=1`) of the core overlay vs an
  AA-only core baseline, 2015→embargo.
- Diebold–Mariano overlay-vs-AA. `new_factors/leakage_audit.py` on the core pinned set.
- **Gate:** ship the overlay into the printed core number only if it beats AA-only OOS (rel-RMSE < 1)
  with headline discipline; else core = AA-only. Record the decision + stats in the docs.

## Phase 5 — Integration

- `model.py main()` prints both blocks: HEADLINE (unchanged) and CORE (AA + overlay contributions
  if shipped, else AA-only, clearly labelled).
- Parallel `data/live_scorecard_core.csv` + report section (`generate_live_report.py`) tracking core
  (core_aa, core_final, core_consensus, actual). Seed with June-2026 (actual core = 2.6).
- Update `docs/final_model.md` / `MODEL_CARD.md` with the core spec + validation results.

## Phase 6 — Visualizations (matplotlib → `plots/core/`, all YoY %)

New `code/plot_core.py` (mirrors existing `plot_*.py`). Use the `dataviz` conventions for palette/
axes and the `captioning-figures` skill to bake a one-sentence narrative caption into each PNG.
`plots/*.png` is gitignored, so these are local inspection artifacts regenerated on demand.

- **F1 — Headline vs Core CPI history.** Line chart, x = date (full history), y = **YoY %**. Series:
  headline CPI YoY, core CPI YoY; shaded headline−core wedge (energy/food contribution); BoE 2% line.
  → `plots/core/cpi_vs_core_history.png`.
- **F2 — Factor screen (SHAP).** Horizontal bar chart of core candidate factors ranked by pre-2015
  mean |SHAP|; passed vs rejected colour-coded with the screen threshold marked.
  → `plots/core/core_factor_screen_shap.png`.
- **F3 — Factor race (rel-RMSE).** Horizontal bar chart of candidates by univariate rel-RMSE vs the
  AA baseline on the core residual (rel-RMSE < 1 = beats AA); pinned set highlighted.
  → `plots/core/core_factor_race_relrmse.png`.
- **F4 — Overlay validation.** Bar chart of core AA-only vs TVP / LGBM / combo (λ=0.5) walk-forward
  RMSE (the Phase-4 gate outcome, visualized). → `plots/core/core_overlay_validation.png`.

## Testing

- Headline regression: nowcast + backtest byte-identical post-refactor (golden June = 2.683).
- Unit tests for target-parameterized `load_matrix` / `nowcast`.
- Recorded core backtest RMSE + DM stats for reproducibility.
- Figure scripts run headless and emit the four PNGs without error.

## Key risks

- **Core history depth** (Phase 0): a short core series shrinks the AA backtest window. Contingency:
  core-specific `aa_start`; document the reduced sample.
- **Overlay may not beat AA** (Phase 4 gate): genuine unknown. Design degrades gracefully — core
  prints AA-only, still satisfying "both numbers".

## Deliverable

`model.py` prints both nowcasts; core is a validated core-specific overlay OR an AA-only fallback per
the gate; four inspection figures under `plots/core/`; research artifacts (screen, race, calibration,
validation) + a parallel core scorecard committed on `core`.
