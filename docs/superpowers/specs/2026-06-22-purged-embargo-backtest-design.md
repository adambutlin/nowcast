# Purged + Embargoed Walk-Forward Backtest — Design

Date: 2026-06-22
Status: approved

## Problem

The production forecaster is `AA + 0.25·TVP + 0.25·LGBM`, where the TVP and LGBM
overlays are trained on the AutoARIMA-residual history `resid = cpi_yoy − AA`. It is
evaluated by the walk-forward backtest in `code/new_factors/two_stage.py:backtest`,
which calls each Stage-2 member's `BaseModel.backtest`
(`code/uk_model_zoo.py`).

That backtest is strictly causal at the year boundary (train = `year < yr`), **but
train and test abut with no gap**. Because the target `cpi_yoy` is a 12-month
difference, the last ≤12 training months share YoY information with the first test
months. This autocorrelation overlap leaks information across the train/test
boundary and optimistically inflates the OOS metrics written to
`data/new_factors/backtest.csv` / `metrics.csv`.

(The model-zoo `ElasticNetCV(cv=5)` plain KFold is a separate, known artifact and is
explicitly out of scope — it is not part of the production model.)

## Goal

Make the **production model's walk-forward backtest** honest by adding:

- **Purging** — when a test fold is defined, drop every training point whose
  label-horizon (the 12-month YoY window) overlaps the test fold.
- **Embargoing** — add a small additional gap (1 month) after the (purged) training
  data ends and before the test fold begins, so regime-shift information cannot leak
  via residual autocorrelation.

Defaults: `horizon = 12`, `embargo = 1` (monthly data, YoY target).

## Scope decisions

1. **Backtest only, not live nowcast.** In live `nowcast()` there is no future test
   label to purge against — residuals up to the last release are genuinely known and
   legitimately usable; purging them would discard real information rather than
   prevent leakage. Purge/embargo's job is to make the *backtest* honest. The live
   path is left unchanged, with the rationale documented in code.

2. **Default-off in the model zoo.** `BaseModel.backtest` gains optional
   `purge_horizon=0, embargo=0` parameters. With the defaults, behavior is
   byte-identical to today, so every existing zoo artifact and the ElasticNetCV
   path are unaffected. Purge/embargo is turned ON only for the production Stage-2
   member backtests.

## Components

### 1. `code/validation.py` (new) — reusable splitter helper

```
purge_embargo(train, test_start, horizon=12, embargo=1) -> DataFrame
```

- `train`: training DataFrame indexed by month-end timestamps.
- `test_start`: first timestamp of the test fold.
- Returns `train` with every row whose date is within `horizon + embargo` months
  before `test_start` removed. Concretely: keep rows where
  `row_date <= test_start - DateOffset(months=horizon + embargo)`.
- Pure index logic; no model dependency; independently unit-testable.
- `horizon=0, embargo=0` is the identity (returns `train` unchanged).

### 2. `BaseModel.backtest` — apply the helper

Add `purge_horizon=0, embargo=0` params. For each test year, after building the
`train` slice (expanding or rolling) and before `_fit_predict_year`, apply
`purge_embargo(train, test_start, purge_horizon, embargo)`. The existing
`len(train) < min_train` guard then naturally skips folds left with too little data.

### 3. `two_stage.py` — turn it on for production

- Add module constants `PURGE_HORIZON = 12`, `EMBARGO = 1`.
- In `backtest()`, pass `purge_horizon=PURGE_HORIZON, embargo=EMBARGO` to each
  Stage-2 member's `.backtest()` call. The AA Stage-1 backtest (univariate anchor)
  and the live `nowcast()` are unchanged.

## Testing

- **Unit** (`purge_embargo`): boundary correctness — a point exactly
  `horizon+embargo` months before `test_start` is kept; one month later is dropped;
  `horizon=0, embargo=0` is identity.
- **Leakage** (mirrors `tests/test_rates_gates.py:TestWalkForwardNoLookahead`):
  run a member `.backtest(..., purge_horizon=12, embargo=1)`, then poison training
  rows that fall *inside* the purge window, re-run, and assert the test-fold
  predictions are unchanged (the purged rows had no influence).
- **Regression**: a member `.backtest()` with default params equals the current
  output (default-off invariant).

## Out of scope

- The model-zoo `ElasticNetCV(cv=5)` KFold.
- Any change to the live `nowcast()` forecast path.
- Re-tuning weights/lambda based on the new (honest) metrics — that is a separate
  governance decision.
