# Design: nowcast_cpi Expansion — Gas Factor, Model Zoo, Rolling Windows

*Date: 2026-06-05*

---

## 1. Scope

Extend the UK CPI nowcasting system across eight areas:
1. Add `gas_eu` (European gas price) as a factor replacing `gas_hh` in UK runs
2. Rename `compare_uk.py` → `nowcast_cpi.py`; delete `ramm_lgbm_uk_v1.py`
3. Debug `nowcast_cpi.py` systematically (fix nowcast-extraction dropna bug + others)
4. Complete PROCESS.md pending code items (excluding CSV drop-ins): k=2 DFM, ElasticNet, post-1992 vs post-2005 window comparison, Shapley screening, `--rmc` run
5. Add rolling-window variants (5-year and 2-year) for **all 10 models** — 30 variants total in comparison table, logic in the backtest runner
6. Full zoo retrain from scratch after all code fixes are in place
7. Shapley factor screening post-retrain; tighten model gate to 1.0× AR(1) (must beat AR(1), not merely approach it)
8. Push all changes to `adambutlin/nowcast`

---

## 2. Factor: `gas_eu`

**Source:** FRED `PNGASEUUSDM` — IMF Primary Commodity Prices: Natural Gas, Europe (USD/mmBtu, monthly, 1960–present).

**Registry entry:**
```python
"gas_eu": dict(
    fetch=lambda: _fred("PNGASEUUSDM"), transform="logret",
    pub_lag=0, candidate=True, csv="gas_eu.csv",
    note="IMF/FRED European natural gas price (PNGASEUUSDM, USD/mmBtu). "
         "UK imported LNG proxy. History from 1960. pub_lag=0. "
         "TTF front-month futures preferred post-2009 — use CSV drop-in when available.")
```

**Rationale:** Henry Hub (`gas_hh`) is a US market price with weak transmission to UK CPI. European NBP/TTF drives UK household energy costs. `PNGASEUUSDM` is the best free proxy with full post-1992 history. `gas_hh` stays in the registry (tagged `region=US`) but is excluded from UK-only runs. When a post-2009 TTF CSV becomes available, it overrides `gas_eu` automatically via the CSV drop-in mechanism.

**Pub_lag:** 0 — IMF commodity prices are spot prices reported for the reference month, available before UK CPI release.

---

## 3. File Operations

| Action | File | Details |
|--------|------|---------|
| Rename | `compare_uk.py` → `nowcast_cpi.py` | No content change on rename; bugs fixed in debug phase |
| Delete | `ramm_lgbm_uk_v1.py` | Confirmed: no imports anywhere; superseded by `uk_model_zoo.py` |
| No change | `ramm_lgbm_v1.py` | US Core CPI MoM model; out of scope this session |
| No change | `test_ramm_lgbm_v1.py` | Covers US model only |

---

## 4. Debug: `nowcast_cpi.py`

Systematic read of `compare_uk.py` targeting:

- **Nowcast extraction** — same class of bug fixed in `ramm_lgbm_v1.py`: `data.dropna()` must not swallow the most-recent-feature row before `latest_x` is captured
- **Publication-lag wiring** — `apply_publication_lags()` called before all model runs, not inside individual models
- **`all_models()` list** — consistent with zoo class list after new classes are added
- **`combine_dynamic()` edge cases** — rolling-12m window behaviour at the start of the backtest
- **`score_backtest()` / `dm_test()`** — correct error orientation (e1 = model, e2 = AR1)

All bugs found are fixed before running any backtest.

---

## 5. PROCESS.md Code Items

### 5.1 Re-run backtest + pub-lag fix
Run after all code changes are in place:
```bash
FRED_API_KEY=<key> .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --train-from 1992
```

### 5.2 `--rmc` run
```bash
FRED_API_KEY=<key> .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --train-from 1992 --rmc
```

### 5.3 k=2 DFM

Add `k` parameter to `DFM.__init__` (default `k=1` for backward compatibility). Expose via `--dfm-k 2` CLI flag. The two latent factors are expected to separate global-risk (VIX/GBP) from domestic-services (rents/vacancies) regimes.

### 5.4 ElasticNet model

New class `ElasticNet` in `uk_model_zoo.py`. Uses `sklearn.linear_model.ElasticNetCV` with 5-fold CV to select alpha and l1_ratio from a grid. AR-augmented (`cpi_lag1 = target.shift(1)`). Expected RMSE between BVAR (0.591) and LSTAR (0.492).

### 5.5 Post-1992 vs post-2005 training window

`--train-from` flag already exists in `nowcast_cpi.py`. Add side-by-side output table comparing metrics for `--train-from 1992` vs `--train-from 2005` in the same run. Flag: `--compare-windows`.

### 5.6 Shapley factor screening

Add `screen_candidates(df, target, threshold=0.001)` to `factors.py`:
- Fit a quick LightGBM on the full factor matrix
- Use `shap.TreeExplainer` to get mean absolute SHAP values per factor
- Return candidate factors above `threshold`; drop the rest
- Called in `nowcast_cpi.py` before model runs when `--shap-screen` flag is set

### 5.7 Push to GitHub
After all runs pass and results are recorded in STATE.md:
```bash
git add -A && git commit -m "..." && git push origin main
```

---

## 6. Rolling-Window Variants — All Models

Every model in the zoo gets two additional rolling-window variants: 5-year (60 months) and 2-year (24 months). This produces 3 × 10 = 30 model variants in the comparison table alongside AR(1).

**Architecture:** Rolling-window logic lives in the backtest runner (`nowcast_cpi.py`), not in individual model classes. `BaseModel` gets a `WINDOW` class attribute (default `None` = expanding). The runner reads `model.WINDOW` and slices training data accordingly before calling `_fit_predict_year(train, test, factors, target)`.

```python
# In BaseModel
WINDOW = None  # None = expanding; int = rolling window in months

# In backtest runner, per model per year:
if model.WINDOW is None:
    train = data[data.index < test_start]
else:
    cutoff = test_start - pd.DateOffset(months=model.WINDOW)
    train = data[(data.index >= cutoff) & (data.index < test_start)]
    if len(train) < 24:
        train = data[data.index < test_start]  # fall back to expanding
```

**New classes** (one pair per existing model, inheriting everything):
```
DFM_Rolling5Y / DFM_Rolling2Y
RAMM_LGBM_Rolling5Y / RAMM_LGBM_Rolling2Y
UCM_Rolling5Y / UCM_Rolling2Y
TVP_Rolling5Y / TVP_Rolling2Y
HMM_Rolling5Y / HMM_Rolling2Y
MS_DFM_Rolling5Y / MS_DFM_Rolling2Y
LSTAR_Rolling5Y / LSTAR_Rolling2Y
BVAR_Rolling5Y / BVAR_Rolling2Y
HiddenRF_Rolling5Y / HiddenRF_Rolling2Y
GBM_Rolling5Y / GBM_Rolling2Y
```

All 30 classes added to `all_models()`. Combined ensembles, DM tests, and factor importance tables all pick them up automatically.

**Full zoo retrain:** After the nowcast-extraction (`dropna`) bug is fixed in `nowcast_cpi.py` and all code changes are in place, retrain the entire zoo from scratch. This is the canonical post-fix result set.

## 7. Factor Screening and Model Gating

### 7.1 Shapley Factor Screening

After full zoo retrain, run `screen_candidates(df, target, threshold)` from `factors.py`:
- Fit a quick LightGBM on the full factor matrix
- `shap.TreeExplainer` → mean absolute SHAP value per factor
- Factors below threshold are dropped from the live factor list for subsequent runs
- Threshold is a parameter (default 0.001); results logged to console

### 7.2 Model Gate: Must Beat AR(1)

Gate tightened from 1.5× to **1.0× AR(1) RMSE**. Any model (expanding or rolling) with RMSE ≥ AR(1) RMSE is:
- Excluded from `Combined-Static`, `Combined-Dynamic`, `Combined-Superstar`
- Flagged in the comparison table as "below AR(1)"
- Dropped from `Combined-Absolute` greedy subset (was already using an RMSE gate, now stricter)

This replaces the previous 1.5× gate used in `greedy_uncorrelated_subset()`.

---

## 8. Success Criteria

- [ ] `gas_eu` loads from FRED, appears in factor matrix, picked up by all models
- [ ] `nowcast_cpi.py` runs end-to-end with `--start 2015 --train-from 1992` without errors
- [ ] All 30 model variants (10 × expanding + 5y + 2y) appear in comparison table
- [ ] Models with RMSE ≥ AR(1) are flagged and excluded from combined ensembles
- [ ] `--rmc` run completes, saves `rmc_*_perf.csv` files
- [ ] `ElasticNet` class passes backtest loop
- [ ] Shapley factor screening runs post-retrain, weak candidates dropped
- [ ] STATE.md updated with new results
- [ ] All changes pushed to `adambutlin/nowcast`
