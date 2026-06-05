# nowcast_cpi Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add European gas factor, rename/clean files, expand model zoo with rolling-window and ElasticNet variants, tighten the AR(1) gate to 1.0×, add Shapley factor screening, add nowcast output, and retrain the full zoo from scratch.

**Architecture:** All 10 base models + 20 rolling-window variants (WINDOW=60/24 months) live in `uk_model_zoo.py`. Rolling-window logic is in `BaseModel.backtest()` via a `WINDOW` class attribute — subclasses only need `WINDOW = N`. The runner in `nowcast_cpi.py` (renamed from `compare_uk.py`) filters combined ensembles to models beating AR(1) before combining. `factors.py` gains `gas_eu` and a `screen_candidates()` function.

**Tech Stack:** Python 3.12, lightgbm, shap, statsmodels, sklearn (ElasticNetCV), fredapi, yfinance, dbnomics, pandas, numpy, scipy, pytest.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `factors.py` | Modify | Add `gas_eu` REGISTRY entry; add `screen_candidates()` |
| `compare_uk.py` | Rename → `nowcast_cpi.py` | Full runner — fix gate, add nowcast section, update CSV names |
| `uk_model_zoo.py` | Modify | Add `WINDOW` to BaseModel; add 20 rolling classes + ElasticNet + DFM-k2 |
| `ramm_lgbm_uk_v1.py` | Delete | Superseded by zoo |
| `test_nowcast_cpi.py` | Create | Tests for new functionality (gas_eu load, screen_candidates, rolling window, ElasticNet, gate) |
| `STATE.md` | Modify | Update with post-retrain results |

---

## Task 1: Add `gas_eu` factor to `factors.py`

**Files:**
- Modify: `factors.py`
- Create: `test_nowcast_cpi.py`

- [ ] **Step 1: Write the failing test**

```python
# test_nowcast_cpi.py
import os
import unittest
from unittest import mock
import pandas as pd
import numpy as np
import factors as F

class TestGasEu(unittest.TestCase):
    def test_gas_eu_in_registry(self):
        self.assertIn("gas_eu", F.REGISTRY)

    def test_gas_eu_fields(self):
        e = F.REGISTRY["gas_eu"]
        self.assertEqual(e["transform"], "logret")
        self.assertEqual(e["pub_lag"], 0)
        self.assertTrue(e["candidate"])
        self.assertIsNotNone(e["fetch"])
        self.assertNotIn("region", e)   # must not be tagged US — must appear in UK runs

    def test_gas_eu_excluded_from_us_region(self):
        # gas_hh has region=US and is excluded from UK runs; gas_eu must not
        hh = F.REGISTRY["gas_hh"]
        self.assertEqual(hh.get("region"), "US")
        eu = F.REGISTRY["gas_eu"]
        self.assertIsNone(eu.get("region"))

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/Adam/Documents/home/quant/ramm-lgbm
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestGasEu -v
```
Expected: FAIL — `gas_eu` not in REGISTRY.

- [ ] **Step 3: Add `gas_eu` entry to `factors.py` REGISTRY**

In `factors.py`, add after the `gas_hh` entry (around line 165):

```python
    "gas_eu": dict(
        fetch=lambda: _fred("PNGASEUUSDM"), transform="logret",
        pub_lag=0, candidate=True, csv="gas_eu.csv",
        note="IMF/FRED European natural gas price (PNGASEUUSDM, USD/mmBtu, 1960-). "
             "UK imported LNG proxy — more relevant to UK CPI than Henry Hub. "
             "pub_lag=0. TTF front-month futures (yfinance TTF=F) preferred post-2009; "
             "override by dropping data/gas_eu.csv."),
```

Also update `RAMM_LGBM.MONO` dict in `uk_model_zoo.py` to add `"gas_eu": 1` (higher gas → higher CPI). Find the MONO dict at line 174:

```python
    MONO = {"oil_brent": 1, "gbpusd": -1, "uk_be5": 1,
            "uk_rents": 1, "uk_paye": 1, "uk_ashe_pay": 1,
            "uk_infl_swap_1y": 1, "gas_hh": 1, "gas_eu": 1, "cpi_lag1": 1}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestGasEu -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add factors.py uk_model_zoo.py test_nowcast_cpi.py
git commit -m "feat: add gas_eu (FRED PNGASEUUSDM) as UK LNG proxy factor"
```

---

## Task 2: File housekeeping — rename and delete

**Files:**
- Rename: `compare_uk.py` → `nowcast_cpi.py`
- Delete: `ramm_lgbm_uk_v1.py`
- Modify: `nowcast_cpi.py` (update CSV output filenames)

- [ ] **Step 1: Rename the file**

```bash
mv /Users/Adam/Documents/home/quant/ramm-lgbm/compare_uk.py \
   /Users/Adam/Documents/home/quant/ramm-lgbm/nowcast_cpi.py
```

- [ ] **Step 2: Update CSV output filenames inside nowcast_cpi.py**

Find and replace the three CSV save lines at the bottom of `main()` in `nowcast_cpi.py`:

Old:
```python
        out.to_csv("compare_uk_backtest.csv", index=False)
        print("\nSaved → compare_uk_backtest.csv")

    mdf.to_csv("compare_uk_metrics.csv")
    print("Saved → compare_uk_metrics.csv")

    if not spa.empty:
        spa.to_csv("compare_uk_spa.csv")
        print("Saved → compare_uk_spa.csv")
```

New:
```python
        out.to_csv("nowcast_cpi_backtest.csv", index=False)
        print("\nSaved → nowcast_cpi_backtest.csv")

    mdf.to_csv("nowcast_cpi_metrics.csv")
    print("Saved → nowcast_cpi_metrics.csv")

    if not spa.empty:
        spa.to_csv("nowcast_cpi_spa.csv")
        print("Saved → nowcast_cpi_spa.csv")
```

- [ ] **Step 3: Delete `ramm_lgbm_uk_v1.py`**

```bash
rm /Users/Adam/Documents/home/quant/ramm-lgbm/ramm_lgbm_uk_v1.py
```

- [ ] **Step 4: Update docstring in `nowcast_cpi.py`**

Replace the first line of the module docstring:

Old: `compare_uk.py — full model comparison for UK CPI YoY nowcasting.`
New: `nowcast_cpi.py — full model comparison and nowcast for UK CPI YoY.`

Also update the Usage block:

Old:
```
  FRED_API_KEY=<key> python compare_uk.py [--start 2015] [--train-from 1992]
  FRED_API_KEY=<key> python compare_uk.py --start 2015 --rmc
```

New:
```
  FRED_API_KEY=<key> python nowcast_cpi.py [--start 2015] [--train-from 1992]
  FRED_API_KEY=<key> python nowcast_cpi.py --start 2015 --rmc
```

- [ ] **Step 5: Verify import still works**

```bash
FRED_API_KEY=x .venv/bin/python -c "import nowcast_cpi; print('ok')"
```
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add nowcast_cpi.py
git rm ramm_lgbm_uk_v1.py
git commit -m "refactor: rename compare_uk.py to nowcast_cpi.py; delete superseded ramm_lgbm_uk_v1.py"
```

---

## Task 3: Add `WINDOW` attribute to `BaseModel` and rolling-window backtest logic

**Files:**
- Modify: `uk_model_zoo.py` (BaseModel class, lines 87–111)
- Modify: `test_nowcast_cpi.py`

- [ ] **Step 1: Write the failing test**

Add to `test_nowcast_cpi.py`:

```python
import uk_model_zoo as Z

class TestRollingWindow(unittest.TestCase):
    def _make_df(self):
        idx = pd.date_range("2010-01-31", periods=180, freq="ME")
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "f1": rng.standard_normal(180),
            "f2": rng.standard_normal(180),
            "cpi_yoy": rng.standard_normal(180) * 0.5 + 3.0,
        }, index=idx)
        return df

    def test_base_model_has_window_none(self):
        self.assertIsNone(Z.BaseModel.WINDOW)

    def test_expanding_window_uses_all_prior_data(self):
        df = self._make_df()

        class DummyModel(Z.BaseModel):
            name = "dummy"
            train_sizes = []
            def _fit_predict_year(self, train, test, factors, target):
                DummyModel.train_sizes.append(len(train))
                return np.zeros(len(test))

        m = DummyModel()
        m.backtest(df, ["f1", "f2"], "cpi_yoy", start_year=2015)
        # Each successive year should have MORE training data
        self.assertEqual(DummyModel.train_sizes, sorted(DummyModel.train_sizes))

    def test_rolling_window_caps_training_data(self):
        df = self._make_df()

        class RollingDummy(Z.BaseModel):
            name = "rolling_dummy"
            WINDOW = 24
            train_sizes = []
            def _fit_predict_year(self, train, test, factors, target):
                RollingDummy.train_sizes.append(len(train))
                return np.zeros(len(test))

        m = RollingDummy()
        m.backtest(df, ["f1", "f2"], "cpi_yoy", start_year=2015)
        # After warmup: each year's training data should be <= 24 months
        # (or min_train=60 fallback if not enough)
        for sz in RollingDummy.train_sizes:
            self.assertLessEqual(sz, 60)  # at most 60 (fallback size)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestRollingWindow -v
```
Expected: FAIL — `BaseModel` has no `WINDOW` attribute.

- [ ] **Step 3: Modify `BaseModel` in `uk_model_zoo.py`**

Replace the `BaseModel` class definition (lines 87–117):

```python
class BaseModel:
    name = "base"
    importance_type = "permutation ΔRMSE"
    has_regimes = False
    WINDOW = None   # None = expanding; int = rolling window in months

    def _fit_predict_year(self, train, test, factors, target):
        raise NotImplementedError

    def backtest(self, df, factors, target, start_year=START_YEAR_DEFAULT, min_train=60):
        d = _prep(df, factors, target)
        rows = []
        for yr in sorted(y for y in d.index.year.unique() if y >= start_year):
            test_start = pd.Timestamp(f"{yr}-01-01")
            if self.WINDOW is None:
                train = d[d.index.year < yr]
            else:
                cutoff = test_start - pd.DateOffset(months=self.WINDOW)
                train = d[(d.index >= cutoff) & (d.index.year < yr)]
                if len(train) < min_train:
                    train = d[d.index.year < yr]   # fall back to expanding
            test = d[d.index.year == yr]
            if len(train) < min_train or len(test) == 0:
                continue
            try:
                preds = self._fit_predict_year(train, test, factors, target)
            except Exception:
                continue
            for date, actual, pred in zip(test.index, test[target].values, preds):
                if np.isfinite(actual) and np.isfinite(pred):
                    rows.append(dict(date=date, actual=float(actual),
                                     pred=float(pred), year=yr))
        return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

    def importance(self, df, factors, target):
        return pd.Series(dtype=float), self.importance_type

    def regimes(self, df, factors, target):
        return None, {}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestRollingWindow -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add uk_model_zoo.py test_nowcast_cpi.py
git commit -m "feat: add WINDOW attribute to BaseModel for rolling-window backtest support"
```

---

## Task 4: Add 20 rolling-window subclasses + DFM-k2 to `uk_model_zoo.py`

**Files:**
- Modify: `uk_model_zoo.py` (append after `GBM` class, before `dm_test`; update `all_models()`)

- [ ] **Step 1: Add 20 rolling-window classes before `dm_test`**

Append the following block in `uk_model_zoo.py` after the `GBM` class definition (line ~881, before `def dm_test`):

```python
# ─────────────────────────────────────────────────────────────────────────────
# ROLLING-WINDOW VARIANTS  (5-year = 60 months, 2-year = 24 months)
# All inherit _fit_predict_year from parent; only WINDOW and name change.
# BaseModel.backtest() slices training data to WINDOW months before test_start,
# falling back to expanding window if < min_train observations remain.
# ─────────────────────────────────────────────────────────────────────────────

class DFM_Rolling5Y(DFM):
    name = "DFM-5Y";  WINDOW = 60

class DFM_Rolling2Y(DFM):
    name = "DFM-2Y";  WINDOW = 24

class RAMM_LGBM_Rolling5Y(RAMM_LGBM):
    name = "RAMM-LGBM-5Y";  WINDOW = 60

class RAMM_LGBM_Rolling2Y(RAMM_LGBM):
    name = "RAMM-LGBM-2Y";  WINDOW = 24

class UCM_Rolling5Y(UCM):
    name = "UCM-5Y";  WINDOW = 60

class UCM_Rolling2Y(UCM):
    name = "UCM-2Y";  WINDOW = 24

class TVP_Rolling5Y(TVP):
    name = "TVP-5Y";  WINDOW = 60

class TVP_Rolling2Y(TVP):
    name = "TVP-2Y";  WINDOW = 24

class HMM_Rolling5Y(HMM):
    name = "HMM-5Y";  WINDOW = 60

class HMM_Rolling2Y(HMM):
    name = "HMM-2Y";  WINDOW = 24

class MS_DFM_Rolling5Y(MS_DFM):
    name = "MS-DFM-5Y";  WINDOW = 60

class MS_DFM_Rolling2Y(MS_DFM):
    name = "MS-DFM-2Y";  WINDOW = 24

class LSTAR_Rolling5Y(LSTAR):
    name = "LSTAR-5Y";  WINDOW = 60

class LSTAR_Rolling2Y(LSTAR):
    name = "LSTAR-2Y";  WINDOW = 24

class BVAR_Rolling5Y(BVAR):
    name = "BVAR-5Y";  WINDOW = 60

class BVAR_Rolling2Y(BVAR):
    name = "BVAR-2Y";  WINDOW = 24

class HiddenRF_Rolling5Y(HiddenRF):
    name = "HiddenRF-5Y";  WINDOW = 60

class HiddenRF_Rolling2Y(HiddenRF):
    name = "HiddenRF-2Y";  WINDOW = 24

class GBM_Rolling5Y(GBM):
    name = "GBM-5Y";  WINDOW = 60

class GBM_Rolling2Y(GBM):
    name = "GBM-2Y";  WINDOW = 24
```

- [ ] **Step 2: Add DFM-k2 class**

Append immediately after the rolling-window block:

```python
class DFM2(DFM):
    """DFM with two latent factors: intended to separate global-risk from domestic-services."""
    name = "DFM-k2"

    def __init__(self):
        super().__init__(k_factors=2)
```

- [ ] **Step 3: Update `all_models()` to include all 21 new classes**

Replace the existing `all_models()` function:

```python
def all_models():
    # MIDAS removed: monthly-only Almon DL is not genuine mixed-frequency.
    base = [DFM(), RAMM_LGBM(), UCM(), TVP(), HMM(), MS_DFM(), LSTAR(),
            BVAR(), HiddenRF(), GBM()]
    rolling_5y = [DFM_Rolling5Y(), RAMM_LGBM_Rolling5Y(), UCM_Rolling5Y(),
                  TVP_Rolling5Y(), HMM_Rolling5Y(), MS_DFM_Rolling5Y(),
                  LSTAR_Rolling5Y(), BVAR_Rolling5Y(), HiddenRF_Rolling5Y(),
                  GBM_Rolling5Y()]
    rolling_2y = [DFM_Rolling2Y(), RAMM_LGBM_Rolling2Y(), UCM_Rolling2Y(),
                  TVP_Rolling2Y(), HMM_Rolling2Y(), MS_DFM_Rolling2Y(),
                  LSTAR_Rolling2Y(), BVAR_Rolling2Y(), HiddenRF_Rolling2Y(),
                  GBM_Rolling2Y()]
    extras = [DFM2()]
    return base + rolling_5y + rolling_2y + extras
```

- [ ] **Step 4: Write and run smoke test**

Add to `test_nowcast_cpi.py`:

```python
class TestAllModels(unittest.TestCase):
    def test_all_models_count(self):
        models = Z.all_models()
        self.assertEqual(len(models), 31)  # 10 base + 10 rolling-5y + 10 rolling-2y + 1 DFM-k2

    def test_rolling_models_have_window(self):
        models = Z.all_models()
        for m in models:
            if "5Y" in m.name:
                self.assertEqual(m.WINDOW, 60, f"{m.name} should have WINDOW=60")
            elif "2Y" in m.name:
                self.assertEqual(m.WINDOW, 24, f"{m.name} should have WINDOW=24")
            elif m.name not in ("DFM-k2",):
                self.assertIsNone(m.WINDOW, f"{m.name} should have WINDOW=None")

    def test_all_model_names_unique(self):
        models = Z.all_models()
        names = [m.name for m in models]
        self.assertEqual(len(names), len(set(names)), "duplicate model names found")
```

```bash
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestAllModels -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add uk_model_zoo.py test_nowcast_cpi.py
git commit -m "feat: add 20 rolling-window variants + DFM-k2 to model zoo (31 models total)"
```

---

## Task 5: Add `ElasticNet` model to `uk_model_zoo.py`

**Files:**
- Modify: `uk_model_zoo.py` (add class before rolling-window block; add to `all_models()`)
- Modify: `test_nowcast_cpi.py`

- [ ] **Step 1: Write the failing test**

Add to `test_nowcast_cpi.py`:

```python
class TestElasticNet(unittest.TestCase):
    def _make_df(self):
        idx = pd.date_range("2005-01-31", periods=240, freq="ME")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "f1": rng.standard_normal(240),
            "f2": rng.standard_normal(240),
            "cpi_yoy": rng.standard_normal(240) * 0.5 + 3.0,
        }, index=idx)
        return df

    def test_elastic_net_in_all_models(self):
        names = [m.name for m in Z.all_models()]
        self.assertIn("ElasticNet", names)

    def test_elastic_net_backtest_runs(self):
        df = self._make_df()
        m = Z.ElasticNet()
        bt = m.backtest(df, ["f1", "f2"], "cpi_yoy", start_year=2015)
        self.assertGreater(len(bt), 0)
        self.assertIn("actual", bt.columns)
        self.assertIn("pred", bt.columns)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestElasticNet -v
```
Expected: FAIL — `ElasticNet` not in zoo.

- [ ] **Step 3: Add `ElasticNet` class to `uk_model_zoo.py`**

Add before the rolling-window block (before the comment line `# ROLLING-WINDOW VARIANTS`):

```python
# ─────────────────────────────────────────────────────────────────────────────
# 11. ELASTICNET
# ─────────────────────────────────────────────────────────────────────────────

class ElasticNet(BaseModel):
    """AR-augmented ElasticNet with cross-validated alpha and l1_ratio.
    Expected RMSE between BVAR and LSTAR."""
    name = "ElasticNet"
    importance_type = "coefficient |value|"
    LAG = "cpi_lag1"

    def _feats(self, factors):
        return factors + [self.LAG]

    def _add_lag(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        return f

    def _fit_predict_year(self, train, test, factors, target):
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler
        feats = self._feats(factors)
        both = self._add_lag(pd.concat([train, test]), target)
        tr = both.loc[train.index].dropna(subset=feats + [target])
        te = both.loc[test.index].fillna(method="ffill").fillna(tr[feats].mean())
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[feats])
        X_te = scaler.transform(te[feats])
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                         cv=5, max_iter=10000, random_state=42)
        m.fit(X_tr, tr[target].values)
        return m.predict(X_te)

    def importance(self, df, factors, target):
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler
        feats = self._feats(factors)
        d = self._add_lag(_prep(df, factors, target), target).dropna(subset=feats + [target])
        scaler = StandardScaler()
        X = scaler.fit_transform(d[feats])
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                         cv=5, max_iter=10000, random_state=42)
        m.fit(X, d[target].values)
        return pd.Series(np.abs(m.coef_), index=feats), self.importance_type
```

- [ ] **Step 4: Add `ElasticNet` to `all_models()`**

In `all_models()`, add to `extras`:

```python
    extras = [DFM2(), ElasticNet()]
```

Also update the count test in `TestAllModels`:
```python
        self.assertEqual(len(models), 32)  # 10 + 10 + 10 + DFM-k2 + ElasticNet
```

- [ ] **Step 5: Run tests**

```bash
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestElasticNet test_nowcast_cpi.py::TestAllModels -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add uk_model_zoo.py test_nowcast_cpi.py
git commit -m "feat: add ElasticNet model (ElasticNetCV, AR-augmented) to zoo"
```

---

## Task 6: Add `screen_candidates()` to `factors.py`

**Files:**
- Modify: `factors.py`
- Modify: `test_nowcast_cpi.py`

- [ ] **Step 1: Write the failing test**

Add to `test_nowcast_cpi.py`:

```python
class TestScreenCandidates(unittest.TestCase):
    def _make_df(self):
        idx = pd.date_range("2005-01-31", periods=200, freq="ME")
        rng = np.random.default_rng(7)
        # f1 correlates with target, f_noise is pure noise
        target = rng.standard_normal(200) * 0.5 + 3.0
        df = pd.DataFrame({
            "f1": target * 0.8 + rng.standard_normal(200) * 0.1,
            "f_noise": rng.standard_normal(200),
            "cpi_yoy": target,
        }, index=idx)
        return df

    def test_screen_candidates_returns_list(self):
        df = self._make_df()
        # Temporarily patch registry so f1 and f_noise are candidates
        with mock.patch.dict(F.REGISTRY, {
            "f1": dict(candidate=True, transform="level", pub_lag=0, fetch=None),
            "f_noise": dict(candidate=True, transform="level", pub_lag=0, fetch=None),
        }):
            result = F.screen_candidates(df, "cpi_yoy", threshold=0.001)
        self.assertIsInstance(result, list)

    def test_screen_candidates_keeps_informative_drops_noise(self):
        df = self._make_df()
        with mock.patch.dict(F.REGISTRY, {
            "f1": dict(candidate=True, transform="level", pub_lag=0, fetch=None),
            "f_noise": dict(candidate=True, transform="level", pub_lag=0, fetch=None),
        }):
            result = F.screen_candidates(df, "cpi_yoy", threshold=0.01)
        self.assertIn("f1", result)
        self.assertNotIn("f_noise", result)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestScreenCandidates -v
```
Expected: FAIL — `F.screen_candidates` not found.

- [ ] **Step 3: Add `screen_candidates()` to `factors.py`**

Add after `apply_publication_lags()` and before `_load_csv`:

```python
def screen_candidates(df, target, threshold=0.001):
    """
    Shapley-based candidate factor screening.

    Fits a quick LightGBM on all candidate factors present in df, computes
    mean |SHAP| per factor, and returns those above threshold.

    Args:
        df:        DataFrame containing factor columns and target column.
        target:    Name of the target column in df.
        threshold: Minimum mean |SHAP| to retain a candidate (default 0.001).

    Returns:
        List of factor names (candidates only) with mean |SHAP| >= threshold.
        Non-candidate (core) factors are never dropped and are not returned.
    """
    import shap
    from lightgbm import LGBMRegressor

    candidates = [n for n in df.columns
                  if n != target and REGISTRY.get(n, {}).get("candidate")]
    if not candidates:
        return []

    sub = df[candidates + [target]].dropna()
    if len(sub) < 30:
        return candidates  # not enough data to screen; keep all

    X = sub[candidates]
    y = sub[target]

    m = LGBMRegressor(n_estimators=200, learning_rate=0.05,
                      num_leaves=15, random_state=42, verbose=-1)
    m.fit(X, y)

    sv = shap.TreeExplainer(m).shap_values(X)
    importance = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)

    kept = list(importance[importance >= threshold].index)
    dropped = [f for f in candidates if f not in kept]
    if dropped:
        print(f"  [screen_candidates] dropped {dropped} (mean |SHAP| < {threshold})")
    return kept
```

- [ ] **Step 4: Run test to verify it passes**

```bash
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestScreenCandidates -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add factors.py test_nowcast_cpi.py
git commit -m "feat: add screen_candidates() Shapley-based factor screening to factors.py"
```

---

## Task 7: Tighten model gate to 1.0× AR(1) in `nowcast_cpi.py`

**Files:**
- Modify: `nowcast_cpi.py` (functions `greedy_uncorrelated_subset` and `main`)
- Modify: `test_nowcast_cpi.py`

- [ ] **Step 1: Write the failing test**

Add to `test_nowcast_cpi.py`:

```python
import nowcast_cpi as NC

class TestModelGate(unittest.TestCase):
    def _make_bt(self, rmse_target):
        """Create a fake backtest DataFrame with given RMSE."""
        idx = pd.date_range("2015-01-31", periods=60, freq="ME")
        rng = np.random.default_rng(0)
        actual = rng.standard_normal(60) + 3.0
        noise = rng.standard_normal(60) * rmse_target
        pred = actual + noise - noise.mean()
        return pd.DataFrame({"actual": actual, "pred": pred}, index=idx)

    def test_greedy_subset_excludes_models_above_ar1(self):
        # model_bad has RMSE > ar1_rmse; model_good has RMSE < ar1_rmse
        bt_good = self._make_bt(0.1)
        bt_bad  = self._make_bt(2.0)
        ar1_rmse = 0.5

        import pandas as pd as pd2  # noqa — already imported
        err_good = (bt_good["actual"] - bt_good["pred"]).rename("good")
        err_bad  = (bt_bad["actual"]  - bt_bad["pred"]).rename("bad")
        err_df   = pd.concat([err_good, err_bad], axis=1).dropna()
        corr_mat = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]],
                                 index=["good", "bad"], columns=["good", "bad"])
        bt_dict = {"good": bt_good, "bad": bt_bad}

        result = NC.greedy_uncorrelated_subset(corr_mat, bt_dict,
                                               rho_threshold=0.5, ar1_rmse=ar1_rmse)
        self.assertIn("good", result)
        self.assertNotIn("bad", result)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestModelGate -v
```
Expected: FAIL — "bad" (RMSE=2.0 > ar1=0.5) is currently included by the 1.5× gate.

- [ ] **Step 3: Fix `greedy_uncorrelated_subset` in `nowcast_cpi.py`**

Replace the body of `greedy_uncorrelated_subset` (lines 221–241):

```python
def greedy_uncorrelated_subset(corr_mat, bt_dict, rho_threshold=0.5, ar1_rmse=None):
    """
    Greedy selection: start with best model (lowest RMSE), add a candidate only if:
      1. max |ρ| with already-selected models < rho_threshold
      2. candidate RMSE < 1.0 × AR(1) RMSE  [must beat AR(1)]
    """
    rmse_map = {}
    for name, bt in bt_dict.items():
        if bt is not None and len(bt) > 0 and name in corr_mat.index:
            rmse_map[name] = float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean()))
    if ar1_rmse is None:
        return []  # cannot gate without AR(1) baseline
    ranked = sorted((n for n in rmse_map if rmse_map[n] < ar1_rmse), key=rmse_map.get)
    selected = []
    for cand in ranked:
        if not selected:
            selected.append(cand); continue
        max_rho = max(abs(corr_mat.loc[cand, s]) for s in selected
                      if s in corr_mat.columns)
        if max_rho < rho_threshold:
            selected.append(cand)
    return selected
```

- [ ] **Step 4: Filter combined ensembles to models beating AR(1) in `main()`**

Find the combined ensemble section in `nowcast_cpi.py` (around line 719). Replace:

```python
    base_bts = {n: bt for n, bt in bt_dict.items() if n != "AR(1)"}
    bt_static  = combine_static(base_bts)
    bt_dynamic = combine_dynamic(base_bts, window=12)
```

With:

```python
    def _beats_ar1(bt, threshold):
        if bt is None or len(bt) == 0 or threshold is None:
            return False
        return float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean())) < threshold

    beating_bts = {n: bt for n, bt in bt_dict.items()
                   if n != "AR(1)" and _beats_ar1(bt, ar1_r)}
    bt_static  = combine_static(beating_bts)
    bt_dynamic = combine_dynamic(beating_bts, window=12)
```

- [ ] **Step 5: Add gate flag to metrics table output**

In the metrics table section of `main()`, after computing `mdf`, flag sub-AR(1) rows. Find the print line (around line 769):

```python
    print(mdf[["rmse","mae","dir_acc","error_var","mape","bias","n"]].to_string(
        float_format=lambda x: f"{x:8.3f}"))
```

Replace with:

```python
    if ar1_r is not None:
        mdf["beats_ar1"] = mdf["rmse"] < ar1_r
    print(mdf[["rmse","mae","dir_acc","beats_ar1","error_var","mape","bias","n"]].to_string(
        float_format=lambda x: f"{x:8.3f}" if isinstance(x, float) else str(x)))
```

- [ ] **Step 6: Run test to verify it passes**

```bash
FRED_API_KEY=x .venv/bin/python -m pytest test_nowcast_cpi.py::TestModelGate -v
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add nowcast_cpi.py test_nowcast_cpi.py
git commit -m "feat: tighten model gate to 1.0× AR(1); filter combined ensembles to beating models"
```

---

## Task 8: Add nowcast output section to `nowcast_cpi.py`

**Files:**
- Modify: `nowcast_cpi.py` (add to `main()` after the save section)

The nowcast is the final-step prediction from each model fit on ALL available data, predicting the most recent complete feature row (where CPI may be unreleased).

- [ ] **Step 1: Add `nowcast()` method to `BaseModel` in `uk_model_zoo.py`**

Add after `regimes()` in `BaseModel`:

```python
    def nowcast(self, df, factors, target):
        """
        Fit on all rows where factors AND target are known.
        Predict on the most recent row where factors are complete
        (target may be NaN — this is the actual nowcast period).
        Returns (prediction, nowcast_date) or (np.nan, None).
        """
        d = _prep(df, factors, target)
        if len(d) == 0:
            return np.nan, None
        # Latest complete feature row (before or after target cutoff)
        feat_cols = list(dict.fromkeys(factors))
        latest = df[feat_cols].dropna()
        if len(latest) == 0:
            return np.nan, None
        latest_row = latest.iloc[[-1]]
        nowcast_date = latest_row.index[0]
        try:
            preds = self._fit_predict_year(d, latest_row, factors, target)
            return float(preds[0]) if len(preds) > 0 else np.nan, nowcast_date
        except Exception:
            return np.nan, nowcast_date
```

- [ ] **Step 2: Override `nowcast()` for AR-augmented models (RAMM_LGBM, ElasticNet)**

The AR-augmented models add `cpi_lag1` which requires a lag. Add override to `RAMM_LGBM`:

```python
    def nowcast(self, df, factors, target):
        d = _prep(df, factors, target)
        if len(d) == 0:
            return np.nan, None
        feats = self._feats(factors)
        both = self._add_lag(df, target)
        latest = both[feats].dropna()
        if len(latest) == 0:
            return np.nan, None
        latest_row = latest.iloc[[-1]]
        nowcast_date = latest_row.index[0]
        tr = both.loc[d.index].dropna(subset=feats + [target])
        try:
            m = self._model(feats)
            m.fit(tr[feats], tr[target])
            return float(m.predict(latest_row)[0]), nowcast_date
        except Exception:
            return np.nan, nowcast_date
```

Add the same override to `ElasticNet` class (after `importance()`):

```python
    def nowcast(self, df, factors, target):
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler
        d = _prep(df, factors, target)
        if len(d) == 0:
            return np.nan, None
        feats = self._feats(factors)
        both = self._add_lag(df, target)
        latest = both[feats].dropna()
        if len(latest) == 0:
            return np.nan, None
        latest_row = latest.iloc[[-1]]
        nowcast_date = latest_row.index[0]
        tr = both.loc[d.index].dropna(subset=feats + [target])
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[feats])
        X_te = scaler.transform(latest_row[feats])
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                         cv=5, max_iter=10000, random_state=42)
        m.fit(X_tr, tr[target].values)
        return float(m.predict(X_te)[0]), nowcast_date
```

- [ ] **Step 3: Add nowcast section to `main()` in `nowcast_cpi.py`**

Add before `if __name__ == "__main__":`, after the save section:

```python
    # ── CURRENT NOWCAST ─────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("CURRENT NOWCAST")
    print("═"*65)
    nowcast_rows = []
    for m in models:
        try:
            val, nc_date = m.nowcast(df, live_facs, target)
            nowcast_rows.append(dict(model=m.name, nowcast=round(val, 3) if np.isfinite(val) else np.nan,
                                     date=str(nc_date.date()) if nc_date else "?"))
        except Exception as e:
            nowcast_rows.append(dict(model=m.name, nowcast=np.nan, date="error"))
    nc_df = pd.DataFrame(nowcast_rows).set_index("model")
    print(nc_df.to_string())
    nc_df.to_csv("nowcast_cpi_nowcast.csv")
    print("\nSaved → nowcast_cpi_nowcast.csv")
```

- [ ] **Step 4: Verify import**

```bash
FRED_API_KEY=x .venv/bin/python -c "import nowcast_cpi; import uk_model_zoo; print('ok')"
```
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add uk_model_zoo.py nowcast_cpi.py
git commit -m "feat: add nowcast() to BaseModel + RAMM_LGBM/ElasticNet overrides; add nowcast output section"
```

---

## Task 9: Add `--shap-screen` flag to `nowcast_cpi.py`

**Files:**
- Modify: `nowcast_cpi.py`

- [ ] **Step 1: Add `--shap-screen` argparse flag**

In `main()`, add to the `argparse` block (after `--quiet`):

```python
    ap.add_argument("--shap-screen", action="store_true",
                    help="run Shapley factor screening; drop candidates below threshold")
    ap.add_argument("--shap-threshold", type=float, default=0.001,
                    help="mean |SHAP| threshold for screen_candidates (default 0.001)")
```

- [ ] **Step 2: Apply screening after factor loading**

After the `print(f" Factors: {live_facs}")` lines (around line 685), add:

```python
    if args.shap_screen:
        print(f"\nRunning Shapley factor screening (threshold={args.shap_threshold}) …")
        kept = F.screen_candidates(df, target, threshold=args.shap_threshold)
        core = F.core_factors()
        live_facs = [f for f in live_facs if f in kept or f in core]
        print(f"  Live factors after screening: {live_facs}")
```

- [ ] **Step 3: Verify flag parses**

```bash
FRED_API_KEY=x .venv/bin/python nowcast_cpi.py --help | grep shap
```
Expected: lines showing `--shap-screen` and `--shap-threshold`.

- [ ] **Step 4: Commit**

```bash
git add nowcast_cpi.py
git commit -m "feat: add --shap-screen flag to nowcast_cpi.py for post-load factor screening"
```

---

## Task 10: Run full zoo retrain + update STATE.md

**Files:**
- Run: `nowcast_cpi.py`
- Modify: `STATE.md`

This task requires a live `FRED_API_KEY`.

- [ ] **Step 1: Run the full backtest (expanding window, all 32 models)**

```bash
FRED_API_KEY=<your_key> .venv/bin/python -W ignore nowcast_cpi.py \
    --start 2015 --train-from 1992 \
    2>&1 | tee nowcast_cpi_run.log
```
Expected: metrics table with 32 models + AR(1), combined ensembles, nowcast output. No Python exceptions.

- [ ] **Step 2: Run with `--shap-screen` to identify weak factors**

```bash
FRED_API_KEY=<your_key> .venv/bin/python -W ignore nowcast_cpi.py \
    --start 2015 --train-from 1992 --shap-screen \
    2>&1 | tee nowcast_cpi_shap.log
```
Note which candidates are dropped. Update `live_facs` selection in STATE.md.

- [ ] **Step 3: Run `--rmc`**

```bash
FRED_API_KEY=<your_key> .venv/bin/python -W ignore nowcast_cpi.py \
    --start 2015 --train-from 1992 --rmc \
    2>&1 | tee nowcast_cpi_rmc.log
```
Expected: `rmc_*_perf.csv` files saved. RMC metamodel metrics printed.

- [ ] **Step 4: Update `STATE.md`**

Replace the "Last Backtest Results" table with new numbers from `nowcast_cpi_metrics.csv`. Add nowcast table from `nowcast_cpi_nowcast.csv`. Update "Pending Rewrites / Gaps" to mark completed items.

---

## Task 11: Push to GitHub

- [ ] **Step 1: Final status check**

```bash
git status
git log --oneline -10
```

- [ ] **Step 2: Commit any remaining changes (STATE.md, CSV outputs)**

```bash
git add STATE.md nowcast_cpi_metrics.csv nowcast_cpi_spa.csv nowcast_cpi_backtest.csv nowcast_cpi_nowcast.csv
git commit -m "results: post-retrain metrics with rolling windows, ElasticNet, 1.0x AR(1) gate"
```

- [ ] **Step 3: Push**

```bash
git push origin main
```
(Confirm with user before running — first push attempt.)

---

## Self-Review Notes

- **Spec §4 (k=2 DFM):** Handled via `DFM2` class inheriting `DFM(k_factors=2)` — no separate `--dfm-k` flag needed since it's always included in `all_models()`.
- **Spec §5 (post-1992 vs post-2005):** The `--train-from` flag already exists. Side-by-side comparison is done by running twice with different `--train-from`. Omitted from this plan — straightforward to do manually; no code change needed.
- **Model count:** 10 base + 10 rolling-5y + 10 rolling-2y + DFM-k2 + ElasticNet = 32. All tests use 32.
- **Rolling-window subclasses for ElasticNet:** Not added (user said "duplicate the final model" for rolling windows; ElasticNet is new, not yet "final"). Can be added in follow-up.
- **`_beats_ar1` helper:** Defined as a local function inside `main()`. Fine for now.
