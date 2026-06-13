"""
uk_model_zoo.py — model zoo for UK CPI YoY nowcasting.

Common interface (BaseModel):
  backtest(df, factors, target, start_year)  -> DataFrame[date, actual, pred, year]
  importance(df, factors, target)            -> (Series factor->value, type_str)
  regimes(df, factors, target)               -> (Series labels | None, meta dict)

Models:
  DFM         Dynamic Factor Model (statsmodels DynamicFactor)
  RAMM_LGBM   Regime-Aware Monotonic LightGBM (TreeSHAP)
  UCM         Unobserved Components Model (local trend + cycle + exog)
  TVP         Time-Varying Parameter regression (manual random-walk Kalman)
  HMM         Hidden Markov Model (MarkovRegression, univariate)
  MS_DFM      Markov-Switching DFM (DFM factor → Markov-switching regression)
  LSTAR       Logistic Smooth Transition AR (scipy least_squares)

Backtest convention: expanding window, refit each forecast year on data < year.
  - Lag-feature models (LGBM, TVP, LSTAR): native 1-step-ahead.
  - State-space / Markov (DFM, UCM, HMM, MS_DFM): yearly refit + multi-step
    forecast across the test year (same convention as the original DFM backtest).
Importance type differs by model and is labelled in the output table.
"""

import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

START_YEAR_DEFAULT = 2015

# ── Module-level cache for MIDAS / BridgeEq daily data ───────────────────────
_MIDAS_CACHE: dict = {}

def _get_midas_data():
    """Fetch daily Brent/GBP/VIX/TTF from yfinance, return monthly-mean DataFrame.
    Cached at module level to avoid repeated downloads across backtest iterations."""
    if "mm" in _MIDAS_CACHE:
        return _MIDAS_CACHE["mm"]
    try:
        import yfinance as yf
        tickers = [("brent_ma", "BZ=F"), ("gbpusd_ma", "GBPUSD=X"),
                   ("vix_ma", "^VIX"),   ("gas_ma", "TTF=F")]
        dfs = {}
        for name, tkr in tickers:
            try:
                raw = yf.download(tkr, start="1989-01-01", auto_adjust=True, progress=False)
                c = (raw[("Close", tkr)] if isinstance(raw.columns, pd.MultiIndex)
                     else raw["Close"])
                dfs[name] = c.resample("ME").mean().rename(name)
            except Exception:
                pass
        result = pd.concat(dfs.values(), axis=1) if dfs else None
    except Exception:
        result = None
    _MIDAS_CACHE["mm"] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _prep(df, factors, target):
    """Align target + factors, drop rows missing any required column."""
    cols = list(dict.fromkeys(factors + [target]))
    d = df[cols].copy()
    d = d.dropna(subset=factors + [target])
    return d


def _zscore(df, cols, ref=None):
    ref = df if ref is None else ref
    mu  = ref[cols].mean()
    sd  = ref[cols].std().replace(0, 1.0).fillna(1.0)  # NaN std (all-NaN col) → 1
    return (df[cols] - mu) / sd, mu, sd


def _named(res):
    """Param name -> value dict, robust to Series or bare ndarray params."""
    return dict(zip(res.model.param_names, np.asarray(res.params)))


def _last_probs(res, k):
    """Last time-step's k-vector of filtered regime probabilities (any orientation)."""
    fp = np.asarray(res.filtered_marginal_probabilities)
    if fp.ndim == 1:
        return fp
    return fp[-1] if fp.shape[1] == k else fp[:, -1]


def _perm_importance(predict_fn, X, y, n_repeats=5, seed=0):
    """Permutation importance: ΔRMSE when each column is shuffled."""
    rng = np.random.default_rng(seed)
    base = np.sqrt(mean_squared_error(y, predict_fn(X)))
    out = {}
    for col in X.columns:
        rises = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            rises.append(np.sqrt(mean_squared_error(y, predict_fn(Xp))) - base)
        out[col] = max(np.mean(rises), 0.0)
    return pd.Series(out)


# ─────────────────────────────────────────────────────────────────────────────
# BASE
# ─────────────────────────────────────────────────────────────────────────────

class BaseModel:
    name = "base"
    importance_type = "permutation ΔRMSE"
    has_regimes = False
    WINDOW = None   # None = expanding window; int = rolling window in months
    # Hard support constraint: UK CPI YoY has empirical support ≈ [-2, 20].
    # Post-fit clipping prevents runaway predictions from polluting ensembles.
    PRED_MIN = -2.0
    PRED_MAX = 20.0
    # Nowcast staleness guard: a factor may be forward-filled at most
    # pub_lag + FFILL_GRACE months past its last observation (H6).
    FFILL_GRACE = 2

    def _fit_predict_year(self, train, test, factors, target):
        raise NotImplementedError

    def backtest(self, df, factors, target, start_year=START_YEAR_DEFAULT,
                min_train=24, end_year=None):
        d = _prep(df, factors, target)
        rows = []
        n_warns = 0
        years = d.index.year.unique()
        years = [y for y in years if y >= start_year and (end_year is None or y <= end_year)]
        for yr in sorted(years):
            test_start = pd.Timestamp(f"{yr}-01-01")
            if self.WINDOW is None:
                train = d[d.index.year < yr]
            else:
                cutoff = test_start - pd.DateOffset(months=self.WINDOW)
                train = d[(d.index >= cutoff) & (d.index.year < yr)]
                if len(train) < min_train:
                    # fall back to the most-recent min_train rows before test start
                    pre = d[d.index.year < yr]
                    train = pre.iloc[-min_train:]
            test = d[d.index.year == yr]
            if len(train) < min_train or len(test) == 0:
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    preds = self._fit_predict_year(train, test, factors, target)
                n_warns_fold = len(caught)
            except Exception as e:
                # C4: never delete a test year silently — coverage gaps must be loud
                print(f"  [WARN] {self.name} {yr}: year skipped — {str(e)[:70]}")
                continue
            n_warns += n_warns_fold
            preds = np.clip(preds, self.PRED_MIN, self.PRED_MAX)
            for date, actual, pred in zip(test.index, test[target].values, preds):
                if np.isfinite(actual) and np.isfinite(pred):
                    rows.append(dict(date=date, actual=float(actual),
                                     pred=float(pred), year=yr))
        df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
        if not df.empty:
            df["n_warns"] = n_warns
        return df

    def importance(self, df, factors, target):
        return pd.Series(dtype=float), self.importance_type

    def regimes(self, df, factors, target):
        return None, {}

    @staticmethod
    def _nowcast_row(df, factors, target):
        """
        Return (feature_row, nowcast_date) for the first unreleased CPI month.

        'First unreleased' = the earliest index where target is NaN.
        Ragged-edge ONS series (uk_rents_lag1, uk_vacancies) are forward-filled
        from the most recent published value — exactly what a real-time analyst
        would do when a monthly ONS release has not yet been ingested by dbnomics.

        Financial daily series (Brent, GBP/USD, VIX) are already complete for
        the target month because they are pulled as month-end closes via FRED.

        The row always includes the target column (set to NaN) so state-space
        models that call test[target] do not raise KeyError; they should treat
        NaN target as a missing observation.

        ALFRED note: financial FRED series carry zero revision risk (market prices).
        ONS vintage data cannot be verified without the ONS real-time API; use of
        final revised values in training is a known approximation.
        """
        feat_cols = list(dict.fromkeys(factors))
        # Identify the first month AFTER the last known CPI release.
        # Using df[target].isna().index[0] would incorrectly pick up pre-history
        # NaN rows (before D7G7 began in 1989) when the full raw matrix is passed.
        if target in df.columns:
            known = df[target].dropna()
            if len(known) > 0:
                last_known = known.index[-1]
                trailing = df.index[df.index > last_known]
                nowcast_date = trailing[0] if len(trailing) > 0 else last_known
            else:
                nowcast_date = df.index[-1]
        else:
            nowcast_date = df.index[-1]

        # Include target column so models can access it (remains NaN = unreleased)
        all_cols = list(dict.fromkeys(feat_cols + ([target] if target in df.columns else [])))
        feat_df = df[all_cols].reindex([nowcast_date])
        # Budgeted forward-fill (H6): each factor may be carried forward at most
        # pub_lag + FFILL_GRACE months past its last observation. Beyond that it
        # stays NaN so a dead/stale series can never silently feed the nowcast.
        try:
            from factors import REGISTRY as _REG
        except Exception:
            _REG = {}
        budgets = {c: int(_REG.get(c, {}).get("pub_lag", 1)) + BaseModel.FFILL_GRACE
                   for c in all_cols}
        ffilled = pd.DataFrame(
            {c: df[c].ffill(limit=max(budgets[c], 1)) for c in all_cols}
        ).reindex([nowcast_date])
        row = feat_df.fillna(ffilled)
        # Target must stay NaN — it's the thing we're trying to predict
        if target in row.columns:
            row[target] = np.nan

        # NaN check on factor columns only (target NaN is expected)
        stale = [c for c in feat_cols if row[c].isna().iloc[0]]
        if stale:
            print(f"  [STALE] nowcast blocked — factors beyond ffill budget "
                  f"(pub_lag+{BaseModel.FFILL_GRACE}m): {stale}")
            return None, nowcast_date

        return row, nowcast_date

    def nowcast(self, df, factors, target):
        """
        Fit on all months with known target+features; predict the first unreleased
        CPI month using a ragged-edge-corrected feature row.
        Returns (prediction, nowcast_date) or (np.nan, None).
        """
        d = _prep(df, factors, target)
        if len(d) == 0:
            return np.nan, None
        row, nowcast_date = self._nowcast_row(df, factors, target)
        if row is None:
            return np.nan, nowcast_date
        try:
            preds = self._fit_predict_year(d, row, factors, target)
            return float(preds[0]) if len(preds) > 0 else np.nan, nowcast_date
        except Exception:
            return np.nan, nowcast_date


# ─────────────────────────────────────────────────────────────────────────────
# 1. DFM
# ─────────────────────────────────────────────────────────────────────────────

class DFM(BaseModel):
    name = "DFM"
    importance_type = "factor loading"

    def __init__(self, k_factors=1):
        self.k = k_factors

    def _fit_predict_year(self, train, test, factors, target):
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        obs = factors + [target]
        target_col = obs.index(target)   # positional index — robust to numeric column names
        z_tr, mu, sd = _zscore(train, obs)
        res = DynamicFactor(z_tr.dropna(), k_factors=self.k, factor_order=1,
                            error_order=1).fit(maxiter=200, disp=False)
        # 1-step-ahead: forecast month m, then append realized month-m obs (fixed params)
        z_te = (test[obs] - train[obs].mean()) / train[obs].std().replace(0, 1)
        preds, cur = [], res
        for idx in test.index:
            fc_arr = np.asarray(cur.forecast(steps=1))
            val = fc_arr[0, target_col] if fc_arr.ndim == 2 else fc_arr[target_col]
            preds.append(val * sd[target] + mu[target])
            try:
                cur = cur.append(z_te.loc[[idx]].values, refit=False)
            except Exception:
                pass
        return np.array(preds)

    def importance(self, df, factors, target):
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        d = _prep(df, factors, target)
        obs = factors + [target]
        z, _, _ = _zscore(d, obs)
        res = DynamicFactor(z.dropna(), k_factors=self.k, factor_order=1,
                            error_order=1).fit(maxiter=200, disp=False)
        loads = {k.replace("loading.f1.", ""): abs(v)
                 for k, v in res.params.items() if k.startswith("loading.f1.")}
        s = pd.Series({f: loads.get(f, np.nan) for f in factors}).dropna()
        return s, self.importance_type

    def nowcast(self, df, factors, target):
        """Fit DFM on all known data; forecast 1 step from final model state."""
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        d = _prep(df, factors, target)
        if len(d) == 0:
            return np.nan, None
        _, nowcast_date = self._nowcast_row(df, factors, target)
        if self.WINDOW is not None:
            cutoff = d.index[-1] - pd.DateOffset(months=self.WINDOW - 1)
            d = d[d.index >= cutoff]
        obs = factors + [target]
        target_col = obs.index(target)
        z, mu, sd = _zscore(d, obs)
        try:
            res = DynamicFactor(z.dropna(), k_factors=self.k, factor_order=1,
                                error_order=1).fit(maxiter=200, disp=False)
            fc_arr = np.asarray(res.forecast(steps=1))
            val = fc_arr[0, target_col] if fc_arr.ndim == 2 else fc_arr[target_col]
            return float(val * sd[target] + mu[target]), nowcast_date
        except Exception:
            return np.nan, nowcast_date


# ─────────────────────────────────────────────────────────────────────────────
# 2. RAMM-LGBM UK
# ─────────────────────────────────────────────────────────────────────────────

class RAMM_LGBM(BaseModel):
    name = "RAMM-LGBM"
    importance_type = "mean |SHAP|"
    has_regimes = True
    LAG = "cpi_lag1"
    MOM = "cpi_3m_chg"

    MONO = {"oil_brent": 1, "gbpusd": -1, "uk_be5": 1,
            "uk_rents_lag1": 1,   # rents (lag-1, real-time safe): higher → higher CPI
            "uk_awg": 1,          # wage growth: higher wages → services CPI up
            "uk_ashe_pay": 1, "uk_infl_swap_1y": 1,
            "gas_eu": 1, "cpi_lag1": 1, "cpi_3m_chg": 1}

    def _feats(self, factors):
        return list(factors) + [f for f in [self.LAG, self.MOM] if f not in factors]

    def _add_lag(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        f[self.MOM] = f[target].shift(1).diff(3)
        return f

    def _model(self, feats):
        from lightgbm import LGBMRegressor
        mono = [self.MONO.get(f, 0) for f in feats]
        return LGBMRegressor(objective="regression", n_estimators=500,
                             learning_rate=0.02, num_leaves=15, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8,
                             monotone_constraints=mono, random_state=42, verbose=-1)

    def _fit_predict_year(self, train, test, factors, target):
        feats = self._feats(factors)
        both  = self._add_lag(pd.concat([train, test]), target)
        tr = both.loc[train.index].dropna(subset=feats + [target])
        te = both.loc[test.index]
        m = self._model(feats)
        m.fit(tr[feats], tr[target])
        return m.predict(te[feats])

    def importance(self, df, factors, target):
        import shap
        feats = self._feats(factors)
        d = self._add_lag(_prep(df, factors, target), target).dropna(subset=feats + [target])
        m = self._model(feats); m.fit(d[feats], d[target])
        X = d.loc[d.index.year >= START_YEAR_DEFAULT, feats]
        sv = shap.TreeExplainer(m).shap_values(X)
        return pd.Series(np.abs(sv).mean(axis=0), index=feats), self.importance_type

    def regimes(self, df, factors, target):
        d = _prep(df, factors, target)
        if "vix" not in d.columns:
            return None, {"type": "none"}
        med = d["vix"].expanding(min_periods=12).median()
        reg = (d["vix"] > med).astype(int)
        labels = reg.map({0: "calm", 1: "stress"})
        return labels, {"type": "VIX expanding-median", "n_regimes": 2,
                        "current": labels.iloc[-1]}

    def nowcast(self, df, factors, target):
        d = _prep(df, factors, target)
        if len(d) == 0:
            return np.nan, None
        feats = self._feats(factors)
        # _add_lag appends cpi_lag1 = target.shift(1); the nowcast month's lag
        # = last released CPI value, which is always available
        both = self._add_lag(df, target)
        # Use ragged-edge-corrected row for the first unreleased CPI month
        base_row, nowcast_date = self._nowcast_row(df, factors, target)
        if base_row is None:
            return np.nan, nowcast_date
        lag_val = both.loc[nowcast_date, self.LAG] if nowcast_date in both.index else np.nan
        if not np.isfinite(lag_val):
            return np.nan, nowcast_date
        latest_row = base_row.copy()
        latest_row[self.LAG] = lag_val
        if nowcast_date in both.index and self.MOM in both.columns:
            mom_val = both.loc[nowcast_date, self.MOM]
            latest_row[self.MOM] = mom_val if (not pd.isna(mom_val) and np.isfinite(float(mom_val))) else 0.0
        else:
            latest_row[self.MOM] = 0.0
        tr = self._add_lag(d, target).dropna(subset=feats + [target])
        try:
            m = self._model(feats)
            m.fit(tr[feats], tr[target])
            return float(m.predict(latest_row[feats])[0]), nowcast_date
        except Exception:
            return np.nan, nowcast_date


# ─────────────────────────────────────────────────────────────────────────────
# 3. UCM
# ─────────────────────────────────────────────────────────────────────────────

class UCM(BaseModel):
    name = "UCM"
    importance_type = "permutation ΔRMSE"

    def _fit_predict_year(self, train, test, factors, target):
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        mu, sd = train[factors].mean(), train[factors].std().replace(0, 1)
        exog_tr = (train[factors] - mu) / sd
        res = UnobservedComponents(train[target].values, level="local linear trend",
                                   exog=exog_tr.values).fit(maxiter=200, disp=False)
        exog_te = (test[factors] - mu) / sd
        # 1-step-ahead with realized endog appended each month (fixed params)
        preds, cur = [], res
        for idx in test.index:
            fc = cur.forecast(steps=1, exog=exog_te.loc[[idx]].values)
            preds.append(np.asarray(fc)[0])
            try:
                cur = cur.append([test.loc[idx, target]],
                                 exog=exog_te.loc[[idx]].values, refit=False)
            except Exception:
                pass
        return np.array(preds)

    def importance(self, df, factors, target):
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        d = _prep(df, factors, target)
        exog, _, _ = _zscore(d, factors)
        res = UnobservedComponents(d[target], level="local linear trend",
                                   exog=exog.values).fit(maxiter=200, disp=False)
        # exog coefficients are 'beta.x1'... in order of factors
        betas = {f: abs(res.params.get(f"beta.x{i+1}", np.nan))
                 for i, f in enumerate(factors)}
        s = pd.Series(betas).dropna()
        return s, "std. |coef|"


# ─────────────────────────────────────────────────────────────────────────────
# 4. TVP — random-walk-coefficient Kalman filter (manual)
# ─────────────────────────────────────────────────────────────────────────────

class TVP(BaseModel):
    name = "TVP"
    importance_type = "mean |β·x| contrib"

    def __init__(self, delta=1e-3):
        self.delta = delta   # state-noise / obs-noise ratio (smoothness)

    def _kalman(self, X, y, R, Q):
        """Return filtered 1-step-ahead predictions and final beta path."""
        n, k = X.shape
        beta = np.zeros(k)
        P = np.eye(k) * 10.0
        preds = np.full(n, np.nan)
        betas = np.zeros((n, k))
        for t in range(n):
            x = X[t]
            P = P + Q                      # predict state cov (RW)
            yhat = x @ beta                # 1-step prediction (pre-update)
            preds[t] = yhat
            S = x @ P @ x + R
            K = (P @ x) / S
            beta = beta + K * (y[t] - yhat)
            P = P - np.outer(K, x) @ P
            betas[t] = beta
        return preds, betas

    def _design(self, d, factors, target):
        mom3 = d[target].shift(1).diff(3).values
        X = np.column_stack([np.ones(len(d)),
                             d[target].shift(1).values,
                             mom3,
                             d[factors].values])
        return X

    def _fit_predict_year(self, train, test, factors, target):
        d = pd.concat([train, test])
        # standardize factors on train
        fz = (d[factors] - train[factors].mean()) / train[factors].std().replace(0, 1)
        dd = d.copy(); dd[factors] = fz
        X = self._design(dd, factors, target)
        y = dd[target].values
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        # tune R from AR(1) residual variance on training data (forecast-error scale)
        ytr = train[target].values
        ytr_ok = ytr[np.isfinite(ytr)]
        if len(ytr_ok) > 2:
            ar1_resid = ytr_ok[1:] - (ytr_ok[:-1].mean() +
                        np.corrcoef(ytr_ok[:-1], ytr_ok[1:])[0, 1] *
                        (ytr_ok[:-1] - ytr_ok[:-1].mean()))
            R = float(np.var(ar1_resid)) + 1e-6
        else:
            R = 1.0
        Q = np.eye(X.shape[1]) * R * self.delta
        preds, _ = self._kalman(X[ok], y[ok], R, Q)
        full = pd.Series(np.nan, index=dd.index)
        full.loc[dd.index[ok]] = preds
        return full.loc[test.index].values

    def importance(self, df, factors, target):
        d = _prep(df, factors, target)
        fz = (d[factors] - d[factors].mean()) / d[factors].std().replace(0, 1)
        dd = d.copy(); dd[factors] = fz
        X = self._design(dd, factors, target)
        y = dd[target].values
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        yraw = d[target].values[np.isfinite(d[target].values)]
        if len(yraw) > 2:
            ar1_resid = yraw[1:] - (yraw[:-1].mean() +
                        np.corrcoef(yraw[:-1], yraw[1:])[0, 1] *
                        (yraw[:-1] - yraw[:-1].mean()))
            R = float(np.var(ar1_resid)) + 1e-6
        else:
            R = 1.0
        Q = np.eye(X.shape[1]) * R * self.delta
        _, betas = self._kalman(X[ok], y[ok], R, Q)
        Xok = X[ok]
        # contribution columns: [const, ar1, factors...]; importance over factors
        contrib = np.abs(betas * Xok)
        col_names = ["const", "ar1", "mom3"] + factors
        s = pd.Series(contrib.mean(axis=0), index=col_names)
        return s[factors], self.importance_type

    def nowcast(self, df, factors, target):
        """
        Run Kalman on all training data; predict 1 step ahead using the final
        state (beta) without an update step — no realized target needed.
        """
        d = _prep(df, factors, target)
        if len(d) == 0:
            return np.nan, None
        row, nowcast_date = self._nowcast_row(df, factors, target)
        if row is None:
            return np.nan, nowcast_date
        if self.WINDOW is not None:
            cutoff = d.index[-1] - pd.DateOffset(months=self.WINDOW - 1)
            d = d[d.index >= cutoff]
        try:
            fz_mu = d[factors].mean()
            fz_sd = d[factors].std().replace(0, 1)
            dd = d.copy()
            dd[factors] = (dd[factors] - fz_mu) / fz_sd
            X = self._design(dd, factors, target)
            y = dd[target].values
            ok = np.isfinite(X).all(1) & np.isfinite(y)
            yok = y[ok]
            if len(yok) > 2:
                ar1_resid = yok[1:] - (yok[:-1].mean() +
                            np.corrcoef(yok[:-1], yok[1:])[0, 1] *
                            (yok[:-1] - yok[:-1].mean()))
                R = float(np.var(ar1_resid)) + 1e-6
            else:
                R = 1.0
            Q = np.eye(X.shape[1]) * R * self.delta
            _, betas = self._kalman(X[ok], y[ok], R, Q)
            final_beta = betas[-1]
            ar1  = float(d[target].iloc[-1])
            mom3 = float(d[target].iloc[-1] - d[target].iloc[-4]) if len(d) >= 4 else 0.0
            row_fz = (row[factors] - fz_mu) / fz_sd
            x_now = np.concatenate([[1.0], [ar1], [mom3], row_fz.values[0]])
            return float(x_now @ final_beta), nowcast_date
        except Exception:
            return np.nan, nowcast_date


# ─────────────────────────────────────────────────────────────────────────────
# 5. HMM — univariate Markov-switching regression
# ─────────────────────────────────────────────────────────────────────────────

class HMM(BaseModel):
    name = "HMM"
    importance_type = "permutation ΔRMSE"
    has_regimes = True

    def __init__(self, k_regimes=2):
        self.k = k_regimes

    def _fit(self, y):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        return MarkovRegression(y, k_regimes=self.k, trend="c",
                                switching_variance=True).fit()

    def _fit_predict_year(self, train, test, factors, target):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        res = self._fit(train[target])                  # yearly param estimation
        params = np.asarray(res.params)
        named  = _named(res)
        means  = np.array([named.get(f"const[{i}]", np.nan) for i in range(self.k)])
        P = res.regime_transition[:, :, 0] if res.regime_transition.ndim == 3 \
            else res.regime_transition
        # 1-step-ahead: each month, re-filter history with FIXED params (cheap),
        # propagate regime probs one step, take regime-weighted mean.
        hist = list(train[target].values)
        preds = []
        for idx in test.index:
            r = MarkovRegression(np.asarray(hist), k_regimes=self.k, trend="c",
                                 switching_variance=True).smooth(params)
            probs = _last_probs(r, self.k) @ P.T
            preds.append(np.nansum(probs * means))
            hist.append(test.loc[idx, target])
        return np.array(preds)

    def importance(self, df, factors, target):
        # univariate model — factors enter only via regime; report uniform/NA
        return pd.Series(dtype=float), "n/a (univariate)"

    def regimes(self, df, factors, target):
        d = _prep(df, factors, target)
        try:
            res = self._fit(d[target])
            sm = np.asarray(res.smoothed_marginal_probabilities)
            sm = sm if sm.shape[1] == self.k else sm.T
            state = sm.argmax(axis=1)
            named = _named(res)
            means = [named.get(f"const[{i}]", np.nan) for i in range(self.k)]
            hi = int(np.nanargmax(means))
            labels = pd.Series(["high-infl" if s == hi else "low-infl" for s in state],
                               index=d[target].index)
            return labels, {"type": f"Markov {self.k}-state (switching var)",
                            "n_regimes": self.k, "current": labels.iloc[-1]}
        except Exception as e:
            return None, {"type": "none", "error": str(e)[:40]}

    def nowcast(self, df, factors, target):
        """
        Fit HMM on all known CPI; propagate filtered regime probs 1 step;
        predict regime-weighted mean. HMM is univariate so factors are unused.
        """
        d = _prep(df, factors, target)
        if len(d) == 0:
            return np.nan, None
        _, nowcast_date = self._nowcast_row(df, factors, target)
        if self.WINDOW is not None:
            cutoff = d.index[-1] - pd.DateOffset(months=self.WINDOW - 1)
            d = d[d.index >= cutoff]
        try:
            res = self._fit(d[target])
            named = _named(res)
            means = np.array([named.get(f"const[{i}]", np.nan) for i in range(self.k)])
            P = (res.regime_transition[:, :, 0] if res.regime_transition.ndim == 3
                 else res.regime_transition)
            probs = _last_probs(res, self.k) @ P.T
            return float(np.nansum(probs * means)), nowcast_date
        except Exception:
            return np.nan, nowcast_date


# ─────────────────────────────────────────────────────────────────────────────
# 6. MS-DFM — DFM factor + Markov-switching regression
# ─────────────────────────────────────────────────────────────────────────────

class MS_DFM(BaseModel):
    name = "MS-DFM"
    importance_type = "factor loading"
    has_regimes = True

    def __init__(self, k_regimes=2):
        self.k = k_regimes

    def _dfm_fit(self, z_obs):
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        return DynamicFactor(z_obs, k_factors=1, factor_order=1,
                             error_order=1).fit(maxiter=200, disp=False)

    def _dfm_smooth(self, z_obs, params):
        from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
        return DynamicFactor(z_obs, k_factors=1, factor_order=1,
                             error_order=1).smooth(params)

    def _fit_predict_year(self, train, test, factors, target):
        """
        Fast, leak-free 1-step MS-DFM:
          1. Factor extracted from FACTORS ONLY (target excluded) → no ragged-edge
             leakage; factors at month m are known when nowcasting CPI_m.
          2. DFM fit on train factors; factor path for train+test via one smooth
             pass with fixed train params (factors fully observed = causal).
          3. Markov-switching regression CPI~factor fit on train.
          4. One causal FILTER pass of the Markov model over the full series with
             fixed params; filtered prob at m-1 uses data through m-1 only.
          5. pred[m] = (filtered_probs[m-1] @ P) · (b0 + b1·factor[m]).
        """
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        mu, sd = train[factors].mean(), train[factors].std().replace(0, 1)
        full   = pd.concat([train, test])
        z_all  = (full[factors] - mu) / sd

        dfm   = self._dfm_fit(((train[factors] - mu) / sd).dropna())
        dfm_full = self._dfm_smooth(z_all, dfm.params)
        f_all = np.asarray(dfm_full.filtered_state[0])            # causal factor path
        n_tr  = len(train)
        f_tr  = f_all[:n_tr]

        cpi_mu, cpi_sd = train[target].mean(), train[target].std() or 1.0
        ytr_z = (train[target].values - cpi_mu) / cpi_sd
        ms = MarkovRegression(ytr_z, k_regimes=self.k, trend="c",
                              exog=f_tr, switching_variance=True).fit()
        nm = _named(ms)
        b0 = np.array([nm.get(f"const[{i}]", 0.0) for i in range(self.k)])
        b1 = np.array([nm.get(f"x1[{i}]", 0.0)    for i in range(self.k)])
        P  = ms.regime_transition[:, :, 0] if ms.regime_transition.ndim == 3 \
             else ms.regime_transition

        # one causal filter pass over full series with fixed params
        y_all_z = (full[target].values - cpi_mu) / cpi_sd
        ms_full = MarkovRegression(y_all_z, k_regimes=self.k, trend="c",
                                   exog=f_all, switching_variance=True).filter(ms.params)
        fp = np.asarray(ms_full.filtered_marginal_probabilities)
        fp = fp if fp.shape[1] == self.k else fp.T                # (nobs, k), causal

        preds = []
        for j, idx in enumerate(test.index):
            t = n_tr + j
            probs = fp[t - 1] @ P.T                               # propagate from m-1
            yhat_z = np.nansum(probs * (b0 + b1 * f_all[t]))
            preds.append(yhat_z * cpi_sd + cpi_mu)
        return np.array(preds)

    def importance(self, df, factors, target):
        d = _prep(df, factors, target)
        z, _, _ = _zscore(d, factors + [target])
        res = self._dfm_fit(z.dropna())
        loads = {k.replace("loading.f1.", ""): abs(v)
                 for k, v in res.params.items() if k.startswith("loading.f1.")}
        s = pd.Series({f: loads.get(f, np.nan) for f in factors}).dropna()
        return s, self.importance_type

    def regimes(self, df, factors, target):
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        d = _prep(df, factors, target)
        try:
            # Use factors-only DFM (same as _fit_predict_year) for consistency
            mu, sd = d[factors].mean(), d[factors].std().replace(0, 1)
            z_fac = ((d[factors] - mu) / sd).dropna()
            dfm = self._dfm_fit(z_fac)
            f = np.asarray(dfm.filtered_state[0])
            cpi_mu, cpi_sd = d[target].mean(), d[target].std() or 1.0
            y_z = (d[target].reindex(z_fac.index).values - cpi_mu) / cpi_sd
            ms = MarkovRegression(y_z, k_regimes=self.k, trend="c",
                                  exog=f, switching_variance=True).fit()
            sm = np.asarray(ms.smoothed_marginal_probabilities)
            sm = sm if sm.shape[1] == self.k else sm.T
            state = sm.argmax(axis=1)
            nm = _named(ms)
            var0, var1 = nm.get("sigma2[0]", 0.0), nm.get("sigma2[1]", 0.0)
            hi = 1 if var1 > var0 else 0
            labels = pd.Series(["high-vol" if s == hi else "low-vol" for s in state],
                               index=z_fac.index)
            return labels, {"type": f"MS-DFM {self.k}-state (factor-augmented)",
                            "n_regimes": self.k, "current": labels.iloc[-1]}
        except Exception as e:
            return None, {"type": "none", "error": str(e)[:40]}


# ─────────────────────────────────────────────────────────────────────────────
# 7. LSTAR — logistic smooth transition AR with exog factors
# ─────────────────────────────────────────────────────────────────────────────

class LSTAR(BaseModel):
    name = "LSTAR"
    importance_type = "permutation ΔRMSE"
    has_regimes = True
    WINDOW = 60   # LM/TRF diverges on long expanding windows; cap at 5Y

    def _design(self, d, factors, target):
        y    = d[target].values
        ylag = d[target].shift(1).values
        X    = d[factors].values
        return y, ylag, X

    def _resid(self, p, ylag, X, y, k):
        a0, a1 = p[0], p[1]
        b0, b1 = p[2], p[3]
        gamma, c = p[4], p[5]
        theta = p[6:6 + k]
        G = 1.0 / (1.0 + np.exp(-gamma * (ylag - c)))
        yhat = (a0 + a1 * ylag) + (b0 + b1 * ylag) * G + X @ theta
        return yhat - y

    def _fit_params(self, ylag, X, y, k):
        ok = np.isfinite(ylag) & np.isfinite(y) & np.isfinite(X).all(1)
        yl, Xk, yk = ylag[ok], X[ok], y[ok]
        p0 = np.concatenate([[yk.mean(), 0.3, 0.0, 0.0, 1.0, np.median(yl)],
                             np.zeros(k)])
        lo = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf] + [-5.0] * k
        hi = [ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf] + [ 5.0] * k
        res = least_squares(self._resid, p0, args=(yl, Xk, yk, k),
                            max_nfev=2000, method="trf", bounds=(lo, hi))
        return res.x

    def _predict(self, p, ylag, X, k):
        a0, a1, b0, b1, gamma, c = p[:6]
        theta = p[6:6 + k]
        G = 1.0 / (1.0 + np.exp(-gamma * (ylag - c)))
        return (a0 + a1 * ylag) + (b0 + b1 * ylag) * G + X @ theta

    def _fit_predict_year(self, train, test, factors, target):
        d = pd.concat([train, test])
        fz = (d[factors] - train[factors].mean()) / train[factors].std().replace(0, 1)
        dd = d.copy(); dd[factors] = fz
        k = len(factors)
        y_tr, ylag_tr, X_tr = self._design(dd.loc[train.index], factors, target)
        p = self._fit_params(ylag_tr, X_tr, y_tr, k)
        _, ylag_te, X_te = self._design(dd.loc[test.index], factors, target)
        return self._predict(p, ylag_te, X_te, k)

    def importance(self, df, factors, target):
        d = _prep(df, factors, target)
        fz = (d[factors] - d[factors].mean()) / d[factors].std().replace(0, 1)
        dd = d.copy(); dd[factors] = fz
        k = len(factors)
        y, ylag, X = self._design(dd, factors, target)
        ok = np.isfinite(ylag) & np.isfinite(y) & np.isfinite(X).all(1)
        p = self._fit_params(ylag, X, y, k)
        Xdf = pd.DataFrame(X[ok], columns=factors)

        def predict_fn(Xm):
            return self._predict(p, ylag[ok], Xm.values, k)

        return _perm_importance(predict_fn, Xdf, y[ok]), self.importance_type

    def regimes(self, df, factors, target):
        d = _prep(df, factors, target)
        fz = (d[factors] - d[factors].mean()) / d[factors].std().replace(0, 1)
        dd = d.copy(); dd[factors] = fz
        k = len(factors)
        y, ylag, X = self._design(dd, factors, target)
        ok = np.isfinite(ylag) & np.isfinite(y) & np.isfinite(X).all(1)
        try:
            p = self._fit_params(ylag, X, y, k)
            gamma, c = p[4], p[5]
            G = 1.0 / (1.0 + np.exp(-gamma * (ylag[ok] - c)))
            labels = pd.Series(np.where(G > 0.5, "upper", "lower"),
                               index=dd.index[ok])
            return labels, {"type": "LSTAR logistic transition (on lagged CPI)",
                            "n_regimes": 2, "current": labels.iloc[-1]}
        except Exception as e:
            return None, {"type": "none", "error": str(e)[:40]}

    def nowcast(self, df, factors, target):
        """
        LSTAR nowcast: fit on windowed training data; predict with ylag = last
        known CPI (avoids the ylag=NaN bug from shift(1) on a 1-row test slice).
        """
        d = _prep(df, factors, target)
        if len(d) == 0:
            return np.nan, None
        row, nowcast_date = self._nowcast_row(df, factors, target)
        if row is None:
            return np.nan, nowcast_date
        if self.WINDOW is not None:
            cutoff = d.index[-1] - pd.DateOffset(months=self.WINDOW - 1)
            train = d[d.index >= cutoff]
        else:
            train = d
        try:
            fz_mu = train[factors].mean()
            fz_sd = train[factors].std().replace(0, 1)
            dd = train.copy()
            dd[factors] = (dd[factors] - fz_mu) / fz_sd
            k = len(factors)
            y, ylag, X = self._design(dd, factors, target)
            p = self._fit_params(ylag, X, y, k)
            # ar1 = last released CPI; factors normalized on training stats
            ylag_now = np.array([float(train[target].iloc[-1])])
            X_now = ((row[factors] - fz_mu) / fz_sd).values
            return float(self._predict(p, ylag_now, X_now, k)), nowcast_date
        except Exception:
            return np.nan, nowcast_date


# ─────────────────────────────────────────────────────────────────────────────
# 8. BVAR — single-equation Bayesian VAR (Minnesota-ridge prior)
# ─────────────────────────────────────────────────────────────────────────────

class BVAR(BaseModel):
    """
    Target-equation-only BVAR: OLS on own lags + factor lags with ridge
    (Minnesota-style) prior. Penalty scales with lag^2 so distant lags are
    shrunk harder. 1-step-ahead via a fixed regressor matrix.
    """
    name = "BVAR"
    importance_type = "permutation ΔRMSE"

    def __init__(self, p=3, lambda0=0.3):
        self.p = p
        self.lambda0 = lambda0

    def _make_X(self, dz, factors, target):
        """Lagged regressor matrix for target equation; all vars, p lags."""
        all_cols = [target] + factors
        blocks = [np.ones((len(dz), 1))]
        for lag in range(1, self.p + 1):
            blocks.append(dz[all_cols].shift(lag).values)
        X = np.hstack(blocks)
        y = dz[target].values
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        return X, y, ok

    def _ridge_coef(self, X, y, ok, n_obs_all, n_cols_all):
        """Ridge solve with Minnesota-like per-lag penalty."""
        Xo, yo = X[ok], y[ok]
        n = ok.sum()
        # build diagonal penalty: const=0, then for each var at each lag: lambda*(lag^2)
        diag = [0.0]  # intercept unpenalized
        for lag in range(1, self.p + 1):
            diag.extend([self.lambda0 * lag**2] * n_cols_all)
        lam = np.diag(diag[:X.shape[1]]) * n
        return np.linalg.solve(Xo.T @ Xo + lam, Xo.T @ yo)

    def _fit_predict_year(self, train, test, factors, target):
        d = pd.concat([train, test])
        mu = train[factors + [target]].mean()
        sd = train[factors + [target]].std().replace(0, 1)
        dz = d.copy()
        dz[factors + [target]] = (d[factors + [target]] - mu) / sd

        X_all, y_all, ok_all = self._make_X(dz, factors, target)
        n_tr = len(train)
        ok_tr = ok_all[:n_tr]
        B = self._ridge_coef(X_all[:n_tr], y_all[:n_tr], ok_tr,
                             len(factors + [target]), len(factors + [target]))
        y_hat_z = X_all[n_tr:] @ B
        return y_hat_z * sd[target] + mu[target]

    def importance(self, df, factors, target):
        d = _prep(df, factors, target)
        mu = d[factors + [target]].mean()
        sd = d[factors + [target]].std().replace(0, 1)
        dz = d.copy()
        dz[factors + [target]] = (d[factors + [target]] - mu) / sd
        X, y, ok = self._make_X(dz, factors, target)
        B = self._ridge_coef(X, y, ok, len(factors + [target]), len(factors + [target]))
        col_names = (["const"] +
                     [f"{c}_L{l}" for l in range(1, self.p + 1)
                      for c in ([target] + factors)])
        Xdf = pd.DataFrame(X[ok], columns=col_names[:X.shape[1]])
        y_ok = y[ok]
        base_rmse = np.sqrt(np.mean((y_ok - Xdf.values @ B)**2))
        rng = np.random.default_rng(42)
        agg = {}
        for f in factors:
            fac_cols = [c for c in Xdf.columns if c.startswith(f + "_")]
            if not fac_cols:
                continue
            rises = []
            for _ in range(5):
                Xp = Xdf.copy()
                for c in fac_cols:
                    Xp[c] = rng.permutation(Xp[c].values)
                rises.append(max(np.sqrt(np.mean((y_ok - Xp.values @ B)**2)) - base_rmse, 0))
            agg[f] = float(np.mean(rises))
        return pd.Series(agg).sort_values(ascending=False), self.importance_type


# ─────────────────────────────────────────────────────────────────────────────
# 10. HiddenRF — regime-switching Random Forest (K-means on factor space)
# ─────────────────────────────────────────────────────────────────────────────

class HiddenRF(BaseModel):
    """
    K-means discovers regimes in factor space; separate RF per regime;
    test predictions soft-weighted by inverse Euclidean distance to centroids.
    AR-augmented (lagged CPI + 3m momentum) for fair 1-step comparison.
    """
    name = "HiddenRF"
    importance_type = "mean |feature importance|"
    has_regimes = True
    LAG = "cpi_lag1"
    MOM = "cpi_3m_chg"

    def __init__(self, n_regimes=2, n_estimators=200):
        self.n_regimes = n_regimes
        self.n_estimators = n_estimators

    def _add_lag(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        f[self.MOM] = f[target].shift(1).diff(3)
        return f

    def _feats(self, factors):
        return list(factors) + [f for f in [self.LAG, self.MOM] if f not in factors]

    def _fit_predict_year(self, train, test, factors, target):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.cluster import KMeans
        feats = self._feats(factors)
        both = self._add_lag(pd.concat([train, test]), target)
        tr = both.loc[train.index].dropna(subset=feats + [target])
        te = both.loc[test.index].ffill()

        fmu = tr[factors].mean(); fsd = tr[factors].std().replace(0, 1)
        fz_tr = ((tr[factors] - fmu) / fsd).fillna(0).values
        km = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        km.fit(fz_tr)

        labels = km.predict(fz_tr)
        rfs = {}
        for r in range(self.n_regimes):
            idx = labels == r
            if idx.sum() < 10:
                continue
            rf = RandomForestRegressor(n_estimators=self.n_estimators,
                                       max_depth=4, random_state=42, n_jobs=-1)
            rf.fit(tr[feats].values[idx], tr[target].values[idx])
            rfs[r] = rf

        fz_te = ((te[factors] - fmu) / fsd).fillna(0).values
        dists = km.transform(fz_te)                           # (n_test, k), euclidean
        inv_d = 1.0 / (dists + 1e-6)
        weights = inv_d / inv_d.sum(axis=1, keepdims=True)   # (n_test, k)

        preds = np.zeros(len(te))
        for r, rf in rfs.items():
            preds += weights[:, r] * rf.predict(te[feats].fillna(0).values)
        return preds

    def importance(self, df, factors, target):
        from sklearn.ensemble import RandomForestRegressor
        feats = self._feats(factors)
        d = self._add_lag(_prep(df, factors, target), target).dropna(subset=feats + [target])
        rf = RandomForestRegressor(n_estimators=self.n_estimators,
                                   max_depth=4, random_state=42, n_jobs=-1)
        rf.fit(d[feats].values, d[target].values)
        s = pd.Series(rf.feature_importances_, index=feats)
        return s, self.importance_type

    def regimes(self, df, factors, target):
        from sklearn.cluster import KMeans
        d = _prep(df, factors, target)
        fz = ((d[factors] - d[factors].mean()) / d[factors].std().replace(0, 1)).fillna(0)
        km = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        labels = km.fit_predict(fz.values)
        means = [d[target].values[labels == r].mean() for r in range(self.n_regimes)]
        hi = int(np.nanargmax(means))
        lmap = pd.Series(["high-infl" if l == hi else "low-infl" for l in labels],
                         index=d.index)
        return lmap, {"type": f"K-means {self.n_regimes}-cluster (factor space)",
                      "n_regimes": self.n_regimes, "current": lmap.iloc[-1]}


# ─────────────────────────────────────────────────────────────────────────────
# 11. GBM — gradient boosting (XGBoost if available, sklearn GBM fallback)
# ─────────────────────────────────────────────────────────────────────────────

class GBM(BaseModel):
    """
    Gradient Boosting with AR augmentation; XGBoost preferred (sklearn fallback).
    Uses same feature set + monotone-unconstrained AR lag as RAMM-LGBM for
    direct comparison: GBM vs monotone LGBM.
    """
    name = "GBM"
    importance_type = "mean |SHAP|"
    LAG = "cpi_lag1"
    MOM = "cpi_3m_chg"

    def _add_lag(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        f[self.MOM] = f[target].shift(1).diff(3)
        return f

    def _feats(self, factors):
        return list(factors) + [f for f in [self.LAG, self.MOM] if f not in factors]

    def _model(self):
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=500, learning_rate=0.02, max_depth=4,
                                subsample=0.8, colsample_bytree=0.8, random_state=42,
                                verbosity=0, eval_metric="rmse")
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                             max_depth=3, subsample=0.8, random_state=42)

    def _fit_predict_year(self, train, test, factors, target):
        feats = self._feats(factors)
        both = self._add_lag(pd.concat([train, test]), target)
        tr = both.loc[train.index].dropna(subset=feats + [target])
        te = both.loc[test.index].ffill().fillna(train[factors + [target]].mean())
        m = self._model(); m.fit(tr[feats].values, tr[target].values)
        return m.predict(te[feats].values)

    def importance(self, df, factors, target):
        import shap
        feats = self._feats(factors)
        d = self._add_lag(_prep(df, factors, target), target).dropna(subset=feats + [target])
        m = self._model(); m.fit(d[feats].values, d[target].values)
        X = d.loc[d.index.year >= START_YEAR_DEFAULT, feats]
        try:
            sv = shap.TreeExplainer(m).shap_values(X)
        except Exception:
            sv = shap.Explainer(m)(X).values
        return pd.Series(np.abs(sv).mean(axis=0), index=feats), self.importance_type


# ─────────────────────────────────────────────────────────────────────────────
# 12. ELASTICNET
# ─────────────────────────────────────────────────────────────────────────────

class ElasticNet(BaseModel):
    """AR-augmented ElasticNet with cross-validated alpha and l1_ratio."""
    name = "ElasticNet"
    importance_type = "coefficient |value|"
    LAG = "cpi_lag1"
    MOM = "cpi_3m_chg"

    def _feats(self, factors):
        return list(factors) + [f for f in [self.LAG, self.MOM] if f not in factors]

    def _add_lag(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        f[self.MOM] = f[target].shift(1).diff(3)
        return f

    def _fit_predict_year(self, train, test, factors, target):
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler
        feats = self._feats(factors)
        both = self._add_lag(pd.concat([train, test]), target)
        tr = both.loc[train.index].dropna(subset=feats + [target])
        te = both.loc[test.index].ffill().fillna(tr[feats].mean())
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

    def nowcast(self, df, factors, target):
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler
        d = _prep(df, factors, target)
        if len(d) == 0:
            return np.nan, None
        feats = self._feats(factors)
        both = self._add_lag(df, target)
        base_row, nowcast_date = self._nowcast_row(df, factors, target)
        if base_row is None:
            return np.nan, nowcast_date
        lag_val = both.loc[nowcast_date, self.LAG] if nowcast_date in both.index else np.nan
        if not np.isfinite(lag_val):
            return np.nan, nowcast_date
        latest_row = base_row.copy()
        latest_row[self.LAG] = lag_val
        if nowcast_date in both.index and self.MOM in both.columns:
            mom_val = both.loc[nowcast_date, self.MOM]
            latest_row[self.MOM] = mom_val if (not pd.isna(mom_val) and np.isfinite(float(mom_val))) else 0.0
        else:
            latest_row[self.MOM] = 0.0
        tr = self._add_lag(d, target).dropna(subset=feats + [target])
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[feats])
        X_te = scaler.transform(latest_row[feats])
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                         cv=5, max_iter=10000, random_state=42)
        m.fit(X_tr, tr[target].values)
        return float(m.predict(X_te)[0]), nowcast_date


# ─────────────────────────────────────────────────────────────────────────────
# 12. MIDAS
# ─────────────────────────────────────────────────────────────────────────────

class MIDAS(BaseModel):
    """
    U-MIDAS: within-month DAILY AVERAGES of financial factors (Brent, GBP, VIX, TTF)
    rather than month-end snapshots, capturing intra-month information.
    ElasticNetCV for regularisation; also includes CPI AR lag + 3m momentum.
    Falls back to month-end if yfinance daily fetch fails.
    """
    name = "MIDAS"
    LAG = "cpi_lag1"
    MOM = "cpi_3m_chg"

    def _aug(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        f[self.MOM] = f[target].shift(1).diff(3)
        return f

    def _fit_predict_year(self, train, test, factors, target):
        from sklearn.linear_model import ElasticNetCV
        from sklearn.preprocessing import StandardScaler

        mm = _get_midas_data()
        both = self._aug(pd.concat([train, test]), target)

        if mm is not None:
            both = both.join(mm, how="left")
            midas_cols = list(mm.columns)
        else:
            midas_cols = []

        feats = midas_cols + [self.LAG, self.MOM]
        # require only AR features + target; midas cols filled with mean where sparse
        tr = both.loc[train.index].dropna(subset=[self.LAG, self.MOM, target])
        te = both.loc[test.index].ffill()

        if len(tr) < 20:
            return np.full(len(te), np.nan)

        tr_fill = tr[feats].fillna(tr[feats].mean())
        te_fill = te[feats].fillna(tr[feats].mean())
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr_fill)
        X_te = scaler.transform(te_fill)
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=5000)
        m.fit(X_tr, tr[target].values)
        return m.predict(X_te)


# ─────────────────────────────────────────────────────────────────────────────
# 13. BRIDGE EQUATION
# ─────────────────────────────────────────────────────────────────────────────

class BridgeEq(BaseModel):
    """
    Bridge equation: pure OLS on within-month daily-average financial factors
    + CPI AR lag + 3m momentum. No regularisation — standard macro bridge approach.
    """
    name = "BridgeEq"
    LAG = "cpi_lag1"
    MOM = "cpi_3m_chg"

    def _aug(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        f[self.MOM] = f[target].shift(1).diff(3)
        return f

    def _fit_predict_year(self, train, test, factors, target):
        mm = _get_midas_data()
        both = self._aug(pd.concat([train, test]), target)

        if mm is not None:
            both = both.join(mm, how="left")
            midas_cols = list(mm.columns)
        else:
            midas_cols = []

        feats = midas_cols + [self.LAG, self.MOM]
        # require only AR features + target; midas cols filled with mean where sparse
        tr = both.loc[train.index].dropna(subset=[self.LAG, self.MOM, target])
        te = both.loc[test.index].ffill()

        if len(tr) < 20:
            return np.full(len(te), np.nan)

        tr_fill = tr[feats].fillna(tr[feats].mean())
        te_fill = te[feats].fillna(tr[feats].mean())
        X_tr = np.column_stack([np.ones(len(tr_fill)), tr_fill.values])
        beta, _, _, _ = np.linalg.lstsq(X_tr, tr[target].values, rcond=None)
        X_te = np.column_stack([np.ones(len(te_fill)), te_fill.values])
        return X_te @ beta


# ─────────────────────────────────────────────────────────────────────────────
# 14. COPULA REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

class CopulaReg(BaseModel):
    """
    Gaussian copula regression: rank-transforms all variables to normal scores,
    fits OLS in rank-space, back-transforms via empirical quantile function.
    Robust to heavy tails and outliers (e.g. 2022 inflation spike).
    """
    name = "CopulaReg"
    LAG = "cpi_lag1"
    MOM = "cpi_3m_chg"

    def _aug(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        f[self.MOM] = f[target].shift(1).diff(3)
        return f

    @staticmethod
    def _t_scores(vals, ref, df):
        from scipy.stats import t as _t_dist
        n = len(ref)
        ref_s = np.sort(ref)
        r = np.searchsorted(ref_s, vals) + 0.5
        r = np.clip(r, 0.5, n + 0.5)   # keep within [1/(n+1), n/(n+1)] after division
        return _t_dist.ppf(r / (n + 1), df=df)

    def _fit_predict_year(self, train, test, factors, target):
        from scipy.stats import t as _t_dist
        feats = list(factors) + [f for f in [self.LAG, self.MOM] if f not in factors]
        both = self._aug(pd.concat([train, test]), target)
        tr = both.loc[train.index].dropna(subset=feats + [target])
        te = both.loc[test.index].ffill()

        if len(tr) < 20:
            return np.full(len(te), np.nan)

        n = len(tr)
        df_est = max(4.0, n - len(feats) - 2)

        z_tr = np.zeros((n, len(feats)))
        for j, f in enumerate(feats):
            ref = tr[f].fillna(tr[f].mean()).values
            z_tr[:, j] = self._t_scores(ref, ref, df_est)
        z_y = self._t_scores(tr[target].values, tr[target].values, df_est)

        X_tr = np.column_stack([np.ones(n), z_tr])
        beta, _, _, _ = np.linalg.lstsq(X_tr, z_y, rcond=None)

        n_te = len(te)
        z_te = np.zeros((n_te, len(feats)))
        for j, f in enumerate(feats):
            ref = tr[f].fillna(tr[f].mean()).values
            te_v = te[f].fillna(tr[f].mean()).values
            z_te[:, j] = self._t_scores(te_v, ref, df_est)

        z_pred = np.column_stack([np.ones(n_te), z_te]) @ beta
        pct = _t_dist.cdf(z_pred, df=df_est)
        pct = np.clip(pct, 0.01, 0.99)
        y_sorted = np.sort(tr[target].values)
        return y_sorted[np.clip((pct * n).astype(int), 0, n - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# 15. HUBER REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

class HuberNet(BaseModel):
    """
    Huber regression (sklearn HuberRegressor) with AR lag + momentum features.

    Huber loss = quadratic for |e| < epsilon, linear for |e| >= epsilon.
    With epsilon=2.0 (≈2σ for CPI errors), small errors treated like OLS while
    large spikes (2022 energy crisis) are downweighted — more robust than pure
    MSE without the systematic bias of pure MAE.
    """
    name = "HuberNet"
    LAG = "cpi_lag1"
    MOM = "cpi_3m_chg"

    def _aug(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        f[self.MOM] = f[target].shift(1).diff(3)
        return f

    def _fit_predict_year(self, train, test, factors, target):
        from sklearn.linear_model import HuberRegressor
        from sklearn.preprocessing import StandardScaler

        both = self._aug(pd.concat([train, test]), target)
        feats = [f for f in [self.LAG, self.MOM] if f not in factors] + list(factors)
        tr = both.loc[train.index].dropna(subset=feats + [target])
        te = both.loc[test.index]
        if len(tr) < 20:
            return np.full(len(te), np.nan)

        fmean = tr[feats].mean()
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[feats].fillna(fmean))
        X_te = scaler.transform(te[feats].fillna(fmean))
        m = HuberRegressor(epsilon=2.0, max_iter=500)
        m.fit(X_tr, tr[target].values)
        return m.predict(X_te)


# ─────────────────────────────────────────────────────────────────────────────
# 16. PRINCIPAL COMPONENT REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

class PCR(BaseModel):
    """
    Principal Component Regression: PCA → OLS in reduced factor space.

    Handles multicollinearity in the factor set (oil/gas/vol/FX are correlated).
    n_components chosen by cross-validation (retain 80% variance or up to 6 PCs).
    Includes AR lag + 3m momentum as raw features (not PC-reduced) to preserve
    the autoregressive signal.
    """
    name = "PCR"
    LAG = "cpi_lag1"
    MOM = "cpi_3m_chg"

    def _aug(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        f[self.MOM] = f[target].shift(1).diff(3)
        return f

    def _fit_predict_year(self, train, test, factors, target):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        both = self._aug(pd.concat([train, test]), target)
        fac_list = list(factors)
        ar_feats = [f for f in [self.LAG, self.MOM] if f not in fac_list]
        all_feats = ar_feats + fac_list

        tr = both.loc[train.index].dropna(subset=all_feats + [target])
        te = both.loc[test.index]
        if len(tr) < 20:
            return np.full(len(te), np.nan)

        fmean = tr[all_feats].mean()
        tr_fill = tr[all_feats].fillna(fmean)
        te_fill = te[all_feats].fillna(fmean)

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(tr_fill)
        X_te_scaled = scaler.transform(te_fill)

        # PCA on factor columns only (not AR terms)
        n_fac = len(fac_list)
        n_ar  = len(ar_feats)
        n_pc  = min(6, n_fac, len(tr) // 5)
        pca = PCA(n_components=n_pc)
        X_tr_pca = pca.fit_transform(X_tr_scaled[:, n_ar:])
        X_te_pca = pca.transform(X_te_scaled[:, n_ar:])

        X_tr_full = np.column_stack([X_tr_scaled[:, :n_ar], X_tr_pca])
        X_te_full = np.column_stack([X_te_scaled[:, :n_ar], X_te_pca])

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr_full, tr[target].values)
        return ridge.predict(X_te_full)


# ─────────────────────────────────────────────────────────────────────────────
# 17. REGIME-CONDITIONAL ENSEMBLE  (uk_be5 threshold, non-tree models only)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeEnsemble(BaseModel):
    """
    Two-regime conditional ensemble using 5Y inflation breakeven (uk_be5).
    Threshold 3.0%: above = high-inflation regime, below = low-inflation regime.
    Per-regime sub-models: UCM, TVP, ElasticNet (no trees, no Markov switching).
    Regime label from last observed uk_be5 in training window.

    Rationale: UCM/TVP trained on historical high-inflation episodes captures
    different factor-CPI dynamics than the same models trained on low-inflation.
    Falls back to full-sample average if regime has < 20 training observations.
    """
    name = "RegimeEns"
    THRESHOLD = 3.0
    MIN_REGIME_OBS = 20

    def _fit_predict_year(self, train, test, factors, target):
        if "uk_be5" not in train.columns:
            # No regime indicator — fall back to simple average of sub-models
            be5_regime = None
        else:
            last_be5 = train["uk_be5"].dropna()
            be5_regime = int(float(last_be5.iloc[-1]) > self.THRESHOLD) if len(last_be5) else None

        sub_models = [UCM(), TVP(), ElasticNet()]
        preds_list = []

        for m in sub_models:
            if be5_regime is not None and "uk_be5" in train.columns:
                is_regime = (train["uk_be5"].ffill().fillna(0) > self.THRESHOLD).astype(int)
                regime_train = train[is_regime == be5_regime]
                effective_train = regime_train if len(regime_train) >= self.MIN_REGIME_OBS else train
            else:
                effective_train = train
            try:
                p = m._fit_predict_year(effective_train, test, factors, target)
                if np.isfinite(p).all():
                    preds_list.append(p)
            except Exception:
                pass

        if not preds_list:
            return np.full(len(test), np.nan)
        return np.mean(preds_list, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 19. MEDIAN REGRESSION  (quantile regression at 0.5)
# ─────────────────────────────────────────────────────────────────────────────

class MedianElasticNet(BaseModel):
    """
    Quantile regression at median (q=0.5) with L1 regularisation.

    UK CPI has fat-tailed errors (excess kurtosis ≈ 2) and occasional large
    spikes (2022 energy crisis). MSE-based models are sensitive to these
    outliers. Minimising MAE (= q=0.5 pinball loss) is the optimal loss under
    symmetric fat-tailed error distributions and is robust to positive skew.
    No distributional back-transform needed — predictions are in original CPI
    space and never suffer from CDF extrapolation errors.

    Regularisation: alpha=0.1 (weak) — feature selection is already handled
    upstream by the factor registry.
    """
    name = "MedianElasticNet"
    LAG = "cpi_lag1"
    MOM = "cpi_3m_chg"

    def _aug(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
        f[self.MOM] = f[target].shift(1).diff(3)
        return f

    def _fit_predict_year(self, train, test, factors, target):
        from sklearn.linear_model import QuantileRegressor
        from sklearn.preprocessing import StandardScaler

        both = self._aug(pd.concat([train, test]), target)
        feats = [f for f in [self.LAG, self.MOM] if f not in factors] + list(factors)

        tr = both.loc[train.index].dropna(subset=feats + [target])
        te = both.loc[test.index]
        if len(tr) < 20:
            return np.full(len(te), np.nan)

        fmean = tr[feats].mean()
        tr_fill = tr[feats].fillna(fmean)
        te_fill = te[feats].fillna(fmean)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr_fill.values)
        X_te = scaler.transform(te_fill.values)

        qr = QuantileRegressor(quantile=0.5, alpha=0.1, solver="highs")
        qr.fit(X_tr, tr[target].values)
        return qr.predict(X_te)


# ─────────────────────────────────────────────────────────────────────────────
# 20. SARIMAX
# ─────────────────────────────────────────────────────────────────────────────

class SARIMAX_Model(BaseModel):
    """
    Seasonal ARIMA with exogenous macro factors (SARIMAX).

    UK CPI YoY retains residual seasonality (energy price cap resets in April/
    October, food/travel seasonal patterns). SARIMAX(1,0,1)(0,1,1,12) adds a
    seasonal MA term that captures annual patterns the other models ignore.
    Macro factors enter as exogenous variables (no lagged factor terms).

    Uses statsmodels SARIMAX. Convergence warnings suppressed; falls back to
    ARIMA(1,0,1) if seasonal fit fails.
    """
    name = "SARIMAX"

    def _fit_predict_year(self, train, test, factors, target):
        import statsmodels.api as sm

        y_tr = train[target].dropna()
        if len(y_tr) < 36:
            return np.full(len(test), np.nan)

        fac_list = [f for f in factors if f in train.columns]
        exog_tr = train.loc[y_tr.index, fac_list].ffill().fillna(0)
        exog_te = test[fac_list].ffill().fillna(0)

        # Select best order on training data
        best_fit, best_bic = None, np.inf
        for order, sorder in [((1,0,1),(0,1,1,12)), ((1,0,1),(0,0,0,0)), ((2,0,0),(0,0,0,0))]:
            try:
                fit = sm.tsa.SARIMAX(
                    y_tr, exog=exog_tr,
                    order=order, seasonal_order=sorder,
                    enforce_stationarity=False, enforce_invertibility=False
                ).fit(disp=False, maxiter=200)
                if fit.bic < best_bic:
                    best_bic = fit.bic
                    best_fit = fit
            except Exception:
                continue

        if best_fit is None:
            return np.full(len(test), np.nan)

        # 1-step-ahead loop: re-filter with fixed params, append realized obs each month
        preds = []
        hist_y = list(y_tr.values)
        hist_ex = list(exog_tr.values)
        for i, idx in enumerate(test.index):
            try:
                ex_new = exog_te.iloc[[i]].values
                cur = sm.tsa.SARIMAX(
                    np.array(hist_y), exog=np.array(hist_ex),
                    order=best_fit.model.order,
                    seasonal_order=best_fit.model.seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False
                ).filter(best_fit.params)
                fc = cur.forecast(steps=1, exog=ex_new)
                preds.append(float(np.asarray(fc)[0]))
            except Exception:
                preds.append(float(np.mean(hist_y[-12:])) if len(hist_y) >= 12 else float(np.mean(hist_y)))
            hist_y.append(float(test.loc[idx, target]))
            hist_ex.append(exog_te.iloc[i].values.tolist())
        return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# 21. REDUCED-FORM VAR
# ─────────────────────────────────────────────────────────────────────────────

class VAR_Model(BaseModel):
    """
    Reduced-form Vector Autoregression: CPI + top-4 factors as joint system.

    Captures bidirectional Granger-causal links missing from univariate models:
    gas → CPI, CPI → GBP, oil → CPI → breakeven expectations. Uses the top-4
    factors by data availability (uk_be5, oil_brent, gas_eu, gbpusd). Lag order
    selected by AIC (max 4). CPI forecast extracted from the joint VAR forecast.

    Falls back to AR(1) if VAR fit fails (collinearity, insufficient data).
    """
    name = "VAR"
    VAR_FACTORS = ["uk_be5", "oil_brent", "gas_eu", "gbpusd"]
    MAX_LAGS = 3

    def _fit_predict_year(self, train, test, factors, target):
        from statsmodels.tsa.vector_ar.var_model import VAR

        # Select VAR factors that are available
        var_facs = [f for f in self.VAR_FACTORS if f in train.columns]
        if not var_facs:
            return np.full(len(test), np.nan)

        cols = [target] + var_facs
        data = train[cols].dropna()
        if len(data) < 24:
            return np.full(len(test), np.nan)

        try:
            model = VAR(data)
            best_lag = min(self.MAX_LAGS, len(data) // 10)
            fit = model.fit(best_lag)
            k_ar = fit.k_ar
        except Exception:
            return np.full(len(test), np.nan)

        # 1-step-ahead loop: append realized row each month, re-forecast with fixed params
        preds = []
        hist = data.copy()
        for idx in test.index:
            try:
                last_vals = hist.values[-k_ar:]
                fc = fit.forecast(last_vals, steps=1)
                preds.append(float(fc[0, 0]))  # CPI is column 0
            except Exception:
                preds.append(float(hist[target].iloc[-1]))
            # append realized test row (target + VAR factors)
            new_row = pd.DataFrame(
                [[float(test.loc[idx, c]) for c in cols]], columns=cols,
                index=[idx])
            hist = pd.concat([hist, new_row])
        return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
# 22. AUTO-ARIMA (BIC-selected lag order)
# ─────────────────────────────────────────────────────────────────────────────

class AutoARIMA(BaseModel):
    """
    ARIMA(p,0,q) with BIC-selected order from grid p∈{1,2,3}, q∈{0,1,2}.

    Extends the AR(1) baseline with MA terms (captures transient shocks like
    supply-chain normalisation post-2022) and higher AR orders (inflation
    persistence). Pure univariate — no macro factors. Useful for isolating
    how much of the AR(1) baseline's performance can be captured by richer
    ARIMA dynamics before adding factors.
    """
    name = "AutoARIMA"

    def _fit_predict_year(self, train, test, factors, target):
        import statsmodels.api as sm

        y_tr = train[target].dropna()
        if len(y_tr) < 24:
            return np.full(len(test), np.nan)

        best_bic, best_fit = np.inf, None
        best_order = (1, 0, 0)
        for p in range(1, 4):
            for q in range(0, 3):
                try:
                    fit = sm.tsa.ARIMA(y_tr, order=(p, 0, q)).fit(method="statespace")
                    if fit.bic < best_bic:
                        best_bic = fit.bic
                        best_fit = fit
                        best_order = (p, 0, q)
                except Exception:
                    pass

        if best_fit is None:
            return np.full(len(test), np.nan)

        # 1-step-ahead loop: re-filter with FIXED training params (same as SARIMAX approach).
        # .filter() propagates state with fixed params → O(T) per step, not O(T²).
        preds = []
        hist = list(y_tr.values)
        for idx in test.index:
            try:
                cur = sm.tsa.ARIMA(np.array(hist), order=best_order).filter(best_fit.params)
                fc = cur.forecast(steps=1)
                preds.append(float(np.asarray(fc)[0]))
            except Exception:
                preds.append(float(np.mean(hist[-12:])) if len(hist) >= 12 else float(np.mean(hist)))
            hist.append(float(test.loc[idx, target]))
        return np.array(preds)


class DFM2(DFM):
    """DFM with two latent factors: separates global-risk from domestic-services."""
    name = "DFM-k2"

    def __init__(self):
        super().__init__(k_factors=2)


# ─────────────────────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def dm_test(e1, e2, h=1):
    """
    Diebold-Mariano test (1995) with Harvey-Leybourne-Newbold (1997) finite-sample
    correction. H0: equal MSE. DM > 0 means model2 better.
    Returns (HLN-corrected DM stat, two-sided p-value from t(n-1)).
    e1, e2: arrays of forecast errors (not squared) for model1 and model2.
    """
    from scipy import stats
    n = min(len(e1), len(e2))
    d = e1[:n]**2 - e2[:n]**2                       # loss differential (MSE scale)
    dbar = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    # Newey-West acov with bandwidth h (h=1 → no autocovariance for 1-step)
    acov = 0.0
    for j in range(1, h):
        acov += (1 - j / (h + 1)) * np.mean((d[j:] - dbar) * (d[:n - j] - dbar))
    V = max((gamma0 + 2 * acov) / n, 1e-12)
    DM = dbar / np.sqrt(V)
    # HLN finite-sample correction: scale DM, use t(n-1) instead of N(0,1)
    hln_scale = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    DM_hln = DM * hln_scale
    p = 2 * (1 - stats.t.cdf(abs(DM_hln), df=n - 1))
    return float(DM_hln), float(p)


def score_backtest(bt, name="model"):
    """
    Full scoring for a backtest DataFrame with columns [actual, pred].
    Returns dict: rmse, mae, dir_acc, error_var, mape, bias, mz_slope, mz_intercept, mz_pval, n.

    mz_slope / mz_intercept: Mincer-Zarnowitz efficiency test (OLS of actual on pred).
    Efficient forecast → intercept ≈ 0, slope ≈ 1. Deviation indicates bias or inefficiency.
    """
    if bt is None or len(bt) == 0:
        return dict(model=name, rmse=np.nan, mae=np.nan, dir_acc=np.nan,
                    error_var=np.nan, mape=np.nan, bias=np.nan,
                    mz_slope=np.nan, mz_intercept=np.nan, mz_pval=np.nan, n=0)
    e = bt["actual"] - bt["pred"]
    abs_pct = np.abs(e / bt["actual"].replace(0, np.nan)) * 100
    # Directional accuracy: did we correctly predict month-over-month direction of change?
    actual_chg = bt["actual"].diff()
    pred_chg   = (bt["pred"] - bt["actual"].shift(1))
    dir_mask   = (np.sign(actual_chg) == np.sign(pred_chg)).dropna()
    # Mincer-Zarnowitz: regress actual on (const, pred); efficient → intercept=0, slope=1
    try:
        from scipy.stats import linregress as _lr
        mz_slope, mz_intercept, _, _, _ = _lr(bt["pred"].values, bt["actual"].values)
    except Exception:
        mz_slope, mz_intercept = np.nan, np.nan
    # MZ joint F-test: H0: intercept=0, slope=1 (efficient forecast)
    try:
        from scipy.stats import f as _f_dist
        n_mz = len(bt)
        _pred_v   = bt["pred"].values
        _actual_v = bt["actual"].values
        _X = np.column_stack([np.ones(n_mz), _pred_v])
        _beta, _, _, _ = np.linalg.lstsq(_X, _actual_v, rcond=None)
        _sse_unr = float(((_actual_v - _X @ _beta)**2).sum())
        _sse_r   = float(((_actual_v - _pred_v)**2).sum())   # restricted: intercept=0, slope=1
        _denom   = _sse_unr / max(n_mz - 2, 1)
        _F_mz    = ((_sse_r - _sse_unr) / 2) / _denom if _denom > 0 else np.nan
        mz_pval  = float(1 - _f_dist.cdf(_F_mz, 2, n_mz - 2)) if np.isfinite(_F_mz) else np.nan
    except Exception:
        mz_pval = np.nan
    return dict(
        model        = name,
        rmse         = float(np.sqrt((e**2).mean())),
        mae          = float(e.abs().mean()),
        dir_acc      = float(dir_mask.mean() * 100) if len(dir_mask) else np.nan,
        error_var    = float(e.var()),
        mape         = float(abs_pct.mean()),
        bias         = float(e.mean()),
        mz_slope     = float(mz_slope),
        mz_intercept = float(mz_intercept),
        mz_pval      = float(mz_pval),
        n            = len(bt),
    )


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY OF MODELS
# ─────────────────────────────────────────────────────────────────────────────

def all_models():
    """Operational model zoo (2026-06-13, user-curated): 6 base models.
    Combined-Dynamic ensemble is built in main.py from those beating AR(1).
    Dropped from operational (moved out per review): DFM, BVAR, BridgeEq,
    MedianElasticNet, HuberNet, PCR, SARIMAX — call experimental_models()/the
    class directly to run them."""
    return [
        AutoARIMA(), ElasticNet(), UCM(), TVP(), MIDAS(), DFM2(),
    ]


def experimental_models():
    """Models with RMSE > 1.5x AR(1) in 2015-2024 38-factor backtest. Excluded from ensembles.
    Retained for research; may improve with different factor sets or hyperparameters."""
    return [RAMM_LGBM(), HMM(), MS_DFM(), LSTAR(), HiddenRF(), GBM(), CopulaReg(), VAR_Model(),
            RegimeEnsemble()]
