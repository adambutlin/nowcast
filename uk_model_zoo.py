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
    mu, sd = ref[cols].mean(), ref[cols].std().replace(0, 1.0)
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

    def _fit_predict_year(self, train, test, factors, target):
        raise NotImplementedError

    def backtest(self, df, factors, target, start_year=START_YEAR_DEFAULT, min_train=24):
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
                    # fall back to the most-recent min_train rows before test start
                    pre = d[d.index.year < yr]
                    train = pre.iloc[-min_train:]
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
        ffilled  = df[all_cols].ffill().reindex([nowcast_date])
        row = feat_df.fillna(ffilled)
        # Target must stay NaN — it's the thing we're trying to predict
        if target in row.columns:
            row[target] = np.nan

        # NaN check on factor columns only (target NaN is expected)
        if row[feat_cols].isna().any(axis=1).iloc[0]:
            return None, nowcast_date   # still some factor genuinely unavailable

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
        z_tr, mu, sd = _zscore(train, obs)
        res = DynamicFactor(z_tr.dropna(), k_factors=self.k, factor_order=1,
                            error_order=1).fit(maxiter=200, disp=False)
        # 1-step-ahead: forecast month m, then append realized month-m obs (fixed params)
        z_te = (test[obs] - train[obs].mean()) / train[obs].std().replace(0, 1)
        preds, cur = [], res
        for idx in test.index:
            fc = cur.forecast(steps=1)
            val = fc[target].values[0] if hasattr(fc, "columns") else np.asarray(fc)[0]
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
        z, mu, sd = _zscore(d, obs)
        try:
            res = DynamicFactor(z.dropna(), k_factors=self.k, factor_order=1,
                                error_order=1).fit(maxiter=200, disp=False)
            fc = res.forecast(steps=1)
            val = fc[target].values[0] if hasattr(fc, "columns") else np.asarray(fc)[0]
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
    LAG = "cpi_lag1"                       # AR-augmentation for fair 1-step nowcast

    # monotone sign by factor name (default 0)
    MONO = {"oil_brent": 1, "gbpusd": -1, "uk_be5": 1,
            "uk_rents": 1, "uk_paye": 1, "uk_ashe_pay": 1,
            "uk_infl_swap_1y": 1, "gas_hh": 1, "gas_eu": 1, "cpi_lag1": 1}

    def _feats(self, factors):
        return factors + [self.LAG]

    def _add_lag(self, frame, target):
        f = frame.copy()
        f[self.LAG] = f[target].shift(1)
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
        # Augment the base feature row with cpi_lag1
        lag_val = both.loc[nowcast_date, self.LAG] if nowcast_date in both.index else np.nan
        if not np.isfinite(lag_val):
            return np.nan, nowcast_date
        latest_row = base_row.copy()
        latest_row[self.LAG] = lag_val
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
        X = np.column_stack([np.ones(len(d)),
                             d[target].shift(1).values,
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
        # tune R/Q from train residual variance
        ytr = train[target].values
        R = np.nanvar(np.diff(ytr)) + 1e-6
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
        R = np.nanvar(np.diff(d[target].values)) + 1e-6
        Q = np.eye(X.shape[1]) * R * self.delta
        _, betas = self._kalman(X[ok], y[ok], R, Q)
        Xok = X[ok]
        # contribution columns: [const, ar1, factors...]; importance over factors
        contrib = np.abs(betas * Xok)
        col_names = ["const", "ar1"] + factors
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
            R = np.nanvar(np.diff(y[ok])) + 1e-6
            Q = np.eye(X.shape[1]) * R * self.delta
            _, betas = self._kalman(X[ok], y[ok], R, Q)
            final_beta = betas[-1]
            # Nowcast feature: [const=1, ar1=last known CPI, normalized factors]
            ar1 = float(d[target].iloc[-1])
            row_fz = (row[factors] - fz_mu) / fz_sd
            x_now = np.concatenate([[1.0], [ar1], row_fz.values[0]])
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
            obs = factors + [target]
            z, _, _ = _zscore(d, obs)
            z = z.dropna()
            dfm = self._dfm_fit(z)
            f = np.asarray(dfm.filtered_state[0])
            ms = MarkovRegression(z[target].values, k_regimes=self.k, trend="c",
                                  exog=f, switching_variance=True).fit()
            sm = np.asarray(ms.smoothed_marginal_probabilities)
            sm = sm if sm.shape[1] == self.k else sm.T
            state = sm.argmax(axis=1)
            nm = _named(ms)
            var0, var1 = nm.get("sigma2[0]", 0.0), nm.get("sigma2[1]", 0.0)
            hi = 1 if var1 > var0 else 0
            labels = pd.Series(["high-vol" if s == hi else "low-vol" for s in state],
                               index=z.index)
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
# 9. MIDAS — Almon polynomial distributed-lag model
# ─────────────────────────────────────────────────────────────────────────────

class MIDAS(BaseModel):
    """
    MIDAS with Almon (Gamma-polynomial) restricted distributed lags.
    Each factor contributes K lags compressed through a degree-d polynomial;
    reduces params from K·|factors| to (d+1)·|factors| while capturing
    humped/decaying lag weight profiles.
    """
    name = "MIDAS"
    importance_type = "permutation ΔRMSE"

    def __init__(self, K=6, degree=2):
        self.K = K
        self.degree = degree

    def _almon_basis(self, s, K, degree):
        """Almon basis: columns are sum_k k^d * s_{t-k} for d=0..degree."""
        cols = {}
        for d in range(degree + 1):
            cols[d] = sum((k**d) * s.shift(k) for k in range(1, K + 1))
        return pd.DataFrame(cols)

    def _build_X(self, df, factors):
        parts = []
        for fac in factors:
            basis = self._almon_basis(df[fac], self.K, self.degree)
            basis.columns = [f"{fac}_A{d}" for d in range(self.degree + 1)]
            parts.append(basis)
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

    def _fit_predict_year(self, train, test, factors, target):
        d = pd.concat([train, test])
        mu = train[factors].mean(); sd = train[factors].std().replace(0, 1)
        dz = d.copy(); dz[factors] = (d[factors] - mu) / sd
        X_all = self._build_X(dz, factors)
        X_tr = X_all.loc[train.index]; y_tr = train[target].values
        ok_tr = np.isfinite(X_tr).all(1) & np.isfinite(y_tr)
        lam = 1e-4 * ok_tr.sum()
        Xo = X_tr[ok_tr].values
        B = np.linalg.solve(Xo.T @ Xo + lam * np.eye(Xo.shape[1]), Xo.T @ y_tr[ok_tr])
        X_te = X_all.loc[test.index]
        ok_te = np.isfinite(X_te).all(1)
        preds = np.full(len(test), np.nan)
        preds[ok_te] = X_te[ok_te].values @ B
        # fill any NaN from short history with train mean
        preds = np.where(np.isnan(preds), train[target].mean(), preds)
        return preds

    def importance(self, df, factors, target):
        d = _prep(df, factors, target)
        mu = d[factors].mean(); sd = d[factors].std().replace(0, 1)
        dz = d.copy(); dz[factors] = (d[factors] - mu) / sd
        X = self._build_X(dz, factors)
        y = d[target].values
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        lam = 1e-4 * ok.sum()
        Xo = X[ok].values
        B = np.linalg.solve(Xo.T @ Xo + lam * np.eye(Xo.shape[1]), Xo.T @ y[ok])
        Xdf = pd.DataFrame(Xo, columns=X.columns)
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
    AR-augmented (lagged CPI) for fair 1-step comparison.
    """
    name = "HiddenRF"
    importance_type = "mean |feature importance|"
    has_regimes = True
    LAG = "cpi_lag1"

    def __init__(self, n_regimes=2, n_estimators=200):
        self.n_regimes = n_regimes
        self.n_estimators = n_estimators

    def _add_lag(self, frame, target):
        f = frame.copy(); f[self.LAG] = f[target].shift(1); return f

    def _feats(self, factors):
        return factors + [self.LAG]

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

    def _add_lag(self, frame, target):
        f = frame.copy(); f[self.LAG] = f[target].shift(1); return f

    def _feats(self, factors):
        return factors + [self.LAG]

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
# 11. ELASTICNET
# ─────────────────────────────────────────────────────────────────────────────

class ElasticNet(BaseModel):
    """AR-augmented ElasticNet with cross-validated alpha and l1_ratio."""
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
        tr = self._add_lag(d, target).dropna(subset=feats + [target])
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(tr[feats])
        X_te = scaler.transform(latest_row[feats])
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                         cv=5, max_iter=10000, random_state=42)
        m.fit(X_tr, tr[target].values)
        return float(m.predict(X_te)[0]), nowcast_date


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
    Diebold-Mariano test (1995). H0: equal MSE. Sign: DM > 0 means model2 better.
    Returns (DM stat, two-sided p-value).
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
    p  = 2 * (1 - stats.norm.cdf(abs(DM)))
    return float(DM), float(p)


def score_backtest(bt, name="model"):
    """
    Full scoring for a backtest DataFrame with columns [actual, pred].
    Returns dict: rmse, mae, dir_acc, error_var, mape, bias.
    """
    if bt is None or len(bt) == 0:
        return dict(model=name, rmse=np.nan, mae=np.nan, dir_acc=np.nan,
                    error_var=np.nan, mape=np.nan, bias=np.nan, n=0)
    e = bt["actual"] - bt["pred"]
    abs_pct = np.abs(e / bt["actual"].replace(0, np.nan)) * 100
    return dict(
        model     = name,
        rmse      = float(np.sqrt((e**2).mean())),
        mae       = float(e.abs().mean()),
        dir_acc   = float((np.sign(bt["actual"]) == np.sign(bt["pred"])).mean() * 100),
        error_var = float(e.var()),
        mape      = float(abs_pct.mean()),
        bias      = float(e.mean()),
        n         = len(bt),
    )


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY OF MODELS
# ─────────────────────────────────────────────────────────────────────────────

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
    extras = [DFM2(), ElasticNet()]
    return base + rolling_5y + rolling_2y + extras
