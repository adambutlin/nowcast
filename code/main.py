"""
nowcast_cpi.py — full model comparison and nowcast for UK CPI YoY.

Runs all 22 models + ensembles, reports:
  1. Metrics table   : RMSE, MAE, DirAcc, MZ slope/intercept, Error Variance, MAPE, Bias, n
  2. DM / SPA table  : HLN-corrected DM stat + p-value vs AR(1) baseline for each model
  3. Error correlation matrix + greedy uncorrelated subset (Spearman ρ<0.5)
  4. Regimes table   : which regimes each model identifies (current state)
  5. Factor importance: top-5 factors per model with importance values
  6. Regime-model-combine: train models per regime, metamodel vs model-regime-combine

Combined models (post-processing, no double-fit):
  Combined-Static    : equal-weight average of all 10 models
  Combined-Dynamic   : monthly inverse-RMSE-weighted (12-month rolling)
  Combined-Superstar : equal-weight of models with DM>0 and p<0.10 vs AR(1)
  Combined-Absolute  : equal-weight of greedy-selected uncorrelated-error subset
  RMC-<method>       : regime-model-combine metamodel (one per regime method)

Mixed-frequency discipline:
  apply_publication_lags() is called on the factor matrix before all model runs.
  pub_lag=0 factors (oil, FX, VIX, PMI, ISM) used contemporaneously — available
  before CPI release (~16th of T+1). pub_lag=1 factors (ONS releases) shifted
  by 1 month so row T contains factor(T-1).

Usage:
  FRED_API_KEY=<key> python nowcast_cpi.py [--start 2015] [--train-from 1997]
  FRED_API_KEY=<key> python nowcast_cpi.py --start 2015 --rmc       # include regime-model-combine
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# project-root-relative output dirs (works whether run from root or code/)
_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA  = os.path.join(_ROOT, "data")
_PLOTS = os.path.join(_ROOT, "plots")
os.makedirs(_DATA,  exist_ok=True)
os.makedirs(_PLOTS, exist_ok=True)

import factors as F
import uk_model_zoo as Z


# ─────────────────────────────────────────────────────────────────────────────
# AR(1) BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def ar1_backtest(df, target, start_year=2015, min_train=60, end_year=None):
    """Expanding-window AR(1): fit rho + mu on train, predict each test obs using realized lag."""
    series = df[target].dropna()
    rows = []
    years = [y for y in series.index.year.unique()
             if y >= start_year and (end_year is None or y <= end_year)]
    for yr in sorted(years):
        train = series[series.index.year < yr]
        test  = series[series.index.year == yr]
        if len(train) < min_train or len(test) == 0:
            continue
        y = train.values
        mu   = float(y.mean())
        rho  = float(np.corrcoef(y[:-1], y[1:])[0, 1]) if len(y) > 1 else 0.0
        for idx in test.index:
            y_prev = series.loc[:idx].dropna().iloc[-2] if len(series.loc[:idx].dropna()) >= 2 else mu
            pred   = mu + rho * (float(y_prev) - mu)
            actual = float(test.loc[idx])
            if np.isfinite(actual) and np.isfinite(pred):
                rows.append(dict(date=idx, actual=actual, pred=pred, year=yr))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED FORECASTS
# ─────────────────────────────────────────────────────────────────────────────

def combine_static(bt_dict):
    """Equal-weight average across models at each date."""
    aligned = pd.DataFrame({n: bt["pred"] for n, bt in bt_dict.items()
                             if bt is not None and len(bt) > 0})
    if aligned.empty:
        return pd.DataFrame()
    pred = aligned.mean(axis=1)
    for n, bt in bt_dict.items():
        if bt is not None and len(bt) > 0 and "actual" in bt.columns:
            actual = bt["actual"]
            break
    else:
        return pd.DataFrame()
    out = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    out["year"] = out.index.year
    return out


def combine_dynamic(bt_dict, window=12):
    """
    Monthly inverse-RMSE weights using a rolling `window`-month lookback.
    At each date t, weight for model m = 1 / RMSE(t-window..t-1).
    """
    aligned = pd.DataFrame({n: bt["pred"] for n, bt in bt_dict.items()
                             if bt is not None and len(bt) > 0})
    if aligned.empty:
        return pd.DataFrame()
    for n, bt in bt_dict.items():
        if bt is not None and len(bt) > 0 and "actual" in bt.columns:
            actual = bt["actual"].reindex(aligned.index)
            break
    else:
        return pd.DataFrame()

    preds = []
    for i, idx in enumerate(aligned.index):
        if i < window:
            preds.append(aligned.iloc[i].mean())
            continue
        window_actual = actual.iloc[i - window:i]
        weights = {}
        for col in aligned.columns:
            window_pred = aligned[col].iloc[i - window:i]
            ok = window_actual.notna() & window_pred.notna()
            if ok.sum() < 3:
                weights[col] = 0.0
                continue
            rmse = np.sqrt(((window_actual[ok] - window_pred[ok])**2).mean())
            weights[col] = 1.0 / (rmse + 1e-6)
        total = sum(weights.values())
        if total == 0:
            preds.append(aligned.iloc[i].mean())
            continue
        w = {c: v / total for c, v in weights.items()}
        pred_t = sum(w[c] * aligned[c].iloc[i] for c in aligned.columns)
        preds.append(pred_t)

    out = pd.DataFrame({"actual": actual, "pred": preds}, index=aligned.index).dropna()
    out["year"] = out.index.year
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LEAKAGE PROBE
# ─────────────────────────────────────────────────────────────────────────────

def probe_leakage(df, factor, target, live_facs, models_subset, start_year=2015):
    """
    Tests whether a factor carries data leakage by comparing 3 backtest scenarios.
    Uses raw df (pre-publication-lag) to show the lag=0 leakage explicitly.
    """
    from scipy.stats import spearmanr

    print(f"\n{'═'*65}")
    print(f"LEAKAGE PROBE: {factor}")
    print(f"{'═'*65}")

    print("  Spearman ρ(factor_t-lag, cpi_yoy_t):")
    common = df[[factor, target]].dropna()
    for lag in range(0, 5):
        shifted = common[factor].shift(lag)
        valid = shifted.notna() & common[target].notna()
        rho, pv = spearmanr(shifted[valid], common[target][valid])
        flag = " ← contemporaneous" if lag == 0 else (" ← use this as feature" if lag == 1 else "")
        print(f"    lag={lag}: ρ={rho:+.4f}  p={pv:.3f}{flag}")

    scenarios = {
        "original":  df.copy(),
        "lagged_1m": df.assign(**{factor: df[factor].shift(1)}),
        "excluded":  df.drop(columns=[factor]),
    }
    print(f"\n  RMSE by scenario (mean over {len(models_subset)} fast models):")
    scenario_rmse = {}
    for label, data in scenarios.items():
        facs = [f for f in live_facs if f in data.columns]
        rmses = []
        for m in models_subset:
            try:
                bt = m.backtest(data, facs, target, start_year=start_year)
                if len(bt) > 0:
                    rmses.append(float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean())))
            except Exception:
                pass
        mean_rmse = float(np.mean(rmses)) if rmses else np.nan
        scenario_rmse[label] = mean_rmse
        print(f"    {label:<12}: {mean_rmse:.4f}")

    orig  = scenario_rmse.get("original",  np.nan)
    lag1  = scenario_rmse.get("lagged_1m", np.nan)
    excl  = scenario_rmse.get("excluded",  np.nan)
    lift  = lag1 - orig
    indep = lag1 - excl
    print(f"\n  Leakage lift   (lagged - original): {lift:+.4f}pp RMSE")
    print(f"  Independent signal (lagged - excl): {indep:+.4f}pp RMSE")
    if abs(lift) > 0.05:
        print(f"  VERDICT: LEAKAGE DETECTED — {factor} at lag=0 provides {lift:.3f}pp unearned RMSE.")
        print(f"           Fixed by apply_publication_lags() in this run.")
    elif indep < -0.02:
        print(f"  VERDICT: clean — genuine independent signal at lag=1.")
    else:
        print(f"  VERDICT: minimal signal at lag=1.")
    return scenario_rmse


# ─────────────────────────────────────────────────────────────────────────────
# ERROR CORRELATION MATRIX & UNCORRELATED SUBSET
# ─────────────────────────────────────────────────────────────────────────────

def error_corr_matrix(bt_dict):
    """Spearman correlation matrix of forecast errors across all models."""
    from scipy.stats import spearmanr
    errors = {}
    for name, bt in bt_dict.items():
        if bt is not None and len(bt) > 0:
            errors[name] = (bt["actual"] - bt["pred"]).rename(name)
    if len(errors) < 2:
        return pd.DataFrame(), pd.DataFrame()
    err_df = pd.concat(errors.values(), axis=1).dropna()
    names = list(err_df.columns)
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                rho, _ = spearmanr(err_df[n1], err_df[n2])
                mat.loc[n1, n2] = mat.loc[n2, n1] = rho
    return err_df, mat


def greedy_uncorrelated_subset(corr_mat, bt_dict, rho_threshold=0.5, ar1_rmse=None):
    """
    Greedy selection: start with best model (lowest RMSE), add a candidate only if:
      1. max |rho| with already-selected models < rho_threshold
      2. candidate RMSE < 1.0 x AR(1) RMSE  [must strictly beat AR(1)]
    Returns [] if ar1_rmse is None (cannot gate without baseline).
    """
    if ar1_rmse is None:
        return []
    rmse_map = {}
    for name, bt in bt_dict.items():
        if bt is not None and len(bt) > 0 and name in corr_mat.index:
            rmse_map[name] = float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean()))
    ranked = sorted((n for n in rmse_map if rmse_map[n] < ar1_rmse), key=rmse_map.get)
    selected = []
    for cand in ranked:
        if not selected:
            selected.append(cand)
            continue
        max_rho = max(abs(corr_mat.loc[cand, s]) for s in selected
                      if s in corr_mat.columns)
        if max_rho < rho_threshold:
            selected.append(cand)
    return selected


def combine_subset(bt_dict, names, label="subset"):
    """Equal-weight combination over a named subset of models."""
    valid = {n: bt_dict[n] for n in names if bt_dict.get(n) is not None and len(bt_dict[n]) > 0}
    return combine_static(valid)


def combine_recursive(bt_dict, bench_bt, rho_threshold=0.5, min_hist=36, min_obs=12):
    """
    Walk-forward selected ensemble (leak-free successor to Combined-Absolute).

    Membership for test year y is decided ONLY on backtest rows from years < y:
      1. AR(1) gate: candidate's historical RMSE < benchmark's historical RMSE
      2. greedy decorrelation: max |Spearman rho| of error series vs already-
         selected members < rho_threshold (insufficient overlap blocks)
    Equal-weight fallback over all models while history < min_hist months or
    when no candidate passes the gate. C1: no statistic computed on year y or
    later ever influences year-y membership.
    """
    cand = {n: bt for n, bt in bt_dict.items()
            if n != "AR(1)" and bt is not None and len(bt) > 0}
    if not cand or bench_bt is None or len(bench_bt) == 0:
        return pd.DataFrame()
    preds = pd.DataFrame({n: bt["pred"] for n, bt in cand.items()}).sort_index()
    actual = bench_bt["actual"].reindex(preds.index)
    for bt in cand.values():
        actual = actual.fillna(bt["actual"].reindex(preds.index))

    rows = []
    for yr in sorted(preds.index.year.unique()):
        hist = preds.index[preds.index.year < yr]
        members = list(preds.columns)                      # fallback: all models
        if len(hist) >= min_hist:
            b_hist = bench_bt[bench_bt.index.year < yr]
            ar1_rmse = (float(np.sqrt(((b_hist["actual"] - b_hist["pred"])**2).mean()))
                        if len(b_hist) else np.nan)
            rmse_map, errs = {}, {}
            for n in preds.columns:
                e = (actual.loc[hist] - preds[n].loc[hist]).dropna()
                if len(e) >= min_obs:
                    rmse_map[n] = float(np.sqrt((e**2).mean()))
                    errs[n] = e
            if np.isfinite(ar1_rmse):
                from scipy.stats import spearmanr
                ranked = sorted((n for n in rmse_map if rmse_map[n] < ar1_rmse),
                                key=rmse_map.get)
                sel = []
                for c in ranked:
                    if not sel:
                        sel.append(c)
                        continue
                    rhos = []
                    for s_ in sel:
                        common = errs[c].index.intersection(errs[s_].index)
                        if len(common) < min_obs:
                            rhos.append(1.0)               # too little overlap: block
                        else:
                            r, _ = spearmanr(errs[c].loc[common], errs[s_].loc[common])
                            rhos.append(abs(r) if np.isfinite(r) else 1.0)
                    if max(rhos) < rho_threshold:
                        sel.append(c)
                if sel:
                    members = sel
        for idx in preds.index[preds.index.year == yr]:
            p = preds.loc[idx, members].dropna()
            a = actual.loc[idx]
            if len(p) > 0 and np.isfinite(a):
                rows.append(dict(date=idx, actual=float(a),
                                 pred=float(p.mean()), year=yr))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# SPA TABLE
# ─────────────────────────────────────────────────────────────────────────────

def spa_table(bt_dict, benchmark_name="AR(1)"):
    """DM test for each model vs benchmark. DM > 0 → model beats benchmark."""
    if benchmark_name not in bt_dict or bt_dict[benchmark_name] is None:
        return pd.DataFrame()
    bench = bt_dict[benchmark_name]
    rows = []
    for name, bt in bt_dict.items():
        if bt is None or len(bt) == 0:
            rows.append(dict(model=name, DM=np.nan, p=np.nan, beats="?", n=0))
            continue
        common = bench.index.intersection(bt.index)
        if len(common) < 10:
            rows.append(dict(model=name, DM=np.nan, p=np.nan, beats="?", n=0))
            continue
        e_bench = (bench.loc[common, "actual"] - bench.loc[common, "pred"]).values
        e_model = (bt.loc[common, "actual"] - bt.loc[common, "pred"]).values
        dm, p = Z.dm_test(e_bench, e_model)
        sig = "**" if p < 0.05 else ("*" if p < 0.10 else "")
        beats = f"yes{sig}" if dm > 0 and p < 0.10 else ("no" if dm <= 0 else f"~{sig}")
        rows.append(dict(model=name, DM=round(dm, 2), p=round(p, 3), beats=beats, n=len(common)))
    return pd.DataFrame(rows).set_index("model")


def gw_test(e1, e2, h=1):
    """Giacomini-White test: regress loss diff on constant + lagged loss diff.
    Tests conditional predictability (time-varying forecast ability).
    H0: no conditional predictability improvement of e1 over e2."""
    from scipy import stats
    d = e1**2 - e2**2
    d = d.dropna()
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    d_lag = d.shift(h).dropna()
    d_curr = d.iloc[h:]
    X = np.column_stack([np.ones(len(d_lag)), d_lag])
    # OLS
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ d_curr.values
    resid = d_curr.values - X @ beta
    s2 = resid @ resid / max(len(d_curr) - 2, 1)
    V = s2 * XtX_inv
    # Wald stat (joint test on both coefficients)
    W = beta @ np.linalg.pinv(V) @ beta
    p = 1 - stats.chi2.cdf(W, df=2)
    return float(W), float(p)


# ─────────────────────────────────────────────────────────────────────────────
# REGIME-MODEL-COMBINE
# ─────────────────────────────────────────────────────────────────────────────

def _regime_labels_hmm(series, k=2):
    """
    Semi-causal HMM regime labels: fit on full data, run forward filter.
    Parameters from full sample (slight lookahead), state sequence from
    causal forward filter. Standard approach in the nowcasting literature.
    Returns pd.Series of 'r0'/'r1' labels on series.dropna().index.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    s = series.dropna()
    try:
        res = MarkovRegression(s.values, k_regimes=k, trend="c",
                               switching_variance=True).fit(disp=False)
        params = np.asarray(res.params)
        filt = MarkovRegression(s.values, k_regimes=k, trend="c",
                                switching_variance=True).filter(params)
        fp = np.asarray(filt.filtered_marginal_probabilities)
        fp = fp if fp.shape[1] == k else fp.T
        pnames = dict(zip(res.model.param_names, np.asarray(res.params)))
        means = [pnames.get(f"const[{i}]", np.nan) for i in range(k)]
        hi = int(np.nanargmax(means))
        states = fp.argmax(axis=1)
        return pd.Series([f"r{1 if st == hi else 0}" for st in states], index=s.index)
    except Exception:
        return pd.Series(np.where(s > s.mean(), "r1", "r0"), index=s.index)


def _regime_labels_hmm_recursive(series, k=2, min_fit=60):
    """Fit HMM on train<yr, forward-filter with fixed params on train+test. No lookahead."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    labels = pd.Series(index=series.index, dtype=object)
    dates = series.index
    years = sorted(series.index.year.unique())
    for yr in years:
        train_mask = dates.year < yr
        if train_mask.sum() < min_fit:
            labels[dates.year == yr] = "r0"
            continue
        train = series[train_mask].dropna()
        if len(train) < min_fit:
            labels[dates.year == yr] = "r0"
            continue
        try:
            res = MarkovRegression(train, k_regimes=k, trend="c", switching_variance=True).fit(disp=False)
            # forward-filter with FIXED training params — no lookahead into test year
            eval_series = series[dates.year <= yr].dropna()
            res2 = MarkovRegression(eval_series, k_regimes=k, trend="c",
                                    switching_variance=True).filter(res.params)
            fp = np.asarray(res2.filtered_marginal_probabilities)
            fp = fp if fp.shape[1] == k else fp.T   # ensure (nobs, k)
            eval_mask = eval_series.index.year == yr
            yr_labels = fp[eval_mask].argmax(axis=1)
            labels[dates.year == yr] = [f"r{int(v)}" for v in yr_labels]
        except Exception:
            labels[dates.year == yr] = "r0"
    return labels


def _regime_labels_lstar(df, factors, target):
    """LSTAR G function: G>0.5 → r1 (upper regime), else r0."""
    d = Z._prep(df, factors, target)
    fz = (d[factors] - d[factors].mean()) / d[factors].std().replace(0, 1)
    dd = d.copy(); dd[factors] = fz
    k = len(factors)
    model = Z.LSTAR()
    y, ylag, X = model._design(dd, factors, target)
    ok = np.isfinite(ylag) & np.isfinite(y) & np.isfinite(X).all(1)
    try:
        p = model._fit_params(ylag[ok], X[ok], y[ok], k)
        gamma, c = p[4], p[5]
        G = 1.0 / (1.0 + np.exp(-gamma * (ylag[ok] - c)))
        return pd.Series(np.where(G > 0.5, "r1", "r0"), index=dd.index[ok])
    except Exception:
        return pd.Series(np.where(d[target] > d[target].mean(), "r1", "r0"),
                         index=d.index)


def _regime_labels_dfm(df, factors, target):
    """DFM latent factor sign: causal filtered state sign."""
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
    d = Z._prep(df, factors, target)
    obs = factors + [target]
    z, _, _ = Z._zscore(d, obs)
    try:
        res = DynamicFactor(z.dropna(), k_factors=1, factor_order=1,
                            error_order=1).fit(maxiter=200, disp=False)
        f = np.asarray(res.filtered_state[0])
        return pd.Series(np.where(f > 0, "r1", "r0"), index=z.dropna().index)
    except Exception:
        return pd.Series(np.where(d[target] > d[target].mean(), "r1", "r0"),
                         index=d.index)


def _regime_labels_dfm_k2(df, factors, target):
    """DFM k=2 regime labels: KMeans-2 on two latent factor scores."""
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
    from sklearn.cluster import KMeans
    d = Z._prep(df, factors, target)
    obs = factors + [target]
    z, _, _ = Z._zscore(d, obs)
    clean = z.dropna()
    try:
        res = DynamicFactor(clean, k_factors=2, factor_order=1,
                            error_order=1).fit(maxiter=300, disp=False)
        F2 = np.asarray(res.filtered_state[:2]).T  # (T, 2)
        km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(F2)
        labels_raw = km.labels_
        # r1 = cluster with higher mean of first factor (higher-activity regime)
        means = [F2[labels_raw == k, 0].mean() for k in range(2)]
        hi = int(np.argmax(means))
        return pd.Series([f"r{1 if l == hi else 0}" for l in labels_raw],
                         index=clean.index)
    except Exception:
        return pd.Series(np.where(d[target] > d[target].mean(), "r1", "r0"),
                         index=d.index)


def _regime_labels_vix(df):
    """Manual VIX-based regime: above expanding median = stress (r1)."""
    if "vix" not in df.columns:
        return None
    med = df["vix"].expanding(min_periods=12).median()
    return pd.Series(np.where(df["vix"] > med, "r1", "r0"), index=df.index).dropna()



def regime_model_combine(df, factors, target, models, start_year=2015,
                          end_year=None, regime_methods=None, min_regime_train=30):
    """
    Regime-first framework: split training data by regime, train each model
    per-regime, drop models that don't beat AR(1) within that regime, build
    a metamodel that selects models based on the current regime signal.

    Architecture (contrast with model-regime-combine):
      model-regime-combine: model.fit(all_data) → model infers regime internally
      regime-model-combine: identify_regime(t) → select_regime_models(t) → ensemble

    Parameters:
      df           : factor+target DataFrame (publication lags already applied)
      factors      : list of factor column names
      target       : target column name
      models       : list of BaseModel instances
      start_year   : first backtest year
      regime_methods: list of strings in {'hmm','lstar','dfm','manual_vix'};
                     default: all four
      min_regime_train: minimum training obs per regime before fitting

    Returns:
      results: dict keyed by method name, each containing:
        'bt'       : metamodel backtest DataFrame
        'perf'     : DataFrame[regime × model → rmse, beats_ar1]
        'surviving': dict {regime → [model_names]}
        'ar1_rmse' : float — AR(1) RMSE on same test set
    """
    if regime_methods is None:
        regime_methods = ["hmm", "lstar", "dfm", "manual_vix"]

    series = df[target].dropna()
    results = {}

    for method in regime_methods:
        print(f"\n  RMC [{method}]: computing regime labels …", end="", flush=True)

        # ── compute causal regime labels ──────────────────────────────────
        try:
            if method == "hmm":
                labels = _regime_labels_hmm_recursive(series, k=2)
            elif method == "lstar":
                labels = _regime_labels_lstar(df, factors, target)
            elif method == "dfm":
                labels = _regime_labels_dfm(df, factors, target)
            elif method == "dfm_k2":
                labels = _regime_labels_dfm_k2(df, factors, target)
            elif method == "manual_vix":
                labels = _regime_labels_vix(df)
            else:
                continue
            if labels is None or len(labels) == 0:
                print(" SKIP (no labels)")
                continue
        except Exception as e:
            print(f" SKIP ({e})")
            continue

        regimes = sorted(labels.unique())
        print(f" regimes={regimes}", end="", flush=True)

        # ── per-regime, per-model backtest ────────────────────────────────
        # regime_bt[regime][model_name] = backtest DataFrame
        regime_bt = {r: {} for r in regimes}

        for m in models:
            for r in regimes:
                regime_idx = labels[labels == r].index
                rows = []
                for yr in sorted(y for y in df.index.year.unique()
                                 if y >= start_year and (end_year is None or y <= end_year)):
                    # regime-r training data: years < yr AND regime == r
                    train_all   = df[df.index.year < yr]
                    train_r_idx = regime_idx.intersection(train_all.index)
                    train_r     = df.loc[train_r_idx]

                    # fall back to full training data if not enough regime-r data
                    train = train_r if len(train_r) >= min_regime_train else train_all

                    # test data: year == yr AND regime == r
                    test_all   = df[df.index.year == yr]
                    test_r_idx = regime_idx.intersection(test_all.index)
                    test_r     = df.loc[test_r_idx]

                    if len(test_r) == 0 or len(train) < min_regime_train:
                        continue

                    try:
                        preds = m._fit_predict_year(train, test_r, factors, target)
                        for date, actual, pred in zip(
                                test_r.index, test_r[target].values, preds):
                            # sanity: CPI YoY can't plausibly exceed ±50%
                            if np.isfinite(actual) and np.isfinite(pred) and abs(pred) < 50:
                                rows.append(dict(date=date, actual=float(actual),
                                                 pred=float(pred), year=yr))
                    except Exception:
                        pass

                if rows:
                    regime_bt[r][m.name] = pd.DataFrame(rows).set_index("date")

        # ── AR(1) RMSE per regime ─────────────────────────────────────────
        ar1_by_regime = {}
        for r in regimes:
            r_idx = labels[labels == r].index
            r_series = series.reindex(r_idx).dropna()
            if len(r_series) < 30:
                ar1_by_regime[r] = np.nan
                continue
            # simple AR(1) on regime-r subset
            y = r_series.values
            rows_ar1 = []
            start_yr_r = r_series.index.year.min() + 5
            for yr in sorted(vy for vy in r_series.index.year.unique() if vy >= start_yr_r):
                y_tr = r_series[r_series.index.year < yr].values
                y_te = r_series[r_series.index.year == yr]
                if len(y_tr) < 20:
                    continue
                mu  = float(y_tr.mean())
                rho = float(np.corrcoef(y_tr[:-1], y_tr[1:])[0, 1]) if len(y_tr) > 1 else 0.0
                for idx in y_te.index:
                    prev = r_series.loc[:idx].dropna()
                    y_prev = float(prev.iloc[-2]) if len(prev) >= 2 else mu
                    pred   = mu + rho * (y_prev - mu)
                    actual = float(y_te.loc[idx])
                    if np.isfinite(actual) and np.isfinite(pred):
                        rows_ar1.append(dict(actual=actual, pred=pred))
            if rows_ar1:
                bt_ar1 = pd.DataFrame(rows_ar1)
                ar1_by_regime[r] = float(
                    np.sqrt(((bt_ar1["actual"] - bt_ar1["pred"])**2).mean()))
            else:
                ar1_by_regime[r] = np.nan

        # ── evaluate per-regime model performance ─────────────────────────
        perf_rows = []
        surviving = {r: [] for r in regimes}

        for r in regimes:
            ar1_r = ar1_by_regime.get(r, np.nan)
            for m in models:
                bt = regime_bt[r].get(m.name)
                if bt is None or len(bt) == 0:
                    perf_rows.append(dict(regime=r, model=m.name, rmse=np.nan,
                                          ar1_rmse=ar1_r, beats_ar1=False, n=0))
                    continue
                rmse = float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean()))
                beats = (not np.isnan(ar1_r)) and (rmse < ar1_r)
                if beats:
                    surviving[r].append(m.name)
                perf_rows.append(dict(regime=r, model=m.name, rmse=round(rmse, 4),
                                      ar1_rmse=round(ar1_r, 4) if not np.isnan(ar1_r) else np.nan,
                                      beats_ar1=beats, n=len(bt)))

        # ── build metamodel predictions ───────────────────────────────────
        # At each test month: use the regime label → average of surviving models
        # Fall back to best overall model (UCM) if no survivors in regime
        all_dates = set()
        for r in regimes:
            for bt in regime_bt[r].values():
                all_dates.update(bt.index.tolist())

        meta_rows = []
        for date in sorted(all_dates):
            # find regime at this date
            if date not in labels.index:
                continue
            r = labels.loc[date]
            surv = surviving.get(r, [])

            preds_at_date = []
            actual_at_date = None
            for m_name in (surv if surv else [m.name for m in models]):
                bt = regime_bt.get(r, {}).get(m_name)
                if bt is not None and date in bt.index:
                    preds_at_date.append(float(bt.loc[date, "pred"]))
                    actual_at_date = float(bt.loc[date, "actual"])

            if preds_at_date and actual_at_date is not None:
                meta_rows.append(dict(date=date, actual=actual_at_date,
                                      pred=float(np.mean(preds_at_date)),
                                      year=date.year))

        bt_meta = (pd.DataFrame(meta_rows).set_index("date")
                   if meta_rows else pd.DataFrame())
        perf_df = pd.DataFrame(perf_rows)

        # ── overall AR(1) on same test dates ─────────────────────────────
        if not bt_meta.empty:
            common_dates = bt_meta.index
            ar1_bt = ar1_backtest(df, target, start_year=start_year, end_year=end_year)
            ar1_common = ar1_bt.reindex(common_dates).dropna()
            ar1_overall = float(np.sqrt(((ar1_common["actual"] - ar1_common["pred"])**2).mean())) \
                if len(ar1_common) > 0 else np.nan
        else:
            ar1_overall = np.nan

        results[method] = dict(
            bt=bt_meta, perf=perf_df, surviving=surviving,
            ar1_rmse=ar1_overall,
            labels=labels, ar1_by_regime=ar1_by_regime,
        )
        n_meta = len(bt_meta)
        rmse_meta = float(np.sqrt(((bt_meta["actual"] - bt_meta["pred"])**2).mean())) \
            if n_meta > 0 else np.nan
        print(f"  → metamodel n={n_meta}  RMSE={rmse_meta:.3f}  AR(1)={ar1_overall:.3f}")

    return results


def print_rmc_results(rmc_results, bt_dict):
    """Print regime-model-combine performance table and comparison."""
    print("\n" + "═"*80)
    print("REGIME-MODEL-COMBINE (RMC) FRAMEWORK")
    print("  Training data split by regime → per-regime model selection → metamodel")
    print("  Compare: regime-model-combine (RMC) vs model-regime-combine (existing)")
    print("═"*80)

    for method, res in rmc_results.items():
        bt = res["bt"]
        perf = res["perf"]
        surviving = res["surviving"]
        ar1 = res["ar1_rmse"]
        ar1_by_r = res.get("ar1_by_regime", {})

        if bt is None or len(bt) == 0:
            print(f"\n  [{method}] No predictions generated.")
            continue

        rmse = float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean()))
        print(f"\n  ── Method: {method.upper()} ──────────────────────────────────────")
        print(f"  Metamodel RMSE = {rmse:.4f}  |  AR(1) RMSE = {ar1:.4f}")
        print(f"  Regimes and surviving models:")
        for r, surv in surviving.items():
            r_ar1 = ar1_by_r.get(r, np.nan)
            print(f"    {r}: AR(1)={r_ar1:.3f}  survivors={surv if surv else '(none — full ensemble)'}")

        if not perf.empty:
            print(f"\n  Per-regime model RMSE (vs regime-specific AR(1)):")
            pivot = perf.pivot_table(values="rmse", index="model", columns="regime",
                                     aggfunc="first")
            pivot_beats = perf.pivot_table(values="beats_ar1", index="model", columns="regime",
                                           aggfunc="first")
            for col in pivot.columns:
                r_ar1 = ar1_by_r.get(col, np.nan)
                ar1_str = f"{r_ar1:.3f}" if not np.isnan(r_ar1) else "?"
                print(f"    Regime {col}  [AR(1)={ar1_str}]:")
                col_data = pivot[col].dropna().sort_values()
                for m_name, rmse_val in col_data.items():
                    beat_str = "✓" if pivot_beats.loc[m_name, col] else " "
                    print(f"      {beat_str} {m_name:<18} RMSE={rmse_val:.4f}")

    # Comparison table: RMC metamodels vs existing model-regime-combine models
    print(f"\n  ── Comparison: RMC metamodels vs model-regime-combine (existing) ────")
    print(f"  {'Model / Method':<28} {'RMSE':>8}  Notes")
    print(f"  {'-'*60}")

    # existing model-regime-combine models
    for name in ("UCM", "TVP", "Combined-Dynamic", "Combined-Superstar"):
        bt_m = bt_dict.get(name)
        if bt_m is not None and len(bt_m) > 0:
            rmse_m = float(np.sqrt(((bt_m["actual"] - bt_m["pred"])**2).mean()))
            print(f"  {'MRC: ' + name:<28} {rmse_m:8.4f}  model-regime-combine (internal)")

    for method, res in rmc_results.items():
        bt = res["bt"]
        if bt is not None and len(bt) > 0:
            rmse = float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean()))
            n_surv = sum(len(v) for v in res["surviving"].values())
            print(f"  {'RMC-' + method:<28} {rmse:8.4f}  {n_surv} total regime-model survivors")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def subsample_rmse(bt_dict, periods):
    """Compute RMSE for each model over each sub-period.
    periods: list of (label, year_start, year_end) tuples."""
    rows = []
    for name, bt in bt_dict.items():
        row = {"model": name}
        for label, y0, y1 in periods:
            if bt is None or len(bt) == 0:
                row[label] = float("nan")
                continue
            sub = bt[(bt.index.year >= y0) & (bt.index.year <= y1)]
            if len(sub) >= 6:
                row[label] = float(np.sqrt(((sub["actual"] - sub["pred"])**2).mean()))
            else:
                row[label] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def common_sample_metrics(bt_dict, benchmark="AR(1)"):
    """
    Score every model on its date-intersection with the benchmark (C4).

    Adds per row:
      ar1_rmse_cs : benchmark RMSE recomputed on the SAME dates as the model
      beats_ar1   : rmse < ar1_rmse_cs (common sample, never cross-sample)
      coverage    : fraction of benchmark dates the model covers
    """
    bench = bt_dict.get(benchmark)
    rows = []
    for name, bt in bt_dict.items():
        if bt is None or len(bt) == 0:
            m = Z.score_backtest(bt, name=name)
            m.update(ar1_rmse_cs=np.nan, beats_ar1=False, coverage=np.nan)
            rows.append(m)
            continue
        if name == benchmark or bench is None or len(bench) == 0:
            m = Z.score_backtest(bt, name=name)
            m.update(ar1_rmse_cs=m["rmse"] if name == benchmark else np.nan,
                     beats_ar1=False,
                     coverage=1.0 if name == benchmark else np.nan)
            rows.append(m)
            continue
        common = bt.index.intersection(bench.index)
        m = Z.score_backtest(bt.loc[common], name=name)
        if len(common) > 0:
            b = bench.loc[common]
            ar1_cs = float(np.sqrt(((b["actual"] - b["pred"])**2).mean()))
        else:
            ar1_cs = np.nan
        m.update(ar1_rmse_cs=ar1_cs,
                 beats_ar1=bool(np.isfinite(ar1_cs) and m["rmse"] < ar1_cs),
                 coverage=round(len(common) / len(bench), 2))
        rows.append(m)
    return pd.DataFrame(rows).set_index("model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",      type=int, default=2015, help="first backtest year")
    ap.add_argument("--end",        type=int, default=2026, help="last backtest year (default 2026)")
    ap.add_argument("--train-from", type=int, default=1997, help="earliest training data year")
    ap.add_argument("--rmc",        action="store_true",
                    help="run regime-model-combine framework (slow: ~3-5 min extra)")
    ap.add_argument("--rmc-methods", nargs="+",
                    default=["hmm", "lstar", "dfm", "manual_vix"],
                    help="which RMC regime methods to run")
    ap.add_argument("--rmc-top-k", type=int, default=None,
                    help="pre-filter to top-N models by RMSE before RMC (default: all)")
    ap.add_argument("--quiet",      action="store_true")
    ap.add_argument("--shap-screen", action="store_true", default=True,
                    help="run Shapley factor screening; drop candidates below threshold (default: on)")
    ap.add_argument("--no-shap-screen", dest="shap_screen", action="store_false",
                    help="disable Shapley factor screening")
    ap.add_argument("--shap-threshold", type=float, default=0.001,
                    help="mean |SHAP| threshold for screen_candidates (default 0.001)")
    ap.add_argument("--always-keep", nargs="*", default=["gas_eu"],
                    help="factor names to always keep after SHAP screen regardless of SHAP score (default: gas_eu)")
    ap.add_argument("--no-always-keep", dest="always_keep", action="store_const", const=[],
                    help="disable always-keep list (allow SHAP to drop gas_eu etc.)")
    ap.add_argument("--target", default="cpi_yoy",
                    help="target series: cpi_yoy (default) or cpi_yoy_long for extended history")
    args = ap.parse_args()

    print("Loading factor matrix …")
    df_raw, status = F.build_matrix()
    live_facs  = [n for n, s in status.items()
                  if s != "unavailable"
                  and n not in ("cpi_yoy", "cpi_yoy_long")  # never use CPI-level as factor
                  and F.REGISTRY.get(n, {}).get("region") != "US"
                  and n != "uk_rents"          # collinear with uk_rents_lag1 after pub-lag
                  and n != "uk_paye"           # identical to uk_awg (both use ONS KAB9)
                  and n not in ("uk_cpih", "uk_services_cpi")  # CPI measures predicting CPI — circular
                  and n != "gas_eu_3m"]  # ablation: adds noise, drops RMSE 0.024 vs gas_eu alone
    target     = args.target
    if target not in df_raw.columns:
        sys.exit(f"{target} unavailable — check dbnomics / data/ CSV.")

    # trim to training start
    df_raw = df_raw[df_raw.index.year >= args.train_from]

    # ── factor liveness gate (H6/C4) ─────────────────────────────────────────
    # A single dead factor truncates EVERY model's panel via _prep's dropna.
    # Drop DEAD series loudly; flag STALE ones (nowcast ffill budget handles them).
    health = F.factor_health(df_raw, live_facs)
    print("\n── Factor liveness ──")
    print(health.to_string())
    health.to_csv(os.path.join(_DATA, "factor_health.csv"))
    dead = health[health["status"] == "DEAD"].index.tolist()
    if dead:
        print(f"  ⚠ dropping DEAD factors (stale >6m — would truncate the backtest "
              f"panel): {dead}")
        live_facs = [f for f in live_facs if f not in dead]
    stale_facs = health[health["status"] == "STALE"].index.tolist()
    if stale_facs:
        print(f"  ⚠ STALE factors (backtest OK to last obs; excluded from nowcast "
              f"by ffill budget): {stale_facs}")

    # ── apply publication lags (mixed-frequency discipline) ─────────────────
    df = F.apply_publication_lags(df_raw, live_facs)
    # regulatory event factors: 0 in periods before their data starts (no event = 0)
    _reg_factors = ["mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]
    for _f in _reg_factors:
        if _f in df.columns:
            df[_f] = df[_f].fillna(0)
    df["cpi_3m_chg"] = df[target].shift(1).diff(3)
    if "cpi_3m_chg" not in live_facs:
        live_facs = live_facs + ["cpi_3m_chg"]
    print(f"  Matrix (raw):       {df_raw.index.min().date()} → {df_raw.index.max().date()} "
          f"({len(df_raw)} months, {len(live_facs)} live factors)")
    print(f"  Matrix (pub-laged): {df.dropna(how='all').index.min().date()} → "
          f"{df.dropna(how='all').index.max().date()}")
    print(f"  Factors: {live_facs}")

    if args.shap_screen:
        print(f"\nRunning Shapley factor screening (threshold={args.shap_threshold}) …")
        screen_df = df[df.index.year < args.start]
        kept = F.screen_candidates(screen_df, target, threshold=args.shap_threshold,
                                    always_keep=args.always_keep)
        core = [n for n in live_facs if not F.REGISTRY.get(n, {}).get("candidate", False)]
        live_facs = [f for f in live_facs if f in kept or f in core]
        print(f"  Live factors after screening: {live_facs}")

    print(f"  pub_lag=0 (contemporaneous): "
          f"{[f for f in live_facs if F.REGISTRY.get(f,{}).get('pub_lag',1)==0]}")
    print(f"  pub_lag≥1 (lagged ONS/other): "
          f"{[f for f in live_facs if F.REGISTRY.get(f,{}).get('pub_lag',1)>=1]}")

    # ── factor correlation matrix (for diversification check) ───────────────
    fac_df = df[live_facs].dropna()
    if len(fac_df) > 10 and len(live_facs) > 1:
        corr_facs = fac_df.corr(method="spearman")
        print(f"\n── Factor Spearman Correlation Matrix ({len(live_facs)} factors) ──")
        print(corr_facs.round(2).to_string())
        # effective rank = sum(eigenvalues)/max(eigenvalue)
        # drop any all-NaN rows/cols (e.g. near-constant factors like ofgem_cap_delta)
        corr_clean = corr_facs.dropna(how="all").dropna(axis=1, how="all").fillna(0)
        np.fill_diagonal(corr_clean.values, 1.0)
        eigvals = np.linalg.eigvalsh(corr_clean.values)
        eff_rank = eigvals.sum() / eigvals.max() if eigvals.max() > 0 else float("nan")
        print(f"  Effective rank (sum/max eigenvalue): {eff_rank:.1f} of {len(live_facs)}")

    # ── run all models ──────────────────────────────────────────────────────
    models   = Z.all_models()
    bt_dict  = {}
    print(f"\nRunning {len(models)} models (backtest {args.start}–{args.end}) …")
    for m in models:
        print(f"  {m.name:<18}", end="", flush=True)
        try:
            bt = m.backtest(df, live_facs, target, start_year=args.start,
                            end_year=args.end)
            bt_dict[m.name] = bt
            n = len(bt)
            rmse = float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean())) if n else np.nan
            print(f"n={n:3d}  RMSE={rmse:.3f}" if n else "FAILED (0 rows)")
        except Exception as e:
            bt_dict[m.name] = None
            print(f"ERROR: {str(e)[:60]}")

    # ── AR(1) baseline ──────────────────────────────────────────────────────
    print("  AR(1)             ", end="", flush=True)
    try:
        bt_ar1 = ar1_backtest(df, target, start_year=args.start, end_year=args.end)
        bt_dict["AR(1)"] = bt_ar1
        n = len(bt_ar1)
        rmse = float(np.sqrt(((bt_ar1["actual"] - bt_ar1["pred"])**2).mean())) if n else np.nan
        print(f"n={n:3d}  RMSE={rmse:.3f}" if n else "FAILED")
    except Exception as e:
        bt_dict["AR(1)"] = None
        print(f"ERROR: {str(e)[:60]}")

    # ── combined ensembles ──────────────────────────────────────────────────
    ar1_r = None
    if bt_dict.get("AR(1)") is not None and len(bt_dict["AR(1)"]) > 0:
        ar1_r = float(np.sqrt(((bt_dict["AR(1)"]["actual"] - bt_dict["AR(1)"]["pred"])**2).mean()))

    # C1: no ex-post membership gate — gating on full-backtest RMSE selected
    # winners with future information. All models enter; equal weights (Static)
    # and causal rolling inverse-RMSE weights (Dynamic) handle model quality.
    member_bts = {n: bt for n, bt in bt_dict.items()
                  if n != "AR(1)" and bt is not None and len(bt) > 0}
    bt_static  = combine_static(member_bts)
    bt_dynamic = combine_dynamic(member_bts, window=12)
    bt_dict["Combined-Static"]  = bt_static  if len(bt_static)  else None
    bt_dict["Combined-Dynamic"] = bt_dynamic if len(bt_dynamic) else None
    for n in ("Combined-Static", "Combined-Dynamic"):
        bt = bt_dict.get(n)
        if bt is not None and len(bt):
            rmse = float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean()))
            print(f"  {n:<18}n={len(bt):3d}  RMSE={rmse:.3f}")

    # ── SPA (preliminary, needed to build superstar set) ────────────────────
    selection_end = (args.start + args.end) // 2
    bt_dict_sel = {k: v[v.index.year <= selection_end] for k, v in bt_dict.items()
                   if v is not None and len(v) > 0}
    spa_prelim = spa_table(bt_dict_sel, benchmark_name="AR(1)")

    # ── superstar combined ───────────────────────────────────────────────────
    if not spa_prelim.empty:
        from statsmodels.stats.multitest import multipletests
        cands = spa_prelim[(spa_prelim["DM"] > 0) &
                           (~spa_prelim.index.isin(["Combined-Static","Combined-Dynamic","AR(1)"]))].copy()
        if len(cands) > 0:
            reject_bh, _, _, _ = multipletests(cands["p"].fillna(1.0), alpha=0.10, method="fdr_bh")
            superstar_names = list(cands.index[reject_bh])
        else:
            superstar_names = []
        print(f"  Superstar BH-corrected (FDR 10%): {superstar_names}")
        bt_super = combine_subset(bt_dict, superstar_names)
        # C1: members were selected on data <= selection_end, so the combo may
        # only be scored on years AFTER the selection window
        if len(bt_super):
            bt_super = bt_super[bt_super.index.year > selection_end]
        bt_dict["Combined-Superstar"] = bt_super if len(bt_super) else None
        if len(bt_super):
            rmse = float(np.sqrt(((bt_super["actual"] - bt_super["pred"])**2).mean()))
            print(f"  Combined-Superstar n={len(bt_super):3d}  RMSE={rmse:.3f}  "
                  f"({superstar_names}; scored {selection_end + 1}+ only)")

    # ── recursive selected ensemble (C1: leak-free Combined-Absolute successor) ─
    # AR(1) gate + greedy error-decorrelation recomputed each year from rows
    # strictly before that year. Replaces the full-sample-selected Combined-Absolute.
    base_bts = {n: bt for n, bt in bt_dict.items() if not n.startswith("Combined-")}
    bt_recur = combine_recursive(base_bts, bt_dict.get("AR(1)"),
                                 rho_threshold=0.5, min_hist=36)
    bt_dict["Combined-Recursive"] = bt_recur if len(bt_recur) else None
    if bt_dict["Combined-Recursive"] is not None:
        rmse = float(np.sqrt(((bt_recur["actual"] - bt_recur["pred"])**2).mean()))
        print(f"  Combined-Recursive  n={len(bt_recur):3d}  RMSE={rmse:.3f}  "
              f"(walk-forward selection, min_hist=36)")

    # ── METRICS TABLE ───────────────────────────────────────────────────────
    print("\n" + "═"*90)
    print("METRICS TABLE")
    print("═"*90)
    # C4: every model scored on its intersection with AR(1)'s dates; benchmark
    # RMSE recomputed per model on the same dates; coverage made explicit.
    mdf = common_sample_metrics(bt_dict, benchmark="AR(1)").sort_values("rmse")
    low_cov = mdf.index[(mdf["coverage"].fillna(1.0) < 0.9)].tolist()
    if low_cov:
        print(f"\n  ⚠ coverage <90% of benchmark dates — RMSE ranks not comparable "
              f"across rows: {low_cov}")
    # add Giacomini-White conditional predictability test columns
    if bt_dict.get("AR(1)") is not None and len(bt_dict["AR(1)"]) > 0:
        bench_gw = bt_dict["AR(1)"]
        gw_stats, gw_ps = {}, {}
        for name, bt in bt_dict.items():
            if bt is None or len(bt) == 0:
                gw_stats[name] = float("nan")
                gw_ps[name]    = float("nan")
                continue
            common = bench_gw.index.intersection(bt.index)
            if len(common) < 10:
                gw_stats[name] = float("nan")
                gw_ps[name]    = float("nan")
                continue
            e_bench = bench_gw.loc[common, "actual"] - bench_gw.loc[common, "pred"]
            e_model = bt.loc[common, "actual"] - bt.loc[common, "pred"]
            w, p = gw_test(e_bench, e_model)
            gw_stats[name] = round(w, 3) if not np.isnan(w) else float("nan")
            gw_ps[name]    = round(p, 3) if not np.isnan(p) else float("nan")
        mdf["gw_stat"] = pd.Series(gw_stats)
        mdf["gw_p"]    = pd.Series(gw_ps)
    cols = ["rmse", "ar1_rmse_cs", "beats_ar1", "coverage", "mae", "dir_acc",
            "mz_slope", "mz_intercept", "mz_pval",
            "error_var", "mape", "bias", "gw_stat", "gw_p", "n"]
    print_cols = [c for c in cols if c in mdf.columns]
    print(mdf[print_cols].to_string(
        float_format=lambda x: f"{x:8.3f}" if isinstance(x, float) else str(x)))
    if ar1_r is not None:
        poor = mdf[mdf["rmse"] > 2.0 * ar1_r].index.tolist()
        if poor:
            print(f"\n  ⚠ Models >2\xd7 AR(1) RMSE: {poor}")

    # ── SUBSAMPLE RMSE ───────────────────────────────────────────────────────
    sub_periods = [("2015-19", 2015, 2019), ("2020-21", 2020, 2021),
                   ("2022-23", 2022, 2023), ("2024+", 2024, 2099)]
    sub_df = subsample_rmse(bt_dict, sub_periods)
    print("\n── Subsample RMSE ──")
    sub_sorted = sub_df.sort_values("2015-19")
    # Add approximate 95% CI: ±RMSE * 1.96/sqrt(2n) per cell
    # Count obs per sub-period from bt_dict
    sub_n = {}
    for label, y0, y1 in sub_periods:
        ar1_bt = bt_dict.get("AR(1)")
        if ar1_bt is not None and len(ar1_bt) > 0:
            sub_n[label] = int(((ar1_bt.index.year >= y0) & (ar1_bt.index.year <= y1)).sum())
        else:
            sub_n[label] = 1
    print(sub_sorted.to_string(float_format=lambda x: f"{x:.3f}"))
    ci_strs = {label: f"  n={sub_n.get(label,0)}, \xb1{1.96/np.sqrt(max(2*sub_n.get(label,1),1)):.2f}\xd7RMSE"
               for label, *_ in sub_periods}
    for label, note in ci_strs.items():
        print(f"    [{label}]{note}")

    # ── DM / SPA TABLE ──────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("DM TEST vs AR(1)  (DM > 0 → model beats AR(1); * p<.10, ** p<.05)")
    print("═"*60)
    spa = spa_table(bt_dict, benchmark_name="AR(1)")
    if not spa.empty:
        print(spa.to_string())

    # ── ERROR CORRELATION MATRIX ─────────────────────────────────────────────
    print("\n" + "═"*70)
    print("ERROR CORRELATION MATRIX  (Spearman ρ, forecast errors)")
    print("═"*70)
    _, full_corr = error_corr_matrix(
        {n: bt for n, bt in bt_dict.items()
         if n not in ("Combined-Static","Combined-Dynamic","Combined-Superstar",
                      "Combined-Recursive","AR(1)")})
    if not full_corr.empty:
        print(full_corr.round(2).to_string())
        uncorr = greedy_uncorrelated_subset(full_corr, bt_dict, rho_threshold=0.5,
                                             ar1_rmse=ar1_r)
        print(f"\n  Greedy uncorrelated subset (|ρ|<0.5): {uncorr}")

    # ── REGIMES TABLE ───────────────────────────────────────────────────────
    print("\n" + "═"*70)
    print("IDENTIFIED REGIMES  (current regime from full-sample fit)")
    print("═"*70)
    print(f"  {'Model':<18} {'Type':<42} {'Current':<12}")
    print(f"  {'-'*70}")
    for m in models:
        if not m.has_regimes:
            continue
        try:
            labels, meta = m.regimes(df, live_facs, target)
            if labels is not None:
                rtype   = meta.get("type", "?")[:42]
                current = str(meta.get("current", "?"))[:12]
                pct_hi  = (labels == current).mean() * 100 if labels is not None else np.nan
                print(f"  {m.name:<18} {rtype:<42} {current:<12}  ({pct_hi:.0f}% of sample)")
        except Exception as e:
            print(f"  {m.name:<18} ERROR: {str(e)[:50]}")

    # ── FACTOR IMPORTANCE TABLE ─────────────────────────────────────────────
    print("\n" + "═"*75)
    print("FACTOR IMPORTANCE  (top 5 per model)")
    print("═"*75)
    print(f"  {'Model':<18} {'Type':<22} {'Top 5 factors (value)'}")
    print(f"  {'-'*73}")
    for m in models:
        try:
            imp, itype = m.importance(df, live_facs, target)
            if imp is None or len(imp) == 0:
                print(f"  {m.name:<18} {itype:<22} n/a")
                continue
            top5 = imp.sort_values(ascending=False).head(5)
            items = "  ".join(f"{k}={v:.3f}" for k, v in top5.items())
            print(f"  {m.name:<18} {itype:<22} {items}")
        except Exception as e:
            print(f"  {m.name:<18} ERROR: {str(e)[:60]}")

    # ── REGIME-MODEL-COMBINE ────────────────────────────────────────────────
    rmc_results = {}
    if args.rmc:
        # optionally pre-filter to top-k models by RMSE before RMC
        if args.rmc_top_k is not None:
            model_rmses = {
                m.name: float(np.sqrt(((bt_dict[m.name]["actual"] - bt_dict[m.name]["pred"])**2).mean()))
                for m in models
                if bt_dict.get(m.name) is not None and len(bt_dict[m.name]) > 0
            }
            top_names = sorted(model_rmses, key=model_rmses.get)[:args.rmc_top_k]
            models_for_rmc = [m for m in models if m.name in top_names]
            print(f"\nRMC top-{args.rmc_top_k} filter: {top_names}")
        else:
            models_for_rmc = models
        print(f"\nRunning regime-model-combine ({args.rmc_methods}) …")
        rmc_results = regime_model_combine(
            df, live_facs, target, models_for_rmc,
            start_year=args.start,
            end_year=args.end,
            regime_methods=args.rmc_methods,
        )
        print_rmc_results(rmc_results, bt_dict)

        # add RMC metamodels to bt_dict + metrics
        rmc_metrics = []
        for method, res in rmc_results.items():
            bt = res["bt"]
            key = f"RMC-{method}"
            bt_dict[key] = bt if (bt is not None and len(bt)) else None
            rmc_metrics.append(Z.score_backtest(bt, name=key))
        if rmc_metrics:
            print("\n" + "═"*60)
            print("RMC METAMODEL METRICS")
            print("═"*60)
            rmc_df = pd.DataFrame(rmc_metrics).set_index("model").sort_values("rmse")
            print(rmc_df[["rmse","mae","dir_acc","bias","n"]].to_string(
                float_format=lambda x: f"{x:8.3f}"))

    # ── SAVE ────────────────────────────────────────────────────────────────
    out_rows = []
    for name, bt in bt_dict.items():
        if bt is not None and len(bt):
            b = bt.copy(); b["model"] = name
            out_rows.append(b)
    if out_rows:
        out = pd.concat(out_rows).reset_index()
        p = os.path.join(_DATA, "nowcast_cpi_backtest.csv")
        out.to_csv(p, index=False)
        print(f"\nSaved → {p}")

    p = os.path.join(_DATA, "nowcast_cpi_metrics.csv")
    mdf.to_csv(p)
    print(f"Saved → {p}")

    if not spa.empty:
        p = os.path.join(_DATA, "nowcast_cpi_spa.csv")
        spa.to_csv(p)
        print(f"Saved → {p}")

    if rmc_results:
        for method, res in rmc_results.items():
            if not res["perf"].empty:
                p = os.path.join(_DATA, f"rmc_{method}_perf.csv")
                res["perf"].to_csv(p, index=False)
                print(f"Saved → {p}")

    # ── CURRENT NOWCAST ─────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("CURRENT NOWCAST")
    print("═" * 65)
    nowcast_rows = []
    for m in models:
        try:
            val, nc_date = m.nowcast(df, live_facs, target)
            nowcast_rows.append(dict(
                model=m.name,
                nowcast=round(val, 3) if np.isfinite(val) else np.nan,
                date=str(nc_date.date()) if nc_date is not None else "?",
            ))
        except Exception:
            nowcast_rows.append(dict(model=m.name, nowcast=np.nan, date="error"))
    nc_df = pd.DataFrame(nowcast_rows).set_index("model")
    # nowcast_lo/hi = nowcast ± model RMSE
    if "rmse" in mdf.columns:
        nc_df = nc_df.join(mdf[["rmse"]].rename(columns={"rmse": "model_rmse"}), how="left")
        nc_df["nowcast_lo"] = nc_df["nowcast"] - nc_df["model_rmse"]
        nc_df["nowcast_hi"] = nc_df["nowcast"] + nc_df["model_rmse"]
        nc_df = nc_df.drop(columns=["model_rmse"])
    print(nc_df.to_string())
    p = os.path.join(_DATA, "nowcast_cpi_nowcast.csv")
    nc_df.to_csv(p)
    print(f"\nSaved → {p}")


if __name__ == "__main__":
    main()
