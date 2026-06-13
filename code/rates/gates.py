"""
rates/gates.py — Deliverable 2 (Gate 1) + Deliverable 3 (Gate 2).

Gate 1  : does my nowcast forecast CPI better than UCL / consensus / naive?
Gate 2  : does my_surprise carry rates-repricing information INCREMENTAL to
          ucl_surprise and market pricing?  HAC-robust, walk-forward.

Gate 2 is the project: PASS gates the MVP model and trading signal.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C

# reuse the HLN-corrected Diebold-Mariano from the CPI zoo
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uk_model_zoo as Z


# ─────────────────────────────────────────────────────────────────────────────
# GATE 1 — forecast accuracy vs benchmarks
# ─────────────────────────────────────────────────────────────────────────────

def gate1_accuracy(panel):
    """Compare |my_nowcast - actual| to UCL / consensus / naive-RW benchmarks.
    Returns a metrics DataFrame; DM>0 means my forecast beats the benchmark."""
    p = panel.dropna(subset=["actual_cpi_mom", "my_nowcast"]).copy()
    actual = p["actual_cpi_mom"]

    preds = {"my_nowcast": p["my_nowcast"]}
    if p["ucl_nowcast"].notna().any():          preds["ucl_nowcast"] = p["ucl_nowcast"]
    if p["economist_consensus"].notna().any():  preds["consensus"]  = p["economist_consensus"]
    preds["naive_rw"] = actual.shift(1)          # last published YoY

    err = {k: (v - actual) for k, v in preds.items()}
    rows = []
    for k, e in err.items():
        e2 = e.dropna()
        rows.append(dict(forecaster=k,
                         rmse=float(np.sqrt((e2**2).mean())),
                         mae=float(e2.abs().mean()),
                         n=int(len(e2))))
    out = pd.DataFrame(rows).set_index("forecaster")

    # DM: my_nowcast vs each benchmark on the common sample
    e_my = err["my_nowcast"]
    for k in [c for c in preds if c != "my_nowcast"]:
        common = e_my.dropna().index.intersection(err[k].dropna().index)
        if len(common) >= 10:
            dm, pv = Z.dm_test(err[k].loc[common].values, e_my.loc[common].values)
            out.loc[k, "DM_vs_my"] = round(dm, 3)
            out.loc[k, "DM_p"]     = round(pv, 4)
            out.loc[k, "my_beats"] = bool(dm > 0)
    if "ucl_nowcast" in out.index and "my_beats" in out.columns:
        out.attrs["my_beats_ucl"] = bool(out.loc["ucl_nowcast", "my_beats"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# GATE 2 — incremental information regression
# ─────────────────────────────────────────────────────────────────────────────

def _design(panel, target, controls=True, exclude_ldi=True):
    """Build (y, X, names). Predictors: my_surprise (always), then ucl_surprise,
    market_implied_expectation, optional regime dummies + days_to_mpc. Drops
    all-NaN / zero-variance columns (keeps my_surprise). Excludes LDI rows."""
    p = panel.copy()
    if exclude_ldi and "ldi_event" in p:
        p = p[p["ldi_event"] != 1]
    cand = ["my_surprise", "ucl_surprise", "market_implied_expectation"]
    cols = [c for c in cand if c in p.columns]
    X = p[cols].copy()
    if controls:
        if "days_to_mpc" in p:
            X["days_to_mpc"] = p["days_to_mpc"]
        if "mpc_regime" in p:
            dummies = pd.get_dummies(p["mpc_regime"], prefix="reg", drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)
    y = p[target]
    # Drop predictor columns that are entirely (or almost) NaN BEFORE the row
    # dropna — otherwise an absent control (e.g. market_implied, or ucl in
    # naive_rw mode) would null out every row. my_surprise is always kept.
    for c in list(X.columns):
        if c == "my_surprise":
            continue
        if X[c].notna().sum() < max(10, int(0.5 * len(X))):
            X = X.drop(columns=c)
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    y = d["y"]; X = d.drop(columns="y")
    # keep my_surprise; drop remaining zero-variance / constant predictors
    keep = [c for c in X.columns
            if c == "my_surprise" or X[c].std(skipna=True) > 1e-12]
    X = X[keep]
    return y, X


def _hac_lags(n):
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def gate2_incremental(panel, target=None, controls=True, exclude_ldi=True,
                      min_train=24):
    """Pooled HAC OLS + nested incremental test + walk-forward OOS.
    Returns dict with the verdict and all stats."""
    target = target or C.PRIMARY_MOVE
    y, X = _design(panel, target, controls=controls, exclude_ldi=exclude_ldi)
    n = len(y)
    res = {"target": target, "n": int(n), "verdict": "INSUFFICIENT_DATA"}
    if n < max(min_train, X.shape[1] + 5) or "my_surprise" not in X.columns:
        return res
    L = _hac_lags(n)

    # full model (HAC-robust)
    Xf = sm.add_constant(X, has_constant="add")
    full = sm.OLS(y.values, Xf.values).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    names = ["const"] + list(X.columns)
    j = names.index("my_surprise")
    beta_my = float(full.params[j]); t_my = float(full.tvalues[j]); p_my = float(full.pvalues[j])

    # reduced model (drop my_surprise) for incremental R^2
    Xr = X.drop(columns="my_surprise")
    if Xr.shape[1] == 0:
        red_r2 = 0.0
        red = sm.OLS(y.values, np.ones((n, 1))).fit()
    else:
        red = sm.OLS(y.values, sm.add_constant(Xr, has_constant="add").values).fit()
        red_r2 = float(red.rsquared)
    incr_r2 = float(full.rsquared) - red_r2

    econ_bp = beta_my * float(X["my_surprise"].std())   # bp move per 1 sigma my_surprise

    # ── walk-forward OOS: fit on past releases, predict current move ─────────
    oos_pred, oos_real = [], []
    Xv = sm.add_constant(X, has_constant="add")
    for i in range(min_train, n):
        Xtr, ytr = Xv.iloc[:i].values, y.iloc[:i].values
        try:
            b = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
        except Exception:
            continue
        oos_pred.append(float(Xv.iloc[i].values @ b))
        oos_real.append(float(y.iloc[i]))
    oos_pred, oos_real = np.array(oos_pred), np.array(oos_real)
    if len(oos_pred) >= 8 and np.std(oos_pred) > 1e-12:
        oos_corr = float(np.corrcoef(oos_pred, oos_real)[0, 1])
        oos_hit  = float(np.mean(np.sign(oos_pred) == np.sign(oos_real)))
        ss_res = float(((oos_real - oos_pred) ** 2).sum())
        ss_tot = float(((oos_real - oos_real.mean()) ** 2).sum()) or np.nan
        oos_r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    else:
        oos_corr = oos_hit = oos_r2 = np.nan

    passed = (t_my >= C.GATE2_T_THRESHOLD
              and incr_r2 >= C.GATE2_INCR_R2_MIN
              and beta_my > 0
              and np.isfinite(oos_corr) and oos_corr >= C.GATE2_OOS_CORR_MIN
              and np.isfinite(oos_r2) and oos_r2 > C.GATE2_OOS_R2_MIN)
    res.update(dict(
        verdict="PASS" if passed else "FAIL",
        anchor_mode=panel.attrs.get("anchor_mode"),
        hac_lags=L, predictors=list(X.columns),
        beta_my_surprise=beta_my, t_my_surprise_HAC=t_my, p_my_surprise_HAC=p_my,
        full_r2=float(full.rsquared), reduced_r2=red_r2, incremental_r2=incr_r2,
        econ_bp_per_1sigma=econ_bp,
        oos_corr=oos_corr, oos_sign_hit=oos_hit, oos_r2=oos_r2, oos_n=int(len(oos_pred)),
    ))
    return res


def run_gates(panel, target=None):
    g1 = gate1_accuracy(panel)
    g2 = gate2_incremental(panel, target=target)
    g1.to_csv(C.GATE1_CSV)
    pd.Series(g2).to_csv(C.GATE2_CSV)
    return g1, g2
