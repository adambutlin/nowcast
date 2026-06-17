"""
Factor race — rank every available factor (core 'live' + 'candidate') by how much it
helps explain the AutoARIMA residual (the only thing Stage-2 can add).

Target: resid_t = CPI_yoy_t - AutoARIMA_t   (walk-forward, expanding window, 2015-2024)

Three rankings:
  1. univ_rel_rmse  — walk-forward single-factor ridge on resid, reconstructed RMSE
                      relative to AA-only (<1 => factor helps when added alone). DECISION metric.
  2. shap_mean_abs  — mean |SHAP| from a gradient-boosted tree on resid (all factors).
  3. perm_dRMSE     — OOS permutation ΔRMSE from the same GBT.
  + |corr| with residual.

Out: data/new_factors/factor_race.csv ; plots/new_factors/factor_race.png
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/new_factors/factor_race.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import factors as F, uk_model_zoo as Z

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "new_factors")
_PLOT = os.path.join(_ROOT, "plots", "new_factors")
os.makedirs(_OUT, exist_ok=True); os.makedirs(_PLOT, exist_ok=True)
TARGET = "cpi_yoy"
AA_START, START, END, TRAIN_FROM = 2001, 2015, 2024, 1997


def aa_residual():
    df, status = F.build_matrix(names=[TARGET])
    df = df[df.index.year >= TRAIN_FROM].resample("ME").last()
    aa = Z.AutoARIMA().backtest(df, [], TARGET, start_year=AA_START, end_year=END)
    aa = aa[(aa.index.year >= START) & (aa.index.year <= END)]
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    return resid, aa


def load_all_factors():
    names = [n for n in F.REGISTRY if n != TARGET]
    df, status = F.build_matrix(names=names)
    live = [n for n in names if status.get(n) != "unavailable"]
    df = df[df.index.year >= TRAIN_FROM]
    df = F.apply_publication_lags(df, live)
    for rf in ["mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]:
        if rf in df.columns:
            df[rf] = df[rf].fillna(0)
    return df.resample("ME").last(), live, status


def wf_univariate_rel_rmse(resid, x, aa):
    """Walk-forward single-factor ridge: resid ~ x. Reconstructed CPI RMSE vs AA-only."""
    from sklearn.linear_model import Ridge
    d = pd.DataFrame({"resid": resid, "x": x, "actual": aa["actual"], "aa": aa["pred"]}).dropna()
    if len(d) < 48:
        return np.nan, np.nan, len(d)
    errs_f, errs_aa = [], []
    for yr in range(START + 2, END + 1):          # need >=2y train
        tr = d[d.index.year < yr]; te = d[d.index.year == yr]
        if len(tr) < 24 or len(te) == 0:
            continue
        mu, sd = tr["x"].mean(), tr["x"].std() or 1.0
        m = Ridge(alpha=1.0).fit(((tr[["x"]] - mu) / sd), tr["resid"])
        rp = m.predict((te[["x"]] - mu) / sd)
        fpred = te["aa"].values + rp
        errs_f += list(te["actual"].values - fpred)
        errs_aa += list(te["actual"].values - te["aa"].values)
    if len(errs_f) < 12:
        return np.nan, np.nan, len(d)
    rf = np.sqrt(np.mean(np.square(errs_f))); ra = np.sqrt(np.mean(np.square(errs_aa)))
    return rf / ra, ra, len(d)


def main():
    print("AutoARIMA residual …")
    resid, aa = aa_residual()
    print("Loading all factors …")
    fac, live, status = load_all_factors()
    feats = [c for c in fac.columns if c in live]
    print(f"available factors: {len(feats)}")

    # align
    X = fac[feats].reindex(resid.index)
    y = resid.copy()
    # common evaluation window: where target + most factors exist
    Xy = X.join(y).loc[resid.index]
    Xy = Xy[Xy.index.year >= START]

    rows = []
    for f in feats:
        rel, ra, n = wf_univariate_rel_rmse(resid, X[f], aa)
        c = X[f].corr(y)
        rows.append(dict(factor=f, region=F.REGISTRY[f].get("region", "UK"),
                         status=status[f], n=n, abs_corr=abs(c) if np.isfinite(c) else np.nan,
                         univ_rel_rmse=rel))
    race = pd.DataFrame(rows).set_index("factor")

    # ---- multivariate GBT: SHAP + permutation importance (attribution) ----
    Xm = X.copy()
    # causal-ish impute: forward-fill then column median (attribution, not forecast)
    Xm = Xm.ffill().fillna(Xm.median(numeric_only=True))
    mask = y.notna() & Xm.notna().all(axis=1)
    Xm, ym = Xm[mask], y[mask]
    if len(Xm) > 40:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.inspection import permutation_importance
        gbt = GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                        learning_rate=0.03, subsample=0.8,
                                        random_state=0).fit(Xm, ym)
        try:
            import shap
            expl = shap.TreeExplainer(gbt)
            sv = expl.shap_values(Xm)
            shap_abs = pd.Series(np.abs(sv).mean(axis=0), index=Xm.columns)
        except Exception as e:
            print("  [shap fallback to feature_importances_]:", str(e)[:60])
            shap_abs = pd.Series(gbt.feature_importances_, index=Xm.columns)
        perm = permutation_importance(gbt, Xm, ym, n_repeats=20, random_state=0,
                                      scoring="neg_root_mean_squared_error")
        perm_s = pd.Series(perm.importances_mean, index=Xm.columns)
        race["shap_mean_abs"] = shap_abs.reindex(race.index)
        race["perm_dRMSE"] = perm_s.reindex(race.index)

    race = race.sort_values("univ_rel_rmse")
    race.to_csv(os.path.join(_OUT, "factor_race.csv"))
    pd.options.display.width = 220; pd.options.display.max_rows = 100
    print("\n=== FACTOR RACE (sorted by univ_rel_rmse; <1 => helps AA when added alone) ===")
    show = race[["region", "status", "n", "abs_corr", "univ_rel_rmse",
                 "shap_mean_abs", "perm_dRMSE"]].round(4)
    print(show.to_string())
    helpers = race[race["univ_rel_rmse"] < 1.0]
    print(f"\nfactors that beat AA standalone (rel<1): {len(helpers)}/{len(race)}")
    print("top by SHAP:", race.sort_values("shap_mean_abs", ascending=False).head(6).index.tolist())
    print("top by perm ΔRMSE:", race.sort_values("perm_dRMSE", ascending=False).head(6).index.tolist())
    print("highlighted new factors:")
    for nf in ["uk_ppi_input", "uk_ppi_output", "deep_sea_freight", "move_index",
               "slope_2s10s", "slope_5s30s", "freightos_fbx"]:
        if nf in race.index:
            r = race.loc[nf]
            print(f"  {nf:16} rel_rmse={r['univ_rel_rmse']:.4f}  shap={r.get('shap_mean_abs',np.nan):.4f}  "
                  f"perm={r.get('perm_dRMSE',np.nan):.4f}  corr={r['abs_corr']:.3f}")
        else:
            print(f"  {nf:16} (unavailable)")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 8))
        r2 = race.dropna(subset=["univ_rel_rmse"]).sort_values("univ_rel_rmse")
        col = ["#27ae60" if v < 1 else "#c0392b" for v in r2["univ_rel_rmse"]]
        ax[0].barh(r2.index, r2["univ_rel_rmse"] - 1.0, color=col)
        ax[0].axvline(0, color="k", lw=0.8)
        ax[0].set_title("univ_rel_rmse − 1  (green<0 = beats AA standalone)")
        ax[0].invert_yaxis(); ax[0].tick_params(labelsize=7)
        if "shap_mean_abs" in race:
            s = race.dropna(subset=["shap_mean_abs"]).sort_values("shap_mean_abs")
            ax[1].barh(s.index, s["shap_mean_abs"], color="#2c7fb8")
            ax[1].set_title("mean |SHAP| on AA residual (GBT)"); ax[1].tick_params(labelsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(_PLOT, "factor_race.png"), dpi=110)
        print("\nplot ->", os.path.join(_PLOT, "factor_race.png"))
    except Exception as e:
        print("plot skipped:", str(e)[:60])
    print("written", os.path.join(_OUT, "factor_race.csv"))


if __name__ == "__main__":
    main()
