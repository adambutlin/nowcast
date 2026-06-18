"""
Residual-target LightGBM benchmark vs the production ensemble.

Target : resid_t = CPI_yoy_t - AutoARIMA_t   (walk-forward, expanding, no lookahead)
Features: production PINNED factor set only (pub-lagged monthly matrix from two_stage).
Models compared: (1) AutoARIMA, (2) production ensemble (AA+0.375BVAR+0.25TVP+0.375MIDAS),
                 (3) AA + Residual-LGBM.
Walk-forward: test years 2015-2024; for year y, train LGBM on resid for months < y
(expanding, from AA_START vintage so pre-2015 training exists). Conservative LGBM (shallow,
regularised) given small n. SHAP (TreeExplainer) + OOS permutation importance.

Out: data/timing/lgbm/{rmse_comparison,shap_summary,permutation_importance}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/residual_lgbm.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
import lightgbm as lgb
import uk_model_zoo as Z, two_stage as TS

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing", "lgbm")
os.makedirs(_OUT, exist_ok=True)
EVAL_START, END = 2015, 2024
LGB = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
           min_child_samples=12, subsample=0.8, colsample_bytree=0.8,
           reg_lambda=1.0, random_state=0, verbose=-1)
WIN = {"full": lambda i: (i.year >= EVAL_START), "2022_23": lambda i: i.year.isin([2022, 2023]),
       "ex_shock": lambda i: (i.year >= EVAL_START) & ~i.year.isin([2022, 2023]),
       "pre_2020": lambda i: (i.year >= EVAL_START) & (i.year <= 2019)}


def rmse(e): e = pd.Series(e).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else np.nan
def mae(e): e = pd.Series(e).dropna(); return float(e.abs().mean()) if len(e) else np.nan
def dm(e1, e2):
    d = np.asarray(e1, float)**2 - np.asarray(e2, float)**2; d = d[np.isfinite(d)]; n = len(d)
    if n < 8 or np.allclose(d, 0): return np.nan, np.nan
    db = d.mean(); dd = d - db; v = (dd @ dd)/n; L = max(1, int(round(n**(1/3))))
    for k in range(1, L+1): v += 2*(1-k/(L+1))*((dd[k:]@dd[:-k])/n)
    if v <= 0: return np.nan, np.nan
    s = db/np.sqrt(v/n); return float(s), float(2*(1-tdist.cdf(abs(s), df=n-1)))


def main():
    df, live, status = TS.load_matrix()
    feats = list(live)
    print("features:", feats)
    # AA over full vintage (so pre-2015 training residuals exist)
    aa = Z.AutoARIMA().backtest(df, [], TS.TARGET, start_year=TS.AA_START, end_year=END)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    X = df[feats].reindex(aa.index)
    data = X.join(resid).join(aa["actual"].rename("actual")).join(aa["pred"].rename("aa"))

    # walk-forward LGBM on residual
    models, lgb_pred = {}, pd.Series(index=aa.index, dtype=float)
    for y in range(EVAL_START, END+1):
        tr = data[data.index.year < y].dropna(subset=["resid"])
        te = data[data.index.year == y]
        if len(tr) < 36 or len(te) == 0: continue
        m = lgb.LGBMRegressor(**LGB).fit(tr[feats], tr["resid"])
        models[y] = m
        lgb_pred.loc[te.index] = m.predict(te[feats])
    data["lgb_resid"] = lgb_pred
    data["aa_lgb"] = data["aa"] + data["lgb_resid"]

    # production ensemble (same factor matrix)
    bt = TS.backtest(df, live)
    ens = bt["forecast"].reindex(aa.index)

    ev = data[(data.index.year >= EVAL_START) & data["lgb_resid"].notna()].copy()
    ev["ens"] = ens.reindex(ev.index)
    a = ev["actual"]

    # PART: rmse comparison by window
    rows = []
    for w, fn in WIN.items():
        m = fn(ev.index); s = ev[m]
        rows.append(dict(window=w, n=len(s),
                         rmse_AA=rmse(s["actual"]-s["aa"]),
                         rmse_ENS=rmse(s["actual"]-s["ens"]),
                         rmse_AA_LGB=rmse(s["actual"]-s["aa_lgb"]),
                         mae_AA=mae(s["actual"]-s["aa"]), mae_ENS=mae(s["actual"]-s["ens"]),
                         mae_AA_LGB=mae(s["actual"]-s["aa_lgb"]),
                         rel_ENS=rmse(s["actual"]-s["ens"])/rmse(s["actual"]-s["aa"]),
                         rel_LGB=rmse(s["actual"]-s["aa_lgb"])/rmse(s["actual"]-s["aa"])))
    cmp = pd.DataFrame(rows).set_index("window")
    cmp.to_csv(os.path.join(_OUT, "rmse_comparison.csv"))
    # DM tests (full)
    f = WIN["full"](ev.index)
    s_la, p_la = dm(ev["actual"][f]-ev["aa"][f], ev["actual"][f]-ev["aa_lgb"][f])    # AA vs AA+LGB
    s_le, p_le = dm(ev["actual"][f]-ev["ens"][f], ev["actual"][f]-ev["aa_lgb"][f])   # ENS vs AA+LGB
    pd.options.display.width = 200
    print("\n=== RMSE COMPARISON (rel<1 beats AA) ===")
    print(cmp[["n","rmse_AA","rmse_ENS","rmse_AA_LGB","rel_ENS","rel_LGB"]].round(4).to_string())
    print(f"\nDM full: AA vs AA+LGB stat={s_la:+.2f} p={p_la:.3f} (>0 => LGB better)")
    print(f"DM full: ENS vs AA+LGB stat={s_le:+.2f} p={p_le:.3f} (>0 => LGB better than ensemble)")

    # SHAP (in-sample attribution on a full-fit LGBM over eval window)
    import shap
    full_tr = data[(data.index.year >= TS.AA_START)].dropna(subset=["resid"])
    mfull = lgb.LGBMRegressor(**LGB).fit(full_tr[feats], full_tr["resid"])
    sv = shap.TreeExplainer(mfull).shap_values(full_tr[feats])
    shap_abs = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(ascending=False)
    shp = pd.DataFrame({"mean_abs_shap": shap_abs, "share_%": 100*shap_abs/shap_abs.sum()})
    shp.to_csv(os.path.join(_OUT, "shap_summary.csv"))
    print("\n=== SHAP (mean|SHAP| on residual, full-fit) ==="); print(shp.round(4).to_string())

    # OOS permutation importance (re-predict eval rows with one feature permuted, per-year models)
    rng = np.random.default_rng(0)
    base = rmse(ev["actual"]-ev["aa_lgb"])
    perm_rows = []
    for c in feats:
        drops = []
        for _ in range(15):
            ev2 = ev.copy(); ev2[c] = rng.permutation(ev2[c].values)
            pr = pd.Series(index=ev2.index, dtype=float)
            for y in range(EVAL_START, END+1):
                te = ev2[ev2.index.year == y]
                if y in models and len(te): pr.loc[te.index] = models[y].predict(te[feats])
            drops.append(rmse(ev2["actual"]-(ev2["aa"]+pr)) - base)
        perm_rows.append(dict(feature=c, perm_dRMSE=float(np.mean(drops))))
    perm = pd.DataFrame(perm_rows).sort_values("perm_dRMSE", ascending=False).set_index("feature")
    perm.to_csv(os.path.join(_OUT, "permutation_importance.csv"))
    print("\n=== OOS permutation importance (ΔRMSE when shuffled) ==="); print(perm.round(4).to_string())
    print("\nwritten rmse_comparison / shap_summary / permutation_importance")


if __name__ == "__main__":
    main()
