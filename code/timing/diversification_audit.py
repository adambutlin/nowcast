"""
Diversification audit. Base = AutoARIMA. Residual members: TVP, BVAR, MIDAS, LGBM.
Forecast = AA + sum_i w_i * resid_i  (resid_i = member_recon_i - AA; renorm over available).

Ensembles:
  A current production : TVP .25 / BVAR .375 / MIDAS .375
  B equal 4            : .25 each
  C inverse-RMSE       : w ~ 1/rmse_i (full-sample; in-sample weights, flagged)
  D error-corr optimal : min-variance w = Σ^{-1}1 / 1'Σ^{-1}1 on member error cov (in-sample)
  E TVP+LGBM 50/50
  F TVP+LGBM+BVAR equal

Metrics RMSE/MAE/OOS-corr by window (full/2022_23/ex_shock/pre_2020) + error-corr matrix +
diversification benefit + DM vs production.

Out: data/timing/divers/{ensemble_metrics,error_corr,dm_tests,weights}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/diversification_audit.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
import lightgbm as lgb
import uk_model_zoo as Z, two_stage as TS

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing", "divers")
os.makedirs(_OUT, exist_ok=True)
EVAL_START, END = 2015, 2024
LGB = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
           min_child_samples=12, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=0, verbose=-1)
WIN = {"full": lambda i: i.year >= EVAL_START, "2022_23": lambda i: i.year.isin([2022, 2023]),
       "ex_shock": lambda i: (i.year >= EVAL_START) & ~i.year.isin([2022, 2023]),
       "pre_2020": lambda i: (i.year >= EVAL_START) & (i.year <= 2019)}
MEMBERS = ["tvp", "bvar", "midas", "lgbm"]


def rmse(e): e = pd.Series(e).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else np.nan
def mae(e): e = pd.Series(e).dropna(); return float(e.abs().mean()) if len(e) else np.nan
def dm(e1, e2):
    d = np.asarray(e1, float)**2 - np.asarray(e2, float)**2; d = d[np.isfinite(d)]; n = len(d)
    if n < 8 or np.allclose(d, 0): return np.nan, np.nan
    db = d.mean(); dd = d-db; v = (dd@dd)/n; L = max(1, int(round(n**(1/3))))
    for k in range(1, L+1): v += 2*(1-k/(L+1))*((dd[k:]@dd[:-k])/n)
    if v <= 0: return np.nan, np.nan
    s = db/np.sqrt(v/n); return float(s), float(2*(1-tdist.cdf(abs(s), df=n-1)))


def build():
    df, live, status = TS.load_matrix()
    aa = Z.AutoARIMA().backtest(df, [], TS.TARGET, start_year=TS.AA_START, end_year=END)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    # TVP/BVAR/MIDAS reconstructed preds (production stack)
    bt = TS.backtest(df, live)
    out = pd.DataFrame({"actual": aa["actual"], "aa": aa["pred"]})
    for tag in ["tvp", "bvar", "midas"]:
        out[f"{tag}_resid"] = (bt[f"{tag}_pred"] - bt["aa_pred"]).reindex(out.index)
    # LGBM walk-forward residual
    feats = list(live)
    data = df[feats].reindex(aa.index).join(resid)
    lgbm_pred = pd.Series(index=aa.index, dtype=float)
    for y in range(EVAL_START, END+1):
        tr = data[data.index.year < y].dropna(subset=["resid"]); te = data[data.index.year == y]
        if len(tr) < 36 or len(te) == 0: continue
        m = lgb.LGBMRegressor(**LGB).fit(tr[feats], tr["resid"])
        lgbm_pred.loc[te.index] = m.predict(te[feats])
    out["lgbm_resid"] = lgbm_pred
    out = out[(out.index.year >= EVAL_START) & out["lgbm_resid"].notna()].dropna(subset=["actual","aa"])
    return out


def ens_forecast(out, weights):
    """weights: dict member->w over MEMBERS subset; renorm over available per row."""
    cols = [m for m in weights if f"{m}_resid" in out.columns]
    W = pd.Series({m: weights[m] for m in cols})
    R = out[[f"{m}_resid" for m in cols]]
    R.columns = cols
    denom = (R.notna().astype(float) * W.values).sum(axis=1).replace(0, np.nan)
    overlay = (R.fillna(0) * W.values).sum(axis=1) / denom
    return out["aa"] + overlay.fillna(0)


def main():
    out = build()
    a = out["actual"]
    # member errors + corr
    merr = pd.DataFrame({m: a - (out["aa"] + out[f"{m}_resid"]) for m in MEMBERS})
    ecorr = merr.corr(); ecorr.to_csv(os.path.join(_OUT, "error_corr.csv"))
    mrmse = {m: rmse(merr[m]) for m in MEMBERS}

    # weights
    inv = {m: 1.0/mrmse[m] for m in MEMBERS}; s = sum(inv.values()); invw = {m: inv[m]/s for m in MEMBERS}
    Sig = merr.cov().values
    w_mv = np.linalg.solve(Sig, np.ones(len(MEMBERS)))
    w_mv = w_mv / w_mv.sum()
    mvw = dict(zip(MEMBERS, w_mv))
    ENS = {
        "A_production": {"tvp":0.25,"bvar":0.375,"midas":0.375},
        "B_equal4":     {m:0.25 for m in MEMBERS},
        "C_invRMSE":    invw,
        "D_minvar":     mvw,
        "E_TVP_LGBM":   {"tvp":0.5,"lgbm":0.5},
        "F_TVP_LGBM_BVAR": {"tvp":1/3,"lgbm":1/3,"bvar":1/3},
    }
    pd.DataFrame(ENS).T.to_csv(os.path.join(_OUT, "weights.csv"))

    rows = []
    fc = {}
    for name, w in ENS.items():
        f = ens_forecast(out, w); fc[name] = f
        for win, fn in WIN.items():
            m = fn(out.index); s2 = out[m]
            e = (s2["actual"] - f[m])
            rows.append(dict(ensemble=name, window=win, n=len(s2), rmse=rmse(e), mae=mae(e),
                             corr=float(np.corrcoef(s2["actual"], f[m])[0,1]) if len(s2)>2 else np.nan,
                             rmse_AA=rmse(s2["actual"]-s2["aa"]),
                             rel=rmse(e)/rmse(s2["actual"]-s2["aa"])))
    met = pd.DataFrame(rows).set_index(["ensemble","window"])
    met.to_csv(os.path.join(_OUT, "ensemble_metrics.csv"))

    # diversification benefit (full): avg member rmse vs ensemble rmse
    avg_mem = np.mean([mrmse[m] for m in MEMBERS]); best_mem = min(mrmse.values())
    # DM vs production (full)
    f_full = WIN["full"](out.index)
    base = a[f_full] - fc["A_production"][f_full]
    dm_rows = []
    for name in ENS:
        s_, p_ = dm(base, a[f_full]-fc[name][f_full])
        dm_rows.append(dict(ensemble=name, rmse_full=rmse(a[f_full]-fc[name][f_full]),
                            DM_vs_prod=s_, p_vs_prod=p_))
    dmt = pd.DataFrame(dm_rows).set_index("ensemble"); dmt.to_csv(os.path.join(_OUT, "dm_tests.csv"))

    pd.options.display.width = 220
    print("=== member full-sample RMSE ===", {k: round(v,4) for k,v in mrmse.items()})
    print("=== error correlation ==="); print(ecorr.round(3).to_string())
    print("\n=== weights ==="); print(pd.DataFrame(ENS).T.round(3).to_string())
    print("\n=== ensemble RMSE by window ===")
    print(met["rmse"].unstack("window").round(4).to_string())
    print("\n=== rel vs AA by window ===")
    print(met["rel"].unstack("window").round(4).to_string())
    print("\n=== DM vs production (full) ===")
    print(dmt.round(4).to_string())
    print(f"\ndiversification: avg member RMSE={avg_mem:.4f} best member={best_mem:.4f} "
          f"production ens={rmse(a[f_full]-fc['A_production'][f_full]):.4f}")
    print("written ensemble_metrics / error_corr / dm_tests / weights")


if __name__ == "__main__":
    main()
