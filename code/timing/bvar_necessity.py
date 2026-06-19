"""
BVAR necessity audit. Tier1 = AA + 0.5 TVP + 0.5 LGBM. Tier2 = AA + (TVP+LGBM+BVAR)/3.

1. full + window RMSE diff (Tier1 vs Tier2) + DM.
2. rolling 5y RMSE diff.
3. corr(BVAR_err, Tier1_err) and corr(BVAR_err, mean(TVP,LGBM)_err).
4. windows: 2022/23 / ex_shock / pre_2020.
5. PPI-ablation: Tier1_noPPI vs Tier2_noPPI (does BVAR help when LGBM has no uk_ppi_input?).

eval 2015-2024, walk-forward. Out: data/timing/bvar/{necessity,rolling,corr}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/bvar_necessity.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
import lightgbm as lgb
import uk_model_zoo as Z, two_stage as TS

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing", "bvar")
os.makedirs(_OUT, exist_ok=True)
EVAL_START, END = 2015, 2024
LGB = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
           min_child_samples=12, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=0, verbose=-1)
WIN = {"full": lambda i: i.year >= EVAL_START, "2022_23": lambda i: i.year.isin([2022, 2023]),
       "ex_shock": lambda i: (i.year >= EVAL_START) & ~i.year.isin([2022, 2023]),
       "pre_2020": lambda i: (i.year >= EVAL_START) & (i.year <= 2019)}


def rmse(e): e = pd.Series(e).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else np.nan
def dm(e1, e2):
    d = np.asarray(e1, float)**2 - np.asarray(e2, float)**2; d = d[np.isfinite(d)]; n = len(d)
    if n < 8 or np.allclose(d, 0): return np.nan, np.nan
    db = d.mean(); dd = d-db; v = (dd@dd)/n; L = max(1, int(round(n**(1/3))))
    for k in range(1, L+1): v += 2*(1-k/(L+1))*((dd[k:]@dd[:-k])/n)
    if v <= 0: return np.nan, np.nan
    s = db/np.sqrt(v/n); return float(s), float(2*(1-tdist.cdf(abs(s), df=n-1)))


def lgbm_wf(data, feats):
    p = pd.Series(index=data.index, dtype=float)
    for y in range(EVAL_START, END+1):
        tr = data[data.index.year < y].dropna(subset=["resid"]); te = data[data.index.year == y]
        if len(tr) < 36 or len(te) == 0: continue
        p.loc[te.index] = lgb.LGBMRegressor(**LGB).fit(tr[feats], tr["resid"]).predict(te[feats])
    return p


def main():
    df, live, status = TS.load_matrix()
    feats = list(live); feats_np = [f for f in feats if f != "uk_ppi_input"]
    aa = Z.AutoARIMA().backtest(df, [], TS.TARGET, start_year=TS.AA_START, end_year=END)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    bt = TS.backtest(df, live)
    o = pd.DataFrame({"actual": aa["actual"], "aa": aa["pred"]})
    o["tvp_r"] = (bt["tvp_pred"] - bt["aa_pred"]).reindex(o.index)
    o["bvar_r"] = (bt["bvar_pred"] - bt["aa_pred"]).reindex(o.index)
    data = df[feats].reindex(aa.index).join(resid)
    o["lgbm_r"] = lgbm_wf(data, feats)
    o["lgbm_np_r"] = lgbm_wf(data, feats_np)
    o = o[(o.index.year >= EVAL_START)].dropna(subset=["actual", "aa", "lgbm_r", "tvp_r", "bvar_r"])
    a = o["actual"]
    T1 = o["aa"] + 0.5*o["tvp_r"] + 0.5*o["lgbm_r"]
    T2 = o["aa"] + (o["tvp_r"] + o["lgbm_r"] + o["bvar_r"]) / 3
    T1n = o["aa"] + 0.5*o["tvp_r"] + 0.5*o["lgbm_np_r"]
    T2n = o["aa"] + (o["tvp_r"] + o["lgbm_np_r"] + o["bvar_r"]) / 3

    rows = []
    for w, fn in WIN.items():
        m = fn(o.index); s = o[m]
        raa = rmse(a[m]-s["aa"])
        st, p = dm(a[m]-T1[m], a[m]-T2[m])   # >0 => T1 worse (T2 better)
        rows.append(dict(window=w, n=len(s), rmse_AA=raa,
                         rmse_T1=rmse(a[m]-T1[m]), rmse_T2=rmse(a[m]-T2[m]),
                         rel_T1=rmse(a[m]-T1[m])/raa, rel_T2=rmse(a[m]-T2[m])/raa,
                         T2_minus_T1=rmse(a[m]-T2[m])-rmse(a[m]-T1[m]),
                         DM_T1_vs_T2=st, p=p,
                         rmse_T1_noPPI=rmse(a[m]-T1n[m]), rmse_T2_noPPI=rmse(a[m]-T2n[m]),
                         T2n_minus_T1n=rmse(a[m]-T2n[m])-rmse(a[m]-T1n[m])))
    nec = pd.DataFrame(rows).set_index("window"); nec.to_csv(os.path.join(_OUT, "necessity.csv"))

    # rolling 5y
    rr = []
    for ys in range(EVAL_START, END-3):
        ye = ys+4; m = (o.index.year>=ys)&(o.index.year<=ye); s = o[m]
        if len(s) < 36: continue
        rr.append(dict(window=f"{ys}-{ye}", rmse_T1=rmse(a[m]-T1[m]), rmse_T2=rmse(a[m]-T2[m]),
                       T2_minus_T1=rmse(a[m]-T2[m])-rmse(a[m]-T1[m])))
    roll = pd.DataFrame(rr).set_index("window"); roll.to_csv(os.path.join(_OUT, "rolling.csv"))

    # error correlations
    e_bvar = a - (o["aa"] + o["bvar_r"])
    e_t1 = a - T1
    e_tl = a - (o["aa"] + 0.5*o["tvp_r"] + 0.5*o["lgbm_r"])  # = e_t1
    e_tvp = a - (o["aa"] + o["tvp_r"]); e_lgb = a - (o["aa"] + o["lgbm_r"])
    corr = pd.Series({
        "corr_BVAR_Tier1err": float(e_bvar.corr(e_t1)),
        "corr_BVAR_TVP": float(e_bvar.corr(e_tvp)),
        "corr_BVAR_LGBM": float(e_bvar.corr(e_lgb)),
        "corr_BVAR_meanTVPLGBM": float(e_bvar.corr((e_tvp+e_lgb)/2)),
    })
    corr.to_csv(os.path.join(_OUT, "corr.csv"))

    pd.options.display.width = 220
    print("=== necessity by window (rel vs AA; T2_minus_T1<0 => BVAR helps) ===")
    print(nec[["n","rel_T1","rel_T2","T2_minus_T1","DM_T1_vs_T2","p"]].round(4).to_string())
    print("\n=== PPI-ablation: Tier1_noPPI vs Tier2_noPPI (does BVAR help when LGBM has no PPI?) ===")
    print(nec[["rmse_T1_noPPI","rmse_T2_noPPI","T2n_minus_T1n"]].round(4).to_string())
    print("\n=== rolling 5y (T2_minus_T1<0 => BVAR helps) ==="); print(roll.round(4).to_string())
    print(f"  windows where BVAR helps: {int((roll['T2_minus_T1']<0).sum())}/{len(roll)}")
    print("\n=== error correlations ==="); print(corr.round(3).to_string())
    print("\nwritten necessity / rolling / corr")


if __name__ == "__main__":
    main()
