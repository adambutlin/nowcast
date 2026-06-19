"""
Bayesian shrinkage audit for Forecast = AA + lambda * Overlay (Overlay = 0.5 TVP + 0.5 LGBM).

Model the AA residual r = actual - AA as Signal + Noise, with the overlay o a noisy estimate.
Normal-normal / MMSE shrinkage: posterior mean of r given o = lambda*o with
  lambda_bayes = Cov(o,r)/Var(o)            (the reliability / attenuation factor)
Decompose overlay variance into signal+noise (shared-with-r interpretation):
  tau2 (signal) = Cov(o,r);  sigma2 (overlay noise) = Var(o) - Cov(o,r);  lambda = tau2/(tau2+sigma2)
Predictive signal-vs-noise: R2 = corr(o,r)^2 = fraction of residual variance the overlay explains.

Walk-forward lambda_bayes (expanding Cov/Var on history < t) + by-window. Compare to empirical
grid lambda (~0.8) and production proposal 0.5.

Out: data/timing/bayes/{by_window,walkforward}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/bayesian_shrinkage.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
import lightgbm as lgb
import uk_model_zoo as Z, two_stage as TS

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing", "bayes"); os.makedirs(_OUT, exist_ok=True)
EVAL_START, END = 2015, 2024
LGB = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
           min_child_samples=12, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=0, verbose=-1)
WIN = {"full": lambda i: i.year >= EVAL_START, "2022_23": lambda i: i.year.isin([2022, 2023]),
       "ex_shock": lambda i: (i.year >= EVAL_START) & ~i.year.isin([2022, 2023]),
       "pre_2020": lambda i: (i.year >= EVAL_START) & (i.year <= 2019)}


def reliab(o, r):
    o, r = np.asarray(o, float), np.asarray(r, float); m = np.isfinite(o) & np.isfinite(r)
    o, r = o[m], r[m]
    vo, vr = o.var(ddof=1), r.var(ddof=1); cov = np.cov(o, r, ddof=1)[0, 1]
    lam = cov/vo if vo > 0 else np.nan
    corr = cov/np.sqrt(vo*vr) if vo > 0 and vr > 0 else np.nan
    return dict(n=len(o), var_o=vo, var_r=vr, cov_or=cov, corr=corr, R2=corr**2,
                lambda_bayes=lam, tau2=cov, sigma2=vo-cov, signal_share=cov/vo if vo > 0 else np.nan)


def main():
    df, live, status = TS.load_matrix()
    aa = Z.AutoARIMA().backtest(df, [], TS.TARGET, start_year=TS.AA_START, end_year=END)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    bt = TS.backtest(df, live)
    o = pd.DataFrame({"actual": aa["actual"], "aa": aa["pred"]})
    o["tvp_r"] = (bt["tvp_pred"] - bt["aa_pred"]).reindex(o.index)
    data = df[live].reindex(aa.index).join(resid)
    lp = pd.Series(index=aa.index, dtype=float)
    for y in range(EVAL_START, END+1):
        tr = data[data.index.year < y].dropna(subset=["resid"]); te = data[data.index.year == y]
        if len(tr) < 36 or len(te) == 0: continue
        lp.loc[te.index] = lgb.LGBMRegressor(**LGB).fit(tr[live], tr["resid"]).predict(te[live])
    o["lgbm_r"] = lp
    o = o[(o.index.year >= EVAL_START)].dropna(subset=["actual","aa","tvp_r","lgbm_r"])
    o["overlay"] = 0.5*o["tvp_r"] + 0.5*o["lgbm_r"]
    o["r"] = o["actual"] - o["aa"]

    # by-window reliability
    rows = []
    for w, fn in WIN.items():
        s = o[fn(o.index)]
        rows.append(dict(window=w, **reliab(s["overlay"], s["r"])))
    bw = pd.DataFrame(rows).set_index("window"); bw.to_csv(os.path.join(_OUT,"by_window.csv"))

    # walk-forward lambda_bayes (expanding history < test year)
    wf = []
    for y in range(EVAL_START+3, END+1):
        h = o[o.index.year < y]
        if len(h) < 24: continue
        rr = reliab(h["overlay"], h["r"])
        wf.append(dict(test_year=y, lambda_bayes=rr["lambda_bayes"], R2=rr["R2"], n_hist=rr["n"]))
    wfdf = pd.DataFrame(wf).set_index("test_year"); wfdf.to_csv(os.path.join(_OUT,"walkforward.csv"))

    pd.options.display.width = 200
    print("=== Bayesian reliability by window ===")
    print(bw[["n","var_o","var_r","cov_or","corr","R2","lambda_bayes","signal_share"]].round(4).to_string())
    print("\n=== walk-forward lambda_bayes (expanding) ===")
    print(wfdf.round(4).to_string())
    print(f"\n  WF lambda_bayes: mean {wfdf['lambda_bayes'].mean():.3f} std {wfdf['lambda_bayes'].std():.3f} "
          f"min {wfdf['lambda_bayes'].min():.3f} max {wfdf['lambda_bayes'].max():.3f}")
    print(f"  WF R2 (signal share, predictive): mean {wfdf['R2'].mean():.3f}")
    print("\n=== comparison ===")
    print(f"  lambda_bayes (full, MMSE slope)      = {bw.loc['full','lambda_bayes']:.3f}")
    print(f"  lambda_bayes shock / ex_shock / pre  = {bw.loc['2022_23','lambda_bayes']:.3f} / "
          f"{bw.loc['ex_shock','lambda_bayes']:.3f} / {bw.loc['pre_2020','lambda_bayes']:.3f}")
    print(f"  predictive R2 (full)                 = {bw.loc['full','R2']:.3f}  (overlay explains this fraction of residual var)")
    print(f"  empirical grid optimum               ~ 0.80")
    print(f"  production proposal                  = 0.50")
    print("\nwritten by_window / walkforward")


if __name__ == "__main__":
    main()
