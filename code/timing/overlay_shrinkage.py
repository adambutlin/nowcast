"""
Overlay-shrinkage calibration. Forecast = AA + lambda * Overlay, Overlay = 0.5 TVP + 0.5 LGBM.
Sweep lambda 0..1 (walk-forward member resids). Windows full/2022_23/ex_shock/pre_2020 + May-2026.
Also: closed-form RMSE-optimal lambda* = sum(ov*r)/sum(ov^2) (r = actual-AA), directional
sign-hit and corr(overlay, AA residual) -> is the overlay a level or just a direction signal?

Out: data/timing/shrink/{rmse_by_lambda,mae_by_lambda,summary}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/overlay_shrinkage.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
import lightgbm as lgb
import uk_model_zoo as Z, two_stage as TS

_ROOT = os.path.dirname(_CODE)
_OUT = os.path.join(_ROOT, "data", "timing", "shrink"); os.makedirs(_OUT, exist_ok=True)
EVAL_START, END = 2015, 2024
LAM = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
LGB = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
           min_child_samples=12, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=0, verbose=-1)
WIN = {"full": lambda i: i.year >= EVAL_START, "2022_23": lambda i: i.year.isin([2022, 2023]),
       "ex_shock": lambda i: (i.year >= EVAL_START) & ~i.year.isin([2022, 2023]),
       "pre_2020": lambda i: (i.year >= EVAL_START) & (i.year <= 2019)}


def rmse(e): e = pd.Series(e).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else np.nan
def mae(e): e = pd.Series(e).dropna(); return float(e.abs().mean()) if len(e) else np.nan
def lam_star(ov, r):
    ov, r = np.asarray(ov, float), np.asarray(r, float); m = np.isfinite(ov) & np.isfinite(r)
    return float((ov[m] @ r[m]) / (ov[m] @ ov[m])) if (ov[m] @ ov[m]) > 0 else np.nan


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
    a = o["actual"]

    # May 2026 overlay from reconstruction
    rc = pd.read_csv(os.path.join(_ROOT,"data","timing","may2026_reconstruction.csv"),index_col=0)["value"]
    may_aa = float(rc["aa_forecast"]); may_ov = 0.5*float(rc["tvp_resid"]) + 0.5*float(rc["lgbm_resid"])
    may_actual = float(rc["actual_cpi"])

    rmse_rows, mae_rows, may_rows = [], [], []
    for lam in LAM:
        f = o["aa"] + lam*o["overlay"]
        rr = {"lambda": lam}; mm = {"lambda": lam}
        for w, fn in WIN.items():
            m = fn(o.index); rr[w] = rmse(a[m]-f[m]); mm[w] = mae(a[m]-f[m])
        rmse_rows.append(rr); mae_rows.append(mm)
        may_f = may_aa + lam*may_ov
        may_rows.append(dict(lambda_=lam, may_forecast=round(may_f,3), may_abs_err=round(abs(may_f-may_actual),3)))
    R = pd.DataFrame(rmse_rows).set_index("lambda"); R.to_csv(os.path.join(_OUT,"rmse_by_lambda.csv"))
    M = pd.DataFrame(mae_rows).set_index("lambda"); M.to_csv(os.path.join(_OUT,"mae_by_lambda.csv"))
    May = pd.DataFrame(may_rows).set_index("lambda_")

    # closed-form lambda* + directional stats per window
    srows = []
    for w, fn in WIN.items():
        m = fn(o.index); s = o[m]
        srows.append(dict(window=w, n=len(s), lambda_star=lam_star(s["overlay"], s["r"]),
                          sign_hit=float((np.sign(s["overlay"])==np.sign(s["r"])).mean()),
                          corr_overlay_resid=float(s["overlay"].corr(s["r"])),
                          mean_abs_overlay=float(s["overlay"].abs().mean()),
                          mean_abs_resid=float(s["r"].abs().mean())))
    S = pd.DataFrame(srows).set_index("window"); S.to_csv(os.path.join(_OUT,"summary.csv"))

    pd.options.display.width = 200
    print("=== RMSE by lambda ==="); print(R.round(4).to_string())
    print("\nargmin RMSE lambda:", {w: float(R[w].idxmin()) for w in WIN})
    print("\n=== MAE by lambda ==="); print(M.round(4).to_string())
    print("argmin MAE lambda:", {w: float(M[w].idxmin()) for w in WIN})
    print("\n=== May 2026 by lambda (actual 2.8; AA=%.2f overlay=%.3f) ===" % (may_aa, may_ov))
    print(May.round(3).to_string())
    print(f"  lambda that nails May: {(may_actual-may_aa)/may_ov:.3f}")
    print("\n=== closed-form lambda* + directional ==="); print(S.round(3).to_string())
    print("\nwritten rmse_by_lambda / mae_by_lambda / summary")


if __name__ == "__main__":
    main()
