"""
Stability audit — members TVP, LGBM, BVAR (base = AutoARIMA). Is LGBM genuine or a
nonlinear wrapper around uk_ppi_input?

Deliverables:
  rolling_rmse.csv       rolling 5y (60-month) rel-rmse vs AA: TVP, BVAR, LGBM, LGBM_noPPI,
                         + ensembles TVP+LGBM, TVP+LGBM_noPPI.
  rolling_shap.csv       rolling/expanding mean|SHAP| shares (PPI share + top factor) over time.
  ppi_ablation.csv       LGBM with vs without uk_ppi_input by window; ΔRMSE; replacement factor.
  month_contribution.csv per-month gain |AA_err|-|member_err|; concentration of LGBM edge.

Walk-forward, expanding, no lookahead. eval 2015-2024.
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/stability_audit.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
import lightgbm as lgb
import uk_model_zoo as Z, two_stage as TS

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing", "stability")
os.makedirs(_OUT, exist_ok=True)
EVAL_START, END = 2015, 2024
LGB = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
           min_child_samples=12, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=0, verbose=-1)


def rmse(e): e = pd.Series(e).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else np.nan


def lgbm_wf(data, feats):
    pred = pd.Series(index=data.index, dtype=float)
    for y in range(EVAL_START, END+1):
        tr = data[data.index.year < y].dropna(subset=["resid"]); te = data[data.index.year == y]
        if len(tr) < 36 or len(te) == 0: continue
        pred.loc[te.index] = lgb.LGBMRegressor(**LGB).fit(tr[feats], tr["resid"]).predict(te[feats])
    return pred


def main():
    df, live, status = TS.load_matrix()
    feats = list(live); feats_noppi = [f for f in feats if f != "uk_ppi_input"]
    aa = Z.AutoARIMA().backtest(df, [], TS.TARGET, start_year=TS.AA_START, end_year=END)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    bt = TS.backtest(df, live)
    out = pd.DataFrame({"actual": aa["actual"], "aa": aa["pred"]})
    out["tvp"] = bt["tvp_pred"].reindex(out.index)
    out["bvar"] = bt["bvar_pred"].reindex(out.index)
    data = df[feats].reindex(aa.index).join(resid)
    out["lgbm"] = aa["pred"] + lgbm_wf(data, feats)
    out["lgbm_noppi"] = aa["pred"] + lgbm_wf(data, feats_noppi)
    out = out[(out.index.year >= EVAL_START)].dropna(subset=["actual", "aa", "lgbm"])
    a = out["actual"]
    members = ["tvp", "bvar", "lgbm", "lgbm_noppi"]
    # ensembles
    out["E"] = out["aa"] + 0.5*((out["tvp"]-out["aa"]) + (out["lgbm"]-out["aa"]))
    out["E_noppi"] = out["aa"] + 0.5*((out["tvp"]-out["aa"]) + (out["lgbm_noppi"]-out["aa"]))

    # 1. rolling 5y rel-rmse
    rows = []
    for ys in range(EVAL_START, END-3):
        ye = ys+4; m = (out.index.year >= ys) & (out.index.year <= ye); s = out[m]
        if len(s) < 36: continue
        raa = rmse(s["actual"]-s["aa"]); rec = dict(window=f"{ys}-{ye}", n=len(s), rmse_AA=raa)
        for k in members + ["E", "E_noppi"]:
            rec[f"rel_{k}"] = rmse(s["actual"]-s[k]) / raa
        rows.append(rec)
    roll = pd.DataFrame(rows).set_index("window"); roll.to_csv(os.path.join(_OUT, "rolling_rmse.csv"))

    # 2. rolling SHAP (expanding to window end)
    import shap
    srows = []
    for ys in range(EVAL_START, END-3):
        ye = ys+4
        tr = data[data.index.year <= ye].dropna(subset=["resid"])
        if len(tr) < 60: continue
        m = lgb.LGBMRegressor(**LGB).fit(tr[feats], tr["resid"])
        sv = pd.Series(np.abs(shap.TreeExplainer(m).shap_values(tr[feats])).mean(0), index=feats)
        sv = sv / (sv.sum() or 1)
        srows.append(dict(window=f"-{ye}", ppi_share=float(sv["uk_ppi_input"]),
                          top_factor=sv.idxmax(), top_share=float(sv.max()),
                          **{f"shap_{c}": float(sv[c]) for c in feats}))
    rshap = pd.DataFrame(srows).set_index("window"); rshap.to_csv(os.path.join(_OUT, "rolling_shap.csv"))

    # 3/4/5. PPI ablation by window + replacement factor (SHAP without ppi, full-fit)
    full_tr = data.dropna(subset=["resid"])
    m_np = lgb.LGBMRegressor(**LGB).fit(full_tr[feats_noppi], full_tr["resid"])
    sv_np = pd.Series(np.abs(shap.TreeExplainer(m_np).shap_values(full_tr[feats_noppi])).mean(0), index=feats_noppi)
    repl = sv_np.idxmax()
    abl = []
    for w, fn in {"full": lambda i: i.year>=EVAL_START, "2022_23": lambda i: i.year.isin([2022,2023]),
                  "ex_shock": lambda i: (i.year>=EVAL_START)&~i.year.isin([2022,2023]),
                  "pre_2020": lambda i: (i.year>=EVAL_START)&(i.year<=2019)}.items():
        m = fn(out.index); s = out[m]; raa = rmse(s["actual"]-s["aa"])
        abl.append(dict(window=w, n=len(s), rel_LGBM=rmse(s["actual"]-s["lgbm"])/raa,
                        rel_LGBM_noPPI=rmse(s["actual"]-s["lgbm_noppi"])/raa,
                        edge_lost=rmse(s["actual"]-s["lgbm_noppi"])/raa - rmse(s["actual"]-s["lgbm"])/raa,
                        replacement_factor=repl))
    ppia = pd.DataFrame(abl).set_index("window"); ppia.to_csv(os.path.join(_OUT, "ppi_ablation.csv"))

    # 6. month contribution / concentration of LGBM edge
    g = (a-out["aa"]).abs() - (a-out["lgbm"]).abs()   # LGBM monthly gain vs AA
    gnp = (a-out["aa"]).abs() - (a-out["lgbm_noppi"]).abs()
    mc = pd.DataFrame({"gain_LGBM": g, "gain_LGBM_noPPI": gnp,
                       "gain_TVP": (a-out["aa"]).abs()-(a-out["tvp"]).abs()}).sort_values("gain_LGBM", ascending=False)
    mc.to_csv(os.path.join(_OUT, "month_contribution.csv"))
    gpos = g[g > 0].sum()
    conc = dict(n=len(g), n_pos=int((g>0).sum()), n_neg=int((g<0).sum()),
                total_gain=float(g.sum()), top3_share=float(g.sort_values(ascending=False).head(3).sum()/gpos) if gpos>0 else np.nan,
                top5_share=float(g.sort_values(ascending=False).head(5).sum()/gpos) if gpos>0 else np.nan,
                top10_share=float(g.sort_values(ascending=False).head(10).sum()/gpos) if gpos>0 else np.nan)

    # 7. TVP complementarity after PPI removal (error correlation)
    et, el, eln = (a-out["tvp"]), (a-out["lgbm"]), (a-out["lgbm_noppi"])
    comp = dict(corr_TVP_LGBM=float(et.corr(el)), corr_TVP_LGBMnoPPI=float(et.corr(eln)),
                rel_LGBMnoPPI=rmse(eln)/rmse(a-out["aa"]),
                rel_E_noppi=rmse(a-out["E_noppi"])/rmse(a-out["aa"]),
                rel_E=rmse(a-out["E"])/rmse(a-out["aa"]))

    pd.options.display.width = 220
    print("=== 1. rolling 5y rel-rmse vs AA ===")
    print(roll[["n","rel_tvp","rel_bvar","rel_lgbm","rel_lgbm_noppi","rel_E","rel_E_noppi"]].round(3).to_string())
    print(f"\n  LGBM rel: mean {roll['rel_lgbm'].mean():.3f} std {roll['rel_lgbm'].std():.3f} worst {roll['rel_lgbm'].max():.3f} | beats AA {int((roll['rel_lgbm']<1).sum())}/{len(roll)}")
    print("\n=== 2. rolling SHAP (PPI share + top factor) ===")
    print(rshap[["ppi_share","top_factor","top_share"]].round(3).to_string())
    print("\n=== 3/4/5. PPI ablation (rel vs AA; edge_lost = noPPI worse) ===")
    print(ppia.round(3).to_string())
    print(f"  replacement top SHAP factor without PPI: {repl}")
    print("\n=== 6. LGBM edge concentration ==="); print(pd.Series(conc).round(3).to_string())
    print("\n=== 7. TVP complementarity after PPI removal ==="); print(pd.Series(comp).round(3).to_string())
    print("\nwritten rolling_rmse / rolling_shap / ppi_ablation / month_contribution")


if __name__ == "__main__":
    main()
