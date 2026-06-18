"""
TVP-vs-LGBM regime audit. (Not HMM / HelpfulStage2 / ObservableShock — a new target.)

ShockAdvantage_t = |Error_LGBM_t| - |Error_TVP_t|   (>0 => TVP wins; <0 => LGBM wins)
TVP_wins = 1[ |err_TVP| < |err_LGBM| ].
Features (causal, pub-lagged): uk_ppi_input, deep_sea_freight, move_index, ofgem_cap_delta,
oil_brent(+|.|), gas_eu(+|.|), imf_all_commodity.

Detectors (walk-forward, expanding, test 2018-2024): linear reg + LGBM reg on ShockAdvantage,
logistic on TVP_wins. Then a SWITCHING ensemble (pick TVP or LGBM by predicted winner / prob)
vs equal-weight E (TVP+LGBM) and F (TVP+LGBM+BVAR).

Out: data/timing/regime/{detector_metrics,importance,switching_compare,winrate_by_window}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/regime_tvp_lgbm.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
import lightgbm as lgb
import factors as F, uk_model_zoo as Z, two_stage as TS

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing", "regime")
os.makedirs(_OUT, exist_ok=True)
EVAL_START, END, DET_START = 2015, 2024, 2018
LGB = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
           min_child_samples=12, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=0, verbose=-1)
FEATS_RAW = ["uk_ppi_input", "deep_sea_freight", "move_index", "ofgem_cap_delta",
             "oil_brent", "gas_eu", "imf_all_commodity"]
WIN = {"full": lambda i: i.year >= DET_START, "2022_23": lambda i: i.year.isin([2022, 2023]),
       "ex_shock": lambda i: (i.year >= DET_START) & ~i.year.isin([2022, 2023])}


def rmse(e): e = pd.Series(e).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else np.nan


def build():
    df, live, status = TS.load_matrix()
    aa = Z.AutoARIMA().backtest(df, [], TS.TARGET, start_year=TS.AA_START, end_year=END)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    bt = TS.backtest(df, live)
    base = pd.DataFrame({"actual": aa["actual"], "aa": aa["pred"]})
    base["tvp_resid"] = (bt["tvp_pred"] - bt["aa_pred"]).reindex(base.index)
    base["bvar_resid"] = (bt["bvar_pred"] - bt["aa_pred"]).reindex(base.index)
    # LGBM walk-forward resid
    lfeats = list(live); data = df[lfeats].reindex(aa.index).join(resid)
    lp = pd.Series(index=aa.index, dtype=float)
    for y in range(EVAL_START, END+1):
        tr = data[data.index.year < y].dropna(subset=["resid"]); te = data[data.index.year == y]
        if len(tr) < 36 or len(te) == 0: continue
        lp.loc[te.index] = lgb.LGBMRegressor(**LGB).fit(tr[lfeats], tr["resid"]).predict(te[lfeats])
    base["lgbm_resid"] = lp
    # features for the detector
    fr, st = F.build_matrix(names=[c for c in FEATS_RAW])
    fr = F.apply_publication_lags(fr, [c for c in FEATS_RAW if st.get(c) != "unavailable"]).resample("ME").last()
    for c in ["ofgem_cap_delta"]:
        if c in fr: fr[c] = fr[c].fillna(0)
    feat = pd.DataFrame(index=base.index)
    for c in FEATS_RAW:
        if c in fr: feat[c] = fr[c].reindex(base.index)
    for c in ["oil_brent", "gas_eu"]:
        if c in feat: feat[f"abs_{c}"] = feat[c].abs()
    base = base.join(feat)
    base = base[(base.index.year >= EVAL_START) & base["lgbm_resid"].notna()].dropna(subset=["actual","aa"])
    return base, [c for c in feat.columns]


def main():
    b, feats = build()
    err_tvp = (b["actual"] - (b["aa"] + b["tvp_resid"])).abs()
    err_lgb = (b["actual"] - (b["aa"] + b["lgbm_resid"])).abs()
    b["shock_adv"] = err_lgb - err_tvp           # >0 TVP wins
    b["tvp_wins"] = (err_tvp < err_lgb).astype(int)
    b = b[b["shock_adv"].notna()].copy()
    print(f"base rate TVP_wins (full eval) = {b['tvp_wins'].mean():.3f}  n={len(b)}")

    # win-rate by window (descriptive: is there a real regime?)
    wr = []
    for w, fn in {"full": lambda i: i.year>=EVAL_START, "2022_23": lambda i: i.year.isin([2022,2023]),
                  "ex_shock": lambda i: (i.year>=EVAL_START)&~i.year.isin([2022,2023]),
                  "pre_2020": lambda i: i.year<=2019}.items():
        m = fn(b.index); s = b[m]
        wr.append(dict(window=w, n=len(s), tvp_win_rate=float(s["tvp_wins"].mean()),
                       mean_shock_adv=float(s["shock_adv"].mean())))
    wrdf = pd.DataFrame(wr).set_index("window"); wrdf.to_csv(os.path.join(_OUT,"winrate_by_window.csv"))
    print("\n=== win-rate by window (descriptive) ==="); print(wrdf.round(3).to_string())

    # walk-forward detectors (test DET_START..END)
    X = b[feats].fillna(b[feats].median());
    p_lin = pd.Series(index=b.index, dtype=float); p_lgb = pd.Series(index=b.index, dtype=float)
    p_log = pd.Series(index=b.index, dtype=float)
    for y in range(DET_START, END+1):
        tr = b[b.index.year < y]; te = b[b.index.year == y]
        if len(tr) < 24 or len(te) == 0: continue
        Xtr, Xte = X.loc[tr.index], X.loc[te.index]
        sc = StandardScaler().fit(Xtr)
        p_lin.loc[te.index] = LinearRegression().fit(sc.transform(Xtr), tr["shock_adv"]).predict(sc.transform(Xte))
        p_lgb.loc[te.index] = lgb.LGBMRegressor(**LGB).fit(Xtr, tr["shock_adv"]).predict(Xte)
        if tr["tvp_wins"].nunique() > 1:
            p_log.loc[te.index] = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), tr["tvp_wins"]).predict_proba(sc.transform(Xte))[:,1]
        else:
            p_log.loc[te.index] = tr["tvp_wins"].mean()
    ev = b[(b.index.year >= DET_START) & p_lin.notna()].copy()
    y = ev["tvp_wins"].values
    def auc(p):
        return roc_auc_score(y, p.loc[ev.index]) if len(np.unique(y))>1 else np.nan
    det = pd.DataFrame([
        dict(detector="linear_reg(shock_adv)", auc=auc((p_lin>0).astype(float)*0+ (p_lin.loc[ev.index].rank()/len(ev))),
             sign_hit=float(((p_lin.loc[ev.index]>0).astype(int)==ev["tvp_wins"]).mean())),
        dict(detector="lgbm_reg(shock_adv)", auc=np.nan,
             sign_hit=float(((p_lgb.loc[ev.index]>0).astype(int)==ev["tvp_wins"]).mean())),
        dict(detector="logistic(tvp_wins)", auc=float(auc(p_log)),
             sign_hit=float(((p_log.loc[ev.index]>=0.5).astype(int)==ev["tvp_wins"]).mean())),
    ]).set_index("detector")
    det["base_rate"] = ev["tvp_wins"].mean()
    det["brier_logistic"] = brier_score_loss(y, p_log.loc[ev.index]) if len(np.unique(y))>1 else np.nan
    det.to_csv(os.path.join(_OUT,"detector_metrics.csv"))
    print("\n=== detector metrics (OOS 2018-2024) ==="); print(det.round(3).to_string())

    # importance (full-fit logistic coef + lgbm gain)
    sc = StandardScaler().fit(X.loc[ev.index])
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(X.loc[ev.index]), ev["tvp_wins"])
    imp = pd.DataFrame({"logit_coef": pd.Series(lr.coef_[0], index=feats)}).sort_values("logit_coef")
    imp.to_csv(os.path.join(_OUT,"importance.csv"))
    print("\n=== logistic coef (standardised; + => predicts TVP wins) ==="); print(imp.round(3).to_string())

    # switching vs averaging
    aa = ev["aa"]
    fc = {}
    fc["E_equal"] = aa + 0.5*ev["tvp_resid"] + 0.5*ev["lgbm_resid"]
    fc["F_equal"] = aa + (ev["tvp_resid"]+ev["lgbm_resid"]+ev["bvar_resid"])/3
    # hard switch by logistic
    pick_tvp = (p_log.loc[ev.index] >= 0.5)
    fc["switch_hard"] = aa + np.where(pick_tvp, ev["tvp_resid"], ev["lgbm_resid"])
    # soft switch by prob
    w = p_log.loc[ev.index].clip(0,1)
    fc["switch_soft"] = aa + w*ev["tvp_resid"] + (1-w)*ev["lgbm_resid"]
    # oracle (perfect foresight) upper bound
    fc["oracle"] = aa + np.where(ev["tvp_wins"]==1, ev["tvp_resid"], ev["lgbm_resid"])
    rows = []
    for name, f in fc.items():
        rec = dict(model=name)
        for w_, fn in WIN.items():
            m = fn(ev.index); s = ev[m]
            rec[f"rmse_{w_}"] = rmse(s["actual"]-f[m])
        rows.append(rec)
    sw = pd.DataFrame(rows).set_index("model"); sw.to_csv(os.path.join(_OUT,"switching_compare.csv"))
    print("\n=== switching vs averaging (RMSE; oracle = perfect-foresight upper bound) ===")
    print(sw.round(4).to_string())
    print("\nwritten detector_metrics/importance/switching_compare/winrate_by_window")


if __name__ == "__main__":
    main()
