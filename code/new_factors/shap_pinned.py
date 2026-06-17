"""
SHAP ranking of the 10 PINNED live factors against the AutoARIMA residual.
Restricts the factor-race GBT/SHAP to the factors actually in the two-stage model.
Out: data/new_factors/shap_pinned.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/new_factors/shap_pinned.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import factors as F, uk_model_zoo as Z, two_stage as TS

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "new_factors")


def main():
    df, live, status = TS.load_matrix()
    print("live factors:", live)
    # AA residual
    base = F.build_matrix(names=[TS.TARGET])[0].resample("ME").last()
    base = base[base.index.year >= TS.TRAIN_FROM]
    aa = Z.AutoARIMA().backtest(base, [], TS.TARGET, start_year=TS.AA_START, end_year=TS.END)
    aa = aa[(aa.index.year >= TS.START) & (aa.index.year <= TS.END)]
    resid = (aa["actual"] - aa["pred"]).rename("resid")

    X = df[live].reindex(resid.index)
    X = X.ffill().fillna(X.median(numeric_only=True))
    m = resid.notna() & X.notna().all(axis=1)
    X, y = X[m], resid[m]

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    gbt = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                    subsample=0.8, random_state=0).fit(X, y)
    try:
        import shap
        sv = shap.TreeExplainer(gbt).shap_values(X)
        shap_abs = pd.Series(np.abs(sv).mean(0), index=X.columns)
    except Exception as e:
        print("shap fallback:", str(e)[:50]); shap_abs = pd.Series(gbt.feature_importances_, index=X.columns)
    perm = permutation_importance(gbt, X, y, n_repeats=30, random_state=0,
                                  scoring="neg_root_mean_squared_error")
    out = pd.DataFrame({
        "shap_mean_abs": shap_abs,
        "perm_dRMSE": pd.Series(perm.importances_mean, index=X.columns),
        "abs_corr": X.apply(lambda c: abs(c.corr(y))),
    }).sort_values("shap_mean_abs", ascending=False)
    out["shap_share_%"] = 100 * out["shap_mean_abs"] / out["shap_mean_abs"].sum()
    out.to_csv(os.path.join(_OUT, "shap_pinned.csv"))
    pd.options.display.width = 200
    print(f"\n=== SHAP RANK — {len(live)} pinned live factors on AA residual (n={len(X)}) ===")
    print(out.round(4).to_string())
    print("\nwritten", os.path.join(_OUT, "shap_pinned.csv"))


if __name__ == "__main__":
    main()
