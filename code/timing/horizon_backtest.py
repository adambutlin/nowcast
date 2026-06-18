"""
Within-reference-month horizon backtest (branch `timing`): when does the edge arrive?

Reuses the intramonth causal as-of panel: build_panel(target, k) gives, for forecast
origin T-k (k calendar days before reference month-end), a panel where every historical
month's features are knowable at T-k (HF financials aggregated only over days <= T-k).
ModelStack runs AutoARIMA (baseline) + BVAR (factor) + TVP (regime_tvp) + MIDAS
(intramonth, U-MIDAS=ElasticNet on HF as-of) walk-forward at that origin.

For each origin k in {30,21,14,10,7,5,2,1}, reconstruct:
  AA                 = baseline
  AA+BVAR            = AA + bvar_resid
  AA+BVAR+MIDAS      = AA + 0.5*bvar_resid + 0.5*midas_resid
  full (production)  = AA + 0.375*bvar + 0.25*tvp + 0.375*midas
and compute RMSE/MAE/corr + edge vs AA + per-member contribution, by window.

Caveat: AutoARIMA baseline is ~origin-invariant in this panel (CPI M-1 vintage assumed
available), so the measured EDGE is cleanly the residual-models' HF/factor contribution at
each origin — exactly "how much does watching the month's data help". MIDAS here is the
HF-ElasticNet U-MIDAS variant (production-equivalent in spirit).

Out: data/timing/{horizon_accuracy,edge_arrival,model_contribution_by_horizon,
                  horizon_by_window,may2026_path}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/horizon_backtest.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE)
import numpy as np, pandas as pd
from intramonth import config as C, panel as P
from intramonth.stack import ModelStack

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing")
os.makedirs(_OUT, exist_ok=True)
TARGET = "cpi_headline_yoy"
ORIGINS = [30, 21, 14, 10, 7, 5, 2, 1]
WEIGHTS = {"factor": 0.375, "regime_tvp": 0.25, "intramonth": 0.375}  # BVAR/TVP/MIDAS
WIN = {"full": lambda i: True, "2022_23": lambda i: i.year in (2022, 2023),
       "ex_shock": lambda i: i.year not in (2022, 2023), "pre_2020": lambda i: i.year <= 2019}


def rmse(e):
    e = pd.Series(e).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else np.nan
def mae(e):
    e = pd.Series(e).dropna(); return float(e.abs().mean()) if len(e) else np.nan


def reconstruct(res):
    """Return DataFrame: actual, aa, aa_bvar, aa_bvar_midas, full + resid_pred per member."""
    aa = res["baseline"]
    df = pd.DataFrame({"actual": aa["actual"], "aa": aa["pred"]})
    m = res["models"]
    def rp(layer):
        x = m.get(layer)
        return x["bt"]["resid_pred"].reindex(df.index) if x else pd.Series(0.0, index=df.index)
    bvar, tvp, mid = rp("factor"), rp("regime_tvp"), rp("intramonth")
    df["bvar_resid"], df["tvp_resid"], df["midas_resid"] = bvar, tvp, mid
    df["aa_bvar"] = df["aa"] + bvar.fillna(0)
    df["aa_bvar_midas"] = df["aa"] + 0.5 * bvar.fillna(0) + 0.5 * mid.fillna(0)
    df["full"] = df["aa"] + (WEIGHTS["factor"] * bvar.fillna(0) +
                            WEIGHTS["regime_tvp"] * tvp.fillna(0) +
                            WEIGHTS["intramonth"] * mid.fillna(0))
    return df.dropna(subset=["actual", "aa"])


def main():
    per_origin = {}
    print("Running horizon backtests (this re-fits the stack per origin)…")
    for k in ORIGINS:
        pan, meta = P.build_panel(TARGET, k=k)
        res = ModelStack(pan, meta, end_year=2024).run()
        df = reconstruct(res)
        per_origin[k] = (df, res)
        e_aa = rmse(df["actual"] - df["aa"]); e_full = rmse(df["actual"] - df["full"])
        print(f"  T-{k:<2}: n={len(df)} rmseAA={e_aa:.4f} rmseFull={e_full:.4f} "
              f"edge={e_aa-e_full:+.4f} contrib bvar/tvp/midas="
              f"{df['bvar_resid'].abs().mean():.3f}/{df['tvp_resid'].abs().mean():.3f}/{df['midas_resid'].abs().mean():.3f}")

    # PART B + C: horizon accuracy + edge (full sample)
    rows_acc, rows_edge, rows_contrib = [], [], []
    for k in ORIGINS:
        df, res = per_origin[k]
        a = df["actual"]
        cols = {"AA": "aa", "AA+BVAR": "aa_bvar", "AA+BVAR+MIDAS": "aa_bvar_midas", "full": "full"}
        rec = {"origin": f"T-{k}", "k": k, "n": len(df)}
        for nm, c in cols.items():
            rec[f"rmse_{nm}"] = rmse(a - df[c]); rec[f"mae_{nm}"] = mae(a - df[c])
            rec[f"corr_{nm}"] = float(np.corrcoef(a, df[c])[0, 1]) if len(df) > 2 else np.nan
        rows_acc.append(rec)
        e = {"origin": f"T-{k}", "k": k, "rmse_AA": rmse(a - df["aa"])}
        for nm, c in cols.items():
            if nm != "AA":
                e[f"edge_{nm}"] = rmse(a - df["aa"]) - rmse(a - df[c])
        rows_edge.append(e)
        rows_contrib.append({"origin": f"T-{k}", "k": k,
                             "contrib_BVAR": df["bvar_resid"].abs().mean(),
                             "contrib_TVP": df["tvp_resid"].abs().mean(),
                             "contrib_MIDAS": df["midas_resid"].abs().mean(),
                             "edge_full": rmse(a - df["aa"]) - rmse(a - df["full"])})
    acc = pd.DataFrame(rows_acc).set_index("origin")
    edge = pd.DataFrame(rows_edge).set_index("origin")
    contrib = pd.DataFrame(rows_contrib).set_index("origin")
    acc.to_csv(os.path.join(_OUT, "horizon_accuracy.csv"))
    edge.to_csv(os.path.join(_OUT, "edge_arrival.csv"))
    contrib.to_csv(os.path.join(_OUT, "model_contribution_by_horizon.csv"))

    # final edge fraction earned by each horizon (vs T = T-1 here, the last origin)
    edge_T = edge.loc["T-1", "edge_full"]
    edge["pct_of_final_edge"] = 100 * edge["edge_full"] / edge_T if abs(edge_T) > 1e-9 else np.nan

    pd.options.display.width = 220
    print("\n=== PART B/C: horizon accuracy + edge (full sample) ===")
    print(acc[["n", "rmse_AA", "rmse_AA+BVAR", "rmse_AA+BVAR+MIDAS", "rmse_full"]].round(4).to_string())
    print("\nedge vs AA (RMSE reduction) + % of final (T-1) edge earned:")
    print(edge[["rmse_AA", "edge_full", "pct_of_final_edge"]].round(4).to_string())

    print("\n=== PART D: contribution by horizon (mean |resid overlay|) ===")
    print(contrib[["contrib_BVAR", "contrib_TVP", "contrib_MIDAS", "edge_full"]].round(4).to_string())

    # PART E: by window
    rows = []
    for k in ORIGINS:
        df, res = per_origin[k]; a = df["actual"]
        for w, fn in WIN.items():
            m = df.index.map(lambda x: fn(x)).values.astype(bool)
            s = df[m]
            if len(s) < 4:
                continue
            rows.append(dict(origin=f"T-{k}", window=w, n=len(s),
                             rmse_AA=rmse(s["actual"]-s["aa"]),
                             rmse_full=rmse(s["actual"]-s["full"]),
                             edge=rmse(s["actual"]-s["aa"])-rmse(s["actual"]-s["full"])))
    byw = pd.DataFrame(rows).set_index(["window", "origin"])
    byw.to_csv(os.path.join(_OUT, "horizon_by_window.csv"))
    print("\n=== PART E: edge by window x horizon ===")
    print(byw["edge"].unstack(0).round(4).to_string())

    print("\nwritten horizon_accuracy / edge_arrival / model_contribution_by_horizon / horizon_by_window")


if __name__ == "__main__":
    main()
