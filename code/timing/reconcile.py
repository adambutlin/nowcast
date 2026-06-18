"""
Reconcile Result A (production two_stage, rel~0.93) vs Result B (intramonth panel,
negative edge at every horizon incl T-1).

Runs BOTH on overlapping dates, extracts per-month AA + member + full forecasts, aligns,
and decomposes the discrepancy into: (i) different AutoARIMA, (ii) different overlay
(factors/MIDAS class), (iii) different evaluation sample.

Out: data/timing/{production_reconstruction,intramonth_reconstruction,convergence_audit,
                  information_difference}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/reconcile.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
import two_stage as TS
from intramonth import config as C, panel as P
from intramonth.stack import ModelStack

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing")
W = {"factor": 0.375, "regime_tvp": 0.25, "intramonth": 0.375}


def rmse(e): e = pd.Series(e).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else np.nan


def production():
    df, live, status = TS.load_matrix()
    bt = TS.backtest(df, live)   # actual, aa_pred, *_pred, forecast (WEIGHTS already wired)
    out = pd.DataFrame({"actual": bt["actual"], "aa": bt["aa_pred"], "full": bt["forecast"],
                        "bvar": bt.get("bvar_pred"), "tvp": bt.get("tvp_pred"),
                        "midas": bt.get("midas_pred")})
    out.to_csv(os.path.join(_OUT, "production_reconstruction.csv"))
    return out, live


def intramonth(k=1):
    pan, meta = P.build_panel("cpi_headline_yoy", k=k)
    res = ModelStack(pan, meta, end_year=2024).run()
    aa = res["baseline"]
    out = pd.DataFrame({"actual": aa["actual"], "aa": aa["pred"]})
    m = res["models"]
    for layer, tag in [("factor", "bvar"), ("regime_tvp", "tvp"), ("intramonth", "midas")]:
        x = m.get(layer)
        out[tag] = (x["bt"]["cpi_pred"].reindex(out.index) if x else np.nan)
        out[tag + "_resid"] = (x["bt"]["resid_pred"].reindex(out.index) if x else 0.0)
    out["full"] = out["aa"] + (W["factor"]*out["bvar_resid"].fillna(0) +
                               W["regime_tvp"]*out["tvp_resid"].fillna(0) +
                               W["intramonth"]*out["midas_resid"].fillna(0))
    out = out.dropna(subset=["actual", "aa"])
    out.to_csv(os.path.join(_OUT, "intramonth_reconstruction.csv"))
    feats = P.factor_columns(meta)
    return out, feats


def main():
    print("Running production two_stage backtest …")
    prod, prod_live = production()
    print("Running intramonth panel at T-1 …")
    intra, intra_feats = intramonth(k=1)

    pr = prod.index.min(), prod.index.max()
    ir = intra.index.min(), intra.index.max()
    print(f"\nproduction sample: {pr[0].date()}..{pr[1].date()} n={len(prod)}")
    print(f"intramonth sample: {ir[0].date()}..{ir[1].date()} n={len(intra)}")

    # PART E: information / structural difference
    info = pd.DataFrame([
        dict(item="factor_set", production=str(sorted(prod_live)),
             intramonth=str(sorted(intra_feats))),
        dict(item="MIDAS_class", production="Z.MIDAS (U-MIDAS, own daily yfinance, FULL-month mean)",
             intramonth="ElasticNet alias on HF as-of (partial-month at T-1)"),
        dict(item="data_window", production="month-END snapshot (resample ME.last), FULL month",
             intramonth="as-of T-1 (HF over days<=T-1; monthly factors ffilled)"),
        dict(item="sample", production=f"{pr[0].date()}..{pr[1].date()} n={len(prod)}",
             intramonth=f"{ir[0].date()}..{ir[1].date()} n={len(intra)}"),
        dict(item="rmse_AA_ownsample", production=f"{rmse(prod['actual']-prod['aa']):.4f}",
             intramonth=f"{rmse(intra['actual']-intra['aa']):.4f}"),
    ])
    info.to_csv(os.path.join(_OUT, "information_difference.csv"), index=False)
    print("\n=== PART E: structural differences ==="); print(info.to_string(index=False))

    # PART C: convergence on COMMON dates
    common = prod.index.intersection(intra.index)
    cv = pd.DataFrame({"actual": prod.loc[common, "actual"],
                       "aa_prod": prod.loc[common, "aa"], "aa_intra": intra.loc[common, "aa"],
                       "full_prod": prod.loc[common, "full"], "full_intra": intra.loc[common, "full"]})
    cv["d_aa"] = cv["aa_intra"] - cv["aa_prod"]
    cv["d_full"] = cv["full_intra"] - cv["full_prod"]
    cv["overlay_prod"] = cv["full_prod"] - cv["aa_prod"]
    cv["overlay_intra"] = cv["full_intra"] - cv["aa_intra"]
    cv.to_csv(os.path.join(_OUT, "convergence_audit.csv"))
    print(f"\n=== PART C: convergence on {len(common)} common months ===")
    print(f"  mean |AA_intra - AA_prod|   = {cv['d_aa'].abs().mean():.4f}  (corr {cv['aa_prod'].corr(cv['aa_intra']):.3f})")
    print(f"  mean |full_intra-full_prod| = {cv['d_full'].abs().mean():.4f}")
    print(f"  mean overlay prod={cv['overlay_prod'].abs().mean():.4f}  intra={cv['overlay_intra'].abs().mean():.4f}")

    # edges on common sample
    def edge(df, idx):
        s = df.loc[idx]
        return rmse(s["actual"]-s["aa"]) - rmse(s["actual"]-s["full"]), rmse(s["actual"]-s["aa"]), rmse(s["actual"]-s["full"])
    eP, aP, fP = edge(prod, common); eI, aI, fI = edge(intra, common)
    print("\n=== edges on COMMON sample (same dates) ===")
    print(f"  PRODUCTION  : rmseAA={aP:.4f} rmseFull={fP:.4f} edge={eP:+.4f} rel={fP/aP:.3f}")
    print(f"  INTRAMONTH  : rmseAA={aI:.4f} rmseFull={fI:.4f} edge={eI:+.4f} rel={fI/aI:.3f}")
    # production edge on its OWN full sample (the 0.93 headline)
    eP0, aP0, fP0 = edge(prod, prod.index)
    eI0, aI0, fI0 = edge(intra, intra.index)
    print("\n=== edges on OWN samples (headline numbers) ===")
    print(f"  PRODUCTION own (n={len(prod)}): edge={eP0:+.4f} rel={fP0/aP0:.3f}")
    print(f"  INTRAMONTH own (n={len(intra)}): edge={eI0:+.4f} rel={fI0/aI0:.3f}")
    print("\nwritten production_reconstruction / intramonth_reconstruction / convergence_audit / information_difference")


if __name__ == "__main__":
    main()
