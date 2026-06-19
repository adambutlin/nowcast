"""
Genesis row: reconstruct May-2026 with the FINAL production model
  Forecast = AA + 0.5*TVP + 0.5*LGBM     (Tier 1)
using only information available before the May-2026 release (cpi known through April).

Also recompute the OLD production model (AA + 0.375 BVAR + 0.25 TVP + 0.375 MIDAS) for
comparison. Consensus=3.0, UCL=3.05, actual=2.8 (stated). No retraining of architecture.

Writes data/timing/may2026_reconstruction.csv and seeds data/live_scorecard.csv (row 1).
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/may2026_reconstruction.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
import lightgbm as lgb
import uk_model_zoo as Z, two_stage as TS

_ROOT = os.path.dirname(_CODE)
LGB = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
           min_child_samples=12, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=0, verbose=-1)
CONSENSUS, UCL, ACTUAL = 3.0, 3.05, 2.8
CPI = os.path.join(_ROOT, "data", "cpi_yoy.csv")


def main():
    # temp: cpi ending April so May is the unreleased nowcast target
    bak = pd.read_csv(CPI, parse_dates=["date"]).set_index("date")["value"]
    apr = bak[bak.index < "2026-05-01"]
    apr.to_frame("value").to_csv(CPI)
    try:
        df, live, status = TS.load_matrix()
        aa_pred, nd = Z.AutoARIMA().nowcast(df, [], TS.TARGET)
        # residual history
        aah = Z.AutoARIMA().backtest(df, [], TS.TARGET, start_year=TS.AA_START, end_year=2024)
        resid = (aah["actual"] - aah["pred"]).reindex(df.index).rename("resid")
        d = df.copy(); d["resid"] = resid
        tvp_resid, _ = Z.TVP().nowcast(d, live, "resid")
        # LGBM live: fit on all known residual, predict May feature row
        data = df[live].join(resid).dropna(subset=["resid"])
        m = lgb.LGBMRegressor(**LGB).fit(data[live], data["resid"])
        may_row = df.loc[[nd], live]
        lgbm_resid = float(m.predict(may_row)[0])
        # OLD production model (two_stage current STAGE2/WEIGHTS)
        old_nc = TS.nowcast(df, live)
    finally:
        bak.to_frame("value").to_csv(CPI)   # restore

    final = float(aa_pred + 0.5*tvp_resid + 0.5*lgbm_resid)
    rec = dict(release_month="2026-05", nowcast_date=str(pd.Timestamp(nd).date()),
               aa_forecast=round(float(aa_pred),3),
               tvp_contribution=round(0.5*float(tvp_resid),3),
               lgbm_contribution=round(0.5*lgbm_resid,3),
               tvp_resid=round(float(tvp_resid),3), lgbm_resid=round(lgbm_resid,3),
               model_forecast=round(final,3),
               old_prod_forecast=round(float(old_nc["forecast"]),3),
               consensus_forecast=CONSENSUS, ucl_forecast=UCL, actual_cpi=ACTUAL,
               model_abs_err=round(abs(final-ACTUAL),3), model_signed_err=round(final-ACTUAL,3))
    out = os.path.join(_ROOT, "data", "timing", "may2026_reconstruction.csv")
    pd.Series(rec).to_frame("value").to_csv(out)
    print("=== May 2026 reconstruction (FINAL model AA + 0.5 TVP + 0.5 LGBM) ===")
    for k, v in rec.items():
        print(f"  {k:22} {v}")

    # seed live scorecard
    sc = pd.DataFrame([dict(
        release_month="2026-05", forecast_date=str(pd.Timestamp(nd).date()),
        model_forecast=round(final,3), aa_forecast=round(float(aa_pred),3),
        consensus_forecast=CONSENSUS, ucl_forecast=UCL, actual_cpi=ACTUAL,
        model_error=round(final-ACTUAL,3), aa_error=round(float(aa_pred)-ACTUAL,3),
        consensus_error=round(CONSENSUS-ACTUAL,3), ucl_error=round(UCL-ACTUAL,3))])
    scpath = os.path.join(_ROOT, "data", "live_scorecard.csv")
    sc.to_csv(scpath, index=False)
    print("\nseeded", scpath)
    print(sc.to_string(index=False))


if __name__ == "__main__":
    main()
