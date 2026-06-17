"""
Stage-2 BVAR/MIDAS allocation sweep with TVP LOCKED at 0.25.

Combo overlay = 0.25*TVP_ov + w_bvar*BVAR_ov + w_midas*MIDAS_ov   (w_bvar+w_midas=0.75),
where *_ov = member_recon - AA. Renormalised over members available per row.

One walk-forward backtest -> member preds; sweep weights cheaply.

Deliverables (data/new_factors/):
  weight_sweep.csv               (Part B: RMSE/MAE/OOS-corr/rel, full sample)
  weight_sweep_robustness.csv    (Part C: RMSE by window)
  live_forecast_sensitivity.csv  (Part D: May-2026 forecast per weight)
  error_correlation.csv          (Part E: member error corr)
  factor_exposure_by_weight.csv  (Part F: combo exposure to factor groups)
  + DM hostile tests printed (Part G)
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/new_factors/alloc_sweep.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
import two_stage as TS

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "new_factors")
W_TVP = 0.25
# (w_bvar, w_midas) summing to 0.75  (spec grid + intermediates)
GRID = [(0.75, 0.00), (0.65, 0.10), (0.55, 0.20), (0.50, 0.25), (0.45, 0.30),
        (0.375, 0.375), (0.30, 0.45), (0.25, 0.50), (0.20, 0.55), (0.10, 0.65), (0.00, 0.75)]
WINDOWS = {"full": lambda i: i.year >= TS.START,
           "2022_23": lambda i: i.year.isin([2022, 2023]),
           "ex_shock": lambda i: ~i.year.isin([2022, 2023]),
           "pre_2020": lambda i: i.year <= 2019}
ENERGY_FX = ["oil_brent", "gas_eu", "gbpusd", "vix"]          # MIDAS daily channel
COSTPUSH = ["uk_ppi_input", "deep_sea_freight", "ofgem_cap_delta"]  # BVAR/TVP monthly channel


def dm(e1, e2):
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    d = e1**2 - e2**2; d = d[np.isfinite(d)]; n = len(d)
    if n < 8 or np.allclose(d, 0):
        return np.nan, np.nan
    db = d.mean(); dd = d - db; v = (dd @ dd) / n; L = max(1, int(round(n ** (1/3))))
    for k in range(1, L + 1):
        v += 2 * (1 - k / (L + 1)) * ((dd[k:] @ dd[:-k]) / n)
    if v <= 0:
        return np.nan, np.nan
    s = db / np.sqrt(v / n)
    return float(s), float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


def rmse(e): e = np.asarray(e, float); e = e[np.isfinite(e)]; return float(np.sqrt((e**2).mean())) if len(e) else np.nan
def mae(e):  e = np.asarray(e, float); e = e[np.isfinite(e)]; return float(np.abs(e).mean()) if len(e) else np.nan


def combo_overlay(ov, wb, wm):
    """ov: DataFrame cols bvar/tvp/midas (member - AA). Row-wise renorm over available."""
    w = {"bvar": wb, "tvp": W_TVP, "midas": wm}
    cols = [c for c in ["bvar", "tvp", "midas"] if c in ov.columns]
    W = pd.Series({c: w[c] for c in cols})
    M = ov[cols]
    denom = (M.notna().astype(float) * W.values).sum(axis=1).replace(0, np.nan)
    return (M.fillna(0) * W.values).sum(axis=1) / denom


def main():
    df, live, status = TS.load_matrix()
    print("live factors:", live)
    bt = TS.backtest(df, live)
    nc = TS.nowcast(df, live)
    aa, actual = bt["aa_pred"], bt["actual"]
    ov = pd.DataFrame({m: bt[f"{m}_pred"] - aa for m in ["bvar", "tvp", "midas"] if f"{m}_pred" in bt})

    # ---- Part B/C: sweep ----
    rows_full, rows_rob, rows_live = [], [], []
    err_by_w = {}
    for wb, wm in GRID:
        fc = aa + combo_overlay(ov, wb, wm).fillna(0)
        err = actual - fc
        err_by_w[(wb, wm)] = err
        # full metrics
        corr = float(np.corrcoef(actual, fc)[0, 1])
        rows_full.append(dict(w_bvar=wb, w_midas=wm, rmse=rmse(err), mae=mae(err),
                              oos_corr=corr, rel_rmse=rmse(err) / rmse(actual - aa)))
        rec = dict(w_bvar=wb, w_midas=wm)
        for wn, fn in WINDOWS.items():
            rec[f"rmse_{wn}"] = rmse(err[fn(bt.index)])
        rows_rob.append(rec)
        # live
        ml = nc["members"]
        ovl = {"bvar": ml.get("bvar", np.nan), "tvp": ml.get("tvp", np.nan), "midas": ml.get("midas", np.nan)}
        wl = {"bvar": wb, "tvp": W_TVP, "midas": wm}
        present = {k: v for k, v in ovl.items() if np.isfinite(v)}
        wsum = sum(wl[k] for k in present) or 1.0
        overlay_l = sum(wl[k] * v for k, v in present.items()) / wsum
        rows_live.append(dict(w_bvar=wb, w_midas=wm, nowcast=float(nc["aa_pred"] + overlay_l)))

    full = pd.DataFrame(rows_full).set_index(["w_bvar", "w_midas"])
    rob = pd.DataFrame(rows_rob).set_index(["w_bvar", "w_midas"])
    live_df = pd.DataFrame(rows_live).set_index(["w_bvar", "w_midas"])
    full.to_csv(os.path.join(_OUT, "weight_sweep.csv"))
    rob.to_csv(os.path.join(_OUT, "weight_sweep_robustness.csv"))
    live_df.to_csv(os.path.join(_OUT, "live_forecast_sensitivity.csv"))

    pd.options.display.width = 200
    print("\n=== PART B  full-sample (TVP=0.25 fixed) ===")
    print(full.round(4).to_string())
    print("\n=== PART C  RMSE by window ===")
    print(rob.round(4).to_string())
    print("\n=== PART D  live May-2026 forecast (actual 2.8) ===")
    print(live_df.round(4).to_string())

    # ---- Part E: error correlation between members ----
    me = pd.DataFrame({m: (actual - bt[f"{m}_pred"]) for m in ["bvar", "tvp", "midas"] if f"{m}_pred" in bt})
    ec = me.corr()
    ec.to_csv(os.path.join(_OUT, "error_correlation.csv"))
    print("\n=== PART E  member error correlation ===")
    print(ec.round(3).to_string())

    # ---- Part F: factor exposure by weight ----
    # member-overlay sensitivity to factor groups (|corr| of member_ov with group factors)
    fac = df.reindex(ov.index)
    def grp_exposure(member, group):
        cols = [c for c in group if c in fac.columns]
        if not cols or member not in ov.columns:
            return np.nan
        cs = [abs(ov[member].corr(fac[c])) for c in cols]
        cs = [c for c in cs if np.isfinite(c)]
        return float(np.mean(cs)) if cs else np.nan
    member_exp = {m: {"costpush": grp_exposure(m, COSTPUSH), "energy_fx": grp_exposure(m, ENERGY_FX)}
                  for m in ov.columns}
    rows_fx = []
    for wb, wm in GRID:
        w = {"bvar": wb, "tvp": W_TVP, "midas": wm}
        cp = sum(w[m] * member_exp[m]["costpush"] for m in ov.columns if np.isfinite(member_exp[m]["costpush"]))
        ef = sum(w[m] * member_exp[m]["energy_fx"] for m in ov.columns if np.isfinite(member_exp[m]["energy_fx"]))
        rows_fx.append(dict(w_bvar=wb, w_midas=wm, exposure_costpush=cp, exposure_energy_fx=ef))
    fx = pd.DataFrame(rows_fx).set_index(["w_bvar", "w_midas"])
    fx.to_csv(os.path.join(_OUT, "factor_exposure_by_weight.csv"))
    print("\n=== PART F  factor-group exposure by weight ===")
    print("member exposures:", {m: {k: round(v, 3) for k, v in d.items()} for m, d in member_exp.items()})
    print(fx.round(4).to_string())

    # ---- Part G: hostile DM vs equal (0.375/0.375) ----
    base = err_by_w[(0.375, 0.375)]
    print("\n=== PART G  DM vs equal-split 0.375/0.375 (stat<0 => combo better) ===")
    for (wb, wm), e in err_by_w.items():
        if (wb, wm) == (0.375, 0.375):
            continue
        s, p = dm(e, base)
        print(f"  ({wb:.3f},{wm:.3f})  DM={s:+.2f}  p={p:.3f}")
    best_full = full["rmse"].idxmin(); best_calm = rob["rmse_ex_shock"].idxmin()
    best_shock = rob["rmse_2022_23"].idxmin()
    print(f"\nbest full = {best_full}  best ex_shock = {best_calm}  best shock = {best_shock}")
    print("written: weight_sweep.csv, weight_sweep_robustness.csv, live_forecast_sensitivity.csv,"
          " error_correlation.csv, factor_exposure_by_weight.csv")


if __name__ == "__main__":
    main()
