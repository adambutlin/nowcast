"""
TVP weight sweep for Stage-2 = {BVAR, TVP, MIDAS}.

Combo overlay = w_tvp*TVP + (1-w_tvp)*mean(BVAR, MIDAS)   (renormalised over available).
Runs the walk-forward member backtest ONCE, then sweeps w_tvp cheaply over the saved
member predictions. Reports RMSE by window + live May-2026 nowcast per weight.

Out: data/new_factors/weight_sweep.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/new_factors/weight_sweep.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import two_stage as TS

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "new_factors")
GRID = [0.0, 0.10, 0.1667, 0.20, 0.25, 0.3333, 0.45, 0.55, 0.65, 0.80, 1.0]
WINDOWS = {"full": lambda i: i.year >= TS.START,
           "2022_23": lambda i: i.year.isin([2022, 2023]),
           "ex_shock": lambda i: ~i.year.isin([2022, 2023]),
           "pre_2020": lambda i: i.year <= 2019}


def weighted_overlay(row_tvp, row_rest_mean, w_tvp):
    """row_rest_mean = mean(bvar,midas). Renorm if a side is NaN."""
    have_tvp = np.isfinite(row_tvp); have_rest = np.isfinite(row_rest_mean)
    if have_tvp and have_rest:
        return w_tvp * row_tvp + (1 - w_tvp) * row_rest_mean
    if have_tvp:
        return row_tvp
    if have_rest:
        return row_rest_mean
    return np.nan


def rmse(e):
    e = np.asarray(e, float); e = e[np.isfinite(e)]
    return float(np.sqrt((e**2).mean())) if len(e) else np.nan


def main():
    df, live, status = TS.load_matrix()
    print("live factors:", live)
    bt = TS.backtest(df, live)            # has bvar_pred/tvp_pred/midas_pred/actual/aa_pred
    nc = TS.nowcast(df, live)             # members overlays + aa_pred
    aa = bt["aa_pred"]; actual = bt["actual"]
    tvp = bt["tvp_pred"] - aa             # member overlays (pred - AA)
    rest = pd.concat([(bt["bvar_pred"] - aa), (bt["midas_pred"] - aa)], axis=1).mean(axis=1)

    # live overlays
    m = nc["members"]
    tvp_l = m.get("tvp", np.nan)
    rest_l = np.nanmean([m.get("bvar", np.nan), m.get("midas", np.nan)])

    rows = []
    for w in GRID:
        ov = pd.Series([weighted_overlay(t, r, w) for t, r in zip(tvp, rest)], index=bt.index)
        fc = aa + ov.fillna(0)
        err = actual - fc
        rec = {"w_tvp": round(w, 4)}
        for wn, fn in WINDOWS.items():
            msk = fn(bt.index)
            rec[f"rmse_{wn}"] = rmse(err[msk])
        # AA-relative on full
        rec["rel_full"] = rec["rmse_full"] / rmse((actual - aa))
        # live nowcast
        ov_l = weighted_overlay(tvp_l, rest_l, w)
        rec["nowcast"] = float(nc["aa_pred"] + (ov_l if np.isfinite(ov_l) else 0.0))
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("w_tvp")
    out.to_csv(os.path.join(_OUT, "weight_sweep.csv"))
    pd.options.display.width = 200
    print(f"\n=== TVP WEIGHT SWEEP (rest = (BVAR+MIDAS)/2; actual May-26 = 2.8) ===")
    print(out.round(4).to_string())
    best_full = out["rmse_full"].idxmin(); best_ex = out["rmse_ex_shock"].idxmin()
    print(f"\nbest w_tvp by full RMSE   = {best_full}  (rmse {out.loc[best_full,'rmse_full']:.4f}, rel {out.loc[best_full,'rel_full']:.4f})")
    print(f"best w_tvp by ex_shock    = {best_ex}  (rmse {out.loc[best_ex,'rmse_ex_shock']:.4f})")
    print(f"equal-weight (0.3333) full rmse = {out.loc[0.3333,'rmse_full']:.4f}, nowcast {out.loc[0.3333,'nowcast']:.3f}")
    print("written", os.path.join(_OUT, "weight_sweep.csv"))


if __name__ == "__main__":
    main()
