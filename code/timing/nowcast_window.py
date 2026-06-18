"""
NOWCAST-WINDOW backtest (branch `timing`).
Question: standing AFTER month-end T but BEFORE the release R (T+15..T+21), does the
production model add value as new information arrives? i.e. is it better at R-1 than at T?

Information-set logic (as-of date d = T + h days):
  A factor's REFERENCE-MONTH-T value enters only if its publication date <= d.
  Publication offsets (days after T) for the live set:
    brent,gas,gbpusd,vix,mpc,ofgem            : 0   (financial month-end / scheduled)
    uk_ppi_input (pub_lag1), gdp (pub_lag2)    : <0  (use pre-T vintage -> always in)
    MIDAS daily {brent,gbp,vix,ttf}            : 0   (reference-month daily complete at T)
    imf_all_commodity                          : +7  (IMF primary commodity, early next month)
    deep_sea_freight (US BLS PPI)              : +13 (mid next month)
  At horizon h, any factor with offset>h uses its PRIOR-month value (one extra lag),
  faithfully simulating "what was published by day d".

AutoARIMA uses CPI history (month T-1 vintage) -> CONSTANT across the whole window (no new
CPI until release). MIDAS factors frozen at T. Only imf(+7)/freight(+13) can move the model.

Horizons: T(+0), T+5, T+10, T+15, R-1(+17 median).
Out: data/timing/{release_calendar,forecast_evolution,incremental_value,attribution,consensus_compare}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/nowcast_window.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE)
sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
from scipy.stats import t as tdist
import factors as F, uk_model_zoo as Z, two_stage as TS

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "timing")
os.makedirs(_OUT, exist_ok=True)

HORIZONS = {"T": 0, "T+5": 5, "T+10": 10, "T+15": 15, "R-1": 17}
OFFSET = {"imf_all_commodity": 7, "deep_sea_freight": 13}   # others <=0
WIN = {"full": lambda i: i.year >= TS.START,
       "ex_shock": lambda i: ~i.year.isin([2022, 2023]),
       "2022_23": lambda i: i.year.isin([2022, 2023])}


def dm(e1, e2):
    d = np.asarray(e1, float)**2 - np.asarray(e2, float)**2
    d = d[np.isfinite(d)]; n = len(d)
    if n < 8 or np.allclose(d, 0):
        return np.nan, np.nan
    db = d.mean(); dd = d - db; v = (dd @ dd) / n; L = max(1, int(round(n**(1/3))))
    for k in range(1, L + 1):
        v += 2 * (1 - k / (L + 1)) * ((dd[k:] @ dd[:-k]) / n)
    if v <= 0:
        return np.nan, np.nan
    s = db / np.sqrt(v / n)
    return float(s), float(2 * (1 - tdist.cdf(abs(s), df=n - 1)))


def rmse(e): e = np.asarray(e, float); e = e[np.isfinite(e)]; return float(np.sqrt((e**2).mean())) if len(e) else np.nan
def mae(e):  e = np.asarray(e, float); e = e[np.isfinite(e)]; return float(np.abs(e).mean()) if len(e) else np.nan


def release_calendar():
    p = pd.read_csv(os.path.join(_ROOT, "data", "rates_event_panel.csv"))
    d = p[["ref_month", "release_date"]].copy()
    d["ref_month"] = pd.to_datetime(d["ref_month"]) + pd.offsets.MonthEnd(0)
    d["release_date"] = pd.to_datetime(d["release_date"])
    d = d.dropna().drop_duplicates("ref_month").sort_values("ref_month").set_index("ref_month")
    d["R_minus_T"] = (d["release_date"] - d.index).dt.days
    d.to_csv(os.path.join(_OUT, "release_calendar.csv"))
    return d


def backtest_at_horizon(h):
    """Production backtest with as-of factor availability at T+h days."""
    df, live, status = TS.load_matrix()
    df = df.copy()
    for f, off in OFFSET.items():
        if f in df.columns and off > h:
            df[f] = df[f].shift(1)          # not yet published -> prior-month value
    Z._MIDAS_CACHE.clear()                  # MIDAS daily frozen at T (no horizon effect)
    bt = TS.backtest(df, live)
    return bt


def main():
    print("=== PART A: release calendar ===")
    cal = release_calendar()
    print(f"n={len(cal)}  R-T min {cal['R_minus_T'].min()} max {cal['R_minus_T'].max()} "
          f"median {cal['R_minus_T'].median()}  (release never < T+{cal['R_minus_T'].min()})")

    # run backtest at each horizon
    bts = {hz: backtest_at_horizon(d) for hz, d in HORIZONS.items()}
    base = bts["T"]
    aa_err = base["actual"] - base["aa_pred"]      # AA constant across window
    idx = base.index

    # PART B: forecast evolution (per-release forecasts at each horizon)
    evo = pd.DataFrame({"actual": base["actual"], "aa_pred": base["aa_pred"]})
    for hz in HORIZONS:
        evo[f"fc_{hz}"] = bts[hz]["forecast"].reindex(idx)
    evo.to_csv(os.path.join(_OUT, "forecast_evolution.csv"))
    # how much does the forecast MOVE across the window?
    move = (evo["fc_R-1"] - evo["fc_T"]).abs()
    print("\n=== PART B: forecast movement T -> R-1 ===")
    print(f"  mean |fc(R-1)-fc(T)| = {move.mean():.4f}pp  max = {move.max():.4f}pp  "
          f"months with any move = {(move>1e-6).sum()}/{len(move)}")
    print("  identical at T, T+5? ", bool(np.allclose(evo['fc_T'], evo['fc_T+5'], atol=1e-9)),
          " | T+15==R-1? ", bool(np.allclose(evo['fc_T+15'], evo['fc_R-1'], atol=1e-9)))

    # PART C: incremental value vs AA at each horizon
    rows = []
    for hz in HORIZONS:
        err = bts[hz]["actual"] - bts[hz]["forecast"]
        for w, fn in WIN.items():
            m = fn(idx)
            rec = dict(horizon=hz, window=w, n=int(m.sum()),
                       rmse_AA=rmse(aa_err[m]), rmse_model=rmse(err[m]),
                       rel_rmse=rmse(err[m]) / rmse(aa_err[m]),
                       mae_model=mae(err[m]),
                       hit=float((np.sign(bts[hz]["forecast"].diff()[m]) ==
                                  np.sign(bts[hz]["actual"].diff()[m])).mean()))
            rows.append(rec)
    inc = pd.DataFrame(rows).set_index(["horizon", "window"])
    inc.to_csv(os.path.join(_OUT, "incremental_value.csv"))
    print("\n=== PART C: model vs AA by horizon ===")
    print(inc.xs("full", level="window")[["rmse_AA", "rmse_model", "rel_rmse", "hit"]].round(4).to_string())

    # PART D: where does the edge arrive? (full sample RMSE by horizon + DM vs T)
    rows = []
    for hz in HORIZONS:
        err = (bts[hz]["actual"] - bts[hz]["forecast"])
        s, p = dm(base["actual"] - base["forecast"], err)   # T vs hz (improvement?)
        rows.append(dict(horizon=hz, rmse_full=rmse(err),
                         d_rmse_vs_T=rmse(err) - rmse(base["actual"] - base["forecast"]),
                         DM_T_vs_hz=s, p=p))
    att = pd.DataFrame(rows).set_index("horizon")
    att.to_csv(os.path.join(_OUT, "attribution.csv"))
    print("\n=== PART D: edge arrival (Δrmse vs T; imf in @T+10, freight in @T+15) ===")
    print(att.round(4).to_string())
    print("  NOTE: MIDAS & TVP financial factors frozen at T -> only imf(+7)/freight(+13) move the model.")

    # PART E: consensus comparison at R-1
    cons = pd.read_csv(os.path.join(_ROOT, "data", "consensus_cpi.csv"),
                       parse_dates=["date"]).set_index("date")["consensus_cpi"]
    r1 = bts["R-1"]
    cmp = pd.DataFrame({"actual": r1["actual"], "model_R1": r1["forecast"],
                        "aa": r1["aa_pred"], "consensus": cons.reindex(idx)}).dropna()
    cmp["err_model"] = (cmp["actual"] - cmp["model_R1"]).abs()
    cmp["err_aa"] = (cmp["actual"] - cmp["aa"]).abs()
    cmp["err_cons"] = (cmp["actual"] - cmp["consensus"]).abs()
    cmp.to_csv(os.path.join(_OUT, "consensus_compare.csv"))
    print("\n=== PART E: at R-1 vs consensus (n=%d, overlap) ===" % len(cmp))
    print(f"  RMSE  model {rmse(cmp['actual']-cmp['model_R1']):.4f}  AA {rmse(cmp['actual']-cmp['aa']):.4f}  "
          f"consensus {rmse(cmp['actual']-cmp['consensus']):.4f}")
    print(f"  MAE   model {cmp['err_model'].mean():.4f}  AA {cmp['err_aa'].mean():.4f}  "
          f"consensus {cmp['err_cons'].mean():.4f}")
    print(f"  model beats consensus in {100*(cmp['err_model']<cmp['err_cons']).mean():.0f}% of months")
    sM, pM = dm(cmp["actual"]-cmp["consensus"], cmp["actual"]-cmp["model_R1"])
    print(f"  DM consensus vs model: stat={sM:+.2f} p={pM:.3f} (>0 => model better)")

    print("\nwritten to", _OUT)


if __name__ == "__main__":
    main()
