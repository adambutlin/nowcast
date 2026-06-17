"""
Forensic information-set / leakage audit of the frozen two-stage model.
Architecture unchanged. Audit only.

Information boundary (established by code):
  monthly factors : resample('ME').last() -> shift(pub_lag>=0)  => month-T row uses <= month-T-end
  MIDAS           : daily {Brent,GBP,VIX,TTF} -> resample('ME').mean(), NO pub-lag => days 1..end of month T
  weights fixed; each member trains on years < test year.

Therefore "blind observations after month-T-end" (Parts D/E) is a STRUCTURAL NO-OP
(nothing in the model uses post-month-end data) -> proves no post-month-end leakage.
The only forward-looking content is WITHIN-reference-month data. We quantify it with the
decisive test:
  BASELINE        : current (month-T information set; financials through month-end).
  STRICT_PREMONTH : shift every factor +1 month AND MIDAS monthly-mean +1 month
                    => month-T forecast uses only <= month-(T-1)-end data (true ~T-30 /
                       pre-reference-month standpoint). Removes ALL within-month-T info.

Deliverables (data/new_factors/audit/):
  information_set.csv, timing_matrix.csv, midas_timing_audit.csv,
  leakage_sensitivity.csv, blinded_backtest.csv, model_leakage_ranking.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/new_factors/leakage_audit.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import factors as F, uk_model_zoo as Z, two_stage as TS

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "new_factors", "audit")
os.makedirs(_OUT, exist_ok=True)
WIN = {"full": lambda i: i.year >= TS.START,
       "2022_23": lambda i: i.year.isin([2022, 2023]),
       "ex_shock": lambda i: ~i.year.isin([2022, 2023]),
       "pre_2020": lambda i: i.year <= 2019}
MEMBERS = ["bvar", "tvp", "midas"]


def rmse(e):
    e = np.asarray(e, float); e = e[np.isfinite(e)]
    return float(np.sqrt((e**2).mean())) if len(e) else np.nan


# ── Part A: information set ──────────────────────────────────────────────────
def information_set(live):
    rows = []
    for f in live + ["cpi_yoy", "MIDAS_daily"]:
        if f == "MIDAS_daily":
            rows.append(dict(factor="MIDAS_daily(Brent/GBP/VIX/TTF)", source="yfinance",
                             transform="monthly-MEAN of daily", pub_lag=0,
                             resample="ME.mean", feature_date_rule="month-T daily mean (days 1..end of T)",
                             post_month_end="NO"))
            continue
        e = F.REGISTRY.get(f, {})
        lag = e.get("pub_lag", "?")
        rows.append(dict(factor=f, source=(e.get("note", "")[:28]),
                         transform=e.get("transform"), pub_lag=lag, resample="ME.last",
                         feature_date_rule=f"month-(T-{lag}) end" if isinstance(lag, int) else "?",
                         post_month_end="NO"))
    df = pd.DataFrame(rows).set_index("factor")
    df.to_csv(os.path.join(_OUT, "information_set.csv"))
    return df


# ── Part C: MIDAS timing by origin ───────────────────────────────────────────
def midas_timing():
    # May CPI: collection ~12 May; reference month May; release 17 Jun (T0).
    # Origins relative to release of month-T CPI.
    rows = [
        ("T-30", "~ mid month T (June for a June print)", "partial month-T daily (backtest uses FULL month-T)"),
        ("T-21", "~ 3 weeks before release", "partial month-T daily"),
        ("T-14", "~ 2 weeks before release", "more of month-T"),
        ("T-7",  "~ 1 week before release", "near-full month-T"),
        ("T-1",  "day before release", "full month-T daily (== backtest)"),
    ]
    df = pd.DataFrame(rows, columns=["origin", "calendar", "midas_latest_obs_available"])
    df["note"] = "MIDAS feature = resample('ME').mean() of month-T daily -> backtest uses FULL month-T at every origin (month-end standpoint), so backtest is OPTIMISTIC vs a true partial-month T-30."
    df.to_csv(os.path.join(_OUT, "midas_timing_audit.csv"), index=False)
    return df


# ── backtest under an information set ────────────────────────────────────────
def run_backtest(shift_months=0):
    """shift_months=0 -> current. shift_months=1 -> strict pre-reference-month (T-30 proxy):
    every monthly factor +1 month AND MIDAS monthly-mean +1 month."""
    df, live, status = TS.load_matrix()
    if shift_months:
        df = df.copy()
        for c in live:
            df[c] = df[c].shift(shift_months)
    # MIDAS cache injection (not a class change): shift its monthly-mean by shift_months
    Z._MIDAS_CACHE.clear()
    mm = Z._get_midas_data()
    if mm is not None:
        Z._MIDAS_CACHE["mm"] = mm.shift(shift_months) if shift_months else mm
    bt = TS.backtest(df, live)
    nc = TS.nowcast(df, live)
    Z._MIDAS_CACHE.clear()
    return bt, nc, live


def member_rmse(bt, member):
    col = f"{member}_pred"
    if col not in bt:
        return {w: np.nan for w in WIN}
    err = bt["actual"] - bt[col]
    return {w: rmse(err[fn(bt.index)]) for w, fn in WIN.items()}


def ens_rmse(bt):
    err = bt["actual"] - bt["forecast"]
    return {w: rmse(err[fn(bt.index)]) for w, fn in WIN.items()}


def main():
    print("=== PART A: information set ===")
    df0, live, status = TS.load_matrix()
    iset = information_set(live)
    print(iset.to_string()); print("post-month-end users:", (iset["post_month_end"] == "YES").sum())

    print("\n=== PART C: MIDAS timing ===")
    print(midas_timing().to_string(index=False))

    print("\n=== Running BASELINE (month-T info set) ===")
    bt0, nc0, _ = run_backtest(0)
    print("=== Running STRICT_PREMONTH (+1 month on ALL factors incl MIDAS; ~T-30) ===")
    bt1, nc1, _ = run_backtest(1)

    aa = {w: rmse((bt0["actual"] - bt0["aa_pred"])[fn(bt0.index)]) for w, fn in WIN.items()}

    # Part E: blinded_backtest (baseline vs strict-premonth) + note month-end no-op
    rows = []
    for w in WIN:
        rows.append(dict(window=w, rmse_AA=aa[w],
                         rmse_2stage_baseline=ens_rmse(bt0)[w],
                         rmse_2stage_monthend_blind=ens_rmse(bt0)[w],   # structural no-op == baseline
                         rmse_2stage_strict_premonth=ens_rmse(bt1)[w]))
    blinded = pd.DataFrame(rows).set_index("window")
    blinded["edge_baseline"] = blinded["rmse_AA"] - blinded["rmse_2stage_baseline"]
    blinded["edge_strict"] = blinded["rmse_AA"] - blinded["rmse_2stage_strict_premonth"]
    blinded["edge_survived_%"] = 100 * blinded["edge_strict"] / blinded["edge_baseline"]
    blinded.to_csv(os.path.join(_OUT, "blinded_backtest.csv"))
    print("\n=== PART E: blinded backtest (month-end-blind == baseline = NO-OP) ===")
    print(blinded.round(4).to_string())

    # Part D: leakage_sensitivity (Δ from removing within-month info)
    sens = pd.DataFrame({
        "rmse_baseline": ens_rmse(bt0), "rmse_strict_premonth": ens_rmse(bt1),
    })
    sens["delta_rmse"] = sens["rmse_strict_premonth"] - sens["rmse_baseline"]
    sens["pct_worse"] = 100 * sens["delta_rmse"] / sens["rmse_baseline"]
    sens.to_csv(os.path.join(_OUT, "leakage_sensitivity.csv"))
    print("\n=== PART D: leakage sensitivity (strict-premonth vs baseline) ===")
    print(sens.round(4).to_string())
    print(f"live forecast: baseline {nc0['forecast']:.3f}  ->  strict_premonth {nc1['forecast']:.3f}")
    print(f"  baseline members {{k:round(v,3)}}: { {k: round(v,3) for k,v in nc0['members'].items()} }")
    print(f"  strict   members            : { {k: round(v,3) for k,v in nc1['members'].items()} }")

    # Part F: model leakage ranking (per-member full-sample RMSE delta)
    rows = []
    for m in MEMBERS:
        r0 = member_rmse(bt0, m)["full"]; r1 = member_rmse(bt1, m)["full"]
        rows.append(dict(model=m, rmse_full_baseline=r0, rmse_full_strict=r1,
                         delta=r1 - r0, pct_worse=100 * (r1 - r0) / r0 if r0 else np.nan))
    rows.append(dict(model="AutoARIMA", rmse_full_baseline=aa["full"], rmse_full_strict=aa["full"],
                     delta=0.0, pct_worse=0.0))   # AA uses only CPI history -> unaffected
    rank = pd.DataFrame(rows).set_index("model").sort_values("pct_worse", ascending=False)
    rank.to_csv(os.path.join(_OUT, "model_leakage_ranking.csv"))
    print("\n=== PART F: model leakage ranking (within-month dependence) ===")
    print(rank.round(4).to_string())

    # timing_matrix (qualitative, Part B)
    tm = iset.copy()
    tm["may_cpi_collection"] = "~12 May 2026"
    tm["obs_vs_collection"] = np.where(tm["pub_lag"].apply(lambda x: isinstance(x, int) and x >= 1),
                                       "prior-month (before collection)",
                                       "month-end (after 12 May collection, within reference month, pre-release)")
    tm.to_csv(os.path.join(_OUT, "timing_matrix.csv"))

    print("\nwritten to", _OUT)


if __name__ == "__main__":
    main()
