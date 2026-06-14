"""
Hostile residual + regime/factor sweep — 2026-06-14.

Exercise 1: Residual benchmark sweep.
  Three residuals: AR(1), AR(2), AutoARIMA.
  Six models: BVAR, DFM, UCM, HMM, TVP, HuberNet.
  Four windows: full (2015-2024), ex-shock (ex 2022/23), ex-covid (ex 2020/21), pre-2020 (2015-2019).
  Metrics: RMSE, MAE, dir_acc, OOS corr, rel_rmse vs benchmark, N.

Exercise 2: Regime + Factor combination sweep (9 pairs).
  Regime: HMM, UCM, TVP.
  Factor: BVAR, DFM, HuberNet.
  Combination: equal-weight average of predictions on the AutoARIMA residual.
  Same four robustness windows.

Outputs:
  data/residual_sweep/benchmark_model_table.csv
  data/residual_sweep/robustness_table.csv
  data/regime_factor_sweep/combination_table.csv
  data/regime_factor_sweep/final_ranking.csv

Run:  FRED_API_KEY=... .venv/bin/python code/sweep_residual_regime.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import factors as F
import uk_model_zoo as Z

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA  = os.path.join(_ROOT, "data")
_OUT1  = os.path.join(_DATA, "residual_sweep")
_OUT2  = os.path.join(_DATA, "regime_factor_sweep")
os.makedirs(_OUT1, exist_ok=True); os.makedirs(_OUT2, exist_ok=True)

TARGET = "cpi_yoy"
PINNED = ["oil_brent", "gas_eu", "uk_quarterly_gdp", "imf_all_commodity",
          "global_supply_chain_pressure"]
REG    = ["mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]
PINNED = PINNED + REG
AA_START = 2001
FULL_START, FULL_END = 2015, 2024
TRAIN_FROM = 1997

# ── Robustness windows ──────────────────────────────────────────────────────
# Window = list of (year_start, year_end) inclusive ranges to INCLUDE
WINDOWS = {
    "full":     [(2015, 2024)],
    "ex_shock": [(2015, 2021), (2024, 2024)],   # exclude 2022/23
    "ex_covid": [(2015, 2019), (2022, 2024)],   # exclude 2020/21
    "pre_2020": [(2015, 2019)],
}

def window_mask(index, ranges):
    mask = pd.Series(False, index=index)
    for s, e in ranges:
        mask |= ((index.year >= s) & (index.year <= e))
    return mask

# ── Fetch data ──────────────────────────────────────────────────────────────
print("Fetching data …")
df_raw, status = F.build_matrix(names=PINNED + [TARGET])
live = [n for n in PINNED if status.get(n) != "unavailable"]
for n in PINNED:
    print(f"  {n:35} {status.get(n)}")
if status.get(TARGET) == "unavailable":
    sys.exit("target unavailable")

df_raw = df_raw[df_raw.index.year >= TRAIN_FROM]
df = F.apply_publication_lags(df_raw, live)
for _f in REG:
    if _f in df.columns:
        df[_f] = df[_f].fillna(0)

# ── AR(p) backtest helper ───────────────────────────────────────────────────
def ar_backtest(series, p, start, end):
    s = series.dropna(); rows = []
    for yr in range(start, end + 1):
        tr = s[s.index.year < yr]; te = s[s.index.year == yr]
        if len(tr) < 30 or len(te) == 0:
            continue
        y = tr.values; n = len(y)
        X = np.column_stack([np.ones(n - p)] + [y[p - l:n - l] for l in range(1, p + 1)])
        beta = np.linalg.lstsq(X, y[p:], rcond=None)[0]
        hist = list(y)
        for idx in te.index:
            xr = np.array([1.0] + [hist[-l] for l in range(1, p + 1)])
            rows.append((idx, float(s.loc[idx]), float(xr @ beta)))
            hist.append(float(s.loc[idx]))
    return pd.DataFrame(rows, columns=["date", "actual", "pred"]).set_index("date")

# ── AutoARIMA baseline ──────────────────────────────────────────────────────
print("\nAutoARIMA walk-forward …")
aa = Z.AutoARIMA()
aa_bt = aa.backtest(df, [], TARGET, start_year=AA_START, end_year=FULL_END)
aa_pred = aa_bt["pred"]
cpi_actual = aa_bt["actual"]

# ── Three residuals ──────────────────────────────────────────────────────────
print("Building three residuals …")
ar1_bt = ar_backtest(df[TARGET], 1, AA_START, FULL_END)
ar2_bt = ar_backtest(df[TARGET], 2, AA_START, FULL_END)

RESID_DEFS = {}
for name, bench_bt in [("AR1", ar1_bt), ("AR2", ar2_bt), ("AA", aa_bt)]:
    actual_s = bench_bt["actual"]
    pred_s   = bench_bt["pred"]
    resid_s  = actual_s - pred_s
    col = f"resid_{name}"
    df[col] = resid_s.reindex(df.index)
    RESID_DEFS[name] = dict(col=col, bench_bt=bench_bt, actual=actual_s, pred=pred_s)

# ── Metrics helper ──────────────────────────────────────────────────────────
def metrics(bt, bench_rmse, label=""):
    """bt: DataFrame[actual, pred]. Returns dict of metrics."""
    if bt is None or len(bt) == 0:
        return dict(model=label, rmse=np.nan, mae=np.nan, dir_acc=np.nan,
                    oos_corr=np.nan, rel_rmse=np.nan, n=0)
    e = bt["actual"] - bt["pred"]
    rmse = float(np.sqrt((e**2).mean()))
    mae  = float(e.abs().mean())
    try:
        oos_corr = float(bt["actual"].corr(bt["pred"]))
    except Exception:
        oos_corr = np.nan
    actual_chg = bt["actual"].diff()
    pred_chg   = bt["pred"] - bt["actual"].shift(1)
    dir_mask   = (np.sign(actual_chg) == np.sign(pred_chg)).dropna()
    dir_acc    = float(dir_mask.mean() * 100) if len(dir_mask) else np.nan
    rel        = rmse / bench_rmse if (bench_rmse and np.isfinite(bench_rmse) and bench_rmse > 0) else np.nan
    return dict(model=label, rmse=rmse, mae=mae, dir_acc=dir_acc,
                oos_corr=oos_corr, rel_rmse=rel, n=len(bt))

def slice_bt(bt, ranges):
    if bt is None or len(bt) == 0:
        return bt
    mask = window_mask(bt.index, ranges)
    return bt[mask]

# ── Exercise 1 — run all models on all three residuals ─────────────────────
print("\n" + "="*70)
print("EXERCISE 1 — Residual benchmark sweep")
print("="*70)

RESID_MODELS = [Z.BVAR(), Z.DFM(), Z.UCM(), Z.HMM(), Z.TVP(), Z.HuberNet()]
MODEL_NAMES  = [m.name for m in RESID_MODELS]

# store all backtest results: ex1_bt[resid_name][model_name] = bt DataFrame
ex1_bt = {r: {} for r in RESID_DEFS}

for res_name, rdef in RESID_DEFS.items():
    col = rdef["col"]
    print(f"\n  --- Residual: {res_name} ---")
    for m in RESID_MODELS:
        print(f"    {m.name} …", end=" ", flush=True)
        try:
            bt = m.backtest(df, live, col, start_year=FULL_START, end_year=FULL_END)
            bt = bt if (bt is not None and len(bt)) else None
        except Exception as e:
            bt = None
            print(f"ERROR: {str(e)[:50]}", end=" ")
        ex1_bt[res_name][m.name] = bt
        n = len(bt) if bt is not None else 0
        print(f"n={n}")

# ── Build Ex1 tables ────────────────────────────────────────────────────────
bench_rows = []
robust_rows = []

for res_name, rdef in RESID_DEFS.items():
    bench_bt = rdef["bench_bt"]
    for win_name, win_ranges in WINDOWS.items():
        # Benchmark RMSE in this window (predicting residual = 0)
        bench_slice = slice_bt(bench_bt, win_ranges)
        bench_rmse  = float(np.sqrt(((bench_slice["actual"] - bench_slice["pred"])**2).mean())) \
                      if bench_slice is not None and len(bench_slice) else np.nan

        # Benchmark row (predict residual = 0 → same as benchmark's own CPI RMSE)
        if win_name == "full":
            bench_rows.append(dict(
                residual=res_name, model=f"{res_name}[bench]",
                **{k: v for k, v in metrics(bench_slice, bench_rmse, f"{res_name}[bench]").items() if k != "model"}
            ))

        for mname in MODEL_NAMES:
            bt = ex1_bt[res_name].get(mname)
            bt_slice = slice_bt(bt, win_ranges)
            m = metrics(bt_slice, bench_rmse, mname)
            robust_rows.append(dict(
                residual=res_name, window=win_name, **{k: v for k, v in m.items()}
            ))

robust_df = pd.DataFrame(robust_rows)

# Full-sample summary table: benchmark × model
full_df = robust_df[robust_df["window"] == "full"].copy()
bench_extra = pd.DataFrame(bench_rows)
# Add benchmark "baseline" rows to full table
all_full = pd.concat([bench_extra.assign(window="full", dir_acc=np.nan, oos_corr=np.nan)
                                  .rename(columns={}),
                       full_df], ignore_index=True)

print("\n" + "="*70)
print("EX1 FULL-SAMPLE — Benchmark × Model RMSE table")
print("="*70)
pivot = full_df.pivot_table(index="model", columns="residual", values="rmse").round(4)
print(pivot.to_string())

print("\n" + "="*70)
print("EX1 FULL-SAMPLE — rel_rmse (< 1.0 = beats benchmark)")
print("="*70)
pivot_rel = full_df.pivot_table(index="model", columns="residual", values="rel_rmse").round(4)
print(pivot_rel.to_string())

print("\n" + "="*70)
print("EX1 ROBUSTNESS — RMSE by window (AutoARIMA residual)")
print("="*70)
aa_rob = robust_df[robust_df["residual"] == "AA"].pivot_table(
    index="model", columns="window", values="rmse").round(4)
print(aa_rob.to_string())

print("\n" + "="*70)
print("EX1 ROBUSTNESS — rel_rmse by window (AutoARIMA residual)")
print("="*70)
aa_rob_rel = robust_df[robust_df["residual"] == "AA"].pivot_table(
    index="model", columns="window", values="rel_rmse").round(4)
print(aa_rob_rel.to_string())

# ── Exercise 2 — Regime + Factor combination sweep ─────────────────────────
print("\n" + "="*70)
print("EXERCISE 2 — Regime + Factor combination (equal-weight average, AA residual)")
print("="*70)

# Combination method: equal-weight average of regime + factor predictions
# Both trained on AutoARIMA residual. Tests whether regime + factor signals
# are complementary (orthogonal) or redundant (correlated).
# Hostile test: combination must beat BOTH standalones to claim genuine value.

REGIME_MODELS  = ["HMM", "UCM", "TVP"]
FACTOR_MODELS  = ["BVAR", "DFM", "HuberNet"]
aa_bt_dict = ex1_bt["AA"]   # model_name -> bt DataFrame (AA residual)

combo_rows = []

for regime_m in REGIME_MODELS:
    for factor_m in FACTOR_MODELS:
        r_bt = aa_bt_dict.get(regime_m)
        f_bt = aa_bt_dict.get(factor_m)
        if r_bt is None or f_bt is None or len(r_bt) == 0 or len(f_bt) == 0:
            print(f"  {regime_m}+{factor_m}: SKIPPED (missing backtest)")
            continue
        # Align on common dates
        common = r_bt.index.intersection(f_bt.index)
        if len(common) < 12:
            print(f"  {regime_m}+{factor_m}: SKIPPED (only {len(common)} common obs)")
            continue
        combo_pred   = 0.5 * r_bt.loc[common, "pred"] + 0.5 * f_bt.loc[common, "pred"]
        combo_actual = r_bt.loc[common, "actual"]   # same residual = same actual
        combo_df     = pd.DataFrame({"actual": combo_actual, "pred": combo_pred})

        label = f"{regime_m}+{factor_m}"
        print(f"  {label}: n={len(combo_df)}")

        # Benchmark RMSE = predict residual = 0 (= AutoARIMA CPI RMSE)
        aa_test = aa_bt[(aa_bt.index.year >= FULL_START) & (aa_bt.index.year <= FULL_END)]
        bench_full_rmse = float(np.sqrt(((aa_test["actual"] - aa_test["pred"])**2).mean()))

        for win_name, win_ranges in WINDOWS.items():
            c_slice = slice_bt(combo_df, win_ranges)
            r_slice = slice_bt(r_bt[["actual", "pred"]], win_ranges)
            f_slice = slice_bt(f_bt[["actual", "pred"]], win_ranges)

            # Window-specific benchmark RMSE
            bench_slice = slice_bt(aa_bt[["actual", "pred"]], win_ranges)
            if bench_slice is not None and len(bench_slice):
                win_bench_rmse = float(np.sqrt(((bench_slice["actual"] - bench_slice["pred"])**2).mean()))
            else:
                win_bench_rmse = np.nan

            m_combo  = metrics(c_slice, win_bench_rmse, label)
            m_regime = metrics(r_slice, win_bench_rmse, regime_m)
            m_factor = metrics(f_slice, win_bench_rmse, factor_m)

            # Does combo beat BOTH standalones?
            beats_regime = m_combo["rmse"] < m_regime["rmse"] if (np.isfinite(m_combo["rmse"]) and np.isfinite(m_regime["rmse"])) else False
            beats_factor = m_combo["rmse"] < m_factor["rmse"] if (np.isfinite(m_combo["rmse"]) and np.isfinite(m_factor["rmse"])) else False
            beats_AA     = m_combo["rmse"] < win_bench_rmse if (np.isfinite(m_combo["rmse"]) and np.isfinite(win_bench_rmse)) else False

            combo_rows.append(dict(
                combo=label, regime_model=regime_m, factor_model=factor_m,
                window=win_name,
                combo_rmse=m_combo["rmse"], combo_rel=m_combo["rel_rmse"],
                combo_mae=m_combo["mae"], combo_dir=m_combo["dir_acc"],
                combo_corr=m_combo["oos_corr"],
                regime_rmse=m_regime["rmse"], factor_rmse=m_factor["rmse"],
                bench_rmse=win_bench_rmse,
                beats_regime=beats_regime, beats_factor=beats_factor,
                beats_AA=beats_AA, n=m_combo["n"]
            ))

combo_df_out = pd.DataFrame(combo_rows)

print("\n" + "="*70)
print("EX2 FULL-SAMPLE — Combo RMSE vs standalones (AutoARIMA residual)")
print("="*70)
full_combo = combo_df_out[combo_df_out["window"] == "full"].copy()
cols_show = ["combo", "combo_rmse", "regime_rmse", "factor_rmse", "bench_rmse",
             "beats_regime", "beats_factor", "beats_AA"]
print(full_combo[cols_show].sort_values("combo_rmse").round(4).to_string(index=False))

print("\n" + "="*70)
print("EX2 ROBUSTNESS — combo rel_rmse by window")
print("="*70)
rob_pivot = combo_df_out.pivot_table(index="combo", columns="window", values="combo_rmse").round(4)
print(rob_pivot.to_string())

print("\n" + "="*70)
print("EX2 ROBUSTNESS — beats_AA (True = combo < AutoARIMA alone)")
print("="*70)
beats_pivot = combo_df_out.pivot_table(index="combo", columns="window", values="beats_AA").round(0)
print(beats_pivot.to_string())

# ── Final ranking ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL RANKING")
print("="*70)

# Collect: all standalone models on AA residual, all combos, all benchmarks
rank_rows = []

# Benchmarks
for res_name, rdef in RESID_DEFS.items():
    for win_name, win_ranges in WINDOWS.items():
        b_slice = slice_bt(rdef["bench_bt"], win_ranges)
        if b_slice is not None and len(b_slice):
            b_rmse = float(np.sqrt(((b_slice["actual"] - b_slice["pred"])**2).mean()))
            rank_rows.append(dict(type="benchmark", name=f"{res_name}[bench]",
                                  window=win_name, rmse=b_rmse, n=len(b_slice)))

# Standalone models on AA residual (full sample)
for mname in MODEL_NAMES:
    for win_name, win_ranges in WINDOWS.items():
        bt = ex1_bt["AA"].get(mname)
        bt_s = slice_bt(bt, win_ranges)
        if bt_s is not None and len(bt_s):
            r = float(np.sqrt(((bt_s["actual"] - bt_s["pred"])**2).mean()))
            rank_rows.append(dict(type="standalone_AA", name=mname, window=win_name,
                                  rmse=r, n=len(bt_s)))

# Combos
for row in combo_rows:
    rank_rows.append(dict(type="combo", name=row["combo"], window=row["window"],
                          rmse=row["combo_rmse"], n=row["n"]))

rank_df = pd.DataFrame(rank_rows)

# Full-sample leaderboard
full_rank = rank_df[rank_df["window"] == "full"].sort_values("rmse")
print("\nFull-sample leaderboard (all models, AA residual benchmarks):")
print(full_rank[["type", "name", "rmse", "n"]].round(4).to_string(index=False))

# Robustness-adjusted: count windows where model beats AA[bench]
aa_bench_by_window = {}
for win_name, win_ranges in WINDOWS.items():
    b_slice = slice_bt(aa_bt, win_ranges)
    if b_slice is not None and len(b_slice):
        aa_bench_by_window[win_name] = float(np.sqrt(((b_slice["actual"] - b_slice["pred"])**2).mean()))

rob_adj = {}
for name, grp in rank_df[rank_df["type"].isin(["standalone_AA", "combo"])].groupby("name"):
    wins = 0
    for _, row in grp.iterrows():
        bench = aa_bench_by_window.get(row["window"], np.nan)
        if np.isfinite(row["rmse"]) and np.isfinite(bench) and row["rmse"] < bench:
            wins += 1
    rob_adj[name] = wins

rob_series = pd.Series(rob_adj, name="windows_beating_AA").sort_values(ascending=False)
print("\nRobustness-adjusted score (windows where model < AA benchmark, max=4):")
print(rob_series.to_string())

# ── Save outputs ─────────────────────────────────────────────────────────────
robust_df.to_csv(os.path.join(_OUT1, "robustness_table.csv"), index=False)
full_df.to_csv(os.path.join(_OUT1, "benchmark_model_table.csv"), index=False)
combo_df_out.to_csv(os.path.join(_OUT2, "combination_table.csv"), index=False)
rank_df.to_csv(os.path.join(_OUT2, "final_ranking.csv"), index=False)
print(f"\nSaved to {_OUT1}/ and {_OUT2}/")

# ── Hostile review summary ──────────────────────────────────────────────────
print("\n" + "="*70)
print("HOSTILE REVIEW — Concerns by model")
print("="*70)

# For each model, check: does it beat AA in all 4 windows?
# If only beats in full or shock windows → fragile

for mname in MODEL_NAMES:
    win_results = {}
    for win_name, win_ranges in WINDOWS.items():
        bt = ex1_bt["AA"].get(mname)
        bt_s = slice_bt(bt, win_ranges)
        bench = aa_bench_by_window.get(win_name, np.nan)
        if bt_s is not None and len(bt_s) and np.isfinite(bench):
            r = float(np.sqrt(((bt_s["actual"] - bt_s["pred"])**2).mean()))
            win_results[win_name] = (r < bench, round(r, 4), round(bench, 4))
        else:
            win_results[win_name] = (None, np.nan, np.nan)

    flags = []
    full_beats = win_results.get("full", (None,))[0]
    shock_beats = win_results.get("ex_shock", (None,))[0]
    covid_beats = win_results.get("ex_covid", (None,))[0]
    pre_beats   = win_results.get("pre_2020", (None,))[0]

    if full_beats and not (covid_beats and pre_beats):
        flags.append("SHOCK-ONLY — only beats baseline in shock/full period")
    if full_beats and not shock_beats:
        flags.append("SHOCK-DEPENDENT — performance driven by 2022/23 spike")
    if not full_beats:
        flags.append("FAILS — does not beat AutoARIMA baseline on full sample")

    win_counts = sum(1 for v in win_results.values() if v[0])
    print(f"\n  {mname}: beats_AA in {win_counts}/4 windows")
    for wn, (beats, rmse, bench) in win_results.items():
        status = "BEAT" if beats else ("FAIL" if beats is False else "N/A")
        print(f"    {wn:10} rmse={rmse:.4f}  bench={bench:.4f}  [{status}]")
    for flag in flags:
        print(f"    *** {flag}")
    if not flags and win_counts == 4:
        print(f"    PASSES all robustness windows — strong candidate")
    elif not flags and win_counts >= 3:
        print(f"    PASSES {win_counts}/4 windows — conditional pass")

print("\n" + "="*70)
print("DONE.")
