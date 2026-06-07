#!/usr/bin/env python3
"""
Forward factor-addition sweep.

Ranks all live candidate factors by pre-start SHAP importance, then runs
all operational models for each prefix: {top-1}, {top-1,top-2}, ..., {all}.

Output: logs/sweep_factors.csv
Columns: k, factor_added, shap_score, model, rmse, n, beats_ar1

Usage:
    FRED_API_KEY=<key> .venv/bin/python -W ignore code/sweep_factors.py \
        --start 2015 --end 2024 --train-from 1992 2>&1 | tee logs/sweep_factors.log
"""

import os
import sys
import time
import warnings
import argparse
import shutil

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

import factors as F
import uk_model_zoo as Z
import main as NC


def _shap_rank_all(df, candidates, target):
    """Return Series of mean |SHAP| for all candidates, sorted descending."""
    sub = df[candidates + [target]].dropna()
    if len(sub) < 30:
        return pd.Series(1.0, index=candidates)  # insufficient data: rank arbitrarily
    X = sub[candidates]
    y = sub[target]
    m = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=4,
                      min_child_samples=30, reg_alpha=2.0, reg_lambda=2.0,
                      random_state=42, verbose=-1)
    m.fit(X, y)
    sv = shap.TreeExplainer(m).shap_values(X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    return imp.sort_values(ascending=False)


def main():
    ap = argparse.ArgumentParser(description="Forward factor-addition sweep")
    ap.add_argument("--start",      type=int, default=2015)
    ap.add_argument("--end",        type=int, default=2024)
    ap.add_argument("--train-from", type=int, default=1992)
    ap.add_argument("--target",     default="cpi_yoy")
    ap.add_argument("--max-k",      type=int, default=None,
                    help="cap SHAP ranking at this many factors (default: all candidates)")
    ap.add_argument("--k-points",   type=int, nargs="+", default=None,
                    help="specific k values to test instead of every 1..max-k "
                         "(e.g. --k-points 1 2 5 10 32). faster sparse checkpoints.")
    ap.add_argument("--output",     default="logs/sweep_factors.csv")
    args = ap.parse_args()

    print(f"=== Forward Factor Sweep: {args.start}–{args.end} ===\n")

    # ── load factor matrix (same pipeline as main.py) ──────────────────────
    target = args.target
    df_raw, status = F.build_matrix()
    live_facs = [
        n for n, s in status.items()
        if s != "unavailable"
        and n not in (target, "cpi_yoy_long")
        and F.REGISTRY.get(n, {}).get("region") != "US"
        and n not in ("uk_rents", "uk_paye", "uk_cpih", "uk_services_cpi", "gas_eu_3m")
    ]
    if target not in df_raw.columns:
        sys.exit(f"{target} unavailable")

    df_raw = df_raw[df_raw.index.year >= args.train_from]
    df = F.apply_publication_lags(df_raw, live_facs)
    df["cpi_3m_chg"] = df[target].shift(1).diff(3)
    if "cpi_3m_chg" not in live_facs:
        live_facs = live_facs + ["cpi_3m_chg"]

    print(f"Live factors ({len(live_facs)}): {live_facs}\n")

    # ── SHAP ranking on pre-start data ──────────────────────────────────────
    candidates = [f for f in live_facs if F.REGISTRY.get(f, {}).get("candidate")]
    screen_df  = df[df.index.year < args.start].dropna(subset=[target])
    valid_cands = [c for c in candidates if c in screen_df.columns
                   and screen_df[c].notna().sum() >= 20]

    print(f"Computing SHAP ranks on {len(screen_df)} pre-{args.start} obs, "
          f"{len(valid_cands)} candidates …")
    importance = _shap_rank_all(screen_df, valid_cands, target)

    print("\nFactor ranking (mean |SHAP|, pre-start data):")
    for i, (fac, imp) in enumerate(importance.items(), 1):
        print(f"  {i:2d}. {fac:<30s} {imp:.6f}")

    ranked = importance.index.tolist()
    if args.max_k is not None:
        ranked = ranked[: args.max_k]
        print(f"  (capped at k={args.max_k})")

    # ── determine which k values to test ────────────────────────────────────
    n_factors = len(ranked)
    if args.k_points is not None:
        k_values = sorted(set(min(k, n_factors) for k in args.k_points if k >= 1))
        print(f"\nSparse checkpoints: k={k_values}  (from --k-points)")
    else:
        k_values = list(range(1, n_factors + 1))

    # ── AR(1) baseline ──────────────────────────────────────────────────────
    bt_ar1   = NC.ar1_backtest(df, target, start_year=args.start, end_year=args.end)
    ar1_rmse = float(np.sqrt(((bt_ar1["actual"] - bt_ar1["pred"]) ** 2).mean()))
    ar1_n    = len(bt_ar1)
    print(f"\nAR(1) baseline: RMSE={ar1_rmse:.4f}  n={ar1_n}\n")

    # ── import ensemble helpers from main ───────────────────────────────────
    from main import combine_static, combine_dynamic, combine_subset, \
                     error_corr_matrix, greedy_uncorrelated_subset

    # ── sweep ───────────────────────────────────────────────────────────────
    models   = Z.all_models()
    results  = []
    n_steps  = len(k_values)
    t_start  = time.time()

    def _bar(step_idx, n, width=40):
        """Return a compact progress bar string."""
        filled   = int(width * step_idx / n)
        bar      = "█" * filled + "░" * (width - filled)
        elapsed  = time.time() - t_start
        avg_step = elapsed / step_idx if step_idx else 0
        eta      = avg_step * (n - step_idx)
        eta_str  = f"{int(eta//60)}m{int(eta%60):02d}s" if eta >= 60 else f"{eta:.0f}s"
        return f"[{bar}] {step_idx}/{n}  {elapsed:.0f}s elapsed  ETA {eta_str}"

    print(f"Running {n_steps} checkpoints × {len(models)} models "
          f"(~{n_steps * len(models)} backtests total) …\n")

    for step_idx, k in enumerate(k_values, 1):
        facs         = ranked[:k]
        factor_added = ranked[k - 1]
        shap_score   = float(importance[factor_added])
        t0 = time.time()

        # ── run all base models (single pass) ───────────────────────────────
        step_rmses = {}
        bt_dict_k  = {}   # keep DataFrames for ensemble building below

        for m in models:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bt = m.backtest(df, facs, target,
                                    start_year=args.start, end_year=args.end)
                if bt.empty or bt["pred"].isna().all():
                    continue
                n_obs = int(bt["pred"].notna().sum())
                rmse  = float(np.sqrt(((bt["actual"] - bt["pred"]) ** 2).mean()))
                results.append({
                    "k": k, "factor_added": factor_added,
                    "shap_score": round(shap_score, 6),
                    "model": m.name, "rmse": round(rmse, 5),
                    "n": n_obs, "beats_ar1": rmse < ar1_rmse,
                })
                step_rmses[m.name] = rmse
                bt_dict_k[m.name]  = bt
            except Exception:
                results.append({
                    "k": k, "factor_added": factor_added,
                    "shap_score": round(shap_score, 6),
                    "model": m.name, "rmse": None,
                    "n": 0, "beats_ar1": False,
                })

        # ── ensemble models (reuse bt_dict_k, no second backtest pass) ──────
        beating = {n: bt for n, bt in bt_dict_k.items()
                   if step_rmses.get(n, float("inf")) < ar1_rmse}

        for ens_name, ens_bt in [
            ("Combined-Static",  combine_static(beating)),
            ("Combined-Dynamic", combine_dynamic(beating, window=12)),
        ]:
            if ens_bt is not None and len(ens_bt) > 0:
                rmse = float(np.sqrt(((ens_bt["actual"] - ens_bt["pred"])**2).mean()))
                results.append({
                    "k": k, "factor_added": factor_added,
                    "shap_score": round(shap_score, 6),
                    "model": ens_name, "rmse": round(rmse, 5),
                    "n": len(ens_bt), "beats_ar1": rmse < ar1_rmse,
                })
                step_rmses[ens_name] = rmse

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, corr_mat = error_corr_matrix(bt_dict_k)
            if not corr_mat.empty:
                uncorr = greedy_uncorrelated_subset(corr_mat, bt_dict_k,
                                                    rho_threshold=0.5, ar1_rmse=ar1_rmse)
                bt_abs = combine_subset(bt_dict_k, uncorr)
                if bt_abs is not None and len(bt_abs) > 0:
                    rmse = float(np.sqrt(((bt_abs["actual"] - bt_abs["pred"])**2).mean()))
                    results.append({
                        "k": k, "factor_added": factor_added,
                        "shap_score": round(shap_score, 6),
                        "model": "Combined-Absolute", "rmse": round(rmse, 5),
                        "n": len(bt_abs), "beats_ar1": rmse < ar1_rmse,
                    })
                    step_rmses["Combined-Absolute"] = rmse
        except Exception:
            pass

        best_model = min(step_rmses, key=step_rmses.get) if step_rmses else "—"
        best_rmse  = step_rmses.get(best_model, float("nan"))
        elapsed    = time.time() - t0
        bar_str    = _bar(step_idx, n_steps)
        beat       = "✓" if (step_rmses and best_rmse < ar1_rmse) else "✗"
        print(
            f"{bar_str}\n"
            f"  k={k:2d} +{factor_added:<26s} "
            f"best={best_rmse:.4f} ({best_model}) {beat}  ({elapsed:.1f}s)\n",
            flush=True,
        )

    # ── save ────────────────────────────────────────────────────────────────
    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nSaved {len(out)} rows → {args.output}")

    # ── summary: optimal k per model ────────────────────────────────────────
    print("\n── Optimal factor count per model ──")
    for mname in sorted(out["model"].unique()):
        m_df = out[(out["model"] == mname) & out["rmse"].notna()]
        if m_df.empty:
            continue
        best = m_df.loc[m_df["rmse"].idxmin()]
        marker = "✓" if best["beats_ar1"] else "✗"
        print(
            f"  {mname:<25s}  opt k={int(best['k']):2d}  "
            f"RMSE={best['rmse']:.4f}  {marker}"
        )

    # ── summary: Combined-Static at each k ──────────────────────────────────
    cs = out[out["model"].str.startswith("Combined-Static")].dropna(subset=["rmse"])
    if not cs.empty:
        print("\n── Combined-Static RMSE vs k ──")
        for _, row in cs.iterrows():
            marker = "✓" if row["beats_ar1"] else "✗"
            print(f"  k={int(row['k']):2d}  {row['factor_added']:<28s}  "
                  f"RMSE={row['rmse']:.4f}  {marker}")


if __name__ == "__main__":
    main()
