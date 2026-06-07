"""
Regenerate nowcast_history_3.png from nowcast_cpi_backtest.csv + metrics.
Usage: python plot_nowcast_history.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA  = os.path.join(_ROOT, "data")
_PLOTS = os.path.join(_ROOT, "plots")
os.makedirs(_PLOTS, exist_ok=True)

BACKTEST_CSV = os.path.join(_DATA,  "nowcast_cpi_backtest.csv")
METRICS_CSV  = os.path.join(_DATA,  "nowcast_cpi_metrics.csv")
NOWCAST_CSV  = os.path.join(_DATA,  "nowcast_cpi_nowcast.csv")
OUT          = os.path.join(_PLOTS, "nowcast_history_3.png")

def main():
    if not os.path.exists(BACKTEST_CSV):
        sys.exit(f"Missing {BACKTEST_CSV}  — run nowcast_cpi.py first")

    bt = pd.read_csv(BACKTEST_CSV, parse_dates=["date"]).set_index("date")
    met = pd.read_csv(METRICS_CSV).set_index("model")

    # ── Pick models to display (top performers + key baselines) ─────────────
    top_models = [
        "Combined-Static",
        "Combined-Dynamic",
        "MedianElasticNet",
        "ElasticNet",
        "RegimeEns",
        "TVP",
        "UCM",
    ]
    show = [m for m in top_models if m in bt["model"].unique()]

    # ── Colour palette ───────────────────────────────────────────────────────
    palette = {
        "Combined-Static":   "#1f77b4",
        "Combined-Dynamic":  "#ff7f0e",
        "MedianElasticNet":  "#2ca02c",
        "ElasticNet":        "#9467bd",
        "RegimeEns":         "#8c564b",
        "TVP":               "#e377c2",
        "UCM":               "#7f7f7f",
        "AR(1)":             "#bcbd22",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1]})
    ax, ax_err = axes

    # ── Actual CPI (from AR(1) backtest rows which are monthly) ─────────────
    ar1_bt = bt[bt["model"] == "AR(1)"]
    actual = ar1_bt["actual"].sort_index()
    ax.plot(actual.index, actual.values, color="black", lw=2.5,
            label="Actual CPI YoY", zorder=10)

    # ── Model predictions ────────────────────────────────────────────────────
    for mname in show:
        sub = bt[bt["model"] == mname].sort_index()
        if len(sub) == 0:
            continue
        rmse = met.loc[mname, "rmse"] if mname in met.index else np.nan
        label = f"{mname}  (RMSE={rmse:.3f})" if not np.isnan(rmse) else mname
        ax.plot(sub.index, sub["pred"].values,
                color=palette.get(mname, None),
                lw=1.5, alpha=0.85, label=label)

    # AR(1) dashed baseline
    ar1_rmse = met.loc["AR(1)", "rmse"] if "AR(1)" in met.index else np.nan
    ax.plot(ar1_bt.index, ar1_bt["pred"].values,
            color=palette["AR(1)"], lw=1, ls="--", alpha=0.6,
            label=f"AR(1)  (RMSE={ar1_rmse:.3f})" if not np.isnan(ar1_rmse) else "AR(1)")

    # ── Nowcast marker ───────────────────────────────────────────────────────
    if os.path.exists(NOWCAST_CSV):
        nc = pd.read_csv(NOWCAST_CSV)
        nc_date = pd.to_datetime(nc["date"].iloc[0])
        nc_vals = nc.set_index("model")["nowcast"]
        for mname in show:
            if mname in nc_vals.index:
                ax.scatter([nc_date], [nc_vals[mname]],
                           color=palette.get(mname, "grey"),
                           s=60, zorder=11, marker="*")

    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylabel("CPI YoY (%)", fontsize=11)
    ax.set_title("UK CPI YoY Nowcast — Backtest 2015–2024 (blind test 2025+)\n"
                 f"30 factors: 26 pub_lag=0 · 4 pub_lag≥1  ·  21 models",
                 fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
    ax.grid(True, alpha=0.3)

    # Shade high-inflation regime (2022-2023)
    ax.axvspan(pd.Timestamp("2021-10-31"), pd.Timestamp("2023-06-30"),
               alpha=0.06, color="red", label="_nolegend_")

    # ── Error panel ──────────────────────────────────────────────────────────
    best = "Combined-Static"
    if best in bt["model"].unique():
        sub = bt[bt["model"] == best].sort_index()
        err = sub["pred"] - sub["actual"]
        ax_err.bar(err.index, err.values,
                   color=np.where(err > 0, "#ff7f7f", "#7fbfff"),
                   width=40, alpha=0.8)
        ax_err.axhline(0, color="black", lw=0.8)
        ax_err.set_ylabel("Error (pp)\nCombined-Static", fontsize=9)
        ax_err.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
