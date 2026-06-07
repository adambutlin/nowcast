"""
nowcast_plot.py — UK CPI YoY: 2026 YTD nowcasts + 6-month forward projections.

Plot:
  - 2024–2025 actual CPI (history context)
  - 2026 YTD 1-step-ahead nowcasts from UCM, TVP, Superstar, AR(1)
  - 6-month ahead forward forecasts with ±1σ and ±2σ error bands
  - Vertical separator at last actual observation

Error bands:
  - 1-step: backtest RMSE (2015–2025)
  - h-step ahead: σ₁ × √h  (random walk scaling)
  - UCM forward: also shows native statsmodels forecast SE

No US factors: gas_hh, us_ism_pmi, us_ppi_all excluded (region='US').

Usage: FRED_API_KEY=<key> .venv/bin/python nowcast_plot.py [--out nowcast_2026.png]
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLOTS = os.path.join(_ROOT, "plots")
os.makedirs(_PLOTS, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

import factors as F
import uk_model_zoo as Z
from compare_uk import ar1_backtest


# ─────────────────────────────────────────────────────────────────────────────
# UCM FULL-DATA FIT + FORWARD FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def ucm_full_fit(df, factors, target):
    """Fit UCM on all available data. Returns (res, last_date, mu_f, sd_f, last_f)."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    d = df[factors + [target]].dropna()
    mu_f = d[factors].mean()
    sd_f = d[factors].std().replace(0, 1)
    exog = (d[factors] - mu_f) / sd_f
    # Use PeriodIndex so statsmodels correctly infers monthly frequency
    # (plain DatetimeIndex with month-end dates is often unrecognized)
    period_idx = d.index.to_period("M")
    endog = pd.Series(d[target].values, index=period_idx, name=target)
    exog_arr = exog.values
    res = UnobservedComponents(
        endog, level="local linear trend", exog=exog_arr
    ).fit(maxiter=300, disp=False)
    last_f = d[factors].iloc[-1]
    last_date = d.index[-1]
    return res, last_date, mu_f, sd_f, last_f


def ucm_forward(res, last_date, last_f, mu_f, sd_f, n_months=6):
    """
    Project UCM n_months ahead (flat factors).
    Returns (dates, point, lower_95, upper_95, se).
    """
    last_f_norm = ((last_f - mu_f) / sd_f).values
    future_exog = np.tile(last_f_norm, (n_months, 1))
    fc = res.get_forecast(steps=n_months, exog=future_exog)
    point = np.asarray(fc.predicted_mean).flatten()
    ci = fc.conf_int(alpha=0.05)
    ci_arr = np.asarray(ci)
    lo95 = ci_arr[:, 0]
    hi95 = ci_arr[:, 1]
    se = np.asarray(fc.se_mean).flatten()
    dates = pd.date_range(last_date, periods=n_months + 1, freq="ME")[1:]
    return dates, point, lo95, hi95, se


# ─────────────────────────────────────────────────────────────────────────────
# TVP FULL-DATA FIT + FORWARD FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def tvp_full_fit(df, factors, target, delta=1e-3):
    """Run TVP Kalman on all data. Returns (last_beta, last_x_unnorm_for_proj, R, data_index)."""
    d = df[factors + [target]].dropna()
    mu_f = d[factors].mean()
    sd_f = d[factors].std().replace(0, 1)
    fz = (d[factors] - mu_f) / sd_f
    dd = d.copy(); dd[factors] = fz
    m = Z.TVP(delta=delta)
    X = m._design(dd, factors, target)
    y = dd[target].values
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    R = np.nanvar(np.diff(d[target].values)) + 1e-6
    Q = np.eye(X.shape[1]) * R * delta
    _, betas = m._kalman(X[ok], y[ok], R, Q)
    last_beta = betas[-1]
    last_cpi = float(d[target].iloc[-1])
    last_f_norm = ((d[factors].iloc[-1] - mu_f) / sd_f).values
    last_x = np.concatenate([[1.0, last_cpi], last_f_norm])
    return last_beta, last_x, R, d.index


def tvp_forward(last_beta, last_x, R, delta, n_factors, n_months=6):
    """
    Project TVP n_months ahead.
    Feed each predicted value back as the AR(1) term for the next step.
    σ_h = sqrt(R + h * R * delta * n_params)  (random walk in betas).
    Returns (point, se_array).
    """
    preds = []
    x = last_x.copy()
    n_params = len(last_beta)
    for h in range(1, n_months + 1):
        yhat = x @ last_beta
        preds.append(yhat)
        x[1] = yhat   # update AR(1) term for next step; factors stay flat
    se = np.array([
        np.sqrt(R * (1 + h * delta * n_params)) for h in range(1, n_months + 1)
    ])
    return np.array(preds), se


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_PLOTS, "nowcast_2026.png"))
    ap.add_argument("--train-from", type=int, default=1992)
    ap.add_argument("--n-ahead", type=int, default=6)
    args = ap.parse_args()

    # ── Load factors (exclude US) ──────────────────────────────────────────
    print("Loading factor matrix …")
    df_raw, status = F.build_matrix()
    live_facs = [n for n, s in status.items()
                 if s != "unavailable" and n != "cpi_yoy"
                 and F.REGISTRY.get(n, {}).get("region") != "US"]
    target = "cpi_yoy"
    if target not in df_raw.columns:
        sys.exit("cpi_yoy unavailable.")

    df_raw = df_raw[df_raw.index.year >= args.train_from]
    df = F.apply_publication_lags(df_raw, live_facs)
    print(f"  Factors ({len(live_facs)}): {live_facs}")

    actual_cpi = df[target].dropna()
    last_actual = actual_cpi.index[-1]
    print(f"  Latest CPI observation: {last_actual.date()}")

    # ── Backtest RMSE for uncertainty scaling ──────────────────────────────
    print("Computing backtest RMSEs (2015–2025) …")
    backtest_rmse = {}
    models_to_plot = [m for m in Z.all_models() if m.name in ("UCM", "TVP")]
    for m in models_to_plot:
        bt = m.backtest(df, live_facs, target, start_year=2015)
        if len(bt):
            backtest_rmse[m.name] = float(
                np.sqrt(((bt["actual"] - bt["pred"])**2).mean()))
    ar1_bt_full = ar1_backtest(df, target, start_year=2015)
    backtest_rmse["AR(1)"] = float(
        np.sqrt(((ar1_bt_full["actual"] - ar1_bt_full["pred"])**2).mean()))
    backtest_rmse["Superstar"] = np.mean([
        backtest_rmse.get("UCM", 0.15), backtest_rmse.get("TVP", 0.20)])
    print(f"  RMSE — UCM: {backtest_rmse.get('UCM','?'):.3f}  "
          f"TVP: {backtest_rmse.get('TVP','?'):.3f}  "
          f"AR(1): {backtest_rmse.get('AR(1)','?'):.3f}")

    # ── 2026 YTD 1-step-ahead nowcasts ─────────────────────────────────────
    print("Generating 2026 YTD nowcasts …")
    nowcasts = {}
    ucm_m = Z.UCM()
    tvp_m = Z.TVP()
    for m in [ucm_m, tvp_m]:
        bt = m.backtest(df, live_facs, target, start_year=2026)
        nowcasts[m.name] = bt if len(bt) else pd.DataFrame()
        if len(bt):
            print(f"  {m.name}: {len(bt)} months  "
                  f"RMSE={np.sqrt(((bt['actual']-bt['pred'])**2).mean()):.3f}")

    ar1_2026 = ar1_backtest(df, target, start_year=2026)
    nowcasts["AR(1)"] = ar1_2026

    # Superstar = equal-weight UCM + TVP
    ucm_bt = nowcasts.get("UCM", pd.DataFrame())
    tvp_bt = nowcasts.get("TVP", pd.DataFrame())
    if len(ucm_bt) and len(tvp_bt):
        common = ucm_bt.index.intersection(tvp_bt.index)
        if len(common):
            pred_s = (ucm_bt.loc[common, "pred"] + tvp_bt.loc[common, "pred"]) / 2
            nowcasts["Superstar"] = pd.DataFrame({
                "actual": ucm_bt.loc[common, "actual"],
                "pred": pred_s})

    # ── Forward 6-month forecasts ──────────────────────────────────────────
    print(f"Generating {args.n_ahead}-month ahead forecasts from {last_actual.date()} …")

    # UCM forward
    print("  Fitting UCM on full data …", end="", flush=True)
    ucm_res, ucm_last_date, ucm_mu_f, ucm_sd_f, ucm_last_f = ucm_full_fit(df, live_facs, target)
    ucm_fwd_dates, ucm_fwd_pt, ucm_fwd_lo, ucm_fwd_hi, ucm_fwd_se = ucm_forward(
        ucm_res, ucm_last_date, ucm_last_f, ucm_mu_f, ucm_sd_f, n_months=args.n_ahead)
    print(f" done. Last fitted date: {ucm_last_date.date()}")
    print(f"  UCM forward: {ucm_fwd_pt}")

    # TVP forward
    print("  Fitting TVP on full data …", end="", flush=True)
    tvp_last_beta, tvp_last_x, tvp_R, tvp_idx = tvp_full_fit(
        df, live_facs, target)
    tvp_fwd_pt, tvp_fwd_se = tvp_forward(
        tvp_last_beta, tvp_last_x, tvp_R, delta=1e-3,
        n_factors=len(live_facs), n_months=args.n_ahead)
    tvp_fwd_dates = pd.date_range(last_actual, periods=args.n_ahead + 1, freq="ME")[1:]
    print(f" done. TVP forward: {tvp_fwd_pt}")

    # Superstar forward = average of UCM + TVP
    ss_fwd_pt = (ucm_fwd_pt + tvp_fwd_pt) / 2
    ss_fwd_se = (ucm_fwd_se + tvp_fwd_se) / 2

    # AR(1) forward: mu + rho*(last_y - mu)  (no mean-reversion for simplicity)
    y_all = actual_cpi.values
    mu_ar = float(y_all.mean())
    rho_ar = float(np.corrcoef(y_all[:-1], y_all[1:])[0, 1])
    last_y = float(actual_cpi.iloc[-1])
    ar1_fwd = np.array([mu_ar + rho_ar**h * (last_y - mu_ar)
                        for h in range(1, args.n_ahead + 1)])
    ar1_fwd_se = backtest_rmse["AR(1)"] * np.sqrt(np.arange(1, args.n_ahead + 1))
    ar1_fwd_dates = pd.date_range(last_actual, periods=args.n_ahead + 1, freq="ME")[1:]

    # ── PLOT ───────────────────────────────────────────────────────────────
    print("Plotting …")
    fig, ax = plt.subplots(figsize=(15, 7))

    COLORS = {
        "UCM": "#2563EB",       # blue
        "TVP": "#D97706",       # amber
        "Superstar": "#059669", # green
        "AR(1)": "#9CA3AF",     # gray
    }

    # history: Jan 2024 onwards
    hist_start = pd.Timestamp("2024-01-01")
    hist = actual_cpi[actual_cpi.index >= hist_start]
    ax.plot(hist.index, hist.values, color="black", linewidth=2.5,
            label="Actual CPI YoY", zorder=6, solid_capstyle="round")

    # ── 2026 YTD nowcasts ─────────────────────────────────────────────────
    plot_order = ["AR(1)", "TVP", "UCM", "Superstar"]
    for name in plot_order:
        bt = nowcasts.get(name)
        if bt is None or len(bt) == 0:
            continue
        c = COLORS[name]
        sig = backtest_rmse.get(name, 0.2)
        lbl = {"UCM": "UCM (Tier 1)", "TVP": "TVP (Tier 1)",
               "Superstar": "Combined-Superstar", "AR(1)": "AR(1) baseline"}.get(name, name)
        ax.plot(bt.index, bt["pred"].values, color=c, linewidth=2.0,
                label=f"{lbl}  [σ={sig:.3f}]", zorder=5, alpha=0.9)
        ax.fill_between(bt.index,
                        bt["pred"].values - sig,
                        bt["pred"].values + sig,
                        color=c, alpha=0.12, zorder=3)
        ax.fill_between(bt.index,
                        bt["pred"].values - 2 * sig,
                        bt["pred"].values + 2 * sig,
                        color=c, alpha=0.06, zorder=2)

    # ── Forward forecasts ─────────────────────────────────────────────────
    fwd_data = {
        "UCM":      (ucm_fwd_dates, ucm_fwd_pt, ucm_fwd_se),
        "TVP":      (tvp_fwd_dates, tvp_fwd_pt, tvp_fwd_se),
        "Superstar": (ucm_fwd_dates, ss_fwd_pt, ss_fwd_se),
        "AR(1)":    (ar1_fwd_dates, ar1_fwd, ar1_fwd_se),
    }
    for name in plot_order:
        if name not in fwd_data:
            continue
        dates, pt, se = fwd_data[name]
        c = COLORS[name]
        # connect to last nowcast if available
        bt = nowcasts.get(name)
        if bt is not None and len(bt):
            connect_x = [bt.index[-1], dates[0]]
            connect_y = [bt["pred"].iloc[-1], pt[0]]
        else:
            connect_x = [last_actual, dates[0]]
            connect_y = [float(actual_cpi.iloc[-1]), pt[0]]
        ax.plot(connect_x, connect_y, color=c, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.plot(dates, pt, color=c, linewidth=2.0, linestyle="--",
                alpha=0.9, zorder=5)
        ax.fill_between(dates, pt - se, pt + se,
                        color=c, alpha=0.12, zorder=3)
        ax.fill_between(dates, pt - 2 * se, pt + 2 * se,
                        color=c, alpha=0.06, zorder=2)
        ax.scatter(dates, pt, color=c, s=30, zorder=6, alpha=0.8)

    # UCM native 95% CI overlay (statsmodels model uncertainty, wider than σ₁)
    ax.fill_between(ucm_fwd_dates, ucm_fwd_lo, ucm_fwd_hi,
                    color=COLORS["UCM"], alpha=0.08, zorder=1,
                    label="UCM 95% CI (model)")

    # ── BoE target line ───────────────────────────────────────────────────
    ax.axhline(2.0, color="#DC2626", linewidth=0.9, linestyle="-.", alpha=0.5,
               label="BoE 2% target", zorder=2)

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_label_text("CPI YoY (%)", fontsize=10)
    ax.set_xlim(hist_start - pd.DateOffset(days=10),
                fwd_data["UCM"][0][-1] + pd.DateOffset(days=20))
    # Floor y-axis at 0 to avoid negative CPI distorting the chart
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(max(ylo, 0.0), yhi)

    # ── Dividers (after ylim set so text y-position is correct) ──────────
    ax.axvline(last_actual, color="#6B7280", linewidth=1.2, linestyle=":",
               alpha=0.7, zorder=7)
    ax.axvspan(last_actual, fwd_data["UCM"][0][-1] + pd.DateOffset(days=10),
               alpha=0.03, color="#6B7280", zorder=1)
    _, yhi2 = ax.get_ylim()
    ax.text(last_actual + pd.DateOffset(days=4), yhi2 * 0.97,
            "← YTD  Forecast →", fontsize=8.5, color="#6B7280",
            va="top", ha="left")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.grid(True, axis="x", alpha=0.12, linewidth=0.5)
    ax.set_axisbelow(True)

    # ── Legend and title ─────────────────────────────────────────────────
    legend_extras = [
        Patch(facecolor="none", edgecolor="none",
              label=f"Bands: ±1σ / ±2σ from backtest RMSE; h-step scaled ×√h"),
        Patch(facecolor="none", edgecolor="none",
              label=f"Factors: {', '.join(live_facs[:5])}{'…' if len(live_facs)>5 else ''}"),
        Patch(facecolor="none", edgecolor="none",
              label=f"No US variables (gas_hh, ISM, US PPI excluded)"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_extras, labels + [p.get_label() for p in legend_extras],
              fontsize=8.5, loc="upper left", framealpha=0.85, ncol=1)

    months_ahead = args.n_ahead
    fwd_end = fwd_data["UCM"][0][-1].strftime("%b %Y")
    ax.set_title(
        f"UK CPI YoY — Nowcasts 2026 YTD + {months_ahead}-month forward (to {fwd_end})\n"
        f"Training: 1992–{df[df[target].notna()].index[-1].year}  |  "
        f"Tier 1: UCM + TVP  |  Combined-Superstar = equal-weight",
        fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {args.out}")

    # ── Print table ────────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Month':<12} {'Actual':>8} {'UCM':>8} {'TVP':>8} {'Superstar':>10} {'AR(1)':>8}")
    print(f"{'─'*70}")
    # YTD
    all_dates = sorted(set(
        list(ucm_bt.index) + list(tvp_bt.index if len(tvp_bt) else []) + list(ar1_2026.index)))
    for d_idx in all_dates:
        actual = float(actual_cpi.loc[d_idx]) if d_idx in actual_cpi.index else np.nan
        ucm_v  = float(ucm_bt.loc[d_idx, "pred"]) if len(ucm_bt) and d_idx in ucm_bt.index else np.nan
        tvp_v  = float(tvp_bt.loc[d_idx, "pred"]) if len(tvp_bt) and d_idx in tvp_bt.index else np.nan
        ss_v   = (ucm_v + tvp_v) / 2 if np.isfinite(ucm_v) and np.isfinite(tvp_v) else np.nan
        ar1_v  = float(ar1_2026.loc[d_idx, "pred"]) if d_idx in ar1_2026.index else np.nan
        print(f"{d_idx.strftime('%b %Y'):<12} "
              f"{actual:>8.2f} {ucm_v:>8.3f} {tvp_v:>8.3f} {ss_v:>10.3f} {ar1_v:>8.3f}")
    # Forward
    print(f"{'─'*70}  ← forward forecasts (no actual yet)")
    for i in range(args.n_ahead):
        d_str = ucm_fwd_dates[i].strftime("%b %Y")
        ucm_v  = ucm_fwd_pt[i]
        tvp_v  = tvp_fwd_pt[i]
        ss_v   = ss_fwd_pt[i]
        ar1_v  = ar1_fwd[i]
        ucm_se = ucm_fwd_se[i]
        tvp_se = tvp_fwd_se[i]
        print(f"{d_str:<12} {'—':>8} {ucm_v:>7.3f}±{ucm_se:.2f} "
              f"{tvp_v:>7.3f}±{tvp_se:.2f} {ss_v:>9.3f} {ar1_v:>8.3f}")


if __name__ == "__main__":
    main()
