"""
Plot AutoARIMA RESIDUALS (not CPI): r_t = CPI_t - AutoARIMA_forecast_t.
Black = actual residual (what factor models must predict).
Coloured = each model's predicted residual.
Zero-line = baseline ("predict r=0" = AutoARIMA alone).

Reconstructed from data/nowcast_cpi_backtest.csv where, per date:
  AutoARIMA row:  actual = true CPI, pred = aa_f
  model row:      actual = true CPI, pred = aa_f + resid_pred
  => actual_resid = AutoARIMA_actual - AutoARIMA_pred
     model_resid  = model_pred - AutoARIMA_pred
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA  = os.path.join(_ROOT, "data"); _PLOTS = os.path.join(_ROOT, "plots")

bt = pd.read_csv(os.path.join(_DATA, "nowcast_cpi_backtest.csv"), parse_dates=["date"])
aa = bt[bt["model"] == "AutoARIMA"].set_index("date").sort_index()
aa_f = aa["pred"]                              # AutoARIMA forecast = aa_f
actual_resid = (aa["actual"] - aa["pred"])     # true residual

RMSE = {"BVAR":0.459,"DFM":0.613,"UCM":0.462,"HMM":0.457,"TVP":0.475,"HuberNet":0.468}
# Colour-blind safe: distinguish by LINESTYLE + MARKER, not colour. All dark.
STYLE = {
    "HMM":      dict(ls="-",            marker="o", me=(0, 9)),
    "BVAR":     dict(ls="--",           marker="s", me=(3, 9)),
    "UCM":      dict(ls="-.",           marker="^", me=(6, 9)),
    "HuberNet": dict(ls=":",            marker="v", me=(1, 9)),
    "TVP":      dict(ls=(0,(3,1,1,1)),  marker="D", me=(4, 9)),
    "DFM":      dict(ls=(0,(5,1)),      marker="x", me=(7, 9)),
}

fig, ax = plt.subplots(figsize=(12, 6))
ax.axhline(0, color="0.4", ls="--", lw=1.3, label="baseline: predict r=0 (AutoARIMA, 0.469)")
ax.plot(actual_resid.index, actual_resid.values, color="black", ls="-", lw=2.8,
        label="actual residual  r = CPI − AutoARIMA", zorder=10)
for nm in ["HMM","BVAR","UCM","HuberNet","TVP","DFM"]:
    sub = bt[bt["model"] == nm].set_index("date").sort_index()
    if len(sub):
        resid_pred = sub["pred"] - aa_f.reindex(sub.index)   # model's predicted residual
        s = STYLE[nm]
        ax.plot(resid_pred.index, resid_pred.values, color="0.25", lw=1.1, alpha=0.9,
                ls=s["ls"], marker=s["marker"], markersize=4.5, markevery=s["me"],
                markerfacecolor="white", markeredgecolor="0.15",
                label=f"{nm} fit ({RMSE[nm]:.3f})")
ax.set_title("AutoARIMA residual — actual vs factor-model fits (2015-2024 walk-forward)")
ax.set_ylabel("residual  (CPI YoY − AutoARIMA forecast,  %pts)")
ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.grid(alpha=0.3)
# annotate residual std + key shock band
ax.text(0.99, 0.03, f"residual std = {actual_resid.std():.3f}",
        transform=ax.transAxes, ha="right", fontsize=9, color="black")
fig.tight_layout(); p = os.path.join(_PLOTS, "ex1_aa_residuals.png")
fig.savefig(p, dpi=130); plt.close(fig)
print(f"Saved {p}")
print(f"actual residual std={actual_resid.std():.3f}, "
      f"range {actual_resid.min():.2f}..{actual_resid.max():.2f}")
