"""Core CPI inspection figures → plots/core/ (all history panels in YoY %).

Four figures, each with a one-sentence narrative caption baked into the PNG (captioning-figures):
  F1 cpi_vs_core_history      headline vs core CPI YoY + energy/food wedge + BoE 2% line
  F2 core_factor_screen_shap  SHAP importance of exogenous core candidates
  F3 core_factor_race_relrmse walk-forward rel-RMSE vs AA (pinned vs dropped/excluded)
  F4 core_overlay_validation  AA vs TVP/LGBM/combo walk-forward RMSE (the T7 gate)

Colours: Okabe-Ito (colourblind-safe by construction). Run:
  PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python code/plot_core.py
"""
import os, textwrap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "plots", "core"); os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(_ROOT, "data")

# Okabe-Ito
BLUE, ORANGE, GREEN, VERM, SKY, PURP, GREY = ("#0072B2", "#E69F00", "#009E73",
                                              "#D55E00", "#56B4E9", "#CC79A7", "#8a8a8a")
INK, MUTED = "#222222", "#6b6b6b"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10,
                     "axes.edgecolor": "#bbbbbb", "axes.linewidth": 0.8,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED})

CAPTIONS = {
    "cpi_vs_core_history":
        "UK headline vs core CPI inflation (YoY %, 1989-2026): core (ex energy, food, alcohol & "
        "tobacco) is far smoother than headline, whose bigger swings come from the energy/food wedge "
        "(shaded) — motivating a separate, more-persistent core nowcast.",
    "core_factor_screen_shap":
        "SHAP importance of exogenous candidates for the core AA residual: output PPI dominates, but "
        "overall attribution is modest — core leaves little for factors to explain.",
    "core_factor_race_relrmse":
        "Core factor race, walk-forward rel-RMSE vs AA-only (<1 = helps): the five pinned drivers "
        "(services PMI, output-PPI, inflation swap, corporate profits, GBP-EER) each beat AA; "
        "composite PMI is dropped as redundant with services, monthly-GDP's data is dead, CPIH is "
        "circular.",
    "core_overlay_validation":
        "Core overlay validation (purged/embargoed walk-forward, 2015-2024, n=120): the shipped "
        "5-factor AA+0.25*TVP+0.25*LGBM overlay cuts AA-only RMSE by ~7% (rel-RMSE 0.934, DM p=0.006), "
        "with TVP the workhorse.",
}


def _style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, color="#e6e6e6", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    cap = CAPTIONS[name]
    chars = int(fig.get_size_inches()[0] * 13)
    fig.text(0.5, -0.02, textwrap.fill(cap, width=chars), ha="center", va="top",
             fontsize=8, style="italic", color="0.35")
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight")
    plt.close(fig); print("wrote", name)


def _read_yoy(fname):
    s = pd.read_csv(os.path.join(DATA, fname))
    s.columns = [c.lower() for c in s.columns]
    return pd.Series(s["value"].values, index=pd.to_datetime(s["date"])).sort_index()


def fig_history():
    head = _read_yoy("cpi_yoy.csv"); core = _read_yoy("uk_core_cpi.csv")
    idx = head.index.union(core.index)
    head = head.reindex(idx); core = core.reindex(idx)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    _style(ax)
    ax.axhline(2.0, color=GREY, ls="--", lw=1, zorder=1)
    ax.text(idx[5], 2.12, "BoE 2% target", color=MUTED, fontsize=8)
    ax.fill_between(idx, core, head, where=head >= core, color=ORANGE, alpha=0.12, zorder=1,
                    label="headline − core (energy/food) wedge")
    ax.fill_between(idx, core, head, where=head < core, color=BLUE, alpha=0.10, zorder=1)
    ax.plot(idx, head, color=BLUE, lw=2, zorder=3, label="Headline CPI YoY")
    ax.plot(idx, core, color=ORANGE, lw=2, zorder=4, label="Core CPI YoY")
    # direct labels at the right edge
    ax.text(idx[-1], head.iloc[-1] + 0.25, f"Headline {head.iloc[-1]:.1f}%", color=BLUE,
            fontsize=9, fontweight="bold", ha="right")
    ax.text(idx[-1], core.iloc[-1] - 0.45, f"Core {core.iloc[-1]:.1f}%", color=ORANGE,
            fontsize=9, fontweight="bold", ha="right")
    ax.set_ylabel("inflation, YoY %"); ax.set_xlabel("")
    ax.set_title("UK headline vs core CPI inflation", fontsize=12, color=INK, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    save(fig, "cpi_vs_core_history")


def _race():
    return pd.read_csv(os.path.join(DATA, "new_factors", "core_factor_race.csv")).set_index("factor")


def fig_screen():
    r = _race()
    exo = r[r["group"] == "exogenous"].dropna(subset=["shap_mean_abs"]).sort_values("shap_mean_abs")
    fig, ax = plt.subplots(figsize=(8, 4.2)); _style(ax)
    ax.barh(exo.index, exo["shap_mean_abs"], color=BLUE, height=0.6, zorder=3)
    for y, v in zip(range(len(exo)), exo["shap_mean_abs"]):
        ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel("mean |SHAP| on core AA residual")
    ax.set_title("Core factor screen — exogenous candidate importance", fontsize=12, loc="left")
    save(fig, "core_factor_screen_shap")


def fig_race():
    r = _race().sort_values("univ_rel_rmse", ascending=False)
    pinned = {"uk_pmi_services", "uk_ppi_output", "uk_infl_swap_1y", "uk_corp_profits", "gbp_eer"}
    colors = []
    for f in r.index:
        if f in pinned: colors.append(GREEN)
        elif f in ("uk_monthly_gdp", "uk_pmi_composite"): colors.append(VERM)  # beat AA but dropped
        elif r.loc[f, "group"] == "cpi_family": colors.append(PURP)  # circular, excluded
        else: colors.append("#c9c9c9")
    fig, ax = plt.subplots(figsize=(8.5, 5)); _style(ax)
    ax.axvline(1.0, color=INK, lw=1.2, zorder=2)
    ax.text(1.002, len(r) - 0.4, "AA-only", fontsize=8, color=INK)
    ax.barh(r.index, r["univ_rel_rmse"], color=colors, height=0.62, zorder=3)
    for y, v in zip(range(len(r)), r["univ_rel_rmse"]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0.9, max(1.05, r["univ_rel_rmse"].max() + 0.02))
    ax.set_xlabel("univariate rel-RMSE vs AA-only  (<1 = factor helps core)")
    ax.set_title("Core factor race", fontsize=12, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, VERM, PURP, "#c9c9c9")]
    ax.legend(handles, ["pinned", "beats AA but dropped (redundant / dead data)",
                        "excluded — circular (CPI-family)", "does not beat AA"],
              loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8)
    save(fig, "core_factor_race_relrmse")


def fig_validation():
    # from code/new_factors/core_validate.py (5-factor overlay, purged+embargoed WF 2015-2024, n=120)
    names = ["AA-only", "TVP member", "LGBM member", "AA + overlay\n(shipped)"]
    rmse = [0.3213, 0.3031, 0.3227, 0.3000]
    colors = [GREY, SKY, SKY, GREEN]
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); _style(ax)
    bars = ax.bar(names, rmse, color=colors, width=0.6, zorder=3)
    ax.axhline(rmse[0], color=GREY, ls="--", lw=1, zorder=2)
    for b, v in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0.29, 0.327)
    ax.set_ylabel("walk-forward RMSE (pp)")
    ax.set_title("Core overlay validation — beats AA-only (rel 0.934, DM p=0.006)", fontsize=11.5, loc="left")
    save(fig, "core_overlay_validation")


def main():
    fig_history(); fig_screen(); fig_race(); fig_validation()
    emitted = {p[:-4] for p in os.listdir(OUT) if p.endswith(".png")}
    assert emitted == set(CAPTIONS), (emitted - set(CAPTIONS), set(CAPTIONS) - emitted)
    print("all 4 core figures written to plots/core/ with captions")


if __name__ == "__main__":
    main()
