"""
intramonth/run.py — end-to-end intramonth nowcasting pipeline (Part I).

Runs the full system for a target and writes:
  data/intramonth/evolution_<target>.csv      forecast by origin (point, mix, regime, drivers)
  data/intramonth/weights_<target>.csv         regime+horizon model weights by origin
  data/scenarios/scenarios_<target>.csv         scenario probabilities + points by origin
  data/scenarios/scenario_table_<target>.csv    full scenario table at the sharpest origin (T-1)
  data/production/recommendation_<target>.csv   final production recommendation
  data/production/attribution_<target>.csv      model/HF attribution at a representative origin
  plots/intramonth/evolution_<target>.png       point + bands + weight evolution + regime
  plots/scenarios/scenarios_<target>.png         scenario-mass shift T-30→T-1 + decomposition

Colour-blind safe plots (linestyle + marker + greyscale + hatching only).

Run:  set -a; . ./.env; set +a
      PYTHONPATH=code .venv/bin/python -m intramonth.run [--target KEY] [--all] [--end-year 2024]
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factors as F
from intramonth import (config as C, targets as T, evolution as E, regime as R,
                        weights as W, panel as P, stack as S, attribution as A)

# colour-blind-safe styles per layer
LSTYLE = {"baseline": dict(ls="-", marker="o"), "factor": dict(ls="--", marker="s"),
          "regime_tvp": dict(ls="-.", marker="^"), "intramonth": dict(ls=":", marker="D")}
LAYER_LABEL = {"baseline": "AutoARIMA", "factor": "BVAR", "regime_tvp": "TVP", "intramonth": "MIDAS"}
SCEN_HATCH = {"base": "", "normalisation": "//", "energy_shock": "\\\\",
              "services_stickiness": "xx", "upside_surprise": "..", "downside_surprise": "--"}


def run_target(target_key, end_year=2024, do_attrib=True):
    res = E.evolve(target_key, end_year=end_year)
    evo = res["evolution"]; scen_long = res["scen_long"]
    nd = res["nowcast_date"]

    # save evolution + weights + scenarios
    evo.to_csv(os.path.join(C.DIR_INTRA_DATA, f"evolution_{target_key}.csv"), index=False)
    wcols = ["origin", "k", "w_baseline", "w_factor", "w_tvp", "w_midas"]
    evo[wcols].to_csv(os.path.join(C.DIR_INTRA_DATA, f"weights_{target_key}.csv"), index=False)
    scen_long.to_csv(os.path.join(C.DIR_SCEN_DATA, f"scenarios_{target_key}.csv"), index=False)
    res["scenarios"][C.ORIGINS[-1]].to_csv(
        os.path.join(C.DIR_SCEN_DATA, f"scenario_table_{target_key}.csv"), index=False)

    # attribution at a representative mid origin (T-7)
    attrib = {}
    if do_attrib:
        y, _ = T.resolve(target_key); y = y.dropna()
        pan, meta = P.build_panel(target_key, k=7)
        stk = S.ModelStack(pan, meta, end_year=end_year)
        run = stk.run(); stk._baseline_bt = run["baseline"]
        labels = W.regime_labels(y); post = R.nowcast_posterior(y)
        w7, _ = W.weights_for_month(run, W.model_errors(run), labels,
                                    pd.Timestamp(y.index[-1]), post, 7)
        ma = A.model_attribution(run, w7)
        hs = A.hf_sensitivity(stk, 7, w7)
        dc = A.driver_class(run, w7, R.driver_tags(pan, pd.Timestamp(y.index[-1])), post)
        attrib = dict(model_attribution=ma, hf_sensitivity=hs["sensitivity"], driver_class=dc)
        pd.DataFrame([dict(kind="model", name=k, value=v) for k, v in ma.items()] +
                     [dict(kind="hf_sens", name=k, value=v) for k, v in hs["sensitivity"].items()]
                     ).to_csv(os.path.join(C.DIR_PROD_DATA, f"attribution_{target_key}.csv"), index=False)

    # production recommendation: sharpest origin (T-1) + dominant model + top scenario
    t1 = evo.iloc[-1]
    scen_t1 = res["scenarios"][C.ORIGINS[-1]]
    top_scen = scen_t1.iloc[0]
    rec = dict(target=target_key, nowcast_date=str(nd.date()),
               best_origin=t1["origin"], point=round(t1["point"], 3),
               lo=round(t1["lo"], 3), hi=round(t1["hi"], 3),
               dominant_model=LAYER_LABEL.get(t1["dominant_model"], t1["dominant_model"]),
               regime=max(res["regime_post"], key=res["regime_post"].get),
               regime_prob=round(max(res["regime_post"].values()), 3),
               top_scenario=top_scen["scenario"], top_scenario_prob=round(top_scen["prob"], 3),
               scen_entropy=round(t1["scen_entropy"], 3),
               w_baseline=round(t1["w_baseline"], 3), w_midas=round(t1["w_midas"], 3),
               w_tvp=round(t1["w_tvp"], 3), w_factor=round(t1["w_factor"], 3))
    pd.DataFrame([rec]).to_csv(
        os.path.join(C.DIR_PROD_DATA, f"recommendation_{target_key}.csv"), index=False)

    _plot_evolution(target_key, res)
    _plot_scenarios(target_key, res)
    return res, rec, attrib


def _plot_evolution(target_key, res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    evo = res["evolution"]; x = -evo["k"].values   # T-30 left, T-1 right
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                                        gridspec_kw={"height_ratios": [2.2, 1.6, 1.2]})
    # (1) point + uncertainty band
    ax1.fill_between(x, evo["lo"], evo["hi"], color="0.8", alpha=0.7, label="±1σ band")
    ax1.plot(x, evo["point"], color="black", ls="-", marker="o", lw=2, label="weighted nowcast")
    ax1.plot(x, evo["e_point"], color="0.4", ls="--", marker="s", ms=4,
             label="scenario E[point]")
    for _, r in evo.iterrows():
        ax1.annotate(f"{r['point']:.2f}", (-r["k"], r["point"]), fontsize=7,
                     textcoords="offset points", xytext=(0, 7), ha="center")
    ax1.set_ylabel(f"{C.TARGETS[target_key]['label']}  %")
    ax1.set_title(f"Intramonth nowcast evolution — {C.TARGETS[target_key]['label']} "
                  f"({res['nowcast_date'].strftime('%b %Y')})")
    ax1.legend(fontsize=8, loc="best"); ax1.grid(alpha=0.3)
    # (2) weight evolution (lines, distinct styles)
    for layer in W.LAYERS:
        col = {"baseline": "w_baseline", "factor": "w_factor",
               "regime_tvp": "w_tvp", "intramonth": "w_midas"}[layer]
        s = LSTYLE[layer]
        ax2.plot(x, evo[col], color="0.2", lw=1.4, ls=s["ls"], marker=s["marker"],
                 ms=5, markerfacecolor="white", markeredgecolor="0.15",
                 label=LAYER_LABEL[layer])
    ax2.set_ylabel("model weight"); ax2.legend(fontsize=8, ncol=4, loc="upper center")
    ax2.grid(alpha=0.3); ax2.set_title("Regime+horizon model weights (sum=1)")
    # (3) driver overlays
    ax3.plot(x, evo["energy_led"], color="black", ls="-", marker="o", ms=4, label="energy-led")
    ax3.plot(x, evo["services_led"], color="0.4", ls="--", marker="s", ms=4, label="services-led")
    ax3.plot(x, evo["policy_tightening"], color="0.6", ls=":", marker="^", ms=4, label="policy-tightening")
    ax3.set_ylabel("driver intensity"); ax3.set_xlabel("forecast origin (days before month-end →)")
    ax3.legend(fontsize=8, ncol=3); ax3.grid(alpha=0.3)
    ax3.set_xticks(x); ax3.set_xticklabels([f"T-{k}" for k in evo["k"]])
    fig.tight_layout()
    p = os.path.join(C.DIR_INTRA_PLOTS, f"evolution_{target_key}.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  saved {p}")


def _plot_scenarios(target_key, res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    scen_long = res["scen_long"]; origins = sorted(scen_long["k"].unique(), reverse=True)
    scens = list(C.SCENARIOS)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.5, 1]})
    # (L) scenario mass shift T-30 → T-1 (stacked bars, hatched)
    xpos = np.arange(len(origins)); bottom = np.zeros(len(origins))
    for s in scens:
        probs = [float(scen_long[(scen_long["k"] == k) & (scen_long["scenario"] == s)]["prob"].iloc[0])
                 if len(scen_long[(scen_long["k"] == k) & (scen_long["scenario"] == s)]) else 0.0
                 for k in origins]
        axL.bar(xpos, probs, bottom=bottom, facecolor="0.85", edgecolor="black",
                hatch=SCEN_HATCH.get(s, ""), lw=0.8, label=s)
        bottom += np.array(probs)
    axL.set_xticks(xpos); axL.set_xticklabels([f"T-{k}" for k in origins])
    axL.set_ylabel("scenario probability"); axL.set_ylim(0, 1)
    axL.set_title(f"Scenario mass shift — {C.TARGETS[target_key]['label']}")
    axL.legend(fontsize=7, loc="upper center", ncol=3)
    # (R) scenario table at T-1: point ± band, prob as bar width annotation
    scen_t1 = res["scenarios"][C.ORIGINS[-1]].sort_values("point")
    ypos = np.arange(len(scen_t1))
    for i, (_, r) in enumerate(scen_t1.iterrows()):
        axR.errorbar(r["point"], i, xerr=[[r["point"] - r["lo"]], [r["hi"] - r["point"]]],
                     fmt="o", color="black", capsize=4, ms=5)
        axR.barh(i, r["prob"], left=0, height=0.0)  # placeholder for alignment
        axR.annotate(f"{r['scenario']} ({r['prob']*100:.0f}%)", (r["point"], i),
                     fontsize=8, textcoords="offset points", xytext=(8, 6))
    axR.set_yticks(ypos); axR.set_yticklabels([])
    axR.set_xlabel(f"{C.TARGETS[target_key]['label']} %")
    axR.set_title("Scenario tree @ T-1 (point ±1σ, prob%)"); axR.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    p = os.path.join(C.DIR_SCEN_PLOTS, f"scenarios_{target_key}.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  saved {p}")


def summary_table(res):
    """Dominant model / regime / top scenario by horizon."""
    evo = res["evolution"]; rows = []
    for _, r in evo.iterrows():
        k = r["k"]; st = res["scenarios"][k].iloc[0]
        rows.append(dict(origin=r["origin"], point=round(r["point"], 3),
                         dominant_model=LAYER_LABEL.get(r["dominant_model"], r["dominant_model"]),
                         top_scenario=st["scenario"], top_scen_prob=round(st["prob"], 3),
                         energy_led=round(r["energy_led"], 2), revision=round(r["revision"], 3)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=C.DEFAULT_TARGET)
    ap.add_argument("--all", action="store_true", help="run all available targets")
    ap.add_argument("--end-year", type=int, default=2024)
    args = ap.parse_args()

    if args.all:
        avail = [k for k, st in T.available_targets().items() if st != "unavailable"]
    else:
        avail = [args.target]
    print(f"Targets: {avail}")

    for tk in avail:
        print(f"\n=== {tk} ===")
        res, rec, attrib = run_target(tk, end_year=args.end_year)
        print(summary_table(res).to_string(index=False))
        print("RECOMMENDATION:", {k: rec[k] for k in
              ["best_origin", "point", "lo", "hi", "dominant_model", "regime",
               "top_scenario", "top_scenario_prob"]})
        if attrib:
            print("model attribution:", {k: round(v, 3) for k, v in attrib["model_attribution"].items()})
            print("driver class:", attrib["driver_class"])


if __name__ == "__main__":
    main()
