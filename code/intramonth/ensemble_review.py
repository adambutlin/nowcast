"""
intramonth/ensemble_review.py — hostile test: do REGIME WEIGHTS earn their place?

Burden of proof on the regime layer. Walk-forward over the eval window at origin T-1
(month-end, full info — the fairest test for the final nowcast), build per-model causal
CPI predictions, then combine them four ways with CAUSAL per-month weights:

  autoarima : baseline only (persistence null)
  flat      : equal 1/N weights (no skill, no regime)
  perf      : softmax(−decayed RMSE/τ)·horizon_prior — performance weights, NO regime
  regime    : weights_for_month — performance weights CONDITIONED on the regime posterior

The regime layer is justified only if `regime` beats `perf` (isolating the regime
conditioning) on OOS RMSE AND the Diebold–Mariano test rejects equal accuracy. Reported
full-sample and on subsamples (ex-2022/23, pre-2020) so a shock-only result is exposed.

Outputs: data/intramonth/ensemble_review.csv, plots/intramonth/ensemble_review.png
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factors as F
import uk_model_zoo as Z
from intramonth import config as C, panel as P, stack as S, regime as R, weights as W

EVAL_START = 2018


def _perf_weights(errs, asof, k, temp=None, halflife=None):
    """Performance weights (no regime): softmax(−decayed RMSE/τ)·horizon_prior."""
    temp = temp or C.WEIGHT_TEMP; halflife = halflife or C.WEIGHT_HALFLIFE
    pri = W.horizon_prior(k)
    rmse = {l: W._decay_rmse(errs.get(l, pd.Series(dtype=float)), asof, halflife) for l in W.LAYERS}
    glob = np.nanmean([v for v in rmse.values() if np.isfinite(v)]) or 0.5
    scores = {l: -(rmse[l] if np.isfinite(rmse[l]) else glob) / temp for l in W.LAYERS}
    mx = max(scores.values()); ex = {l: np.exp(scores[l] - mx) * pri[l] for l in W.LAYERS}
    z = sum(ex.values()) or 1.0
    return {l: ex[l] / z for l in W.LAYERS}


def run(target="cpi_headline_yoy", end_year=2024):
    y, _ = F.load_factor("cpi_yoy"); y = y.dropna()
    labels = W.regime_labels(y)
    post_df, _, _ = R.filtered_posteriors(y)

    pan, meta = P.build_panel(target, k=1)
    stk = S.ModelStack(pan, meta, end_year=end_year)
    run_out = stk.run()
    errs = W.model_errors(run_out)

    # wide per-model CPI predictions + actual over the eval window
    preds = {"baseline": run_out["baseline"]["pred"]}
    actual = run_out["baseline"]["actual"]
    for layer, m in run_out["models"].items():
        if m:
            preds[layer] = m["bt"]["cpi_pred"]
    idx = actual.index[(actual.index.year >= EVAL_START) & (actual.index.year <= end_year)]
    wide = pd.DataFrame({l: preds[l].reindex(idx) for l in W.LAYERS if l in preds})
    wide["actual"] = actual.reindex(idx)
    wide = wide.dropna(subset=["actual"])

    rows = []
    for M in wide.index:
        a = wide.loc[M, "actual"]
        pv = {l: wide.loc[M, l] for l in W.LAYERS if l in wide.columns and np.isfinite(wide.loc[M, l])}
        if not pv:
            continue
        # causal regime posterior for M = filtered posterior at the prior month
        prior_months = post_df.index[post_df.index < M]
        rp = (post_df.loc[prior_months[-1]].to_dict() if len(prior_months)
              else {r: 1/3 for r in C.REGIMES})
        rp = {r: rp.get(r, 0.0) for r in C.REGIMES}; s = sum(rp.values()) or 1; rp = {r: rp[r]/s for r in rp}

        w_flat = {l: 1.0 / len(pv) for l in pv}
        w_perf = _perf_weights(errs, M, 1)
        w_reg, _ = W.weights_for_month(run_out, errs, labels, M, rp, 1)

        def comb(w):
            num = sum(w.get(l, 0) * pv[l] for l in pv); den = sum(w.get(l, 0) for l in pv)
            return num / den if den > 0 else np.nan
        rows.append(dict(date=M, actual=a, autoarima=pv.get("baseline", np.nan),
                         flat=comb(w_flat), perf=comb(w_perf), regime=comb(w_reg)))
    df = pd.DataFrame(rows).set_index("date")
    return df, run_out


def _rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def evaluate(df):
    schemes = ["autoarima", "flat", "perf", "regime"]
    err = {s: (df["actual"] - df[s]).dropna() for s in schemes}
    common = err["regime"].index
    for s in schemes:
        err[s] = err[s].reindex(common)
    windows = {"full": common,
               "ex_2022_23": common[(common.year < 2022) | (common.year > 2023)],
               "pre_2020": common[common.year < 2020]}
    rmse_tab = {}
    for wn, ix in windows.items():
        rmse_tab[wn] = {s: _rmse(err[s].reindex(ix).dropna()) for s in schemes}
    rmse_df = pd.DataFrame(rmse_tab).T

    # Diebold-Mariano: regime vs each alternative (full sample)
    dm = {}
    for alt in ["flat", "perf", "autoarima"]:
        e1 = err[alt].values; e2 = err["regime"].values
        stat, p = Z.dm_test(e1, e2)   # >0 means regime better
        dm[f"regime_vs_{alt}"] = dict(DM=stat, p=p,
                                      better=("regime" if stat > 0 else alt),
                                      significant=bool(p < 0.05))
    return rmse_df, dm


def verdict(rmse_df, dm):
    full = rmse_df.loc["full"]
    reg, perf = full["regime"], full["perf"]
    improves = reg < perf
    dm_perf = dm["regime_vs_perf"]
    lines = []
    lines.append(f"Full-sample RMSE: AutoARIMA={full['autoarima']:.4f} flat={full['flat']:.4f} "
                 f"perf={full['perf']:.4f} regime={full['regime']:.4f}")
    lines.append(f"Regime vs Perf (isolates regime conditioning): "
                 f"ΔRMSE={reg-perf:+.4f} ({'regime better' if improves else 'NO improvement'}), "
                 f"DM={dm_perf['DM']:.2f} p={dm_perf['p']:.3f} "
                 f"{'(significant)' if dm_perf['significant'] else '(NOT significant)'}")
    if not improves or not dm_perf["significant"]:
        lines.append("VERDICT: regime weighting does NOT significantly beat plain performance "
                     "weighting OOS. The regime layer does not earn its place as an alpha source; "
                     "treat it as interpretation/communication only.")
    else:
        lines.append("VERDICT: regime weighting significantly improves OOS RMSE vs performance "
                     "weighting — the regime conditioning earns its place.")
    # subsample honesty
    if "pre_2020" in rmse_df.index:
        pre = rmse_df.loc["pre_2020"]
        lines.append(f"Pre-2020: perf={pre['perf']:.4f} regime={pre['regime']:.4f} "
                     f"({'regime better' if pre['regime'] < pre['perf'] else 'regime worse/equal'}).")
    return "\n".join(lines)


def main():
    df, run_out = run()
    rmse_df, dm = evaluate(df)
    os.makedirs(C.DIR_INTRA_DATA, exist_ok=True)
    df.to_csv(os.path.join(C.DIR_INTRA_DATA, "ensemble_review_preds.csv"))
    rmse_df.to_csv(os.path.join(C.DIR_INTRA_DATA, "ensemble_review.csv"))
    print("=" * 66 + "\nENSEMBLE HOSTILE REVIEW — do regime weights earn their place?\n" + "=" * 66)
    print("\nRMSE by scheme and window:")
    print(rmse_df.round(4).to_string())
    print("\nDiebold-Mariano (regime vs alt; DM>0 = regime better):")
    for k, v in dm.items():
        print(f"  {k:22} DM={v['DM']:+.2f} p={v['p']:.3f} better={v['better']} "
              f"sig={v['significant']}")
    v = verdict(rmse_df, dm)
    print("\n" + "=" * 66 + "\nVERDICT\n" + "=" * 66 + "\n" + v)
    with open(os.path.join(C.DIR_INTRA_DATA, "ensemble_verdict.txt"), "w") as f:
        f.write(v)
    _plot(rmse_df, df)
    return rmse_df, dm, v


def _plot(rmse_df, df):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]})
    schemes = ["autoarima", "flat", "perf", "regime"]
    hatches = ["", "//", "xx", ".."]
    xpos = np.arange(len(rmse_df.index)); wbar = 0.2
    for i, s in enumerate(schemes):
        axL.bar(xpos + (i - 1.5) * wbar, rmse_df[s].values, wbar, facecolor="0.8",
                edgecolor="black", hatch=hatches[i], label=s)
    axL.set_xticks(xpos); axL.set_xticklabels(rmse_df.index, fontsize=8)
    axL.set_ylabel("RMSE"); axL.set_title("Ensemble RMSE by weighting scheme")
    axL.legend(fontsize=8); axL.grid(alpha=0.3, axis="y")
    # error scatter regime vs perf
    e_reg = (df["actual"] - df["regime"]); e_perf = (df["actual"] - df["perf"])
    axR.plot(df.index, e_perf, color="0.5", ls="--", marker="s", ms=3, label="perf error")
    axR.plot(df.index, e_reg, color="black", ls="-", marker="o", ms=3, label="regime error")
    axR.axhline(0, color="0.3", lw=0.8)
    axR.set_ylabel("forecast error (actual − pred)"); axR.set_title("Regime vs Perf errors over time")
    axR.legend(fontsize=8); axR.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(C.DIR_INTRA_PLOTS, "ensemble_review.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
