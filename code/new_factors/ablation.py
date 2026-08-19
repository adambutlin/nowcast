"""Factor ablation — does the overlay still work without a given PINNED factor?

    PYTHONPATH=code python code/new_factors/ablation.py uk_quarterly_gdp [--target core]

Runs the production spec (AA + λ·(0.5·TVP + 0.5·LGBM)) as a purged + embargoed
walk-forward twice — with the full pinned set and with the factor removed — and
reports RMSE, rel-RMSE against AA-only, and a Diebold-Mariano test of the two
overlays against each other. Method matches the core validation gate
(new_factors/core_validate.py) so the numbers are comparable to the ship decision.

It also reports what the factor costs OPERATIONALLY: `BaseModel._nowcast_row`
blocks a model outright when any pinned factor is past its ffill budget, so a
factor that publishes more slowly than monthly can kill the TVP leg outright in
the live path. The ablation counts those blocked months both ways.
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402

import uk_model_zoo as Z                                  # noqa: E402
import two_stage as TS                                    # noqa: E402
import production.model as PM                             # noqa: E402
from core_validate import dm_stat                         # noqa: E402
from targets import HEADLINE, CORE                        # noqa: E402

END = 2024          # walk-forward scoring window end, as in the core gate


def walk_forward(df, live, target, aa, resid, months):
    """Overlay forecast per month, purged + embargoed. Returns a Series."""
    dfr = df.copy()
    dfr["resid"] = resid.reindex(dfr.index)
    tvp = Z.TVP().backtest(dfr, live, "resid", start_year=target.start, end_year=END,
                           purge_horizon=TS.PURGE_HORIZON, embargo=TS.EMBARGO)
    lgbm = pd.Series({t: _safe_lgbm(df, live, resid, t) for t in months})
    ow = target.overlay_weights
    overlay = ow["tvp"] * tvp["pred"].reindex(months) + ow["lgbm"] * lgbm.reindex(months)
    return aa["pred"].reindex(months) + target.lam * overlay, tvp["pred"].reindex(months), lgbm


def _safe_lgbm(df, live, resid, t):
    try:
        return PM._lgbm_resid(df, live, resid, t, TS.PURGE_HORIZON, TS.EMBARGO)
    except Exception:
        return np.nan


def blocked_months(df, live, target, months):
    """Months where _nowcast_row refuses to build a feature row (TVP leg dies)."""
    out = []
    for t in months:
        row, _ = Z.BaseModel._nowcast_row(df, live, target.name, nowcast_date=t)
        if row is None:
            out.append(t)
    return out


def _score(actual, pred, aa_pred):
    d = pd.DataFrame({"a": actual, "p": pred, "aa": aa_pred}).dropna()
    e_fc = (d["a"] - d["p"]).values
    e_aa = (d["a"] - d["aa"]).values
    rmse = float(np.sqrt(np.mean(e_fc ** 2)))
    return dict(n=len(d), rmse=rmse, mae=float(np.mean(np.abs(e_fc))),
                rel=rmse / float(np.sqrt(np.mean(e_aa ** 2))), err=e_fc, index=d.index)


def main(drop, target):
    df, live, _ = TS.load_matrix(target=target)
    if drop not in live:
        sys.exit(f"{drop!r} is not in the live pinned set for {target.name}: {live}")
    ablated = [f for f in live if f != drop]

    aa = Z.AutoARIMA().backtest(df, [], target.name, start_year=target.aa_start, end_year=None)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    months = aa.index[(aa.index.year >= target.start) & (aa.index.year <= END)]
    actual = aa["actual"].reindex(months)
    aa_pred = aa["pred"].reindex(months)

    print(f"=== ABLATION: {target.label} — dropping {drop!r} ===")
    print(f"  pinned live : {live}")
    print(f"  ablated     : {ablated}")
    print(f"  window      : {months.min().date()} … {months.max().date()}  "
          f"(purged {TS.PURGE_HORIZON}+{TS.EMBARGO})\n")

    results = {}
    for tag, factors in (("full", live), ("ablated", ablated)):
        print(f"  walk-forward [{tag}] …")
        fc, tvp, lgbm = walk_forward(df, factors, target, aa, resid, months)
        results[tag] = _score(actual, fc, aa_pred)
        results[tag]["tvp_missing"] = int(tvp.isna().sum())
        results[tag]["lgbm_missing"] = int(lgbm.reindex(months).isna().sum())

    rmse_aa = float(np.sqrt(np.mean((actual - aa_pred).dropna() ** 2)))
    print(f"\n  {'variant':<9} {'n':>4} {'RMSE':>8} {'MAE':>8} {'rel-RMSE':>9} "
          f"{'TVP gaps':>9} {'LGBM gaps':>10}")
    print(f"  {'AA-only':<9} {results['full']['n']:>4} {rmse_aa:>8.4f} {'—':>8} {1.0:>9.4f}"
          f" {'—':>9} {'—':>10}")
    for tag in ("full", "ablated"):
        r = results[tag]
        print(f"  {tag:<9} {r['n']:>4} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r['rel']:>9.4f}"
              f" {r['tvp_missing']:>9} {r['lgbm_missing']:>10}")

    # Sub-period breakdown: a factor worth keeping should not owe its whole case to
    # one window (and one that hurts should hurt in more than one).
    e = {tag: pd.Series(results[tag]["err"], index=results[tag]["index"])
         for tag in ("full", "ablated")}
    e_aa = (actual - aa_pred).dropna()
    print(f"\n  {'window':<12} {'n':>4} {'RMSE full':>10} {'RMSE ablated':>13} {'Δ':>8} {'rel full':>9} {'rel abl':>8}")
    edges = list(range(target.start, END + 1, 5)) + [END + 1]
    for lo, hi in zip(edges, edges[1:]):
        sel = lambda x: x[(x.index.year >= lo) & (x.index.year < hi)]
        f, a, aa_e = sel(e["full"]).dropna(), sel(e["ablated"]).dropna(), sel(e_aa).dropna()
        if not len(f):
            continue
        rf, ra = float(np.sqrt((f ** 2).mean())), float(np.sqrt((a ** 2).mean()))
        raa = float(np.sqrt((aa_e ** 2).mean()))
        print(f"  {f'{lo}-{hi-1}':<12} {len(f):>4} {rf:>10.4f} {ra:>13.4f} {ra-rf:>+8.4f}"
              f" {rf/raa:>9.4f} {ra/raa:>8.4f}")

    common = results["full"]["index"].intersection(results["ablated"]["index"])
    e_full = pd.Series(results["full"]["err"], index=results["full"]["index"]).reindex(common)
    e_abl = pd.Series(results["ablated"]["err"], index=results["ablated"]["index"]).reindex(common)
    dm, p = dm_stat(e_full.values, e_abl.values)      # d>0 => ablated better than full
    print(f"\n  Diebold-Mariano (ablated vs full, n={len(common)}): DM {dm:+.3f}, "
          f"one-sided p(ablated better) = {p:.3f}")
    delta = results["ablated"]["rmse"] - results["full"]["rmse"]
    print(f"  ΔRMSE from dropping {drop}: {delta:+.4f} "
          f"({'WORSE without it' if delta > 0 else 'BETTER without it'})")

    live_months = df.index[(df.index >= months.min())]
    b_full = blocked_months(df, live, target, live_months)
    b_abl = blocked_months(df, ablated, target, live_months)
    print(f"\n  Live-path feature-row blocks (TVP leg dies, forecast falls back to LGBM alone):")
    print(f"    with {drop:<20}: {len(b_full):>3} / {len(live_months)} months"
          + (f"  most recent: {[str(t.date()) for t in b_full[-3:]]}" if b_full else ""))
    print(f"    without {drop:<17}: {len(b_abl):>3} / {len(live_months)} months"
          + (f"  most recent: {[str(t.date()) for t in b_abl[-3:]]}" if b_abl else ""))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("factor", nargs="?", default="uk_quarterly_gdp")
    ap.add_argument("--target", choices=("headline", "core"), default="headline")
    a = ap.parse_args()
    main(a.factor, HEADLINE if a.target == "headline" else CORE)
