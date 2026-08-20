"""Pinned-set race — energy channel (crude vs pump price) and the missing FX channel.

    PYTHONPATH=code python code/new_factors/energy_fx_race.py [--target headline]

Two open questions the incumbent HEADLINE pinned set never answered:

  1. The motor-fuel channel runs off `oil_brent` (crude), not `uk_pump_price` (the direct
     CPI 07.2.2 input, corr 0.993 with its MoM vs 0.831 for a month-end crude close).
     docs/MODEL_CARD.md lists this as a known gap; docs/postmortem_2026_07.md §6 blocks
     the swap on "the factor race + validation gate". This is that race.
  2. There is no FX factor at all. `gbpusd` was a candidate=False "core factor" in the
     pre-two_stage main.py era and was silently not carried into targets.HEADLINE.pinned
     at d25f185. It ranks 12/50 in the univariate race (rel 0.987) — ahead of three
     factors that ARE pinned. Core CPI pins `gbp_eer` off a real race; headline pins none.

Method mirrors new_factors/ablation.py and the core validation gate so the numbers are
comparable to prior ship decisions: production spec `AA + λ·(0.5·TVP + 0.5·LGBM)` run as a
purged + embargoed walk-forward per arm, scored as RMSE / rel-RMSE against AA-only, with a
Diebold-Mariano test of each arm against the incumbent set.

Caveat reported per arm: `uk_pump_price` (DESNZ) starts 2003-07 against 1997 for the rest of
the matrix, so arms carrying it train on a shorter history. That is a real operational cost
of the factor, not a bug, but it confounds a pure like-for-like read — hence `train_from`
and the effective-n columns in the output.

Out: data/new_factors/energy_fx_race.csv
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402

import factors as F                                       # noqa: E402
import uk_model_zoo as Z                                  # noqa: E402
import two_stage as TS                                    # noqa: E402
from ablation import walk_forward, blocked_months, _score  # noqa: E402
from core_validate import dm_stat                         # noqa: E402
from targets import HEADLINE, CORE                        # noqa: E402

END = 2024
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "new_factors")

PUMP, CRUDE, FX = "uk_pump_price", "oil_brent", "gbpusd"


def arms(pinned):
    """The five arms. `pinned` is the incumbent live set for the target."""
    base = list(pinned)
    no_crude = [f for f in base if f != CRUDE]
    return {
        "base":        base,
        "+pump":       base + [PUMP],
        "swap_pump":   no_crude + [PUMP],
        "+fx":         base + [FX],
        "+pump+fx":    base + [PUMP, FX],
    }


def load_matrix_with(extra, target):
    """two_stage.load_matrix, but with `extra` factors appended to the pinned request.

    Loading one matrix over the union of every arm's factors keeps the AA residual and the
    month index identical across arms — the arms then differ only in which columns the
    overlay is allowed to see, which is what the DM test needs to be paired.
    """
    pinned = list(target.pinned) + [e for e in extra if e not in target.pinned]
    return TS.load_matrix(pinned=pinned, target=target)


def first_obs(df, name):
    s = df[name].dropna() if name in df.columns else pd.Series(dtype=float)
    return s.index[0] if len(s) else None


def main(target):
    union = [PUMP, FX]
    df, live_union, status = load_matrix_with(union, target)
    incumbent = [f for f in target.pinned if f in live_union]

    for n in union:
        if n not in live_union:
            sys.exit(f"{n!r} unavailable (status={status.get(n)}) — cannot race it")

    aa = Z.AutoARIMA().backtest(df, [], target.name, start_year=target.aa_start, end_year=None)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    months = aa.index[(aa.index.year >= target.start) & (aa.index.year <= END)]
    actual = aa["actual"].reindex(months)
    aa_pred = aa["pred"].reindex(months)
    rmse_aa = float(np.sqrt(np.mean((actual - aa_pred).dropna() ** 2)))

    print(f"=== PINNED-SET RACE: {target.label} ===")
    print(f"  incumbent   : {incumbent}")
    print(f"  window      : {months.min().date()} … {months.max().date()}  "
          f"(purged {TS.PURGE_HORIZON}+{TS.EMBARGO}, n={len(months)})")
    print(f"  AA-only RMSE: {rmse_aa:.4f}")
    for n in union + [CRUDE]:
        fo = first_obs(df, n)
        print(f"  {n:<22} first obs {fo.date() if fo is not None else '—'}  status {status.get(n)}")
    print()

    A = arms(incumbent)
    live_months = df.index[df.index >= months.min()]
    results, rows = {}, []
    for tag, facs in A.items():
        print(f"  walk-forward [{tag}] ({len(facs)} factors) …", flush=True)
        fc, tvp, lgbm = walk_forward(df, facs, target, aa, resid, months)
        r = _score(actual, fc, aa_pred)
        r["tvp_missing"] = int(tvp.isna().sum())
        r["lgbm_missing"] = int(lgbm.reindex(months).isna().sum())
        r["blocked"] = len(blocked_months(df, facs, target, live_months))
        r["factors"] = facs
        results[tag] = r

    # Paired DM of each arm against the incumbent set, on their common months.
    e = {t: pd.Series(results[t]["err"], index=results[t]["index"]) for t in A}
    print(f"\n  {'arm':<11} {'k':>3} {'n':>4} {'RMSE':>8} {'MAE':>8} {'rel-RMSE':>9} "
          f"{'DM vs base':>11} {'p':>7} {'TVPgap':>7} {'LGBMgap':>8} {'blocked':>8}")
    print(f"  {'AA-only':<11} {0:>3} {results['base']['n']:>4} {rmse_aa:>8.4f} {'—':>8} "
          f"{1.0:>9.4f} {'—':>11} {'—':>7} {'—':>7} {'—':>8} {'—':>8}")
    for tag, r in results.items():
        if tag == "base":
            dm, p = np.nan, np.nan
        else:
            common = e["base"].dropna().index.intersection(e[tag].dropna().index)
            # d = e_base^2 - e_arm^2 ; DM>0 => arm beats the incumbent set
            dm, p = dm_stat(e["base"].reindex(common).values, e[tag].reindex(common).values)
        print(f"  {tag:<11} {len(r['factors']):>3} {r['n']:>4} {r['rmse']:>8.4f} {r['mae']:>8.4f} "
              f"{r['rel']:>9.4f} {dm:>11.3f} {p:>7.3f} {r['tvp_missing']:>7} "
              f"{r['lgbm_missing']:>8} {r['blocked']:>8}")
        rows.append(dict(arm=tag, k=len(r["factors"]), n=r["n"], rmse=r["rmse"], mae=r["mae"],
                         rel_rmse=r["rel"], dm_vs_base=dm, p_beats_base=p,
                         tvp_gaps=r["tvp_missing"], lgbm_gaps=r["lgbm_missing"],
                         blocked_months=r["blocked"], factors=" ".join(r["factors"])))

    # Sub-period: an arm worth shipping should not owe its case to one window.
    print(f"\n  Sub-period RMSE (5y blocks):")
    edges = list(range(target.start, END + 1, 5)) + [END + 1]
    hdr = "  " + f"{'window':<12}" + "".join(f"{t:>11}" for t in A)
    print(hdr)
    for lo, hi in zip(edges, edges[1:]):
        cells, any_row = "", False
        for tag in A:
            x = e[tag]
            sel = x[(x.index.year >= lo) & (x.index.year < hi)].dropna()
            if len(sel):
                any_row = True
                cells += f"{float(np.sqrt((sel ** 2).mean())):>11.4f}"
            else:
                cells += f"{'—':>11}"
        if any_row:
            print(f"  {f'{lo}-{hi-1}':<12}{cells}")

    # Shock months carry the overlay's whole case (docs/final_model.md) — check there too.
    shock = e["base"].index[(e["base"].index.year >= 2021) & (e["base"].index.year <= 2023)]
    print(f"\n  Shock window 2021-2023 (n={len(shock)}):")
    for tag in A:
        sel = e[tag].reindex(shock).dropna()
        print(f"    {tag:<11} RMSE {float(np.sqrt((sel ** 2).mean())):.4f}")

    out = os.path.join(_OUT, f"energy_fx_race{'' if target is HEADLINE else '_core'}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  wrote {os.path.relpath(out, _ROOT)}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("headline", "core"), default="headline")
    a = ap.parse_args()
    main(HEADLINE if a.target == "headline" else CORE)
