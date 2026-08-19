"""Core overlay validation gate (T7). Purged+embargoed walk-forward of the core overlay
(AA + lam*(0.5*TVP + 0.5*LGBM) on the pinned core factors) vs AA-only, 2015-2024.

Reuses the FROZEN components for methodological consistency:
  - TVP member: uk_model_zoo.TVP().backtest(..., purge_horizon, embargo)
  - LGBM member: production.model._lgbm_resid (the exact frozen per-month purged LGBM)
Diebold-Mariano tests overlay vs AA. Ship the overlay iff rel-RMSE < 1; else core = AA-only.

Run: set -a; . ./.env; set +a; PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python -u code/new_factors/core_validate.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import uk_model_zoo as Z, two_stage as TS
import production.model as PM
from targets import CORE

END = 2024


def dm_stat(e_aa, e_fc, h=1):
    """Diebold-Mariano (squared-error loss). d = e_aa^2 - e_fc^2; >0 => overlay better.
    Returns (DM ~ N(0,1), one-sided p that overlay beats AA)."""
    from scipy import stats
    d = np.asarray(e_aa) ** 2 - np.asarray(e_fc) ** 2
    n = len(d)
    dbar = d.mean()
    g0 = np.var(d, ddof=0)
    # Newey-West with (h-1) lags; h=1 -> just g0
    var = g0 / n
    if var <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var)
    p_overlay_better = 1 - stats.norm.cdf(dm)
    return dm, p_overlay_better


def main():
    df, live, _ = TS.load_matrix(target=CORE)
    print("pinned core factors live:", live)
    aa = Z.AutoARIMA().backtest(df, [], CORE.name, start_year=CORE.aa_start, end_year=None)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    dfr = df.copy(); dfr["resid"] = resid.reindex(dfr.index)

    print("TVP member walk-forward (purged+embargoed) …")
    tvp_bt = Z.TVP().backtest(dfr, live, "resid", start_year=CORE.start, end_year=END,
                              purge_horizon=TS.PURGE_HORIZON, embargo=TS.EMBARGO)
    months = tvp_bt.index

    print("LGBM member walk-forward (frozen _lgbm_resid per month) …")
    lgbm = pd.Series({t: _safe_lgbm(df, live, resid, t) for t in months})

    ow = CORE.overlay_weights
    overlay = ow["tvp"] * tvp_bt["pred"] + ow["lgbm"] * lgbm.reindex(months)
    aa_pred = aa["pred"].reindex(months)
    actual = aa["actual"].reindex(months)
    fc = aa_pred + CORE.lam * overlay

    d = pd.DataFrame({"actual": actual, "aa": aa_pred, "overlay_fc": fc}).dropna()
    e_aa = (d["actual"] - d["aa"]).values
    e_fc = (d["actual"] - d["overlay_fc"]).values
    rmse_aa = float(np.sqrt(np.mean(e_aa ** 2)))
    rmse_fc = float(np.sqrt(np.mean(e_fc ** 2)))
    rel = rmse_fc / rmse_aa
    dm, p = dm_stat(e_aa, e_fc)

    print("\n=== CORE OVERLAY VALIDATION (purged+embargoed WF, 2015-2024) ===")
    print(f"  n months        : {len(d)}")
    print(f"  RMSE AA-only    : {rmse_aa:.4f}")
    print(f"  RMSE overlay    : {rmse_fc:.4f}")
    print(f"  rel-RMSE        : {rel:.4f}   ({'overlay beats AA' if rel < 1 else 'overlay WORSE than AA'})")
    print(f"  Diebold-Mariano : {dm:.3f}  (one-sided p overlay<AA = {p:.3f})")
    ship = rel < 1.0
    print(f"\n  GATE DECISION   : {'SHIP core overlay' if ship else 'FALLBACK — core prints AA-only'}")
    # also individual members for the viz/record
    for tag, ser in [("TVP", tvp_bt['pred']), ("LGBM", lgbm)]:
        m = (aa['pred'].reindex(months) + ser.reindex(months))
        dd = pd.DataFrame({"a": actual, "m": m}).dropna()
        r = float(np.sqrt(np.mean((dd['a'] - dd['m']) ** 2)))
        print(f"    member {tag:4s} recon RMSE: {r:.4f}  (rel {r/rmse_aa:.4f})")
    return dict(n=len(d), rmse_aa=rmse_aa, rmse_overlay=rmse_fc, rel=rel, dm=dm, p=p, ship=ship)


def _safe_lgbm(df, live, resid, t):
    """Per-month LGBM with the BACKTEST purge/embargo switched on explicitly.

    _lgbm_resid defaults to no purge (the live path legitimately trains on every
    released residual); a walk-forward must purge the label-horizon overlap or the
    gate scores itself on leaked information. Governance: docs/final_model.md §11."""
    try:
        return PM._lgbm_resid(df, live, resid, t, TS.PURGE_HORIZON, TS.EMBARGO)
    except Exception:
        return np.nan


if __name__ == "__main__":
    main()
