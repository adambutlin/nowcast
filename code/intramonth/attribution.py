"""
intramonth/attribution.py — model & factor attribution (Part G).

Explains a nowcast at an origin:
  model_attribution   : weight × off-baseline move per layer → normalized contribution
  hf_sensitivity      : Δ(weighted point) when each HF block is shocked ±σ at the
                        nowcast row (energy / fx / vol) → which HF input moved the forecast
  lobo_factors        : leave-one-block-out ΔRMSE on the factor model (energy / fx / rates
                        / monthly blocks) → which factor blocks the model relies on
  driver_class        : labels the nowcast as persistence- / regime- / factor-shock-driven

All causal: uses the same as-of panel and the precomputed baseline.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intramonth import config as C, panel as P
from intramonth.stack import RESID, _zoo_class, _feats

# factor blocks for leave-one-block-out
BLOCKS = {
    "energy_hf":  ["brent_ret", "brent_lvl", "gas_ret", "gas_lvl"],
    "fx_vol_hf":  ["gbp_ret", "gbp_lvl", "vix_ret", "vix_lvl"],
    "energy_mon": ["oil_brent", "gas_eu", "imf_all_commodity", "ofgem_cap_delta"],
    "macro_mon":  ["uk_quarterly_gdp"],
    "policy_mon": ["mpc_rate_change", "mpc_vote_split", "budget_event"],
}


def model_attribution(run_out, weights):
    """Per-layer contribution = weight × mean|off-baseline move|, normalized to 100%."""
    contrib = {}
    contrib["baseline"] = weights.get("baseline", 0.0) * 1e-9  # baseline = persistence anchor
    for layer, m in run_out["models"].items():
        if m:
            contrib[layer] = weights.get(layer, 0.0) * m["contribution"]
    tot = sum(contrib.values()) or 1.0
    return {k: v / tot for k, v in sorted(contrib.items(), key=lambda x: -x[1])}


def hf_sensitivity(stk, k, weights, sigma=0.1):
    """Δ(weighted point) per HF block shocked +σ at the nowcast row. CPI pts."""
    from intramonth import hf_data as H
    daily = H.get_daily()
    base_ext, nd = P.extend_to_nowcast(stk.panel, stk.meta, k, daily=daily)
    resid_known = (stk._baseline_bt["actual"] - stk._baseline_bt["pred"])

    def weighted_point(ext):
        aa = _zoo_class(stk.stack["baseline"])()
        aa_now, _ = aa.nowcast(ext, [], stk.target)
        ext = ext.copy(); ext[RESID] = resid_known.reindex(ext.index)
        per = {"baseline": aa_now}
        for layer in ["factor", "regime_tvp", "intramonth"]:
            cls = _zoo_class(stk.stack[layer]); feats = _feats(layer, stk.meta)
            try:
                r, _ = cls().nowcast(ext, feats, RESID)
                per[layer] = aa_now + r if np.isfinite(r) else np.nan
            except Exception:
                per[layer] = np.nan
        num = den = 0.0
        for layer, w in weights.items():
            v = per.get(layer, np.nan)
            if np.isfinite(v) and w > 0:
                num += w * v; den += w
        return num / den if den else np.nan

    p0 = weighted_point(base_ext)
    sens = {}
    hfcols = {"energy_hf": ["brent_ret", "gas_ret"], "fx_hf": ["gbp_ret"], "vol_hf": ["vix_ret"]}
    for blk, cols in hfcols.items():
        ext = base_ext.copy()
        for c in cols:
            if c in ext.columns and np.isfinite(ext.loc[nd, c]):
                ext.loc[nd, c] = ext.loc[nd, c] + sigma
        sens[blk] = weighted_point(ext) - p0
    return dict(base_point=p0, sensitivity=sens)


def lobo_factors(panel, meta, baseline_bt, factor_model=None, start_year=None, end_year=2024):
    """Leave-one-block-out ΔRMSE on the factor model (higher = block matters more)."""
    factor_model = factor_model or C.STACK["factor"]
    start_year = start_year or C.TRAIN_FROM + 4
    pan = panel.copy()
    resid = (baseline_bt["actual"] - baseline_bt["pred"]).reindex(pan.index)
    pan[RESID] = resid
    allf = _feats("factor", meta)
    cls = _zoo_class(factor_model)

    def rmse(feats):
        try:
            bt = cls().backtest(pan, feats, RESID, start_year=start_year, end_year=end_year)
            if bt is None or not len(bt):
                return np.nan
            return float(np.sqrt(((bt["actual"] - bt["pred"]) ** 2).mean()))
        except Exception:
            return np.nan

    full = rmse(allf)
    out = {"_full": full}
    for blk, cols in BLOCKS.items():
        drop = [c for c in allf if c not in cols]
        if len(drop) == len(allf):       # block not present → skip
            continue
        out[blk] = rmse(drop) - full      # ΔRMSE from removing the block
    return out


def driver_class(run_out, weights, drivers, regime_post):
    """Classify what drives the nowcast: persistence / regime-shift / factor-shock."""
    w_base = weights.get("baseline", 0)
    off = 1 - w_base
    shock_p = regime_post.get("shock", 0) + regime_post.get("disinflation", 0)
    energy = drivers.get("energy_led", 0)
    if w_base > 0.5:
        return "persistence-driven (baseline dominant)"
    if shock_p > 0.4:
        return "regime-shift-driven (non-normal regime mass)"
    if energy > 0.5 or weights.get("intramonth", 0) > 0.3:
        return "factor-shock-driven (HF/energy)"
    return "mixed (factor + persistence)"


if __name__ == "__main__":
    import factors as F
    from intramonth import stack as S, regime as R, weights as W
    y, _ = F.load_factor("cpi_yoy"); y = y.dropna()
    pan, meta = P.build_panel(C.DEFAULT_TARGET, k=7)
    stk = S.ModelStack(pan, meta, end_year=2024)
    run = stk.run(); stk._baseline_bt = run["baseline"]
    labels = W.regime_labels(y); post = R.nowcast_posterior(y)
    w, _ = W.weights_for_month(run, W.model_errors(run), labels,
                               pd.Timestamp(y.index[-1]), post, 7)
    print("model attribution:", {k: round(v, 3) for k, v in model_attribution(run, w).items()})
    print("hf sensitivity:", {k: round(v, 4) for k, v in hf_sensitivity(stk, 7, w)["sensitivity"].items()})
    print("driver class:", driver_class(run, w, R.driver_tags(pan, pd.Timestamp(y.index[-1])), post))
