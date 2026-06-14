"""
intramonth/weights.py — regime-dependent, horizon-dependent model weights (Part D).

Final weight for model m at origin T-k and month M:

    w_m  ∝  Σ_r  post[r] · softmax_r(−RMSE_{m,r}/τ) · horizon_prior_m(k)

where
  post[r]              causal regime posterior for month M (regime.nowcast_posterior),
  RMSE_{m,r}           exp-decay (half-life) weighted error of model m within regime r,
                       using ONLY months strictly before M (causal, walk-forward),
  horizon_prior_m(k)   smooth tilt: baseline up at T-30 (little HF), MIDAS up at T-1
                       (HF accrued), factor/TVP mild mid-month boost.

This is NOT full-sample inverse-RMSE: errors are regime-conditional, half-life decayed,
softmaxed, and combined through the live regime posterior and the horizon prior, so the
mix differs by regime and by horizon and updates every month.

Weights always sum to 1.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intramonth import config as C, regime as R

LAYERS = ["baseline", "factor", "regime_tvp", "intramonth"]


def regime_labels(y):
    """Causal hard regime label per month = argmax of the filtered posterior."""
    post, _, _ = R.filtered_posteriors(pd.Series(y).dropna())
    return post.idxmax(axis=1)


def _decay_rmse(err, asof, halflife):
    """Exp-decay (half-life, months) weighted RMSE of errors strictly before asof."""
    e = err.dropna()
    e = e[e.index < asof]
    if not len(e):
        return np.nan
    age = (asof.to_period("M") - e.index.to_period("M")).map(lambda x: x.n).astype(float)
    w = 0.5 ** (age / max(halflife, 1))
    return float(np.sqrt(np.sum(w * e.values ** 2) / np.sum(w)))


def horizon_prior(k):
    """Smooth per-layer tilt by intramonth proximity. Returns dict layer->multiplier."""
    cov = 1.0 - k / 30.0                  # proxy HF coverage: 0 at T-30, ~1 at T-1
    cov = float(np.clip(cov, 0, 1))
    pri = {
        "baseline":   0.5 + 0.5 * (1 - cov),   # strong early (little HF)
        "factor":     0.6 + 0.2 * cov,
        "regime_tvp": 0.6 + 0.2 * cov,
        "intramonth": 0.3 + 0.7 * cov,         # strong late (HF filled)
    }
    return pri


def model_errors(run_out):
    """Extract per-layer reconstructed-error Series (incl. baseline=actual−pred)."""
    errs = {}
    aa = run_out["baseline"]
    errs["baseline"] = (aa["actual"] - aa["pred"])
    for layer, m in run_out["models"].items():
        errs[layer] = m["bt"]["err"] if m else pd.Series(dtype=float)
    return errs


def weights_for_month(run_out, errs, labels, asof, regime_post, k,
                      temp=None, halflife=None):
    """
    Causal weights at month `asof`, origin T-k, given the live regime posterior.
    Returns (weights dict layer->w, diagnostics dict).
    """
    temp = temp or C.WEIGHT_TEMP
    halflife = halflife or C.WEIGHT_HALFLIFE
    pri = horizon_prior(k)

    # regime-conditional decayed RMSE per layer
    rc_rmse = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        e = errs.get(layer, pd.Series(dtype=float))
        for r in C.REGIMES:
            mask = labels.reindex(e.index) == r
            er = e[mask.fillna(False)]
            rc_rmse[layer][r] = _decay_rmse(er, asof, halflife)

    # also a regime-agnostic fallback RMSE (when a regime has no history)
    fallback = {layer: _decay_rmse(errs.get(layer, pd.Series(dtype=float)), asof, halflife)
                for layer in LAYERS}
    glob = np.nanmean([v for v in fallback.values() if np.isfinite(v)]) or 0.5

    raw = {layer: 0.0 for layer in LAYERS}
    for r in C.REGIMES:
        post_r = regime_post.get(r, 0.0)
        if post_r <= 0:
            continue
        # softmax over −RMSE/τ within regime r
        scores = {}
        for layer in LAYERS:
            rm = rc_rmse[layer][r]
            if not np.isfinite(rm):
                rm = fallback[layer] if np.isfinite(fallback[layer]) else glob
            scores[layer] = -rm / temp
        mx = max(scores.values())
        ex = {layer: np.exp(scores[layer] - mx) for layer in LAYERS}
        z = sum(ex.values()) or 1.0
        for layer in LAYERS:
            raw[layer] += post_r * (ex[layer] / z) * pri[layer]

    z = sum(raw.values()) or 1.0
    w = {layer: raw[layer] / z for layer in LAYERS}
    diag = dict(regime_post=dict(regime_post), horizon_prior=pri,
                rc_rmse={l: {r: rc_rmse[l][r] for r in C.REGIMES} for l in LAYERS})
    return w, diag


def combine(run_out, weights, month=None):
    """Weighted reconstructed CPI forecast path (and per-month if month=None)."""
    aa = run_out["baseline"]
    parts, wsum = None, 0.0
    paths = {"baseline": aa["pred"]}
    for layer, m in run_out["models"].items():
        if m:
            paths[layer] = m["bt"]["cpi_pred"]
    idx = aa.index
    acc = pd.Series(0.0, index=idx); wtot = pd.Series(0.0, index=idx)
    for layer, w in weights.items():
        if layer in paths and w > 0:
            p = paths[layer].reindex(idx)
            acc = acc.add(p * w, fill_value=0.0)
            wtot = wtot.add(p.notna().astype(float) * w, fill_value=0.0)
    combined = acc / wtot.replace(0, np.nan)
    return combined


if __name__ == "__main__":
    import factors as F
    from intramonth import panel as P, stack as S
    y, _ = F.load_factor("cpi_yoy"); y = y.dropna()
    labels = regime_labels(y)
    pan, meta = P.build_panel(C.DEFAULT_TARGET, k=1)
    run = S.ModelStack(pan, meta, end_year=2024).run()
    errs = model_errors(run)
    post = R.nowcast_posterior(y)
    asof = pd.Timestamp("2024-12-31")
    for k in (30, 14, 1):
        w, diag = weights_for_month(run, errs, labels, asof, post, k)
        print(f"T-{k:<2}", {l: round(v, 3) for l, v in w.items()}, "Σ=", round(sum(w.values()), 4))
