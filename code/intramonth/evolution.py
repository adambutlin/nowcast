"""
intramonth/evolution.py — intramonth forecast evolution T-30 → T-1 (Part F).

For a target, walks the forecast origins and at each one produces:
  point forecast, model mix (weights), regime posterior, driver tags,
  scenario probabilities, uncertainty band, and the revision vs the prior origin.

The latent inflation regime is slow (monthly, from the target's own history, causal),
while WITHIN the month the high-frequency data sharpens the point, shifts the model
weights (horizon prior), moves the driver overlays, and re-splits scenario mass — so
the evolution is genuine even though the core regime posterior is month-stamped.

Efficiency: the AutoARIMA baseline is univariate (origin-invariant) so it is computed
ONCE and reused across all origins; only the residual models (which consume HF as-of
features) and the live nowcast are recomputed per origin.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factors as F
from intramonth import (config as C, panel as P, stack as S, regime as R,
                        weights as W, scenarios as SC, hf_data as H)


def _perturbed_nowcast(stk, k, sigma_energy):
    """Live per-model nowcasts at origin k under energy perturbation {-1,0,+1}·σ."""
    daily = H.get_daily()
    base_panel = stk.panel
    meta = stk.meta
    out = {}
    for mult in (-1, 0, +1):
        ext, nd = P.extend_to_nowcast(base_panel, meta, k, daily=daily)
        # perturb HF energy columns on the nowcast row (regime-continuation lever)
        for col in ("brent_ret", "gas_ret"):
            if col in ext.columns and np.isfinite(ext.loc[nd, col]):
                ext.loc[nd, col] = ext.loc[nd, col] + mult * sigma_energy
        # baseline nowcast (energy-invariant) + residual model nowcasts
        from intramonth.stack import RESID, _zoo_class, _feats
        aa = _zoo_class(stk.stack["baseline"])()
        aa_now, _ = aa.nowcast(ext, [], stk.target)
        resid_known = (stk._baseline_bt["actual"] - stk._baseline_bt["pred"])
        ext[RESID] = resid_known.reindex(ext.index)
        per = {"baseline": aa_now}
        for layer in ["factor", "regime_tvp", "intramonth"]:
            cls = _zoo_class(stk.stack[layer]); feats = _feats(layer, meta)
            try:
                r, _ = cls().nowcast(ext, feats, RESID if C.RESIDUAL_FRAMEWORK else stk.target)
                per[layer] = (aa_now + r) if (C.RESIDUAL_FRAMEWORK and np.isfinite(r)) else r
            except Exception:
                per[layer] = np.nan
        out[mult] = (per, nd)
    return out


def _weighted_point(per_model, weights):
    num = den = 0.0
    for layer, w in weights.items():
        v = per_model.get(layer, np.nan)
        if np.isfinite(v) and w > 0:
            num += w * v; den += w
    return num / den if den > 0 else np.nan


def evolve(target_key=None, origins=None, end_year=2024):
    target_key = target_key or C.DEFAULT_TARGET
    origins = origins or C.ORIGINS
    y, st = (F.load_factor(C.TARGETS[target_key]["source"])
             if C.TARGETS[target_key]["kind"] == "yoy" else (None, None))
    from intramonth import targets as T
    y, st = T.resolve(target_key)
    y = y.dropna()
    labels = W.regime_labels(y)
    regime_post = R.nowcast_posterior(y)

    # baseline once (origin-invariant)
    pan0, meta0 = P.build_panel(target_key, k=origins[0])
    stk0 = S.ModelStack(pan0, meta0, end_year=end_year)
    baseline_bt = stk0.baseline()
    sig_energy = float(C.SCEN_ENERGY_SIGMA * (
        (H.get_daily()[["brent", "gas"]].pct_change().dropna().tail(500).std().mean())
        if H.get_daily() is not None else 0.1))

    rows, scen_tables, weight_rows, regime_rows = [], {}, [], []
    prev_pt = None
    asof_month = pd.Timestamp(y.index[-1])     # last released month (causal anchor)

    for k in origins:
        pan, meta = P.build_panel(target_key, k=k)
        stk = S.ModelStack(pan, meta, end_year=end_year)
        stk._baseline_bt = baseline_bt
        run = stk.run_with_baseline(baseline_bt)
        errs = W.model_errors(run)
        w, wdiag = W.weights_for_month(run, errs, labels, asof_month, regime_post, k)

        # live nowcasts under energy perturbation
        stk._baseline_bt = baseline_bt
        pert = _perturbed_nowcast(stk, k, sig_energy)
        per0, nd = pert[0]
        pert_points = {m: _weighted_point(pert[m][0], w) for m in (-1, 0, +1)}
        base_pt = pert_points[0]

        # model dispersion + skew from per-model nowcasts (excl. NaN)
        vals = np.array([per0[l] for l in W.LAYERS if np.isfinite(per0.get(l, np.nan))])
        sigma = max(np.average([run["models"][l]["rmse"] for l in run["models"]
                                if run["models"][l]] + [run["baseline_rmse"]]), 1e-3)
        dispersion = float(np.std(vals) / sigma) if len(vals) > 1 else 0.0
        skew_up = float(np.mean(vals > base_pt)) if len(vals) else 0.5

        drivers = R.driver_tags(pan, asof_month)
        scen = SC.build_scenarios(regime_post, drivers, pert_points, sigma, base_pt,
                                  dispersion, skew_up,
                                  top_model=max(w, key=w.get))
        scen_tables[k] = scen

        revision = (base_pt - prev_pt) if prev_pt is not None else 0.0
        prev_pt = base_pt
        dom = max(w, key=w.get)
        rows.append(dict(origin=f"T-{k}", k=k, point=base_pt,
                         lo=base_pt - sigma, hi=base_pt + sigma, sigma=sigma,
                         revision=revision, dominant_model=dom,
                         w_baseline=w["baseline"], w_factor=w["factor"],
                         w_tvp=w["regime_tvp"], w_midas=w["intramonth"],
                         p_disinflation=regime_post["disinflation"],
                         p_normal=regime_post["normal"], p_shock=regime_post["shock"],
                         energy_led=drivers["energy_led"], services_led=drivers["services_led"],
                         policy_tightening=drivers["policy_tightening"],
                         scen_entropy=SC.scenario_entropy(scen),
                         e_point=SC.expected_forecast(scen),
                         hf_coverage=pan.attrs.get("cover", np.nan), nowcast_date=nd))
        for s in C.SCENARIOS:
            r = scen[scen["scenario"] == s]
            weight_rows.append(dict(origin=f"T-{k}", k=k, **{l: w[l] for l in W.LAYERS}))
            regime_rows.append(dict(origin=f"T-{k}", k=k, scenario=s,
                                    prob=float(r["prob"].iloc[0]) if len(r) else 0.0,
                                    point=float(r["point"].iloc[0]) if len(r) else np.nan))

    evo = pd.DataFrame(rows)
    scen_long = pd.DataFrame(regime_rows)
    return dict(target=target_key, evolution=evo, scenarios=scen_tables,
                scen_long=scen_long, regime_post=regime_post, nowcast_date=nd)


if __name__ == "__main__":
    res = evolve()
    e = res["evolution"]
    cols = ["origin", "point", "sigma", "revision", "dominant_model",
            "w_baseline", "w_midas", "p_normal", "energy_led", "e_point"]
    print(f"target={res['target']} nowcast_date={res['nowcast_date'].date()} "
          f"regime_post={ {k: round(v,3) for k,v in res['regime_post'].items()} }")
    print(e[cols].round(3).to_string(index=False))
