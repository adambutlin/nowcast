"""
intramonth/scenarios.py — probabilistic regime scenario tree (Part E).

NOT a fan chart. Produces a scenario TABLE with explicit probabilities that sum to 100%,
each row carrying a point nowcast, an uncertainty interval, and named model/factor
contributors. Scenario mass is derived from the causal regime posterior plus model
dispersion (tail mass), so it shifts with horizon as HF information sharpens both.

Probability construction (sums to 1 by algebra):
  regime mass:
     normalisation        <- post[disinflation]
     base                 <- post[normal]
     energy_shock         <- post[shock] · energy_share
     services_stickiness  <- post[shock] · (1 - energy_share)
  tail carve-out (model disagreement):
     tail = TAIL_MAX · clip(dispersion, 0, 1)
     each regime-mass scenario scaled by (1 - tail)
     upside_surprise   <- tail · skew_up
     downside_surprise <- tail · (1 - skew_up)

Point forecasts come from the weighted model mix evaluated under energy perturbations
(−σ / 0 / +σ) plus regime tilts; uncertainty = regime-conditional combined RMSE.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intramonth import config as C

TAIL_MAX = 0.30        # max probability mass routed to surprise tails
Z_TAIL   = 1.5         # σ multiple for surprise point forecasts


def _energy_share(drivers):
    e = max(drivers.get("energy_led", 0.0), 1e-6)
    s = max(drivers.get("services_led", 0.0), 1e-6)
    return float(e / (e + s))


def build_scenarios(regime_post, drivers, pert_points, sigma, base_pt,
                    dispersion, skew_up, top_model="?"):
    """
    regime_post : dict regime->prob (sums to 1)
    drivers     : dict driver->intensity (energy_led, services_led, policy_tightening)
    pert_points : dict {-1: pt_lowE, 0: pt_base, +1: pt_highE} weighted nowcasts
    sigma       : regime-conditional combined uncertainty (1σ, CPI pts)
    base_pt     : central weighted nowcast (= pert_points[0])
    dispersion  : normalized model disagreement in [0, ~1]
    skew_up     : fraction of probability the upside tail should carry [0,1]
    Returns DataFrame[scenario, prob, point, lo, hi, regime, model_contrib, factor_contrib, note]
    """
    eshare = _energy_share(drivers)
    tail = TAIL_MAX * float(np.clip(dispersion, 0, 1))
    scale = 1.0 - tail

    # regime-mass scenarios
    mass = {
        "normalisation":       regime_post.get("disinflation", 0.0) * scale,
        "base":                regime_post.get("normal", 0.0) * scale,
        "energy_shock":        regime_post.get("shock", 0.0) * eshare * scale,
        "services_stickiness": regime_post.get("shock", 0.0) * (1 - eshare) * scale,
        "upside_surprise":     tail * float(np.clip(skew_up, 0, 1)),
        "downside_surprise":   tail * (1 - float(np.clip(skew_up, 0, 1))),
    }

    pt_lowE, pt_base, pt_highE = pert_points.get(-1), pert_points.get(0), pert_points.get(+1)
    pol = drivers.get("policy_tightening", 0.0)
    points = {
        "normalisation":       pt_lowE - 0.10,                         # energy down + disinfl tilt
        "base":                pt_base,
        "energy_shock":        pt_highE,                               # energy up
        "services_stickiness": pt_base + 0.5 * sigma * (1 + pol),      # sticky services hold up
        "upside_surprise":     base_pt + Z_TAIL * sigma,
        "downside_surprise":   base_pt - Z_TAIL * sigma,
    }
    factor_contrib = {
        "normalisation":       "energy (Brent/TTF ↓)",
        "base":                "balanced",
        "energy_shock":        "energy (Brent/TTF ↑)",
        "services_stickiness": "services / wages, policy" + (" tightening" if pol > 0.3 else ""),
        "upside_surprise":     "broad upside (model disagreement)",
        "downside_surprise":   "broad downside (model disagreement)",
    }
    regime_of = {s: C.SCENARIOS[s]["regime"] for s in mass}

    rows = []
    for s in C.SCENARIOS:
        p = mass.get(s, 0.0); pt = points.get(s, base_pt)
        rows.append(dict(scenario=s, prob=p, point=pt,
                         lo=pt - sigma, hi=pt + sigma, regime=regime_of[s],
                         model_contrib=top_model, factor_contrib=factor_contrib.get(s, ""),
                         note=C.SCENARIOS[s].get("driver", "")))
    df = pd.DataFrame(rows)
    # normalize defensively (algebra already sums to 1, guard float drift)
    tot = df["prob"].sum()
    if tot > 0:
        df["prob"] = df["prob"] / tot
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    return df


def expected_forecast(scen_df):
    """Probability-weighted point = explains the central nowcast from the tree."""
    return float((scen_df["prob"] * scen_df["point"]).sum())


def scenario_entropy(scen_df):
    """Shannon entropy of the scenario distribution (uncertainty of the regime mix)."""
    p = scen_df["prob"].values
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


if __name__ == "__main__":
    # synthetic smoke test
    post = {"disinflation": 0.2, "normal": 0.6, "shock": 0.2}
    drivers = {"energy_led": 0.7, "services_led": 0.3, "policy_tightening": 0.4}
    pert = {-1: 2.9, 0: 3.04, +1: 3.18}
    df = build_scenarios(post, drivers, pert, sigma=0.42, base_pt=3.04,
                         dispersion=0.5, skew_up=0.6, top_model="TVP")
    print(df.round(3).to_string(index=False))
    print(f"Σprob={df['prob'].sum():.4f}  E[point]={expected_forecast(df):.3f}  "
          f"entropy={scenario_entropy(df):.3f}")
