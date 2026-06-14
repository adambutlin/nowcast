"""
intramonth/regime.py — latent regime detector / gating layer (Part C-Layer5, Part D).

Core probabilistic object: a 3-state Markov-switching model on the target's own
history (statsmodels MarkovRegression, switching variance). States are relabelled by
their estimated mean into {disinflation, normal, shock}. All posteriors are CAUSAL:
filtered probabilities use data through t only; the nowcast-month posterior is the
last filtered posterior propagated ONE step by the transition matrix.

On top of the latent posterior we add interpretable, deterministic DRIVER OVERLAYS
(energy_led / services_led / policy_tightening) from factor momentum — used by the
scenario layer to route regime mass into named scenarios.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intramonth import config as C


def _fit(y, k=None):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    k = k or C.N_REGIME_STATES
    res = MarkovRegression(np.asarray(y, float), k_regimes=k, trend="c",
                           switching_variance=True).fit()
    return res


def _relabel(res, k):
    """Map raw state index -> regime name by ascending estimated mean."""
    named = dict(zip(res.model.param_names, np.asarray(res.params)))
    means = np.array([named.get(f"const[{i}]", np.nan) for i in range(k)])
    order = np.argsort(np.nan_to_num(means, nan=0.0))          # low→high mean
    names = C.REGIMES if k == len(C.REGIMES) else [f"r{i}" for i in range(k)]
    return {int(state): names[rank] for rank, state in enumerate(order)}, means


def filtered_posteriors(y):
    """
    Causal filtered regime posteriors over the whole history.
    Returns DataFrame[date × regime] (probabilities sum to 1 per row) + transition P.
    """
    y = pd.Series(y).dropna()
    k = C.N_REGIME_STATES
    res = _fit(y, k)
    lab, means = _relabel(res, k)
    fp = np.asarray(res.filtered_marginal_probabilities)
    fp = fp if fp.shape[1] == k else fp.T                      # (nobs, k) causal
    cols = [lab[i] for i in range(k)]
    post = pd.DataFrame(fp, index=y.index, columns=cols)
    post = post[[r for r in C.REGIMES if r in post.columns]]   # ordered
    P = (res.regime_transition[:, :, 0] if res.regime_transition.ndim == 3
         else res.regime_transition)
    return post, P, lab


def nowcast_posterior(y):
    """
    Causal regime posterior for the NEXT (unreleased) month: last filtered posterior
    propagated one step by the transition matrix P. Returns dict regime->prob.
    """
    post, P, lab = filtered_posteriors(y)
    last = post.iloc[-1].reindex([lab[i] for i in range(C.N_REGIME_STATES)]).values
    # P is in raw-state order; map last (regime-name order) back to state order
    inv = {v: kk for kk, v in lab.items()}
    state_vec = np.zeros(C.N_REGIME_STATES)
    for rname, p in post.iloc[-1].items():
        state_vec[inv[rname]] = p
    prop = state_vec @ P.T
    out = {lab[i]: float(prop[i]) for i in range(C.N_REGIME_STATES)}
    # temper: p ∝ p**β (β<1 widens an over-confident HMM posterior) — causal
    beta = C.REGIME_TEMPER
    tem = {r: max(out.get(r, 0.0), 0.0) ** beta for r in C.REGIMES}
    s = sum(tem.values()) or 1.0
    return {r: tem[r] / s for r in C.REGIMES}


def driver_tags(panel, asof_idx):
    """
    Deterministic interpretable overlays at a given panel row (causal: uses values
    on/under asof_idx only). Returns dict with intensities in [0,1].
      energy_led        : HF energy momentum (brent_ret+gas_ret) vs trailing dispersion
      services_led      : domestic-demand proxy (uk_quarterly_gdp) high & energy not leading
      policy_tightening : cumulative recent Bank-Rate change > 0
    """
    hist = panel.loc[:asof_idx]
    row = panel.loc[asof_idx]
    def _z(col, val):
        s = hist[col].dropna() if col in hist.columns else pd.Series(dtype=float)
        if len(s) < 12 or not np.isfinite(val):
            return 0.0
        mu, sd = s.iloc[:-1].mean(), s.iloc[:-1].std() or 1.0
        return float((val - mu) / sd)

    e_ret = np.nansum([row.get("brent_ret", np.nan), row.get("gas_ret", np.nan)])
    e_hist = (hist.get("brent_ret", pd.Series(dtype=float)).fillna(0)
              + hist.get("gas_ret", pd.Series(dtype=float)).fillna(0))
    e_sd = e_hist.iloc[:-1].std() or 1.0
    energy = float(np.clip(abs(e_ret) / (2 * e_sd), 0, 1)) if np.isfinite(e_ret) else 0.0

    gdp_z = _z("uk_quarterly_gdp", row.get("uk_quarterly_gdp", np.nan))
    services = float(np.clip((gdp_z + 1) / 2, 0, 1)) * (1 - energy)

    rate_recent = (panel.get("mpc_rate_change", pd.Series(dtype=float))
                   .loc[:asof_idx].tail(6).sum())
    policy = float(np.clip(rate_recent / 75.0, 0, 1)) if np.isfinite(rate_recent) else 0.0  # 75bp→1

    return dict(energy_led=energy, services_led=services, policy_tightening=policy)


def regime_summary(y):
    """Diagnostic: current filtered regime + nowcast posterior."""
    post, P, lab = filtered_posteriors(y)
    now = nowcast_posterior(y)
    cur = post.iloc[-1].idxmax()
    return dict(current_regime=cur, current_posterior=post.iloc[-1].to_dict(),
                nowcast_posterior=now)


if __name__ == "__main__":
    import factors as F
    y, _ = F.load_factor("cpi_yoy")
    y = y.dropna()
    s = regime_summary(y)
    print("current regime:", s["current_regime"])
    print("current posterior:", {k: round(v, 3) for k, v in s["current_posterior"].items()})
    print("nowcast posterior:", {k: round(v, 3) for k, v in s["nowcast_posterior"].items()})
