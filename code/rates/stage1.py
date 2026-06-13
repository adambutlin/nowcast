"""
rates/stage1.py — Part C (forecast gap) + Part D (Stage 1 kill test).

Gap_t      = my_nowcast_t - market_implied_t           (= panel.my_surprise when anchor=market_implied)
zGap_t     = (Gap_t - mean_{<t}Gap) / std_{<t}Gap       (expanding, STRICTLY causal: shift(1))
volGap_t   = Gap_t / std_{<t}(Gap)                       (causal vol scaling)

Stage 1:   ActualSurprise_t = a + b*zGap_t + e_t
           ActualSurprise_t = actual_cpi_t - market_implied_t   (= panel.actual_surprise)

PASS  <=>  b>0 AND in-sample HAC t>=2 AND walk-forward OOS corr>=0 AND OOS R^2>=0.
If FAIL: stop (the gap has no forecast content beyond market pricing).
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C


# ─────────────────────────────────────────────────────────────────────────────
# Part C — causal gap transforms
# ─────────────────────────────────────────────────────────────────────────────

def causal_gaps(panel):
    """Return DataFrame[gap_raw, gap_z, gap_vol] aligned to panel index, all
    using only information dated < t (expanding stats are shift(1))."""
    gap = panel["my_surprise"].astype(float)               # my_nowcast - market_implied
    mu  = gap.expanding(min_periods=12).mean().shift(1)
    sd  = gap.expanding(min_periods=12).std().shift(1)
    out = pd.DataFrame({
        "gap_raw": gap,
        "gap_z":   (gap - mu) / sd,
        "gap_vol": gap / sd,
    }, index=panel.index)
    return out


def _hac_lags(n):
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


# ─────────────────────────────────────────────────────────────────────────────
# Part D — Stage 1
# ─────────────────────────────────────────────────────────────────────────────

def stage1_test(panel, min_train=24, exclude_ldi=True, plot=True):
    p = panel.copy()
    if exclude_ldi and "ldi_event" in p:
        p = p[p["ldi_event"] != 1]
    gaps = causal_gaps(p)
    d = pd.concat([p["actual_surprise"].rename("y"), gaps["gap_z"].rename("x")],
                  axis=1).dropna()
    n = len(d)
    res = {"stage": 1, "anchor_mode": panel.attrs.get("anchor_mode"),
           "n": int(n), "verdict": "INSUFFICIENT_DATA"}
    if n < min_train + 5:
        return res
    y, x = d["y"].values, d["x"].values
    L = _hac_lags(n)

    # in-sample HAC OLS
    X = sm.add_constant(x)
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    a, b = float(fit.params[0]), float(fit.params[1])
    t_b, p_b = float(fit.tvalues[1]), float(fit.pvalues[1])

    # walk-forward OOS
    oos_pred, oos_real = [], []
    for i in range(min_train, n):
        bb = np.linalg.lstsq(X[:i], y[:i], rcond=None)[0]
        oos_pred.append(float(X[i] @ bb)); oos_real.append(float(y[i]))
    oos_pred, oos_real = np.array(oos_pred), np.array(oos_real)
    if len(oos_pred) >= 8 and np.std(oos_pred) > 1e-12:
        oos_corr = float(np.corrcoef(oos_pred, oos_real)[0, 1])
        ss_res = float(((oos_real - oos_pred) ** 2).sum())
        ss_tot = float(((oos_real - oos_real.mean()) ** 2).sum()) or np.nan
        oos_r2 = float(1 - ss_res / ss_tot) if ss_tot else np.nan
        oos_hit = float(np.mean(np.sign(oos_pred) == np.sign(oos_real)))
    else:
        oos_corr = oos_r2 = oos_hit = np.nan

    # ── MECHANICAL-IDENTITY GUARD ────────────────────────────────────────────
    # If market_implied is a slow/horizon-mismatched anchor, gap and surprise
    # both collapse to the CPI level and the test degenerates to Gate-1 forecast
    # accuracy. Detect it: (1) replace the anchor with a CONSTANT and re-fit — if
    # the slope barely changes, the market series adds nothing; (2) the gap should
    # be largely ORTHOGONAL to the forecast level for a genuine surprise.
    b_placebo, corr_placebo = _placebo_constant_anchor(p)
    rel_change = abs(b - b_placebo) / max(abs(b), 1e-9)
    gap_level_corr = float(p.loc[d.index, "my_surprise"].corr(p.loc[d.index, "my_nowcast"]))
    mechanical = (rel_change < 0.15) or (abs(gap_level_corr) > 0.70)

    passed = (b > 0 and t_b >= C.GATE2_T_THRESHOLD
              and np.isfinite(oos_corr) and oos_corr >= 0
              and np.isfinite(oos_r2) and oos_r2 >= 0
              and not mechanical)
    verdict = "INVALID_MECHANICAL" if mechanical else ("PASS" if passed else "FAIL")
    res.update(dict(verdict=verdict, hac_lags=L,
                    a=a, b=b, t_b_HAC=t_b, p_b_HAC=p_b, in_sample_r2=float(fit.rsquared),
                    oos_corr=oos_corr, oos_r2=oos_r2, oos_sign_hit=oos_hit,
                    oos_n=int(len(oos_pred)),
                    b_placebo_const_anchor=float(b_placebo),
                    slope_rel_change_vs_placebo=float(rel_change),
                    gap_vs_forecast_level_corr=gap_level_corr,
                    mechanical_identity=bool(mechanical)))
    if plot:
        _scatter(d["x"].values, d["y"].values, a, b, res)
    return res


def _placebo_constant_anchor(p):
    """Re-run Stage 1 with market_implied replaced by its (full-sample) mean —
    i.e. a CONSTANT anchor. If the slope ~matches the real one, the market
    series carries no information and the real PASS is a mechanical identity.
    Returns (slope, corr)."""
    const = float(p["market_implied_expectation"].mean())
    gap = (p["my_nowcast"] - const)
    mu = gap.expanding(min_periods=12).mean().shift(1)
    sd = gap.expanding(min_periods=12).std().shift(1)
    z = (gap - mu) / sd
    surprise = p["actual_cpi_mom"] - const
    d = pd.concat([surprise.rename("y"), z.rename("x")], axis=1).dropna()
    if len(d) < 20 or d["x"].std() < 1e-9:
        return np.nan, np.nan
    slope = float(np.polyfit(d["x"].values, d["y"].values, 1)[0])
    return slope, float(d["x"].corr(d["y"]))


def _scatter(x, y, a, b, res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(C.PLOTS, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, s=18, alpha=0.6)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax.plot(xs, a + b * xs, "r-", lw=1.5,
                label=f"b={b:.3f}  t={res['t_b_HAC']:.2f}")
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("standardized forecast gap  (my_nowcast - market_implied)")
        ax.set_ylabel("actual surprise  (actual - market_implied)")
        ax.set_title(f"Stage 1 — {res['verdict']}  (anchor={res['anchor_mode']}, n={res['n']})")
        ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(C.PLOTS, "stage1_scatter.png"), dpi=110)
        plt.close(fig)
    except Exception:
        pass
