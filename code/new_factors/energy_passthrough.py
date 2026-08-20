"""Energy → CPI pass-through study: which energy series, at which horizon and lag.

    PYTHONPATH=code:code/new_factors python code/new_factors/energy_passthrough.py

`docs/energy_fx_race.md` raced energy factors into the pinned set and got an underpowered
null. That race asked "does swapping this column change the headline RMSE" — it never asked
whether the energy factors are even *specified* correctly. They are all registered as
contemporaneous monthly log returns (`transform="logret"`, lag 0), which is the wrong shape
for the economics:

  - DIRECT   motor fuel (CPI 07.2.2, ~3% of the basket) and household energy pass through in
             weeks, and household energy is administered — it arrives as an Ofgem cap step
             already handled by the RegEvent layer, not as a price correlation.
  - INDIRECT energy is an input cost to everything transported, heated or manufactured. That
             leg shows up in output prices over quarters, not weeks.
  - EXPECTED energy price spikes are salient, move inflation expectations, and feed wage
             bargaining — the slowest leg, and the one that persists after spot reverses.

A contemporaneous 1-month log return can only see the first channel. This script measures
all three by scanning the lead-lag surface and fitting distributed-lag regressions.

IDENTIFICATION. Headline CPI mechanically contains energy, so a headline correlation mixes
the direct arithmetic effect with the indirect one. `uk_core_cpi` is ONS DKO8 — ex energy,
food, alcohol and tobacco — so energy enters it only through costs and expectations. Any
energy→core pass-through is therefore second-round by construction. `uk_services_cpi` is the
most wage-intensive aggregate and isolates the bargaining leg further.

INFERENCE. Both sides are 12-month changes sampled monthly, so residuals are autocorrelated
by construction and OLS/Pearson p-values are badly oversized. Every test here uses
Newey-West HAC with lag 12+k. Correlations are also scanned over a grid, so a single peak
is not evidence — the reported `shape` column checks whether the lag profile is a smooth
hump (economically credible) or an isolated spike (grid noise).

Out: data/new_factors/energy_passthrough_{surface,dlag,best}.csv
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import statsmodels.api as sm                              # noqa: E402

import factors as F                                       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "new_factors")
os.makedirs(_OUT, exist_ok=True)

MAX_LAG = 24
WINDOWS = [1, 3, 6, 12]        # months of cumulative log change on the energy side
START = "1997-01-31"


# ─────────────────────────────────────────────────────────────────────────────
# RAW SERIES  (factors.load_factor applies the registry transform; this study
# builds its own, so it needs the pre-transform levels with the same CSV/live
# splice semantics.)
# ─────────────────────────────────────────────────────────────────────────────
def raw_series(name):
    e = F.REGISTRY[name]
    csv_path = os.path.join(F.DATA_DIR, e["csv"]) if e.get("csv") else None
    csv_raw = None
    if csv_path and os.path.exists(csv_path):
        try:
            csv_raw = F._read_csv_raw(csv_path)
        except Exception:
            pass
    if csv_raw is not None and len(csv_raw):
        if not (e.get("csv_frozen") or e["fetch"] is None):
            try:
                live = e["fetch"]()
                fresher = live[live.index > csv_raw.index[-1]].dropna()
                if len(fresher):
                    csv_raw = pd.concat([csv_raw, fresher]).sort_index()
            except Exception:
                pass
        return csv_raw.resample("ME").last().rename(name)
    if e["fetch"] is not None:
        return e["fetch"]().resample("ME").last().rename(name)
    raise RuntimeError(f"{name}: unavailable")


def energy_panel():
    """Energy price LEVELS, month-end. Includes GBP-denominated crude and gas.

    UK pass-through runs off the sterling cost of imported energy, not the dollar
    cost: a Brent move that is offset by GBP/USD does not raise a UK firm's input
    bill. Dollar-only crude therefore mismeasures the shock whenever FX is moving,
    which is exactly when it matters. Building the GBP legs also folds the FX
    channel — the one `docs/energy_fx_race.md` found had been dropped by accident —
    into the energy factor rather than needing a separate column.
    """
    brent = raw_series("oil_brent")
    ttf = raw_series("gas_eu")
    pump = raw_series("uk_pump_price")
    usdgbp = raw_series("gbpusd")             # FRED DEXUSUK: USD per GBP
    out = {
        "brent_usd": brent,
        "brent_gbp": brent / usdgbp,
        "ttf_usd": ttf,
        "ttf_gbp": ttf / usdgbp,
        "pump_gbp": pump,                     # already pence/litre, sterling
    }
    return pd.DataFrame(out).sort_index()


def target_panel():
    names = ["cpi_yoy", "uk_core_cpi", "uk_services_cpi", "uk_be5", "uk_awg"]
    df, status = F.build_matrix(names=names)
    for n in names:
        if status.get(n) == "unavailable":
            print(f"  [warn] target {n} unavailable — dropped")
    return df, status


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def hac_corr(x, y, nw_lag):
    """Correlation plus a Newey-West HAC t-test of the slope in y ~ a + b·x.

    The correlation is scale-free and the HAC t-stat tests the same null (b=0), so
    this reports the association and honest inference on it together. Pearson's own
    p-value would be meaningless: both sides are overlapping 12-month changes.
    """
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(d) < 36:
        return np.nan, np.nan, np.nan, len(d)
    r = d["x"].corr(d["y"])
    X = sm.add_constant(d[["x"]].values)
    m = sm.OLS(d["y"].values, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(nw_lag)})
    return r, float(m.tvalues[1]), float(m.pvalues[1]), len(d)


def profile_shape(rs):
    """Classify a lag profile: a credible pass-through is a smooth hump, not a spike.

    `rs` is the correlation as a function of lag. Returns "hump" when the profile
    keeps the sign of its peak over a contiguous run of >=5 lags around the peak and
    the peak is not more than ~1.6x its neighbours, "spike" when the peak stands
    alone, "flat" when nothing reaches |r|>0.15.
    """
    v = np.asarray(rs, dtype=float)
    if not np.isfinite(v).any() or np.nanmax(np.abs(v)) < 0.15:
        return "flat"
    k = int(np.nanargmax(np.abs(v)))
    pk = v[k]
    sgn = np.sign(pk)
    run, i = 1, k - 1
    while i >= 0 and np.isfinite(v[i]) and np.sign(v[i]) == sgn and abs(v[i]) > 0.4 * abs(pk):
        run += 1; i -= 1
    i = k + 1
    while i < len(v) and np.isfinite(v[i]) and np.sign(v[i]) == sgn and abs(v[i]) > 0.4 * abs(pk):
        run += 1; i += 1
    return "hump" if run >= 5 else "spike"


# ─────────────────────────────────────────────────────────────────────────────
# A. LEAD-LAG SURFACE
# ─────────────────────────────────────────────────────────────────────────────
def surface(energy, targets, tgt_names):
    rows = []
    for ecol in energy.columns:
        lvl = np.log(energy[ecol].replace(0, np.nan))
        for w in WINDOWS:
            chg = (lvl - lvl.shift(w)).rename(f"{ecol}_{w}m")
            for tname in tgt_names:
                y = targets[tname]
                rs = []
                for k in range(MAX_LAG + 1):
                    r, t, p, n = hac_corr(chg.shift(k), y, nw_lag=12 + k)
                    rs.append(r)
                    rows.append(dict(energy=ecol, window=w, target=tname, lag=k,
                                     corr=r, hac_t=t, hac_p=p, n=n))
                sh = profile_shape(rs)
                for rr in rows[-(MAX_LAG + 1):]:
                    rr["shape"] = sh
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# B. DISTRIBUTED LAG — cumulative pass-through
# ─────────────────────────────────────────────────────────────────────────────
def dlag(energy, targets, tgt_names, max_j=18, step=3):
    """y_t ~ Σ_j b_j · Δlog(energy)_{t-j}. Σb_j is total pass-through; HAC inference.

    Monthly log changes at every third lag keeps the design matrix conditionable
    (adjacent monthly energy changes are near-collinear) while still spanning 18
    months — long enough to reach the expectations/wage leg.
    """
    rows = []
    lags = list(range(0, max_j + 1, step))
    for ecol in energy.columns:
        lvl = np.log(energy[ecol].replace(0, np.nan))
        d1 = (lvl - lvl.shift(1))
        for tname in tgt_names:
            X = pd.DataFrame({f"L{j}": d1.shift(j) for j in lags})
            d = pd.concat([targets[tname].rename("y"), X], axis=1).dropna()
            if len(d) < 60:
                continue
            m = sm.OLS(d["y"].values, sm.add_constant(d[X.columns].values)).fit(
                cov_type="HAC", cov_kwds={"maxlags": 18})
            b = m.params[1:]
            cum = float(b.sum())
            # HAC variance of the sum of coefficients: 1'V1 over the slope block.
            V = m.cov_params()[1:, 1:]
            se = float(np.sqrt(np.ones(len(b)) @ V @ np.ones(len(b))))
            t = cum / se if se > 0 else np.nan
            from scipy import stats as st
            rows.append(dict(energy=ecol, target=tname, n=len(d),
                             cum_beta=cum, cum_se=se, cum_t=t,
                             cum_p=float(2 * (1 - st.norm.cdf(abs(t)))) if se > 0 else np.nan,
                             peak_lag=int(lags[int(np.argmax(np.abs(b)))]),
                             r2=float(m.rsquared)))
    return pd.DataFrame(rows)


def main():
    print("Fetching energy levels and targets …")
    energy = energy_panel()
    energy = energy[energy.index >= START]
    targets, status = target_panel()
    tgt_names = [n for n in ["cpi_yoy", "uk_core_cpi", "uk_services_cpi", "uk_be5", "uk_awg"]
                 if status.get(n) != "unavailable"]
    targets = targets[targets.index >= START]

    print(f"\n  energy legs : {list(energy.columns)}")
    for c in energy.columns:
        s = energy[c].dropna()
        print(f"    {c:<12} n={len(s):>4}  {s.index[0].date()} … {s.index[-1].date()}")
    print(f"  targets     : {tgt_names}")
    for c in tgt_names:
        s = targets[c].dropna()
        print(f"    {c:<18} n={len(s):>4}  {s.index[0].date()} … {s.index[-1].date()}")

    print(f"\nA. Lead-lag surface (lags 0…{MAX_LAG}, windows {WINDOWS}, HAC 12+k) …")
    surf = surface(energy, targets, tgt_names)
    surf.to_csv(os.path.join(_OUT, "energy_passthrough_surface.csv"), index=False)

    print(f"\nB. Distributed lag (Σ over lags 0,3,…,18; HAC 18) …")
    dl = dlag(energy, targets, tgt_names)
    dl.to_csv(os.path.join(_OUT, "energy_passthrough_dlag.csv"), index=False)

    # ── peak per (energy, window, target) ────────────────────────────────────
    best = (surf.dropna(subset=["corr"])
                .assign(abscorr=lambda d: d["corr"].abs())
                .sort_values("abscorr", ascending=False)
                .groupby(["energy", "window", "target"], as_index=False).first())
    best.to_csv(os.path.join(_OUT, "energy_passthrough_best.csv"), index=False)

    for tname in tgt_names:
        sub = best[best.target == tname].sort_values("abscorr", ascending=False)
        print(f"\n  ── peak |corr| vs {tname} ─────────────────────────────────")
        print(f"  {'energy':<11}{'win':>4}{'lag':>5}{'corr':>8}{'HAC t':>8}{'HAC p':>8}"
              f"{'n':>6}  shape")
        for _, r in sub.iterrows():
            print(f"  {r.energy:<11}{int(r.window):>4}{int(r.lag):>5}{r['corr']:>8.3f}"
                  f"{r.hac_t:>8.2f}{r.hac_p:>8.3f}{int(r.n):>6}  {r['shape']}")

    print(f"\n  ── cumulative pass-through Σb (distributed lag) ───────────────")
    print(f"  {'energy':<11}{'target':<18}{'Σb':>9}{'HAC t':>8}{'p':>8}{'peak lag':>9}{'R²':>7}{'n':>6}")
    for _, r in dl.sort_values(["target", "cum_t"], ascending=[True, False]).iterrows():
        print(f"  {r.energy:<11}{r.target:<18}{r.cum_beta:>9.3f}{r.cum_t:>8.2f}"
              f"{r.cum_p:>8.3f}{int(r.peak_lag):>9}{r.r2:>7.3f}{int(r.n):>6}")

    print(f"\n  wrote data/new_factors/energy_passthrough_{{surface,dlag,best}}.csv")
    return surf, dl, best


if __name__ == "__main__":
    main()
