"""
rates/consensus.py — Parts B/C/F. Consensus benchmark for the CPI surprise test.

REAL consensus (survey median per release) is licensed (Bloomberg/Reuters ECO)
or survivorship-risky (Investing.com backfills its 'consensus' column). Neither
is cleanly acquirable offline. So:

  * build_consensus_proxy()  — strongest OFFLINE-constructible consensus: a pure
    UNIVARIATE walk-forward forecast (AutoARIMA, else AR(1)) already in the CPI
    backtest. The my_nowcast - consensus gap then = the FACTOR information my
    nowcast adds over CPI's own time-series. This is a *lower bar* than a survey
    (professional forecasters >= univariate), so a PASS here is necessary, not
    sufficient; a FAIL is decisive.
  * scrape_investing_consensus() — best-effort acquisition STUB for the real
    survey median; writes data/consensus_cpi.csv [date,consensus_cpi].
  * validate_consensus() — Part C checks before use.
"""

import os
import numpy as np
import pandas as pd

from . import config as C

CONSENSUS_CPI_CSV = os.path.join(C.DATA, "consensus_cpi.csv")   # [date, consensus_cpi]
PROXY_MODELS = ["AutoARIMA", "AR(1)"]   # univariate, in priority order


# ─────────────────────────────────────────────────────────────────────────────
# Part B — build / acquire consensus
# ─────────────────────────────────────────────────────────────────────────────

def build_consensus_proxy(save=True):
    """Causal univariate consensus from the CPI backtest's walk-forward preds.
    Each value was fit on data strictly before its forecast year -> causal,
    point-in-time-safe. Returns Series [date -> consensus_cpi]."""
    bt = pd.read_csv(C.BACKTEST_CSV, parse_dates=["date"])
    w = bt.pivot_table(index="date", columns="model", values="pred", aggfunc="first")
    s = None
    for m in PROXY_MODELS:
        if m in w.columns:
            s = w[m] if s is None else s.fillna(w[m])
    if s is None:
        raise RuntimeError("no univariate model (AutoARIMA/AR(1)) in backtest for consensus proxy")
    s = s.dropna().rename("consensus_cpi"); s.index.name = "date"
    if save:
        s.to_csv(CONSENSUS_CPI_CSV)
    return s


def scrape_investing_consensus():
    """STUB — best-effort scrape of the real survey median (Investing.com UK CPI
    economic-calendar history). Anti-bot + JS + survivorship risk: treat output
    as UNVALIDATED until validate_consensus() passes. Returns Series or None."""
    try:
        import requests
        try:
            import certifi
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass
        # Investing.com loads history via an internal POST endpoint; the exact
        # eventId/payload must be filled in by hand from the network tab. Left as
        # a stub so this never silently fabricates data.
        raise NotImplementedError("fill eventId/headers from Investing.com network tab, "
                                  "or drop data/consensus_cpi.csv manually")
    except Exception as e:
        print(f"  [scrape_investing_consensus] unavailable: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Part C — validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_consensus(consensus, actual, release_dates=None, ref_dates=None):
    """Returns dict of checks. consensus/actual indexed by ref month."""
    d = pd.concat([consensus.rename("c"), actual.rename("a")], axis=1).dropna()
    e = d["a"] - d["c"]
    naive = actual.shift(1)
    en = (actual - naive).dropna()
    out = dict(
        n=int(len(d)),
        # 1) predicts CPI
        rmse_vs_actual=float(np.sqrt((e**2).mean())),
        rmse_naive_rw=float(np.sqrt((en**2).mean())),
        beats_naive=bool(np.sqrt((e**2).mean()) <= np.sqrt((en**2).mean())),
        corr_consensus_actual=float(d["c"].corr(d["a"])),
        # 4) realistic surprise distribution (mean ~ 0, finite std, not absurd)
        surprise_mean=float(e.mean()),
        surprise_std=float(e.std()),
        surprise_reasonable=bool(abs(e.mean()) < 0.5 and 0.05 < e.std() < 3.0),
    )
    # 3) available before release (ref month strictly precedes release date)
    if release_dates is not None and ref_dates is not None:
        ok = all(pd.Timestamp(rd) > pd.Timestamp(rm)
                 for rm, rd in zip(ref_dates, release_dates))
        out["available_before_release"] = bool(ok)
    out["valid"] = bool(out["beats_naive"] and out["surprise_reasonable"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Part F — subperiod robustness
# ─────────────────────────────────────────────────────────────────────────────

LEAVE_OUT = {
    "ex_2022_23":  lambda i: ~i.year.isin([2022, 2023]),
    "ex_covid":    lambda i: ~((i >= "2020-03-01") & (i <= "2021-06-30")),
    "energy_2021_23": lambda i: (i >= "2021-09-01") & (i <= "2023-06-30"),   # keep only
    "pre_2020":    lambda i: i.year < 2020,
}


def robustness(panel, stage1_fn):
    """Re-run Stage 1 on subperiods. stage1_fn(panel)->dict with b,t,verdict."""
    rows = []
    full = stage1_fn(panel); rows.append(dict(window="full", **_pick(full)))
    for name, mask in LEAVE_OUT.items():
        sub = panel[mask(panel.index)]
        if len(sub) < 40:
            rows.append(dict(window=name, verdict="TOO_SMALL", n=len(sub))); continue
        rows.append(dict(window=name, **_pick(stage1_fn(sub))))
    return pd.DataFrame(rows).set_index("window")


def _pick(r):
    return {k: r.get(k) for k in ("verdict", "n", "b", "t_b_HAC", "oos_corr",
                                  "mechanical_identity")}
