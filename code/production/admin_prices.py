"""Administered-price layer — the Ofgem energy price cap, applied as arithmetic.

WHY THIS IS NOT A FACTOR. `ofgem_cap_delta` has been in the PINNED set since the
freeze and contributes nothing: the column is zero in ~93% of months, so LightGBM
never pays for a split on it (gain 0.0%) and the TVP dilutes it across eight
standardised regressors. Meanwhile the AutoARIMA residual in cap-reset months has
RMSE 0.83 against 0.32 in every other month — the model's error is concentrated
almost entirely in the months this layer covers (docs/postmortem_2026_07.md).

A quarterly step in a regulated price is not something to learn. It is announced
~6 weeks before it takes effect (so it is legitimately available at month-end T),
it lands in one month, and its size in CPI terms is arithmetic:

    AdminStep(T) = kappa * beta * cap_pct_change(T)          [pp of CPI YoY]

`cap_pct_change` is the announced change in the cap; `beta` converts 1% of cap
into pp of CPI, estimated through the origin on PRIOR reset months only; `kappa`
is a governance haircut in the spirit of lambda. Adding a step to the level is
valid because CPI_YoY(T) = CPI_YoY(T-1) + (MoM_T - MoM_{T-12}): the anchor knows
the base month, and the cap reset is precisely the part of MoM_T it cannot know.

Beta is NOT 1/1. The cap governs standard variable tariffs; CPI prices all
tariffs (fixed deals reprice later) and standing charges move separately, so
historically ~1.8pp of CPI per 100% of cap.

Walk-forward over reset months, beta re-estimated from prior resets only:
RMSE 0.730 -> 0.671 (rel 0.918) at MIN_PRIOR=8, kappa=0.75; on the longer window
(MIN_PRIOR=4) rel 0.824. Better in 7/11 and 10/15 months respectively. Governance
entry: docs/final_model.md §11.
"""
import os
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(_ROOT, "data")
CAP_CSV = os.path.join(DATA, "ofgem_cap.csv")
CACHE = os.path.join(DATA, "admin_prices")

# Cap resets take effect on the 1st of these months (semi-annual pre-Oct-2022,
# quarterly since). A cap row dated elsewhere is an ANNOUNCEMENT, not a reset.
RESET_MONTHS = (1, 4, 7, 10)
KAPPA = 0.75          # governance haircut on the arithmetic step
MIN_PRIOR = 8         # resets required before beta is trusted at all
MIN_PCT = 0.05        # |cap change| below this is a rounding artefact, not a reset

# ONS MM23 series: CPI component indices and their annual weights (per 1000).
_COMPONENTS = {"elec": ("d7dt", "CJXA.A"), "gas": ("d7du", "CJXB.A")}


def _cached(name, fetch):
    """Fetch, cache to data/admin_prices/<name>.csv, fall back to the cache."""
    path = os.path.join(CACHE, f"{name}.csv")
    try:
        s = fetch()
        if len(s):
            os.makedirs(CACHE, exist_ok=True)
            s.rename("value").to_frame().to_csv(path, index_label="date")
            return s
    except Exception as e:
        print(f"  [admin_prices] {name} fetch failed ({e}) — using cache")
    if os.path.exists(path):
        d = pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"]
        return d
    return pd.Series(dtype=float)


def cap_level():
    """Ofgem default-tariff cap, £/yr for a typical dual-fuel direct-debit household.

    Kept on a SINGLE consumption basis: Ofgem re-based its Typical Domestic
    Consumption Values in July 2026 (the same cap prints as £1,862 old-basis or
    £1,663 new), and a level series that switches basis mid-stream produces a
    fictitious step. Only ratios of this series are ever used.
    """
    s = pd.read_csv(CAP_CSV, parse_dates=["date"]).set_index("date")["value"]
    return s.sort_index().resample("ME").last()


def cap_pct_change(level=None):
    """Announced % change in the cap, indexed by the month it takes effect."""
    s = cap_level() if level is None else level
    pct = s.pct_change().mul(100).replace([np.inf, -np.inf], np.nan).dropna()
    return pct[pct.index.month.isin(RESET_MONTHS) & (pct.abs() > MIN_PCT)]


def realised_contribution():
    """Realised CPI contribution (pp) of household electricity + gas, by month.

    weight/1000 * MoM% for each fuel — what the cap step actually did to CPI.
    """
    out = None
    for fuel, (idx_code, weight_code) in _COMPONENTS.items():
        idx = _cached(f"cpi_{fuel}_index", lambda c=idx_code: _ons(c))
        wts = _cached(f"cpi_{fuel}_weight", lambda c=weight_code: _weights(c))
        if not len(idx) or not len(wts):
            return pd.Series(dtype=float)
        mom = idx.pct_change().mul(100)
        w = pd.Series(wts.values, index=wts.index.year).groupby(level=0).last()
        w_month = pd.Series(mom.index.year, index=mom.index).map(w).ffill()
        c = (w_month / 1000) * mom
        out = c if out is None else out.add(c, fill_value=0.0)
    return out.dropna()


def _ons(code):
    import factors as F
    return F._ons_timeseries(code, "economy/inflationandpriceindices")


def _weights(code):
    """Annual CPI weight series (per 1000), dated to the year it applies from."""
    from dbnomics import fetch_series
    d = fetch_series("ONS", "MM23", code).dropna(subset=["value"])
    idx = pd.to_datetime([f"{p}-01-01" for p in d["original_period"]])
    return pd.Series(d["value"].values, index=idx).sort_index()


def passthrough(before=None, pct=None, contrib=None):
    """beta: pp of CPI per 1% cap change, least squares through the origin.

    `before` restricts the fit to resets STRICTLY earlier than that month, which
    is what makes a walk-forward evaluation honest. Returns (beta, n_used).
    """
    pct = cap_pct_change() if pct is None else pct
    contrib = realised_contribution() if contrib is None else contrib
    joined = pd.DataFrame({"pct": pct}).join(contrib.rename("contrib"), how="inner").dropna()
    if before is not None:
        joined = joined[joined.index < pd.Timestamp(before)]
    if len(joined) == 0:
        return np.nan, 0
    x, y = joined["pct"].values, joined["contrib"].values
    return float((x * y).sum() / (x * x).sum()), len(joined)


def admin_step(nd, kappa=KAPPA, min_prior=MIN_PRIOR, pct=None, contrib=None):
    """The administered-price adjustment for month `nd`, in pp of CPI YoY.

    Zero unless `nd` is a cap-reset month with an announced change and beta has
    enough prior resets to be worth trusting. Returns (step, detail-dict) so the
    caller can log exactly why it fired or did not.
    """
    nd = pd.Timestamp(nd)
    pct = cap_pct_change() if pct is None else pct
    cap_pct = float(pct[nd]) if nd in pct.index else 0.0
    beta, n = passthrough(before=nd, pct=pct, contrib=contrib)
    detail = dict(cap_pct=round(cap_pct, 3), beta=None if beta != beta else round(beta, 5),
                  n_prior_resets=n, kappa=kappa)
    if cap_pct == 0.0:
        detail["reason"] = "no cap reset announced for this month"
        return 0.0, detail
    if n < min_prior or beta != beta:
        detail["reason"] = f"beta not trusted ({n} prior resets < {min_prior})"
        return 0.0, detail
    step = kappa * beta * cap_pct
    detail["reason"] = "applied"
    return float(step), detail


# ── keeping the cap series current ───────────────────────────────────────────

OFGEM_RATES_URL = ("https://www.ofgem.gov.uk/information-consumers/energy-advice-households/"
                   "energy-price-cap-unit-rates-and-standing-charges")
# Fixed consumption basis for turning unit rates into a comparable bill. The
# absolute numbers are irrelevant — only the quarter-on-quarter RATIO is used, so
# this is immune to Ofgem's TDCV revisions.
TDCV_KWH = dict(elec=2700, gas=11500)


def fetch_current_cap_rates(url=OFGEM_RATES_URL):
    """Scrape Ofgem's unit-rate table: the current and previous quarter.

    Returns {'periods': [...], 'elec': [(unit_p, standing_p), ...], 'gas': [...]}
    with two entries each, oldest first. Ofgem publishes the cap ~6 weeks ahead,
    so the current quarter is on this page before it takes effect.
    """
    import re
    import requests
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = re.search(r"<table.*?</table>", html, re.S)
    if not table:
        raise RuntimeError("Ofgem unit-rate table not found — page layout changed")
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", table.group(0)).replace("&nbsp;", " "))
    periods = re.findall(r"(\d+ \w+ to \d+ \w+ \d{4})", text)
    units = [float(v) for v in re.findall(r"([\d.]+) pence per kWh", text)]
    standing = [float(v) for v in re.findall(r"([\d.]+) pence daily standing charge", text)]
    if len(units) < 4 or len(standing) < 4:
        raise RuntimeError(f"Ofgem table parsed {len(units)} unit rates, expected 4")
    # Table order: electricity (prev, curr), gas (prev, curr).
    return dict(periods=periods[:2], elec=list(zip(units[0:2], standing[0:2])),
                gas=list(zip(units[2:4], standing[2:4])))


def implied_cap_pct(rates=None):
    """Quarter-on-quarter % change in a fixed-consumption bill from unit rates.

    The basis-stable way to extend the cap series: bills are rebuilt at a fixed
    TDCV, so an Ofgem consumption revision cannot masquerade as a price change.
    """
    rates = fetch_current_cap_rates() if rates is None else rates
    prev, curr = _bill(rates, 0), _bill(rates, 1)
    return (curr / prev - 1) * 100, dict(periods=rates["periods"],
                                         prev_bill=round(prev, 2), curr_bill=round(curr, 2))


def _bill(rates, i):
    return sum(rates[f][i][0] * TDCV_KWH[f] / 100 + rates[f][i][1] * 365 / 100
               for f in ("elec", "gas"))


_MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}


def _period_start(label):
    """'1 July to 30 September 2026' -> Timestamp('2026-07-31') (month-end index)."""
    import re
    m = re.match(r"\d+ (\w+) to \d+ \w+ (\d{4})", label.strip())
    if not m:
        raise ValueError(f"unparsable Ofgem period label: {label!r}")
    return (pd.Timestamp(int(m.group(2)), _MONTHS[m.group(1)], 1) + pd.offsets.MonthEnd(0))


def refresh_cap_csv(path=CAP_CSV, rates=None):
    """Append any cap quarter Ofgem has published that the CSV does not have yet.

    The cap used to be a hand-maintained file, so it silently ended at the last
    month somebody remembered to add — that is how a +13% reset entered the
    July-2026 nowcast as a zero. Rows are written as a fixed-TDCV bill rebuilt
    from the published unit rates, which keeps the series on one consumption
    basis across Ofgem's TDCV revisions. Returns a list of (month, value) added.
    """
    rates = fetch_current_cap_rates() if rates is None else rates
    s = cap_level()
    added = []
    for i, label in enumerate(rates["periods"]):
        month = _period_start(label)
        if month in s.index and pd.notna(s.get(month)):
            continue
        value = round(_bill(rates, i), 2)
        with open(path, "a") as fh:
            fh.write(f"{month.date()},{value}\n")
        added.append((str(month.date()), value))
    return added
