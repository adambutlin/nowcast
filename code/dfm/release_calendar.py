"""
dfm/release_calendar.py — when did each observation actually become public?

The repo's existing convention is an integer `pub_lag` in months
(factors.apply_publication_lags) plus, in code/timing/nowcast_window.py, a
per-series offset in days. Neither is the real calendar. This module builds a
dated table so a month-T value enters the information set only at the timestamp
it was genuinely published.

Sources, in precedence order (recorded per row in the `source` column):
  ons_api      real ONS release page  (https://www.ons.gov.uk/releases/<slug>/data)
  provider_doc documented publication offset for a non-ONS provider
  rule         fallback only: 3rd Wednesday of the following month

The ONS release slug is NOT deterministic across eras — verified 2026-08-19:
  2016-01  ukconsumerpriceindicesjan2016          (abbreviated month, old stem)
  2019-06  ukconsumerpriceinflationjune2019       (full month, mid stem)
  2022-03  consumerpriceinflationukmarch2022      (modern stem)
  pre-2016 not published as a release page at all -> rule fallback
so several slug patterns are tried per month, newest first.

The rule is materially wrong: 2022-03 published 13 Apr against a rule date of
20 Apr, and 2026-06 published 22 Jul against 15 Jul. The rule can land BEFORE
the true release (2026-06: rule=15 Jul, actual=22 Jul) — a silent fallback there
is not merely imprecise, it is look-ahead bias, since it would admit a value
into the information set before it was genuinely published. So `ons_release_date`
only ever caches a *confirmed* answer (a real timestamp, or a confirmed 404);
anything else — timeout, DNS blip, non-404 HTTP error, malformed JSON, a 200
with no release_date — raises loudly instead of degrading silently to `rule`.
Fallback share is therefore always visible, never silently absorbed.
"""
import calendar
import json
import urllib.error
import urllib.request

import pandas as pd

from dfm import config as C

_UA = {"User-Agent": "Mozilla/5.0 (UK CPI nowcast research; contact via repo)"}
_RELEASE_URL = "https://www.ons.gov.uk/releases/{slug}/data"
_MEMO = {}

# Empirically, ONS release pages exist from 2016-01 onward; 2015-06 and earlier
# 404 on every slug pattern (verified 2026-08-19). Months before this are known
# absences, not lookups — do not hammer the ONS site to rediscover that.
_ONS_RELEASE_PAGES_FROM = pd.Timestamp("2016-01-01")

# Non-ONS publication offsets, in days after the end of the reference month.
# Extends the offsets already encoded in code/timing/nowcast_window.py.
PROVIDER_OFFSET_DAYS = {
    # market / financial closes — the month-end value is known at month-end
    "oil_brent": 0, "gas_eu": 0, "gas_hh": 0, "gbpusd": 0, "gbp_eur": 0,
    "uk_gilt_10y": 0, "uk_be5": 1, "slope_2s10s": 1, "vix": 0, "move_index": 0,
    "oil_vol_6m": 0, "copper_price": 0, "timber_price": 0, "wheat_price": 0,
    "vegetable_oil_price": 0, "metals_index": 0,
    # IMF primary commodity prices — published early the following month
    "imf_all_commodity": 7, "food_price_index": 7,
    # US BLS PPI-derived series — mid the following month
    "deep_sea_freight": 13, "semiconductors_ppi": 13, "chemicals_ppi": 13,
    # BIS effective exchange rates
    "gbp_eer": 14,
    # administered / scheduled events — announced, so known at month-end
    "mpc_rate_change": 0, "ofgem_cap_delta": 0, "budget_event": 0,
}

# ONS series: their release is looked up per reference month. Series published in
# the CPI bulletin share its date; the others carry their own documented offset
# until a per-series lookup is added (Plan 4).
ONS_CPI_BULLETIN = {"cpi_index", "cpi_yoy", "uk_core_cpi", "uk_services_cpi"}
ONS_OTHER_OFFSET_DAYS = {
    "uk_ppi_input": 16, "uk_ppi_output": 16,     # PPI, published with CPI
    "uk_quarterly_gdp": 40,                       # first quarterly estimate
    "uk_retail_sales": 21, "uk_vacancies": 42,
    "uk_awg": 42, "uk_rents": 16, "uk_house_prices": 47,
}


def slug_candidates(ref_month):
    """ONS release-page slugs to try for a reference month, newest pattern first."""
    m = pd.Timestamp(ref_month)
    full = calendar.month_name[m.month].lower()
    abbr = calendar.month_abbr[m.month].lower()
    year = m.year
    stems = ["consumerpriceinflationuk", "ukconsumerpriceinflation",
             "ukconsumerpriceindices", "consumerpriceinflation"]
    out = []
    for stem in stems:
        for mm in (full, abbr):
            slug = f"{stem}{mm}{year}"
            if slug not in out:
                out.append(slug)
    return out


def _fetch_release_page(slug):
    """Return the release timestamp for an ONS release slug.

    Returns None ONLY for a confirmed HTTP 404 (the release page genuinely
    does not exist). Everything else that keeps us from getting a confirmed
    answer — a non-404 HTTP error, a connection/timeout error, malformed
    JSON, or a 200 response missing description.release_date — raises a
    RuntimeError naming the slug. A transient failure must never be mistaken
    for a confirmed absence: doing so would let `ons_release_date` fall back
    to `rule`, which can predate the true release and admit data early.
    """
    req = urllib.request.Request(_RELEASE_URL.format(slug=slug), headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"ONS fetch failed for slug '{slug}': HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"ONS fetch failed for slug '{slug}': {e}") from e

    try:
        payload = json.loads(body)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"ONS fetch failed for slug '{slug}': malformed JSON") from e

    rd = (payload.get("description") or {}).get("release_date")
    if not rd:
        raise RuntimeError(
            f"ONS fetch failed for slug '{slug}': 200 response missing "
            f"description.release_date")
    return pd.Timestamp(rd).tz_localize(None)


def ons_release_date(ref_month):
    """Actual ONS CPI release timestamp for `ref_month`, or None if unpublished.

    Only ever caches a confirmed answer. A transient fetch failure raises
    (see `_fetch_release_page`) and is NOT written to `_MEMO`, so a failed
    month is retried cleanly on the next run rather than being poisoned to
    None for the process lifetime.
    """
    key = pd.Timestamp(ref_month).to_period("M")
    if key in _MEMO:
        return _MEMO[key]
    if pd.Timestamp(ref_month) < _ONS_RELEASE_PAGES_FROM:
        # Known absence, not a lookup: every slug pattern 404s pre-2016.
        _MEMO[key] = None
        return None
    found = None
    for slug in slug_candidates(ref_month):
        found = _fetch_release_page(slug)
        if found is not None:
            break
    _MEMO[key] = found
    return found


def rule_release_date(ref_month):
    """Fallback: 3rd Wednesday of the month after the reference month."""
    nxt = pd.Timestamp(ref_month) + pd.offsets.MonthBegin(1)
    first = pd.Timestamp(nxt.year, nxt.month, 1)
    # 0=Mon .. 2=Wed
    delta = (2 - first.weekday()) % 7
    return first + pd.Timedelta(days=delta + 14)


def resolve_cpi_release(ref_month):
    """(timestamp, source) for the CPI bulletin covering `ref_month`."""
    ts = ons_release_date(ref_month)
    if ts is not None:
        return ts, "ons_api"
    return rule_release_date(ref_month), "rule"
