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
20 Apr, and 2026-06 published 22 Jul against 15 Jul. Fallback share is therefore
reported, never silently absorbed.
"""
import calendar
import json
import os
import urllib.request

import pandas as pd

from dfm import config as C

_UA = {"User-Agent": "Mozilla/5.0 (UK CPI nowcast research; contact via repo)"}
_RELEASE_URL = "https://www.ons.gov.uk/releases/{slug}/data"
_MEMO = {}

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
    """Return the release timestamp for an ONS release slug, or None if 404."""
    req = urllib.request.Request(_RELEASE_URL.format(slug=slug), headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            payload = json.load(r)
    except Exception:
        return None
    rd = (payload.get("description") or {}).get("release_date")
    if not rd:
        return None
    return pd.Timestamp(rd).tz_localize(None)


def ons_release_date(ref_month):
    """Actual ONS CPI release timestamp for `ref_month`, or None if unpublished."""
    key = pd.Timestamp(ref_month).to_period("M")
    if key in _MEMO:
        return _MEMO[key]
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
