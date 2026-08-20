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
import time
import urllib.error
import urllib.request

import pandas as pd

from dfm import config as C

_UA = {"User-Agent": "Mozilla/5.0 (UK CPI nowcast research; contact via repo)"}
_RELEASE_URL = "https://www.ons.gov.uk/releases/{slug}/data"
_MEMO = {}

# Politeness and resilience against the ONS rate limiter. The build makes a few
# hundred requests against a real public site; without spacing it trips HTTP 429
# partway through (observed 2026-08-19 on 'consumerpriceinflationukaugust2019').
_REQUEST_SPACING_SECONDS = 0.35
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_RETRY_BACKOFF_SECONDS = [5.0, 15.0, 45.0]
_RETRY_MAX_WAIT_SECONDS = 120.0

# The slug stem that most recently resolved, as (stem, uses_full_month_name).
# ONS naming is stable within an era, so trying last month's winning pattern
# first collapses the usual 4-8 requests per month down to 1.
_LAST_GOOD = None

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


# ONS is not consistent even within a stem: September 2018 was published as
# "sept2018", not the calendar.month_abbr "sep2018" (found 2026-08-20 while
# chasing the single mid-sample month that fell back to the rule). Extra
# spellings are tried after the standard two.
_EXTRA_MONTH_SPELLINGS = {9: ["sept"]}


def slug_candidates(ref_month):
    """ONS release-page slugs to try for a reference month, newest pattern first."""
    m = pd.Timestamp(ref_month)
    full = calendar.month_name[m.month].lower()
    abbr = calendar.month_abbr[m.month].lower()
    year = m.year
    spellings = [full, abbr] + _EXTRA_MONTH_SPELLINGS.get(m.month, [])
    stems = ["consumerpriceinflationuk", "ukconsumerpriceinflation",
             "ukconsumerpriceindices", "consumerpriceinflation"]
    out = []
    for stem in stems:
        for mm in spellings:
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
    body = None
    for attempt in range(len(_RETRY_BACKOFF_SECONDS) + 1):
        time.sleep(_REQUEST_SPACING_SECONDS)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                body = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            # 429/503 are the site asking us to slow down, not a verdict on
            # whether this release exists. Back off and retry rather than
            # failing the whole build — but still raise if it never clears,
            # because a silent `rule` fallback can predate the true release.
            if e.code in _RETRYABLE_STATUS and attempt < len(_RETRY_BACKOFF_SECONDS):
                wait = e.headers.get("Retry-After") if e.headers else None
                try:
                    wait = float(wait)
                except (TypeError, ValueError):
                    wait = _RETRY_BACKOFF_SECONDS[attempt]
                time.sleep(min(wait, _RETRY_MAX_WAIT_SECONDS))
                continue
            raise RuntimeError(f"ONS fetch failed for slug '{slug}': HTTP {e.code}") from e
        except urllib.error.URLError as e:
            # Not retried: a connection/DNS/timeout failure is an environment
            # problem, not the site throttling us. Fail fast and loudly so the
            # operator fixes it and reruns, rather than sleeping through it.
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
    for slug in _ordered_candidates(ref_month):
        found = _fetch_release_page(slug)
        if found is not None:
            _remember_stem(slug, ref_month)
            break
    _MEMO[key] = found
    return found


def _stem_of(slug, ref_month):
    """Strip the month name and year off a slug, leaving the stem."""
    m = pd.Timestamp(ref_month)
    for mm in (calendar.month_name[m.month].lower(), calendar.month_abbr[m.month].lower()):
        suffix = f"{mm}{m.year}"
        if slug.endswith(suffix):
            return slug[: -len(suffix)], mm
    return slug, None


def _remember_stem(slug, ref_month):
    """Cache the (stem, month-name style) that just worked."""
    global _LAST_GOOD
    stem, mm = _stem_of(slug, ref_month)
    if mm is not None:
        _LAST_GOOD = (stem, mm == calendar.month_name[pd.Timestamp(ref_month).month].lower())


def _ordered_candidates(ref_month):
    """Slug candidates, with the pattern that last worked tried first.

    ONS slug style changes by era but is locally stable, so the stem that
    resolved the previous month almost always resolves this one. Without this
    the build issues up to 8 requests per month against a real third-party
    site and reliably trips its rate limiter.
    """
    cands = slug_candidates(ref_month)
    if _LAST_GOOD is None:
        return cands
    stem, use_full = _LAST_GOOD
    m = pd.Timestamp(ref_month)
    mm = (calendar.month_name if use_full else calendar.month_abbr)[m.month].lower()
    preferred = f"{stem}{mm}{m.year}"
    if preferred in cands:
        cands = [preferred] + [c for c in cands if c != preferred]
    return cands


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


def series_release(series, ref_month):
    """(timestamp, source) for one series' observation of one reference month."""
    if series in ONS_CPI_BULLETIN:
        return resolve_cpi_release(ref_month)
    end = pd.Timestamp(ref_month) + pd.offsets.MonthEnd(0)
    if series in ONS_OTHER_OFFSET_DAYS:
        return end + pd.Timedelta(days=ONS_OTHER_OFFSET_DAYS[series]), "provider_doc"
    if series in PROVIDER_OFFSET_DAYS:
        return end + pd.Timedelta(days=PROVIDER_OFFSET_DAYS[series]), "provider_doc"
    raise KeyError(f"no publication rule for series {series!r} — add it to "
                   f"PROVIDER_OFFSET_DAYS or ONS_OTHER_OFFSET_DAYS")


def build_calendar(series_names, months):
    """Dated release table: one row per (series, reference month)."""
    rows = []
    for name in series_names:
        for m in months:
            end = pd.Timestamp(m) + pd.offsets.MonthEnd(0)
            ts, src = series_release(name, end)
            rows.append(dict(series=name, ref_period=end, release_ts=ts, source=src))
    cal = pd.DataFrame(rows, columns=["series", "ref_period", "release_ts", "source"])
    bad = cal[cal["release_ts"] < cal["ref_period"]]
    if len(bad):
        raise ValueError(f"{len(bad)} rows publish before their reference period ends:\n{bad.head()}")
    return cal


def save_calendar(cal, path=None):
    path = path or C.CALENDAR_CSV
    cal.to_csv(path, index=False)
    return path


def load_calendar(path=None):
    path = path or C.CALENDAR_CSV
    return pd.read_csv(path, parse_dates=["ref_period", "release_ts"])


def coverage_report(cal):
    """Rows and share by source. The `rule` share is the honesty number."""
    n = cal.groupby("source").size().rename("n")
    rep = n.to_frame()
    rep["share"] = rep["n"] / len(cal)
    for src in ("ons_api", "provider_doc", "rule"):
        if src not in rep.index:
            rep.loc[src] = [0, 0.0]
    rep["n"] = rep["n"].astype(int)
    return rep.sort_index()


if __name__ == "__main__":
    from dfm import panel as P   # series list

    months = pd.date_range(f"{C.EVAL_START}-01-31", "2026-07-31", freq="ME")
    cal = build_calendar(P.all_series(), months)
    path = save_calendar(cal)
    rep = coverage_report(cal)
    print(f"wrote {path}  ({len(cal)} rows)")
    print(rep.to_string())
    cpi = cal[cal["series"] == "cpi_index"]
    n_rule = int((cpi["source"] == "rule").sum())
    print(f"\nCPI bulletin: {len(cpi) - n_rule}/{len(cpi)} real ONS dates, "
          f"{n_rule} rule fallbacks")
