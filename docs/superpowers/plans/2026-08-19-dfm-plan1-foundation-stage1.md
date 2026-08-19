# Mixed-Frequency Bayesian DFM — Plan 1: Foundation and Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the causal data foundation (dated release calendar, as-of panel) and Stage 1 — a classical Stock-Watson mixed-frequency dynamic factor model with ragged-edge Kalman filtering — scored on a purged walk-forward against AutoARIMA, AR(1) and a replication of the frozen production model.

**Architecture:** A new, self-contained `code/dfm/` package. Data flows: `release_calendar.py` (when was each value published) + `daily.py` (raw daily series) → `panel.py` (as-of observation matrix, ragged) → `state_space.py` (sequential univariate Kalman filter + Durbin-Koopman simulation smoother) → `sampler.py` (Gibbs blocks) → `model.py` (`DFMNowcaster`) → `backtest.py` (walk-forward, purge/embargo, checkpointed) → `experiments.py` (arm runner). `benchmarks.py` wraps AA / AR(1) / production. Nothing in `code/production/`, `code/factors.py`, `code/targets.py`, `code/new_factors/two_stage.py` or `code/uk_model_zoo.py` is modified — they are imported read-only, and a test enforces it.

**Tech Stack:** Python 3.11 in the `quant` conda env (`/opt/anaconda3/envs/quant/bin/python`) — numpy 2.4.2, numba 0.64.0, pandas 2.3.3, scipy 1.17.0, statsmodels 0.14.6, arviz 0.23.4, pytest. No JAX/Stan/numpyro.

**Spec:** [docs/superpowers/specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md](../specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md)

**Branch:** `dfm-research` (already created, off `main`).

---

## Environment rules — read before running anything

From `docs/CLAUDE.md`, these are not optional:

- **Only** `/opt/anaconda3/envs/quant/bin/python`. Never `.venv`, never conda `base` — `import pandas` takes 10–16 minutes there and looks like a hang.
- **Never run Python processes in parallel** on this machine. They serialise on the macOS security scanner and each import balloons to minutes.
- Any command that fetches from the network (FRED, dbnomics, ONS, yfinance) needs `dangerouslyDisableSandbox: true` on a **foreground** Bash call. Background-launched fetches get re-sandboxed and hang.

Throughout this plan, `$Q` means `/opt/anaconda3/envs/quant/bin/python` and every run command assumes `PYTHONPATH=code` from the repo root `/Users/Adam/Documents/home/quant/nowcast`.

Set it up once per shell:

```bash
cd /Users/Adam/Documents/home/quant/nowcast
export Q=/opt/anaconda3/envs/quant/bin/python
set -a; . ./.env; set +a          # FRED_API_KEY
```

---

## File Structure

| File | Responsibility |
|---|---|
| `code/dfm/__init__.py` | package marker, version constant |
| `code/dfm/config.py` | paths, walk-forward windows, `K`, draw counts, origin offsets — single source of truth |
| `code/dfm/release_calendar.py` | ONS release-date resolver, provider offsets, calendar assembly, coverage report |
| `code/dfm/daily.py` | raw **daily** series fetch + on-disk cache (the registry discards intraday resolution) |
| `code/dfm/panel.py` | series specs, block map, as-of observation matrix, standardization |
| `code/dfm/target.py` | CPI index D7BT, MoM transform, MoM→YoY reconstruction |
| `code/dfm/state_space.py` | sequential univariate Kalman filter + Durbin-Koopman simulation smoother (numba) |
| `code/dfm/sampler.py` | Gibbs blocks: loadings, idiosyncratic AR + variance, factor AR |
| `code/dfm/model.py` | `DFMConfig`, `DFMNowcaster.fit/nowcast` |
| `code/dfm/backtest.py` | walk-forward harness, purge/embargo, checkpoint + resume |
| `code/dfm/benchmarks.py` | AutoARIMA / AR(1) / frozen-production replication on this harness |
| `code/dfm/experiments.py` | timing probe, arm runner, results CSV |
| `code/tests/test_dfm_isolation.py` | production-untouched gate |
| `code/tests/test_dfm_calendar.py` | release calendar invariants |
| `code/tests/test_dfm_panel.py` | as-of panel leakage tests |
| `code/tests/test_dfm_target.py` | YoY reconstruction identity |
| `code/tests/test_dfm_state_space.py` | filter/smoother correctness on synthetic data |
| `code/tests/test_dfm_sampler.py` | per-block parameter recovery |
| `code/tests/test_dfm_backtest.py` | walk-forward causality + checkpoint resume |

Outputs: `data/dfm/` (calendar, daily cache, checkpoints, results), `plots/dfm/`, `docs/dfm_backtest.md`.

---

## Task 1: Package scaffold and the production-isolation gate

**Files:**
- Create: `code/dfm/__init__.py`, `code/dfm/config.py`
- Test: `code/tests/test_dfm_isolation.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_isolation.py`:

```python
"""
Production-isolation gate for the DFM research branch.

The frozen production model must be byte-identical to `main` on this branch.
Spec: docs/superpowers/specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md (Non-goal)
"""
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FROZEN_PATHS = [
    "code/production",
    "code/factors.py",
    "code/targets.py",
    "code/new_factors/two_stage.py",
    "code/uk_model_zoo.py",
]


def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args],
                          capture_output=True, text=True, timeout=60)


def test_frozen_production_untouched():
    """No diff against main on any frozen production path."""
    if _git("rev-parse", "--verify", "main").returncode != 0:
        pytest.skip("no local main branch to diff against")
    out = _git("diff", "main", "--stat", "--", *FROZEN_PATHS)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        "frozen production paths differ from main:\n" + out.stdout)


def test_dfm_package_imports():
    """The dfm package imports without pulling in production side effects."""
    import dfm
    from dfm import config

    assert dfm.__version__
    assert config.K_FACTORS == 7
    assert config.EVAL_START == 2015 and config.EVAL_END == 2024
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_isolation.py -q
```

Expected: `test_frozen_production_untouched` PASSES (nothing changed yet), `test_dfm_package_imports` FAILS with `ModuleNotFoundError: No module named 'dfm'`.

- [ ] **Step 3: Write the minimal implementation**

Create `code/dfm/__init__.py`:

```python
"""
dfm — mixed-frequency Bayesian dynamic factor model for UK CPI (research branch).

This package is a RESEARCH ALTERNATIVE to the frozen production model. It imports
code/production, code/factors.py, code/targets.py and code/uk_model_zoo.py
READ-ONLY, for benchmarking, and must never modify them
(enforced by code/tests/test_dfm_isolation.py).

Spec: docs/superpowers/specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md
"""

__version__ = "0.1.0"
```

Create `code/dfm/config.py`:

```python
"""
dfm/config.py — single source of truth for paths, windows and model dimensions.

Everything downstream reads from here so the whole pipeline changes behaviour by
editing this file, never by hard-coding in a module.
"""
import os

_THIS = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(_THIS)
ROOT = os.path.dirname(CODE)

DATA = os.path.join(ROOT, "data", "dfm")
PLOTS = os.path.join(ROOT, "plots", "dfm")
DAILY_CACHE = os.path.join(DATA, "daily")
CHECKPOINTS = os.path.join(DATA, "checkpoints")
for _d in (DATA, PLOTS, DAILY_CACHE, CHECKPOINTS):
    os.makedirs(_d, exist_ok=True)

CALENDAR_CSV = os.path.join(DATA, "release_calendar.csv")

# ── model dimensions ─────────────────────────────────────────────────────────
K_FACTORS = 7          # 1 pervasive `global` + 6 block factors; fixed across all arms
FACTOR_ORDER = 2       # lags in the factor VAR
IDIO_ORDER = 1         # idiosyncratic AR(1), carried in the state vector

# ── walk-forward windows (match production exactly) ──────────────────────────
PANEL_START = 1997     # earliest panel month
EVAL_START = 2015      # first scored year
EVAL_END = 2024        # last scored year; 2025+ is blind
BLIND_FROM = 2025      # never used to fit or tune
HOLDOUT_MONTHS = ["2026-05", "2026-06", "2026-07"]   # live scored months, held out

# Purge + embargo. The overlay label (cpi_yoy residual) is a 12-month difference;
# the standalone label (CPI MoM) is a 1-month difference. See spec §6.4.
PURGE_OVERLAY, EMBARGO_OVERLAY = 12, 1
PURGE_STANDALONE, EMBARGO_STANDALONE = 1, 1

# ── MCMC ─────────────────────────────────────────────────────────────────────
DRAWS_FIRST, BURN_FIRST = 2000, 1000     # first walk-forward fit
DRAWS_WARM, BURN_WARM = 800, 200         # warm-started fits thereafter
SEED = 20260819

# ── forecast path origins, in days relative to reference month-end ───────────
# Negative = before month-end. Positive = after, inside the release window.
# "R-1" is resolved per month from the actual release date (release_calendar).
ORIGIN_OFFSETS = [-30, -21, -14, -7, 0, 5, 10]
ORIGIN_LABELS = ["T-30", "T-21", "T-14", "T-7", "T", "T+5", "T+10", "R-1"]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_isolation.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add code/dfm/__init__.py code/dfm/config.py code/tests/test_dfm_isolation.py
git commit -m "feat(dfm): package scaffold, config, production-isolation gate"
```

---

## Task 2: ONS release-date resolver

The ONS release page slug is **not** deterministic across eras. Verified 2026-08-19:

| ref month | working slug | release date |
|---|---|---|
| 2016-01 | `ukconsumerpriceindicesjan2016` | 2016-02-16 (a **Tuesday**) |
| 2019-06 | `ukconsumerpriceinflationjune2019` | 2019-07-17 |
| 2022-03 | `consumerpriceinflationukmarch2022` | 2022-04-13 |
| 2024-11 | `consumerpriceinflationuknovember2024` | 2024-12-18 |
| 2026-06 | `consumerpriceinflationukjune2026` | 2026-07-22 |
| 2013-01, 2015-06 | none of the patterns | **404 — pre-2016 unavailable** |

The 3rd-Wednesday rule is wrong by up to 7 days (2022-03 actual 13 Apr vs rule 20 Apr; 2026-06 actual 22 Jul vs rule 15 Jul), which is why this task exists.

**Files:**
- Create: `code/dfm/release_calendar.py`
- Test: `code/tests/test_dfm_calendar.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_calendar.py`:

```python
"""
Release-calendar resolver: slug patterns, caching, rule fallback.

Network tests are marked `network` and skipped by default; the offline tests use
a monkeypatched fetcher so the logic is testable with no egress.
"""
import sys, os
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import release_calendar as RC


class TestSlugCandidates:
    def test_generates_known_working_slugs(self):
        """Every empirically-verified slug appears among the candidates."""
        assert "ukconsumerpriceindicesjan2016" in RC.slug_candidates(pd.Timestamp("2016-01-31"))
        assert "ukconsumerpriceinflationjune2019" in RC.slug_candidates(pd.Timestamp("2019-06-30"))
        assert "consumerpriceinflationukmarch2022" in RC.slug_candidates(pd.Timestamp("2022-03-31"))
        assert "consumerpriceinflationuknovember2024" in RC.slug_candidates(pd.Timestamp("2024-11-30"))

    def test_newest_pattern_is_tried_first(self):
        """Most releases are recent, so the modern stem should be candidate 0."""
        cands = RC.slug_candidates(pd.Timestamp("2024-11-30"))
        assert cands[0] == "consumerpriceinflationuknovember2024"


class TestRuleFallback:
    def test_third_wednesday(self):
        """2024-11 reference month -> 3rd Wednesday of Dec 2024 = 18 Dec."""
        assert RC.rule_release_date(pd.Timestamp("2024-11-30")) == pd.Timestamp("2024-12-18")

    def test_rule_is_always_a_wednesday(self):
        for m in pd.date_range("2015-01-31", "2026-06-30", freq="ME"):
            assert RC.rule_release_date(m).weekday() == 2


class TestOnsReleaseDate:
    def test_uses_first_slug_that_resolves(self, monkeypatch):
        seen = []

        def fake(slug):
            seen.append(slug)
            return pd.Timestamp("2024-12-18 07:00:00") if slug.startswith("consumerpriceinflationuk") else None

        monkeypatch.setattr(RC, "_fetch_release_page", fake)
        RC._MEMO.clear()
        ts = RC.ons_release_date(pd.Timestamp("2024-11-30"))
        assert ts == pd.Timestamp("2024-12-18 07:00:00")
        assert len(seen) == 1, "should stop at the first slug that resolves"

    def test_returns_none_when_no_slug_resolves(self, monkeypatch):
        monkeypatch.setattr(RC, "_fetch_release_page", lambda slug: None)
        RC._MEMO.clear()
        assert RC.ons_release_date(pd.Timestamp("2013-01-31")) is None

    def test_memoises(self, monkeypatch):
        calls = []
        monkeypatch.setattr(RC, "_fetch_release_page",
                            lambda slug: calls.append(slug) or pd.Timestamp("2024-12-18"))
        RC._MEMO.clear()
        RC.ons_release_date(pd.Timestamp("2024-11-30"))
        RC.ons_release_date(pd.Timestamp("2024-11-30"))
        assert len(calls) == 1


class TestResolve:
    def test_prefers_api_and_labels_source(self, monkeypatch):
        monkeypatch.setattr(RC, "ons_release_date", lambda m: pd.Timestamp("2024-12-18 07:00"))
        ts, src = RC.resolve_cpi_release(pd.Timestamp("2024-11-30"))
        assert src == "ons_api" and ts == pd.Timestamp("2024-12-18 07:00")

    def test_falls_back_to_rule_and_labels_it(self, monkeypatch):
        monkeypatch.setattr(RC, "ons_release_date", lambda m: None)
        ts, src = RC.resolve_cpi_release(pd.Timestamp("2013-01-31"))
        assert src == "rule" and ts == pd.Timestamp("2013-02-20")


@pytest.mark.network
def test_live_ons_lookup():
    """Real ONS fetch — run with `-m network`."""
    RC._MEMO.clear()
    assert RC.ons_release_date(pd.Timestamp("2024-11-30")).date() == pd.Timestamp("2024-12-18").date()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_calendar.py -q -m "not network"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dfm.release_calendar'`.

- [ ] **Step 3: Write the implementation**

Create `code/dfm/release_calendar.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_calendar.py -q -m "not network"
```

Expected: `9 passed`.

Then confirm the live path once, with network:

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_calendar.py -q -m network
```

Expected: `1 passed`. This call needs `dangerouslyDisableSandbox: true`.

- [ ] **Step 5: Register the `network` and `slow` markers**

The repo has no `pytest.ini`, `setup.cfg` or `pyproject.toml`, so adding one would move pytest's rootdir for the 124 existing tests. Register the markers in the existing `code/tests/conftest.py` instead — append to it, keeping the two `sys.path` lines already there:

```python

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "network: test performs a live network fetch "
                   "(deselect with '-m \"not network\"')")
    config.addinivalue_line(
        "markers", "slow: test runs a full walk-forward "
                   "(deselect with '-m \"not slow\"')")
```

- [ ] **Step 6: Commit**

```bash
git add code/dfm/release_calendar.py code/tests/test_dfm_calendar.py code/tests/conftest.py
git commit -m "feat(dfm): ONS release-date resolver with multi-era slug patterns and rule fallback"
```

---

## Task 3: Assemble and cache the full release calendar

**Files:**
- Modify: `code/dfm/release_calendar.py` (add `build_calendar`, `load_calendar`, `coverage_report`)
- Test: `code/tests/test_dfm_calendar.py` (append `TestBuildCalendar`)

- [ ] **Step 1: Write the failing test**

Append to `code/tests/test_dfm_calendar.py`:

```python
class TestBuildCalendar:
    def _months(self):
        return pd.date_range("2024-01-31", "2024-06-30", freq="ME")

    def test_every_row_has_timestamp_and_source(self, monkeypatch):
        monkeypatch.setattr(RC, "ons_release_date", lambda m: pd.Timestamp(m) + pd.Timedelta(days=18))
        cal = RC.build_calendar(["cpi_index", "oil_brent", "imf_all_commodity"], self._months())
        assert set(cal.columns) == {"series", "ref_period", "release_ts", "source"}
        assert cal["release_ts"].notna().all()
        assert cal["source"].isin({"ons_api", "provider_doc", "rule"}).all()
        assert len(cal) == 3 * 6

    def test_release_never_precedes_end_of_reference_period(self, monkeypatch):
        monkeypatch.setattr(RC, "ons_release_date", lambda m: pd.Timestamp(m) + pd.Timedelta(days=18))
        cal = RC.build_calendar(["cpi_index", "oil_brent", "uk_ppi_input"], self._months())
        assert (cal["release_ts"] >= cal["ref_period"]).all()

    def test_market_series_release_at_month_end(self, monkeypatch):
        monkeypatch.setattr(RC, "ons_release_date", lambda m: pd.Timestamp(m) + pd.Timedelta(days=18))
        cal = RC.build_calendar(["oil_brent"], self._months())
        assert (cal["release_ts"] == cal["ref_period"]).all()
        assert (cal["source"] == "provider_doc").all()

    def test_coverage_report_counts_rule_fallbacks(self, monkeypatch):
        # ONS lookup fails -> every cpi_index row falls back to the rule
        monkeypatch.setattr(RC, "ons_release_date", lambda m: None)
        cal = RC.build_calendar(["cpi_index", "oil_brent"], self._months())
        rep = RC.coverage_report(cal)
        assert rep.loc["rule", "n"] == 6
        assert rep.loc["provider_doc", "n"] == 6
        assert abs(rep.loc["rule", "share"] - 0.5) < 1e-9

    def test_roundtrip_csv(self, monkeypatch, tmp_path):
        monkeypatch.setattr(RC, "ons_release_date", lambda m: pd.Timestamp(m) + pd.Timedelta(days=18))
        path = tmp_path / "cal.csv"
        cal = RC.build_calendar(["cpi_index", "oil_brent"], self._months())
        RC.save_calendar(cal, str(path))
        back = RC.load_calendar(str(path))
        pd.testing.assert_frame_equal(cal.reset_index(drop=True), back.reset_index(drop=True))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_calendar.py::TestBuildCalendar -q
```

Expected: FAIL — `AttributeError: module 'dfm.release_calendar' has no attribute 'build_calendar'`.

- [ ] **Step 3: Write the implementation**

Append to `code/dfm/release_calendar.py`:

```python
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
    return rep.sort_index()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_calendar.py -q -m "not network"
```

Expected: `14 passed`.

- [ ] **Step 5: Build the real calendar and commit it**

This fetches ~140 ONS release pages, one per reference month 2015-01 → 2026-07. It is slow (~3–5 min) and needs `dangerouslyDisableSandbox: true`. Add a `__main__` block to `code/dfm/release_calendar.py`:

```python
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
```

**This step depends on Task 5 (`panel.all_series`) and is executed at the end of Task 5, not here.** Commit the module now:

```bash
git add code/dfm/release_calendar.py code/tests/test_dfm_calendar.py
git commit -m "feat(dfm): assemble, cache and coverage-report the dated release calendar"
```

---

## Task 4: Raw daily series store

`factors.py` resamples to month-end **inside** every fetcher (`_fred`, `_yf`, `_dbnomics` all end in `.resample("ME").last()`), so intraday resolution is gone by the time a factor is loaded. The as-of panel needs raw daily values, and `factors.py` may not be modified — so `code/dfm/daily.py` fetches and caches its own.

**Files:**
- Create: `code/dfm/daily.py`
- Test: `code/tests/test_dfm_daily.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_daily.py`:

```python
"""Raw daily series store: caching, partial-month aggregation."""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import daily as D


@pytest.fixture
def series():
    idx = pd.date_range("2024-03-01", "2024-03-31", freq="D")
    return pd.Series(np.arange(1.0, len(idx) + 1.0), index=idx, name="oil_brent")


class TestPartialMonth:
    def test_level_takes_last_value_on_or_before_asof(self, series):
        got = D.partial_month(series, pd.Timestamp("2024-03-31"),
                              asof=pd.Timestamp("2024-03-10"), how="level")
        assert got == 10.0

    def test_logret_sums_within_month_only(self, series):
        """Return is measured against the last value of the PRIOR month."""
        prior = pd.Series([100.0], index=[pd.Timestamp("2024-02-29")])
        s = pd.concat([prior, series])
        got = D.partial_month(s, pd.Timestamp("2024-03-31"),
                              asof=pd.Timestamp("2024-03-05"), how="logret")
        assert got == pytest.approx(np.log(5.0 / 100.0))

    def test_returns_nan_when_no_observation_by_asof(self, series):
        got = D.partial_month(series, pd.Timestamp("2024-03-31"),
                              asof=pd.Timestamp("2024-02-28"), how="level")
        assert np.isnan(got)

    def test_future_days_do_not_leak(self, series):
        """Injecting values after asof must not change the answer."""
        before = D.partial_month(series, pd.Timestamp("2024-03-31"),
                                 asof=pd.Timestamp("2024-03-10"), how="level")
        poisoned = series.copy()
        poisoned.loc[pd.Timestamp("2024-03-20")] = 9999.0
        after = D.partial_month(poisoned, pd.Timestamp("2024-03-31"),
                                asof=pd.Timestamp("2024-03-10"), how="level")
        assert before == after


class TestCache:
    def test_writes_and_reads_cache(self, tmp_path, monkeypatch, series):
        monkeypatch.setattr(D.C, "DAILY_CACHE", str(tmp_path))
        calls = []
        monkeypatch.setitem(D.FETCHERS, "oil_brent", lambda: calls.append(1) or series)
        first = D.load_daily("oil_brent")
        second = D.load_daily("oil_brent")
        assert len(calls) == 1, "second load must hit the cache"
        pd.testing.assert_series_equal(first, second)

    def test_force_refetches(self, tmp_path, monkeypatch, series):
        monkeypatch.setattr(D.C, "DAILY_CACHE", str(tmp_path))
        calls = []
        monkeypatch.setitem(D.FETCHERS, "oil_brent", lambda: calls.append(1) or series)
        D.load_daily("oil_brent")
        D.load_daily("oil_brent", force=True)
        assert len(calls) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_daily.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dfm.daily'`.

- [ ] **Step 3: Write the implementation**

Create `code/dfm/daily.py`:

```python
"""
dfm/daily.py — raw DAILY series, cached to disk.

factors.py resamples to month-end inside every fetcher, so the registry cannot
supply intraday resolution and must not be modified (production isolation). This
module fetches the daily series the as-of panel needs and caches each one to
data/dfm/daily/<name>.csv so a multi-hour sweep is reproducible offline.

partial_month() is the mixed-frequency aggregator: given daily observations and
an as-of timestamp, it returns the month's value USING ONLY days <= asof.
Levels take the last value in the window; returns are measured against the last
value of the prior month. Days after asof are invisible by construction.
"""
import os

import numpy as np
import pandas as pd

from dfm import config as C

LONG_START = "1989-01-01"


def _fred_daily(series_id):
    from fredapi import Fred
    s = Fred(api_key=os.getenv("FRED_API_KEY")).get_series(
        series_id, observation_start=LONG_START)
    s.index = pd.to_datetime(s.index)
    return s.dropna().sort_index()


def _yf_daily(ticker):
    import yfinance as yf
    raw = yf.download(ticker, start=LONG_START, auto_adjust=True, progress=False)
    c = (raw[("Close", ticker)] if isinstance(raw.columns, pd.MultiIndex) else raw["Close"])
    return c.dropna().sort_index()


# Daily-resolution members of the panel. Everything else is natively monthly or
# lower and is read through factors.build_matrix in panel.py.
FETCHERS = {
    "oil_brent":    lambda: _fred_daily("DCOILBRENTEU"),
    "gas_hh":       lambda: _fred_daily("DHHNGSP"),
    "gbpusd":       lambda: _fred_daily("DEXUSUK"),
    "uk_gilt_10y":  lambda: _fred_daily("IRLTLT01GBM156N"),
    "vix":          lambda: _yf_daily("^VIX"),
    "gas_eu":       lambda: _yf_daily("TTF=F"),
    "gbp_eur":      lambda: _yf_daily("GBPEUR=X"),
    "copper_price": lambda: _yf_daily("HG=F"),
    "wheat_price":  lambda: _yf_daily("ZW=F"),
}


def _cache_path(name):
    return os.path.join(C.DAILY_CACHE, f"{name}.csv")


def load_daily(name, force=False):
    """Daily series for `name`, from cache when available."""
    path = _cache_path(name)
    if not force and os.path.exists(path):
        s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"]
        return s.rename(name).sort_index()
    if name not in FETCHERS:
        raise KeyError(f"{name!r} has no daily fetcher — add one to dfm.daily.FETCHERS")
    s = FETCHERS[name]().rename(name)
    s.index.name = "date"
    s.rename("value").to_frame().to_csv(path)
    return s.sort_index()


def partial_month(series, ref_month, asof, how="level"):
    """
    Month value for `ref_month` using only observations dated <= `asof`.

    how="level"  -> last value inside [month_start, asof]
    how="logret" -> log(last value inside window / last value of the prior month)

    Returns NaN when the window holds no observation (the cell is then MISSING in
    the panel and the Kalman filter simply skips its update).
    """
    end = pd.Timestamp(ref_month) + pd.offsets.MonthEnd(0)
    start = end.replace(day=1)
    asof = pd.Timestamp(asof)
    win = series[(series.index >= start) & (series.index <= min(asof, end))].dropna()
    if len(win) == 0:
        return np.nan
    last = float(win.iloc[-1])
    if how == "level":
        return last
    if how == "logret":
        prior = series[series.index < start].dropna()
        if len(prior) == 0 or float(prior.iloc[-1]) <= 0 or last <= 0:
            return np.nan
        return float(np.log(last / float(prior.iloc[-1])))
    raise ValueError(f"unknown aggregation {how!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_daily.py -q
```

Expected: `6 passed`.

- [ ] **Step 5: Warm the cache** (network; `dangerouslyDisableSandbox: true`)

```bash
PYTHONPATH=code $Q -c "
from dfm import daily as D
for n in D.FETCHERS:
    try:
        s = D.load_daily(n, force=True)
        print(f'{n:14s} n={len(s):6d}  {s.index[0].date()} -> {s.index[-1].date()}')
    except Exception as e:
        print(f'{n:14s} FAILED {type(e).__name__}: {str(e)[:60]}')
"
```

Expected: every series prints a row count and a date range. Record any failure — a series that cannot be fetched daily falls back to its monthly registry value in Task 5 and must be listed in the model card.

- [ ] **Step 6: Commit**

```bash
git add code/dfm/daily.py code/tests/test_dfm_daily.py
git commit -m "feat(dfm): raw daily series store with as-of partial-month aggregation"
```

Do **not** commit `data/dfm/daily/*.csv` — add `data/dfm/daily/` to `.gitignore` (cache, refetchable).

---

## Task 5: Panel — series specs and block map

**Files:**
- Create: `code/dfm/panel.py`
- Test: `code/tests/test_dfm_panel.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_panel.py`:

```python
"""Panel specs and block map (as-of matrix is tested in Task 6)."""
import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import panel as P
from dfm import config as C


class TestSpecs:
    def test_thirty_two_series_plus_reg_block(self):
        assert len(P.PANEL_SERIES) == 32
        assert len(P.REG_SERIES) == 3

    def test_every_series_has_a_block_and_a_transform(self):
        for name, spec in P.PANEL_SERIES.items():
            assert spec["block"] in P.BLOCKS, f"{name} has unknown block {spec['block']}"
            assert spec["transform"] in {"level", "logret", "yoy", "diff"}
            assert spec["freq"] in {"daily", "monthly", "quarterly"}

    def test_block_sizes_are_three_to_eight(self):
        for block, members in P.block_members().items():
            assert 3 <= len(members) <= 8, f"{block} has {len(members)} members"

    def test_six_blocks_plus_global_equals_k_factors(self):
        assert len(P.BLOCKS) + 1 == C.K_FACTORS

    def test_all_series_includes_target_and_reg(self):
        allser = P.all_series()
        assert "cpi_index" in allser
        assert set(P.REG_SERIES).issubset(allser)
        assert len(allser) == 32 + 3 + 1

    def test_daily_series_are_marked_daily(self):
        from dfm import daily as D
        for name, spec in P.PANEL_SERIES.items():
            if name in D.FETCHERS:
                assert spec["freq"] == "daily", f"{name} has a daily fetcher but freq={spec['freq']}"

    def test_quarterly_series_is_gdp_only(self):
        q = [n for n, s in P.PANEL_SERIES.items() if s["freq"] == "quarterly"]
        assert q == ["uk_quarterly_gdp"]

    def test_no_lseg_series_in_headline_panel(self):
        lseg = {"uk_pmi_services", "uk_pmi_composite", "uk_infl_swap_1y", "uk_corp_profits"}
        assert not (set(P.PANEL_SERIES) & lseg)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_panel.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dfm.panel'`.

- [ ] **Step 3: Write the implementation**

Create `code/dfm/panel.py`:

```python
"""
dfm/panel.py — the mixed-frequency observation panel.

32 series in 6 economic blocks (spec §1.6), plus CPI as MoM log-change and a
3-series observed exogenous reg block. Every series loads on the pervasive
`global` factor; `block` names the second factor it loads on under the
structured-loading arms (E5/E6).

`freq` decides how a month's value is formed at an as-of timestamp:
  daily      partial-month aggregation over days <= asof   (dfm.daily)
  monthly    the whole month, visible only once published  (release calendar)
  quarterly  Mariano-Murasawa 3-month aggregation; intra-quarter months missing

No LSEG series appear here: a multi-hour sweep must not depend on the Workspace
desktop app being open (spec §1.7).
"""
import numpy as np
import pandas as pd

import factors as F
from dfm import config as C
from dfm import daily as D
from dfm import release_calendar as RC

BLOCKS = ["activity", "nominal", "energy", "supply", "fx", "food"]

# name -> block, transform, frequency
PANEL_SERIES = {
    # ── activity (7) ─────────────────────────────────────────────────────────
    "uk_quarterly_gdp":  dict(block="activity", transform="yoy",    freq="quarterly"),
    "uk_retail_sales":   dict(block="activity", transform="yoy",    freq="monthly"),
    "uk_vacancies":      dict(block="activity", transform="yoy",    freq="monthly"),
    "uk_awg":            dict(block="activity", transform="yoy",    freq="monthly"),
    "uk_house_prices":   dict(block="activity", transform="yoy",    freq="monthly"),
    "vix":               dict(block="activity", transform="level",  freq="daily"),
    "move_index":        dict(block="activity", transform="level",  freq="monthly"),
    # ── nominal (8) ──────────────────────────────────────────────────────────
    "uk_ppi_output":     dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_ppi_input":      dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_core_cpi":       dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_services_cpi":   dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_rents":          dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_be5":            dict(block="nominal",  transform="level",  freq="monthly"),
    "uk_gilt_10y":       dict(block="nominal",  transform="diff",   freq="daily"),
    "slope_2s10s":       dict(block="nominal",  transform="level",  freq="monthly"),
    # ── energy (5) ───────────────────────────────────────────────────────────
    "oil_brent":         dict(block="energy",   transform="logret", freq="daily"),
    "gas_eu":            dict(block="energy",   transform="logret", freq="daily"),
    "gas_hh":            dict(block="energy",   transform="logret", freq="daily"),
    "imf_all_commodity": dict(block="energy",   transform="logret", freq="monthly"),
    "oil_vol_6m":        dict(block="energy",   transform="level",  freq="monthly"),
    # ── supply (6) ───────────────────────────────────────────────────────────
    "deep_sea_freight":  dict(block="supply",   transform="logret", freq="monthly"),
    "semiconductors_ppi":dict(block="supply",   transform="logret", freq="monthly"),
    "chemicals_ppi":     dict(block="supply",   transform="logret", freq="monthly"),
    "metals_index":      dict(block="supply",   transform="logret", freq="monthly"),
    "copper_price":      dict(block="supply",   transform="logret", freq="daily"),
    "timber_price":      dict(block="supply",   transform="logret", freq="monthly"),
    # ── fx (3) ───────────────────────────────────────────────────────────────
    "gbp_eer":           dict(block="fx",       transform="diff",   freq="monthly"),
    "gbpusd":            dict(block="fx",       transform="logret", freq="daily"),
    "gbp_eur":           dict(block="fx",       transform="logret", freq="daily"),
    # ── food (3) — the only genuinely new channel vs production ──────────────
    "food_price_index":  dict(block="food",     transform="logret", freq="monthly"),
    "wheat_price":       dict(block="food",     transform="logret", freq="daily"),
    "vegetable_oil_price": dict(block="food",   transform="logret", freq="monthly"),
}

# Observed exogenous shifters. Sparse, mostly-zero administered events cannot
# identify a latent factor, so they enter the measurement equation directly.
REG_SERIES = ["mpc_rate_change", "ofgem_cap_delta", "budget_event"]

TARGET_SERIES = "cpi_index"


def block_members():
    """block -> list of series names loading on that block factor."""
    out = {b: [] for b in BLOCKS}
    for name, spec in PANEL_SERIES.items():
        out[spec["block"]].append(name)
    return out


def all_series():
    """Every series the release calendar must cover."""
    return [TARGET_SERIES] + list(PANEL_SERIES) + list(REG_SERIES)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_panel.py -q
```

Expected: `8 passed`. If `test_block_sizes_are_three_to_eight` or the count test fails, fix the dict — do not weaken the test.

- [ ] **Step 5: Verify every panel series is actually fetchable** (network; `dangerouslyDisableSandbox: true`)

```bash
PYTHONPATH=code $Q -c "
import factors as F
from dfm import panel as P
names = [n for n in P.PANEL_SERIES if n in F.REGISTRY] + [n for n in P.REG_SERIES if n in F.REGISTRY]
missing = [n for n in list(P.PANEL_SERIES) + P.REG_SERIES if n not in F.REGISTRY]
print('NOT IN REGISTRY:', missing)
df, status = F.build_matrix(names=names)
dead = [n for n, s in status.items() if s == 'unavailable']
print('UNAVAILABLE:', dead)
print('matrix', df.shape, df.index.min().date(), '->', df.index.max().date())
" 2>&1 | tail -20
```

Expected: `NOT IN REGISTRY: []` and `UNAVAILABLE: []`. **If any series is missing or unavailable, stop** — either find a registry equivalent or drop it and rebalance the block so it still has ≥3 members, then update the spec's §1.6 table in the same commit.

- [ ] **Step 6: Build and commit the real release calendar**

Now that `panel.all_series()` exists, run the Task 3 `__main__` (network; `dangerouslyDisableSandbox: true`, ~3–5 min):

```bash
PYTHONPATH=code $Q -u -m dfm.release_calendar
```

Expected: a coverage table plus a line like `CPI bulletin: 128/139 real ONS dates, 11 rule fallbacks`. Pre-2016 months fall back to the rule — that is known and reported, not a failure.

- [ ] **Step 7: Commit**

```bash
git add code/dfm/panel.py code/tests/test_dfm_panel.py data/dfm/release_calendar.csv
git commit -m "feat(dfm): panel series specs, 6-block map, and the built release calendar"
```

---

## Task 6: Panel — the as-of observation matrix

**Files:**
- Modify: `code/dfm/panel.py` (add `asof`, `_monthly_matrix`, `standardize`)
- Test: `code/tests/test_dfm_panel.py` (append `TestAsOf`)

- [ ] **Step 1: Write the failing test**

Append to `code/tests/test_dfm_panel.py`:

```python
import numpy as np
import pandas as pd


@pytest.fixture
def fake_panel(monkeypatch):
    """A 3-series panel with a hand-built calendar, no network."""
    months = pd.date_range("2023-01-31", "2024-06-30", freq="ME")
    monthly = pd.DataFrame({
        "uk_ppi_input": np.linspace(1.0, 2.0, len(months)),
        "imf_all_commodity": np.linspace(-0.5, 0.5, len(months)),
    }, index=months)

    daily_idx = pd.date_range("2022-12-01", "2024-06-30", freq="D")
    brent = pd.Series(np.linspace(50.0, 90.0, len(daily_idx)), index=daily_idx)

    cal = pd.concat([
        P.RC.build_calendar(["uk_ppi_input", "imf_all_commodity", "oil_brent"], months),
    ], ignore_index=True)

    monkeypatch.setattr(P, "_monthly_matrix", lambda names, start=None: monthly)
    monkeypatch.setattr(P.D, "load_daily", lambda name, force=False: brent)
    monkeypatch.setattr(P, "load_calendar_cached", lambda: cal)
    specs = {
        "uk_ppi_input": dict(block="nominal", transform="yoy", freq="monthly"),
        "imf_all_commodity": dict(block="energy", transform="logret", freq="monthly"),
        "oil_brent": dict(block="energy", transform="logret", freq="daily"),
    }
    monkeypatch.setattr(P, "PANEL_SERIES", specs)
    monkeypatch.setattr(P, "REG_SERIES", [])
    return months


class TestAsOf:
    def test_returns_one_row_per_month_and_one_column_per_series(self, fake_panel):
        m = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        assert list(m.columns) == ["uk_ppi_input", "imf_all_commodity", "oil_brent"]
        assert m.index[-1] == pd.Timestamp("2024-06-30")

    def test_unpublished_month_is_missing(self, fake_panel):
        """uk_ppi_input for month T publishes T+16d, so at T it is not yet out."""
        m = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        assert np.isnan(m.loc[pd.Timestamp("2024-06-30"), "uk_ppi_input"])
        assert not np.isnan(m.loc[pd.Timestamp("2024-05-31"), "uk_ppi_input"])

    def test_market_series_is_available_at_month_end(self, fake_panel):
        m = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        assert not np.isnan(m.loc[pd.Timestamp("2024-06-30"), "oil_brent"])

    def test_earlier_origin_has_no_more_information(self, fake_panel):
        late = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        early = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-09"))
        assert early.notna().sum().sum() <= late.notna().sum().sum()

    def test_no_future_leak(self, fake_panel, monkeypatch):
        """Poisoning daily values dated after asof must not change a single cell."""
        base = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-10"))
        orig = P.D.load_daily("oil_brent")

        def poisoned(name, force=False):
            s = orig.copy()
            s[s.index > pd.Timestamp("2024-06-10")] = 1e6
            return s

        monkeypatch.setattr(P.D, "load_daily", poisoned)
        P._CACHE.clear()
        after = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-10"))
        pd.testing.assert_frame_equal(base, after)

    def test_standardize_is_causal(self, fake_panel):
        """Means and sds come from the training rows only, never the test row."""
        m = P.asof(pd.Timestamp("2024-06-30"), pd.Timestamp("2024-06-30"))
        z, mu, sd = P.standardize(m, through=pd.Timestamp("2024-03-31"))
        train = m.loc[:pd.Timestamp("2024-03-31")]
        assert mu["imf_all_commodity"] == pytest.approx(train["imf_all_commodity"].mean())
        assert sd["imf_all_commodity"] == pytest.approx(train["imf_all_commodity"].std())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_panel.py::TestAsOf -q
```

Expected: FAIL — `AttributeError: module 'dfm.panel' has no attribute 'asof'`.

- [ ] **Step 3: Write the implementation**

Append to `code/dfm/panel.py`:

```python
_CACHE = {}


def load_calendar_cached():
    """The dated release table, loaded once per process."""
    if "cal" not in _CACHE:
        _CACHE["cal"] = RC.load_calendar()
    return _CACHE["cal"]


def _monthly_matrix(names, start=None):
    """Month-end matrix for the natively-monthly/quarterly series (raw, NO pub-lag).

    Publication timing is applied by `asof` from the dated calendar, so the
    integer pub_lag shift must NOT also be applied here — that would double-count
    the delay.
    """
    key = ("monthly", tuple(sorted(names)))
    if key in _CACHE:
        return _CACHE[key]
    raw, status = F.build_matrix(names=list(names))
    dead = [n for n, s in status.items() if s == "unavailable"]
    if dead:
        raise RuntimeError(f"panel series unavailable: {dead}")
    raw = raw[raw.index.year >= (start or C.PANEL_START)].resample("ME").last()
    _CACHE[key] = raw
    return raw


def _release_lookup():
    key = "rel"
    if key not in _CACHE:
        cal = load_calendar_cached()
        _CACHE[key] = {(r.series, pd.Timestamp(r.ref_period).to_period("M")): r.release_ts
                       for r in cal.itertuples()}
    return _CACHE[key]


def _released_by(series, ref_month, asof_ts):
    """Has `series`' observation of `ref_month` been published by `asof_ts`?"""
    ts = _release_lookup().get((series, pd.Timestamp(ref_month).to_period("M")))
    if ts is None:
        # No calendar row (e.g. a month outside the built range): fall back to the
        # documented offset rather than silently admitting the observation.
        ts, _ = RC.series_release(series, ref_month)
    return pd.Timestamp(ts) <= pd.Timestamp(asof_ts)


def asof(ref_month, asof_ts, start=None):
    """
    Observation matrix visible at `asof_ts`, for months up to and including
    `ref_month`. Unpublished cells are NaN — the Kalman filter skips them.

    Daily series are aggregated over days <= asof_ts within their month
    (dfm.daily.partial_month). Quarterly series carry their value in the last
    month of the quarter, with intra-quarter months NaN (Mariano-Murasawa).
    """
    ref_month = pd.Timestamp(ref_month) + pd.offsets.MonthEnd(0)
    asof_ts = pd.Timestamp(asof_ts)
    names = list(PANEL_SERIES) + list(REG_SERIES)

    monthly_names = [n for n in names if PANEL_SERIES.get(n, {}).get("freq") != "daily"]
    monthly = _monthly_matrix(monthly_names, start=start)
    index = monthly.index[monthly.index <= ref_month]

    cols = {}
    for name in names:
        spec = PANEL_SERIES.get(name, dict(freq="monthly", transform="level"))
        if spec["freq"] == "daily":
            series = D.load_daily(name)
            how = "level" if spec["transform"] in ("level", "diff") else "logret"
            vals = [D.partial_month(series, m, asof_ts, how=how) for m in index]
            col = pd.Series(vals, index=index, dtype=float)
            if spec["transform"] == "diff":
                col = col.diff()
        else:
            col = monthly[name].reindex(index).astype(float) if name in monthly.columns \
                else pd.Series(np.nan, index=index)
        # publication mask — the whole point of the module
        mask = [_released_by(name, m, asof_ts) for m in index]
        col = col.where(pd.Series(mask, index=index), np.nan)
        cols[name] = col

    out = pd.DataFrame(cols, index=index)

    # Reg series are event dummies: NaN inside their observed span means "no event"
    # (0), but NOT past their last observation — that is how a +13% Ofgem cap reset
    # entered the July-2026 nowcast as a 0 (docs/postmortem_2026_07.md §3).
    for rf in REG_SERIES:
        if rf in out.columns and rf in monthly.columns:
            observed = monthly[rf].dropna()
            if len(observed):
                span = out.index <= observed.index[-1]
                out.loc[span, rf] = out.loc[span, rf].fillna(0.0)

    # Quarterly: keep the value only in the quarter's final month.
    for name, spec in PANEL_SERIES.items():
        if spec["freq"] == "quarterly" and name in out.columns:
            keep = out.index.month.isin([3, 6, 9, 12])
            out.loc[~keep, name] = np.nan
    return out


def standardize(mat, through):
    """Z-score with means/sds computed on rows <= `through` only (causal)."""
    train = mat.loc[:pd.Timestamp(through)]
    mu = train.mean()
    sd = train.std().replace(0.0, 1.0).fillna(1.0)
    return (mat - mu) / sd, mu, sd


def staleness_report(asof_ts):
    """Series whose last observation is >6 months before `asof_ts`."""
    mat = asof(asof_ts, asof_ts)
    rows = []
    for name in mat.columns:
        obs = mat[name].dropna()
        last = obs.index[-1] if len(obs) else pd.NaT
        stale = np.nan if pd.isna(last) else (pd.Timestamp(asof_ts).to_period("M")
                                              - last.to_period("M")).n
        rows.append(dict(series=name, last_obs=last, months_stale=stale,
                         status="DEAD" if (pd.isna(stale) or stale > 6) else "LIVE"))
    return pd.DataFrame(rows).set_index("series")
```

Add `from dfm import release_calendar as RC` is already at the top; also add `RC` and `D` re-exports used by the tests — they are already module attributes via the imports.

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_panel.py -q
```

Expected: `14 passed`.

- [ ] **Step 5: Sanity-check the real panel** (network; `dangerouslyDisableSandbox: true`)

```bash
PYTHONPATH=code $Q -c "
import pandas as pd
from dfm import panel as P
m = P.asof(pd.Timestamp('2026-06-30'), pd.Timestamp('2026-06-30'))
print('shape', m.shape)
print('non-missing by series (last 24 months):')
print(m.tail(24).notna().sum().sort_values().to_string())
print()
print(P.staleness_report(pd.Timestamp('2026-06-30')).query(\"status=='DEAD'\").to_string())
"
```

Expected: a 32+3 column matrix; the staleness report may list `uk_house_prices` and similar — that is known (spec §1.7) and handled by the filter, but record it.

- [ ] **Step 6: Commit**

```bash
git add code/dfm/panel.py code/tests/test_dfm_panel.py
git commit -m "feat(dfm): as-of observation matrix with dated publication masking and leakage tests"
```

---

## Task 7: Target — CPI index, MoM, and YoY reconstruction

**Files:**
- Create: `code/dfm/target.py`
- Test: `code/tests/test_dfm_target.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_target.py`:

```python
"""CPI target: MoM transform and the MoM -> YoY reconstruction identity."""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import target as T


@pytest.fixture
def index_series():
    """A synthetic CPI index growing 0.2%/month for 3 years."""
    idx = pd.date_range("2020-01-31", periods=36, freq="ME")
    return pd.Series(100.0 * np.exp(np.arange(36) * 0.002), index=idx, name="cpi_index")


class TestMoM:
    def test_is_100x_log_difference(self, index_series):
        mom = T.mom_from_index(index_series)
        assert mom.iloc[1] == pytest.approx(0.2, abs=1e-9)
        assert np.isnan(mom.iloc[0])

    def test_length_matches_input(self, index_series):
        assert len(T.mom_from_index(index_series)) == len(index_series)


class TestYoYReconstruction:
    def test_identity_recovers_yoy_exactly(self, index_series):
        """YoY built from carry + modelled MoM equals YoY straight off the index."""
        mom = T.mom_from_index(index_series)
        direct = 100.0 * (index_series / index_series.shift(12) - 1.0)
        for m in index_series.index[13:]:
            got = T.yoy_from_mom(float(mom.loc[m]), index_series, m)
            assert got == pytest.approx(float(direct.loc[m]), abs=1e-9)

    def test_carry_uses_only_published_months(self, index_series):
        """The carry term must not touch the index value of month T itself."""
        m = index_series.index[20]
        truncated = index_series.copy()
        truncated.loc[m] = np.nan
        assert T.carry(truncated, m) == pytest.approx(T.carry(index_series, m))

    def test_raises_when_history_too_short(self, index_series):
        with pytest.raises(ValueError, match="12 months"):
            T.carry(index_series, index_series.index[5])


class TestRoundingFloor:
    def test_one_dp_rounding_stays_under_the_documented_floor(self, index_series):
        """Rounding the index to 1 d.p. must not move YoY by more than 0.135pp."""
        rounded = index_series.round(1)
        mom = T.mom_from_index(index_series)
        errs = []
        for m in index_series.index[13:]:
            errs.append(abs(T.yoy_from_mom(float(mom.loc[m]), rounded, m)
                            - T.yoy_from_mom(float(mom.loc[m]), index_series, m)))
        assert max(errs) < 0.135
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_target.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dfm.target'`.

- [ ] **Step 3: Write the implementation**

Create `code/dfm/target.py`:

```python
"""
dfm/target.py — the CPI target in MoM space, and the YoY reconstruction.

Modelling MoM instead of YoY is the single most important choice in this design
(spec §3.2). CPI YoY has lag-1 autocorrelation 0.983, so 11/12 of its variance is
mechanical carry-over the factors would be forced to re-learn; MoM has 0.048. The
repo's prior DFMs modelled YoY and scored ~1.97 RMSE against AutoARIMA's 0.467.

The carry term is fully known at T — every index level it uses is already
published — so all model risk concentrates on the single unknown increment:

    c_T   = 100 * (I_{T-1} / I_{T-12} - 1)
    YoY_T = 100 * ((1 + c_T/100) * exp(pi_T / 100) - 1)

The registry carries only YoY (D7G7), so the index D7BT is defined here, reusing
factors._dbnomics. That keeps the diff to factors.py at zero.
"""
import numpy as np
import pandas as pd

import factors as F
from dfm import config as C

_CACHE = {}


def cpi_index(force=False):
    """UK CPI all-items index D7BT (ONS MM23), month-end."""
    if not force and "idx" in _CACHE:
        return _CACHE["idx"]
    s = F._dbnomics("ONS", "MM23", "D7BT.M")
    s = s.resample("ME").last().dropna().rename("cpi_index")
    _CACHE["idx"] = s
    return s


def mom_from_index(index):
    """pi_t = 100 * dlog(I_t). First observation is NaN."""
    return (100.0 * np.log(index).diff()).rename("cpi_mom")


def carry(index, ref_month):
    """
    Base-effect carry for `ref_month`: 100*(I_{T-1}/I_{T-12} - 1).

    Uses only months strictly before `ref_month`, all of them published by T.
    """
    m = pd.Timestamp(ref_month) + pd.offsets.MonthEnd(0)
    prev = m - pd.offsets.MonthEnd(1)
    base = m - pd.offsets.MonthEnd(12)
    if prev not in index.index or base not in index.index:
        raise ValueError(f"need 12 months of index history before {m.date()}")
    i_prev, i_base = float(index.loc[prev]), float(index.loc[base])
    if not np.isfinite(i_prev) or not np.isfinite(i_base) or i_base == 0:
        raise ValueError(f"non-finite index values for the carry at {m.date()}")
    return 100.0 * (i_prev / i_base - 1.0)


def yoy_from_mom(mom, index, ref_month):
    """Rebuild CPI YoY for `ref_month` from the known carry and a modelled MoM."""
    c = carry(index, ref_month)
    return 100.0 * ((1.0 + c / 100.0) * np.exp(mom / 100.0) - 1.0)


def training_frame(asof_ts=None):
    """DataFrame [cpi_index, cpi_mom] through the last published month."""
    idx = cpi_index()
    if asof_ts is not None:
        idx = idx[idx.index <= pd.Timestamp(asof_ts)]
    return pd.DataFrame({"cpi_index": idx, "cpi_mom": mom_from_index(idx)})
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_target.py -q
```

Expected: `7 passed`.

- [ ] **Step 5: Verify the reconstruction against official D7G7** (network; `dangerouslyDisableSandbox: true`)

```bash
PYTHONPATH=code $Q -c "
import numpy as np, pandas as pd
import factors as F
from dfm import target as T
idx = T.cpi_index()
mom = T.mom_from_index(idx)
official, _ = F.load_factor('cpi_yoy')
rows = []
for m in idx.index[13:]:
    if m in official.index and np.isfinite(official.loc[m]):
        rows.append(abs(T.yoy_from_mom(float(mom.loc[m]), idx, m) - float(official.loc[m])))
r = pd.Series(rows)
print(f'n={len(r)}  max={r.max():.4f}  p99={r.quantile(0.99):.4f}  mean={r.mean():.4f}')
assert r.max() < 0.20, 'reconstruction drift exceeds the documented floor'
print('OK — within the documented rounding floor')
"
```

Expected: `max` around 0.07–0.14 pp, matching spec §3.3. If it exceeds 0.20, stop and investigate before building anything on top.

- [ ] **Step 6: Commit**

```bash
git add code/dfm/target.py code/tests/test_dfm_target.py
git commit -m "feat(dfm): CPI index target, MoM transform and YoY reconstruction identity"
```

---

## Task 8: State space — sequential univariate Kalman filter

The state vector for Stage 1 is `s_t = [f_t, f_{t-1}, e_{1t}, ..., e_{Mt}]`, length `2K + M` where `K=7` and `M = 33` (32 panel series + CPI MoM). Observation `i` is `y_it = [λ_i, 0_K, unit_i]' s_t` with **zero** measurement noise — all noise lives in the state, which is what makes the idiosyncratic AR(1) terms filterable and the missing-data handling exact.

**Files:**
- Create: `code/dfm/state_space.py`
- Test: `code/tests/test_dfm_state_space.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_state_space.py`:

```python
"""Kalman filter and simulation smoother on synthetic data with known parameters."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import state_space as SS


def simulate(T=200, K=2, M=6, seed=0):
    """Generate factors, loadings and observations from a known DFM."""
    rng = np.random.default_rng(seed)
    Phi = np.diag([0.7, 0.5])
    Lam = rng.normal(size=(M, K))
    Lam[:K, :K] = np.tril(Lam[:K, :K])
    np.fill_diagonal(Lam[:K, :K], np.abs(np.diag(Lam[:K, :K])) + 0.5)
    rho = np.full(M, 0.3)
    sig = np.full(M, 0.5)
    f = np.zeros((T, K))
    for t in range(1, T):
        f[t] = Phi @ f[t - 1] + rng.normal(size=K)
    e = np.zeros((T, M))
    for t in range(1, T):
        e[t] = rho * e[t - 1] + rng.normal(size=M) * sig
    y = f @ Lam.T + e
    return dict(y=y, f=f, Lam=Lam, Phi=Phi, rho=rho, sig=sig, K=K, M=M, T=T)


@pytest.fixture(scope="module")
def sim():
    return simulate()


class TestFilter:
    def test_filter_runs_and_returns_expected_shapes(self, sim):
        out = SS.kalman_filter(sim["y"], sim["Lam"], np.eye(sim["K"]),
                               sim["Phi"], sim["rho"], sim["sig"])
        n_state = 2 * sim["K"] + sim["M"]
        assert out["a_filt"].shape == (sim["T"], n_state)
        assert out["P_filt"].shape == (sim["T"], n_state, n_state)
        assert np.isfinite(out["loglik"])

    def test_filtered_factors_track_the_truth(self, sim):
        out = SS.kalman_filter(sim["y"], sim["Lam"], np.eye(sim["K"]),
                               sim["Phi"], sim["rho"], sim["sig"])
        for k in range(sim["K"]):
            c = np.corrcoef(out["a_filt"][20:, k], sim["f"][20:, k])[0, 1]
            assert abs(c) > 0.85, f"factor {k} correlation {c:.3f}"

    def test_missing_observations_are_skipped_not_imputed(self, sim):
        y = sim["y"].copy()
        y[50:60, 0] = np.nan
        out = SS.kalman_filter(y, sim["Lam"], np.eye(sim["K"]),
                               sim["Phi"], sim["rho"], sim["sig"])
        assert np.isfinite(out["loglik"])
        assert np.all(np.isfinite(out["a_filt"]))

    def test_all_missing_row_leaves_state_at_the_prediction(self, sim):
        y = sim["y"].copy()
        y[100, :] = np.nan
        out = SS.kalman_filter(y, sim["Lam"], np.eye(sim["K"]),
                               sim["Phi"], sim["rho"], sim["sig"])
        prev = out["a_filt"][99]
        pred = SS.transition_matrix(sim["Phi"], sim["rho"]) @ prev
        assert np.allclose(out["a_filt"][100], pred, atol=1e-8)

    def test_loglik_matches_statsmodels_on_a_shared_model(self):
        """Zero idio AR, one factor: a model statsmodels can also express."""
        from statsmodels.tsa.statespace.mlemodel import MLEModel
        s = simulate(T=120, K=1, M=3, seed=7)
        rho = np.zeros(s["M"])
        ours = SS.kalman_filter(s["y"], s["Lam"], np.eye(1), np.array([[0.7]]),
                                rho, s["sig"])["loglik"]

        class Ref(MLEModel):
            def __init__(self, endog, Lam, phi, sig):
                super().__init__(endog, k_states=1, initialization="stationary")
                self["design"] = Lam.reshape(-1, 1)
                self["transition"] = np.array([[phi]])
                self["selection"] = np.ones((1, 1))
                self["state_cov"] = np.ones((1, 1))
                self["obs_cov"] = np.diag(sig ** 2)

        ref = Ref(s["y"], s["Lam"], 0.7, s["sig"]).loglike([])
        assert abs(ours - ref) / abs(ref) < 1e-6


class TestSimulationSmoother:
    def test_draws_recover_the_true_factor_path(self, sim):
        rng = np.random.default_rng(11)
        draws = np.stack([
            SS.simulation_smoother(sim["y"], sim["Lam"], np.eye(sim["K"]),
                                   sim["Phi"], sim["rho"], sim["sig"], rng)
            for _ in range(30)])
        mean_f = draws[:, :, :sim["K"]].mean(axis=0)
        for k in range(sim["K"]):
            c = np.corrcoef(mean_f[20:, k], sim["f"][20:, k])[0, 1]
            assert abs(c) > 0.85, f"factor {k} correlation {c:.3f}"

    def test_draws_are_not_identical(self, sim):
        rng = np.random.default_rng(3)
        a = SS.simulation_smoother(sim["y"], sim["Lam"], np.eye(sim["K"]),
                                   sim["Phi"], sim["rho"], sim["sig"], rng)
        b = SS.simulation_smoother(sim["y"], sim["Lam"], np.eye(sim["K"]),
                                   sim["Phi"], sim["rho"], sim["sig"], rng)
        assert not np.allclose(a, b)

    def test_draw_spread_shrinks_when_noise_shrinks(self, sim):
        rng = np.random.default_rng(5)
        def spread(sig_scale):
            d = np.stack([
                SS.simulation_smoother(sim["y"], sim["Lam"], np.eye(sim["K"]), sim["Phi"],
                                       sim["rho"], sim["sig"] * sig_scale, rng)
                for _ in range(20)])
            return d[:, 20:, :sim["K"]].std(axis=0).mean()
        assert spread(0.2) < spread(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_state_space.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dfm.state_space'`.

- [ ] **Step 3: Write the implementation**

Create `code/dfm/state_space.py`:

```python
"""
dfm/state_space.py — Kalman filter and Durbin-Koopman simulation smoother.

State:   s_t = [f_t (K), f_{t-1} (K), e_t (M)]          length n = 2K + M
Trans:   f_t = Phi1 f_{t-1} + Phi2 f_{t-2} + eta_t,   e_it = rho_i e_{i,t-1} + eps_it
Obs:     y_it = lam_i' f_t + e_it                      (no separate measurement noise)

Idiosyncratic errors are carried IN the state rather than quasi-differenced away,
because idiosyncratic stochastic volatility (Stage 2) makes quasi-differencing
ill-defined. All observation noise therefore lives in the state, and the
measurement equation is exact.

The update is SEQUENTIAL (univariate): observations are applied one at a time, so
a missing value is simply a skipped update — no interpolation, no EM fill, and
cost is O(N k^2) rather than O(N^3) per period. This is what makes the ragged
edge native.

Pure numeric: no pandas, no data fetch, fully testable on synthetic data.
"""
import numpy as np

try:
    from numba import njit
except ImportError:                                   # pragma: no cover
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not a else a[0]


def transition_matrix(Phi, rho):
    """
    Block companion transition for [f_t, f_{t-1}, e_t].

    Phi : (K, K) for factor_order 1, or (K, 2K) stacking [Phi1 Phi2].
    rho : (M,) idiosyncratic AR(1) coefficients.
    """
    Phi = np.atleast_2d(np.asarray(Phi, dtype=float))
    K = Phi.shape[0]
    Phi1 = Phi[:, :K]
    Phi2 = Phi[:, K:2 * K] if Phi.shape[1] >= 2 * K else np.zeros((K, K))
    M = len(rho)
    n = 2 * K + M
    T = np.zeros((n, n))
    T[:K, :K] = Phi1
    T[:K, K:2 * K] = Phi2
    T[K:2 * K, :K] = np.eye(K)
    T[2 * K:, 2 * K:] = np.diag(np.asarray(rho, dtype=float))
    return T


def state_cov(Q_f, sig, K, M):
    """Innovation covariance of the state, diag(Q_f, 0_K, sig^2)."""
    n = 2 * K + M
    Q = np.zeros((n, n))
    Q[:K, :K] = np.asarray(Q_f, dtype=float)
    Q[2 * K:, 2 * K:] = np.diag(np.asarray(sig, dtype=float) ** 2)
    return Q


def design_matrix(Lam, K, M):
    """Z with rows [lam_i, 0_K, unit_i]."""
    Z = np.zeros((M, 2 * K + M))
    Z[:, :K] = np.asarray(Lam, dtype=float)
    Z[:, 2 * K:] = np.eye(M)
    return Z


def _initial_state(T, Q, n):
    """Stationary initialisation, with a diffuse fallback if the VAR is explosive."""
    a0 = np.zeros(n)
    try:
        vecP = np.linalg.solve(np.eye(n * n) - np.kron(T, T), Q.reshape(-1))
        P0 = vecP.reshape(n, n)
        if not np.all(np.isfinite(P0)) or np.any(np.diag(P0) <= 0):
            raise np.linalg.LinAlgError
        P0 = 0.5 * (P0 + P0.T)
    except np.linalg.LinAlgError:
        P0 = np.eye(n) * 1e4
    return a0, P0


@njit(cache=True)
def _filter_core(y, Z, T, Q, a0, P0):
    """Sequential univariate filter. Returns filtered states, covs and loglik."""
    n_obs, M = y.shape
    n = a0.shape[0]
    a_filt = np.zeros((n_obs, n))
    P_filt = np.zeros((n_obs, n, n))
    a = a0.copy()
    P = P0.copy()
    loglik = 0.0
    log2pi = np.log(2.0 * np.pi)
    for t in range(n_obs):
        if t > 0:
            a = T @ a
            P = T @ P @ T.T + Q
            P = 0.5 * (P + P.T)
        for i in range(M):
            yi = y[t, i]
            if np.isnan(yi):
                continue                      # missing -> skip the update entirely
            Zi = Z[i]
            PZ = P @ Zi
            F = Zi @ PZ
            if F <= 1e-12:
                continue
            v = yi - Zi @ a
            a = a + PZ * (v / F)
            P = P - np.outer(PZ, PZ) / F
            P = 0.5 * (P + P.T)
            loglik += -0.5 * (log2pi + np.log(F) + v * v / F)
        a_filt[t] = a
        P_filt[t] = P
    return a_filt, P_filt, loglik


def kalman_filter(y, Lam, Q_f, Phi, rho, sig):
    """Filter the panel. `y` is (T, M) with NaN for unobserved cells."""
    y = np.asarray(y, dtype=float)
    M = y.shape[1]
    K = np.atleast_2d(Lam).shape[1]
    T = transition_matrix(Phi, rho)
    Q = state_cov(Q_f, sig, K, M)
    Z = design_matrix(Lam, K, M)
    a0, P0 = _initial_state(T, Q, T.shape[0])
    a_filt, P_filt, loglik = _filter_core(y, Z, T, Q, a0, P0)
    return dict(a_filt=a_filt, P_filt=P_filt, loglik=loglik, Z=Z, T=T, Q=Q)


def _smooth_means(y, Z, T, Q, a0, P0):
    """Disturbance-free RTS smoother; returns smoothed state means (T, n)."""
    a_filt, P_filt, _ = _filter_core(y, Z, T, Q, a0, P0)
    n_obs, n = a_filt.shape
    a_sm = np.zeros_like(a_filt)
    a_sm[-1] = a_filt[-1]
    P_sm = P_filt[-1].copy()
    for t in range(n_obs - 2, -1, -1):
        P_pred = T @ P_filt[t] @ T.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        G = P_filt[t] @ T.T @ np.linalg.pinv(P_pred)
        a_sm[t] = a_filt[t] + G @ (a_sm[t + 1] - T @ a_filt[t])
        P_sm = P_filt[t] + G @ (P_sm - P_pred) @ G.T
    return a_sm


def simulation_smoother(y, Lam, Q_f, Phi, rho, sig, rng):
    """
    Durbin-Koopman (2002) simulation smoother: one draw of the full state path.

    Simulate s+ and y+ from the model, smooth both the real and simulated data,
    and form  s_draw = s+ + E[s | y] - E[s+ | y+].  The missing-data pattern of
    y is copied onto y+ so the two smoothers condition on the same information.
    """
    y = np.asarray(y, dtype=float)
    n_obs, M = y.shape
    K = np.atleast_2d(Lam).shape[1]
    T = transition_matrix(Phi, rho)
    Q = state_cov(Q_f, sig, K, M)
    Z = design_matrix(Lam, K, M)
    n = T.shape[0]
    a0, P0 = _initial_state(T, Q, n)

    # forward simulation
    chol = np.linalg.cholesky(Q + np.eye(n) * 1e-10)
    s_plus = np.zeros((n_obs, n))
    try:
        s_plus[0] = a0 + np.linalg.cholesky(P0 + np.eye(n) * 1e-10) @ rng.standard_normal(n)
    except np.linalg.LinAlgError:
        s_plus[0] = a0 + np.sqrt(np.diag(P0)) * rng.standard_normal(n)
    for t in range(1, n_obs):
        s_plus[t] = T @ s_plus[t - 1] + chol @ rng.standard_normal(n)
    y_plus = s_plus @ Z.T
    y_plus[np.isnan(y)] = np.nan            # identical missingness pattern

    a_hat = _smooth_means(y, Z, T, Q, a0, P0)
    a_plus_hat = _smooth_means(y_plus, Z, T, Q, a0, P0)
    return s_plus + a_hat - a_plus_hat
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_state_space.py -q
```

Expected: `9 passed`. The statsmodels log-likelihood comparison is the one that catches sign and scaling errors — if it fails, fix the filter, not the test.

**If numba rejects `_filter_core`** (a `TypingError` on `np.outer` or the matmuls): drop the `@njit(cache=True)` decorator, confirm the tests pass in pure numpy, and re-time it in Task 14. Correctness first — the filter is called once per Gibbs iteration, so a slow filter changes the budget, not the results. Record the decision in `docs/dfm_backtest.md` under Compute.

- [ ] **Step 5: Commit**

```bash
git add code/dfm/state_space.py code/tests/test_dfm_state_space.py
git commit -m "feat(dfm): sequential univariate Kalman filter and Durbin-Koopman simulation smoother"
```

---

## Task 9: Sampler — loadings, idiosyncratic AR and variances

**Files:**
- Create: `code/dfm/sampler.py`
- Test: `code/tests/test_dfm_sampler.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_sampler.py`:

```python
"""Each Gibbs block must recover its own parameters from a known DGP."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import sampler as S


class TestDrawLoadings:
    def test_recovers_known_loadings(self):
        rng = np.random.default_rng(0)
        T, K, M = 400, 2, 5
        f = rng.normal(size=(T, K))
        Lam = rng.normal(size=(M, K))
        sig = np.full(M, 0.3)
        y = f @ Lam.T + rng.normal(size=(T, M)) * sig
        draws = np.stack([S.draw_loadings(y, f, np.zeros(M), sig, rng) for _ in range(50)])
        assert np.abs(draws.mean(axis=0) - Lam).max() < 0.1

    def test_respects_a_zero_mask(self):
        rng = np.random.default_rng(1)
        T, K, M = 200, 3, 4
        f = rng.normal(size=(T, K))
        y = rng.normal(size=(T, M))
        mask = np.ones((M, K), dtype=bool)
        mask[:, 2] = False                       # third factor pinned off
        got = S.draw_loadings(y, f, np.zeros(M), np.ones(M), rng, mask=mask)
        assert np.all(got[:, 2] == 0.0)

    def test_shrinks_toward_zero_with_a_tight_prior(self):
        rng = np.random.default_rng(2)
        T, K, M = 60, 2, 3
        f = rng.normal(size=(T, K))
        y = f @ np.ones((M, K)).T + rng.normal(size=(T, M))
        loose = np.stack([S.draw_loadings(y, f, np.zeros(M), np.ones(M), rng,
                                          prior_var=100.0) for _ in range(40)]).mean(0)
        tight = np.stack([S.draw_loadings(y, f, np.zeros(M), np.ones(M), rng,
                                          prior_var=0.01) for _ in range(40)]).mean(0)
        assert np.abs(tight).mean() < np.abs(loose).mean()


class TestDrawIdio:
    def test_recovers_ar_coefficient_and_variance(self):
        rng = np.random.default_rng(3)
        T, M = 600, 3
        rho_true, sig_true = np.array([0.6, 0.2, -0.4]), np.array([0.5, 1.0, 0.8])
        e = np.zeros((T, M))
        for t in range(1, T):
            e[t] = rho_true * e[t - 1] + rng.normal(size=M) * sig_true
        rho_d, sig_d = [], []
        for _ in range(60):
            r, s = S.draw_idio(e, rng)
            rho_d.append(r); sig_d.append(s)
        assert np.abs(np.mean(rho_d, axis=0) - rho_true).max() < 0.08
        assert np.abs(np.mean(sig_d, axis=0) - sig_true).max() < 0.10

    def test_rejects_nonstationary_draws(self):
        rng = np.random.default_rng(4)
        T, M = 300, 2
        e = np.cumsum(rng.normal(size=(T, M)), axis=0)     # random walk, rho -> 1
        for _ in range(30):
            rho, _ = S.draw_idio(e, rng)
            assert np.all(np.abs(rho) < 1.0)


class TestDrawFactorAR:
    def test_recovers_diagonal_ar_coefficients(self):
        rng = np.random.default_rng(5)
        T, K = 500, 2
        phi_true = np.array([0.7, -0.3])
        f = np.zeros((T, K))
        for t in range(1, T):
            f[t] = phi_true * f[t - 1] + rng.normal(size=K)
        draws = np.stack([S.draw_factor_ar(f, rng, order=1, diagonal=True) for _ in range(60)])
        est = np.array([draws[:, k, k].mean() for k in range(K)])
        assert np.abs(est - phi_true).max() < 0.08

    def test_diagonal_mode_has_zero_off_diagonals(self):
        rng = np.random.default_rng(6)
        f = np.random.default_rng(7).normal(size=(200, 3))
        Phi = S.draw_factor_ar(f, rng, order=1, diagonal=True)
        assert np.allclose(Phi - np.diag(np.diag(Phi)), 0.0)

    def test_returns_stationary_companion(self):
        rng = np.random.default_rng(8)
        f = np.cumsum(np.random.default_rng(9).normal(size=(300, 2)), axis=0)
        for _ in range(20):
            Phi = S.draw_factor_ar(f, rng, order=1, diagonal=True)
            assert np.max(np.abs(np.linalg.eigvals(Phi))) < 1.0


class TestNormalisation:
    def test_upper_block_is_lower_triangular_with_positive_diagonal(self):
        rng = np.random.default_rng(10)
        Lam = rng.normal(size=(6, 3))
        f = rng.normal(size=(100, 3))
        Lam_n, f_n = S.normalise(Lam.copy(), f.copy())
        upper = Lam_n[:3, :3]
        assert np.allclose(np.triu(upper, 1), 0.0)
        assert np.all(np.diag(upper) > 0)

    def test_normalisation_preserves_the_common_component(self):
        rng = np.random.default_rng(11)
        Lam = rng.normal(size=(6, 3))
        f = rng.normal(size=(100, 3))
        before = f @ Lam.T
        Lam_n, f_n = S.normalise(Lam.copy(), f.copy())
        assert np.allclose(before, f_n @ Lam_n.T, atol=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_sampler.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dfm.sampler'`.

- [ ] **Step 3: Write the implementation**

Create `code/dfm/sampler.py`:

```python
"""
dfm/sampler.py — Gibbs blocks for the DFM.

Stage 1 (E1) needs three blocks plus the factor draw from state_space:
  draw_loadings   Lam | f, y, sig          series-by-series Bayesian regression
  draw_idio       rho, sig | e             idiosyncratic AR(1) + variance
  draw_factor_ar  Phi | f                  factor VAR (diagonal in E1-E3)

Later stages add SV, outlier indicators and hyperpriors; those blocks land in
Plans 2 and 3 and must not change the signatures here.

Every block is a conjugate draw with a weakly-informative prior, so each one is
independently testable by parameter recovery on synthetic data.
"""
import numpy as np


def _stationary(Phi):
    """Is the companion form of Phi stable?"""
    return np.max(np.abs(np.linalg.eigvals(Phi))) < 0.999


def draw_loadings(y, f, gamma_z, sig, rng, mask=None, prior_var=10.0):
    """
    Draw Lam (M, K) row by row from  y_i - gamma_i = lam_i' f + e_i,  e_i ~ N(0, sig_i^2).

    mask : optional (M, K) boolean; False entries are pinned to zero (E5/E6 block
           restrictions). Rows with no free entries return zeros.
    """
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    M, K = y.shape[1], f.shape[1]
    out = np.zeros((M, K))
    for i in range(M):
        free = np.ones(K, dtype=bool) if mask is None else np.asarray(mask[i], dtype=bool)
        if not free.any():
            continue
        obs = np.isfinite(y[:, i])
        if obs.sum() <= free.sum():
            continue
        X = f[obs][:, free]
        target = y[obs, i] - gamma_z[i]
        prec = X.T @ X / sig[i] ** 2 + np.eye(free.sum()) / prior_var
        V = np.linalg.inv(prec)
        m = V @ (X.T @ target / sig[i] ** 2)
        V = 0.5 * (V + V.T)
        out[i, free] = rng.multivariate_normal(m, V)
    return out


def draw_idio(e, rng, prior_var=10.0, a0=3.0, b0=1.0, max_tries=20):
    """
    Draw (rho, sig) for each idiosyncratic AR(1) series from its own residuals.

    Returns (rho (M,), sig (M,)). Non-stationary rho draws are rejected and
    retried; after `max_tries` the previous OLS estimate is clipped into (-0.99, 0.99).
    """
    e = np.asarray(e, dtype=float)
    T, M = e.shape
    rho = np.zeros(M)
    sig = np.ones(M)
    for i in range(M):
        x, y = e[:-1, i], e[1:, i]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 10 or np.dot(x, x) <= 0:
            continue
        # variance first, conditional on an OLS rho
        r_ols = float(np.dot(x, y) / np.dot(x, x))
        resid = y - r_ols * x
        a_n = a0 + len(x) / 2.0
        b_n = b0 + 0.5 * float(resid @ resid)
        s2 = b_n / rng.gamma(a_n, 1.0)          # InvGamma(a_n, b_n) draw
        sig[i] = np.sqrt(s2)
        # then rho | sig
        prec = np.dot(x, x) / s2 + 1.0 / prior_var
        v = 1.0 / prec
        m = v * (np.dot(x, y) / s2)
        r = m + np.sqrt(v) * rng.standard_normal()
        tries = 0
        while abs(r) >= 1.0 and tries < max_tries:
            r = m + np.sqrt(v) * rng.standard_normal()
            tries += 1
        rho[i] = r if abs(r) < 1.0 else float(np.clip(r_ols, -0.99, 0.99))
    return rho, sig


def draw_factor_ar(f, rng, order=1, diagonal=True, prior_var=1.0,
                   minnesota_cross=0.05, max_tries=50):
    """
    Draw the factor VAR coefficient matrix Phi, shape (K, K*order).

    diagonal=True  -> each factor is an independent AR(order) (arms E1-E3).
    diagonal=False -> full VAR under a Minnesota prior: own lags get `prior_var`,
                      cross lags the much tighter `minnesota_cross` (arms E4+).
    Non-stationary draws are rejected; on repeated failure the coefficients are
    scaled back toward zero until the companion form is stable.
    """
    f = np.asarray(f, dtype=float)
    T, K = f.shape
    X = np.column_stack([f[order - l - 1:T - l - 1] for l in range(order)])
    Y = f[order:]
    n_coef = K * order
    Phi = np.zeros((K, n_coef))
    for k in range(K):
        if diagonal:
            cols = [l * K + k for l in range(order)]
        else:
            cols = list(range(n_coef))
        Xk = X[:, cols]
        pv = np.full(len(cols), prior_var, dtype=float)
        if not diagonal:
            for j, c in enumerate(cols):
                if c % K != k:
                    pv[j] = minnesota_cross
        prec = Xk.T @ Xk + np.diag(1.0 / pv)
        V = np.linalg.inv(prec)
        m = V @ (Xk.T @ Y[:, k])
        V = 0.5 * (V + V.T)
        Phi[k, cols] = rng.multivariate_normal(m, V)

    comp = _companion(Phi, K, order)
    tries = 0
    while not _stationary(comp) and tries < max_tries:
        Phi *= 0.95
        comp = _companion(Phi, K, order)
        tries += 1
    return Phi


def _companion(Phi, K, order):
    if order == 1:
        return Phi[:, :K]
    C = np.zeros((K * order, K * order))
    C[:K, :] = Phi
    C[K:, :-K] = np.eye(K * (order - 1))
    return C


def normalise(Lam, f):
    """
    Identify the factors: make the top K x K block of Lam lower-triangular with a
    positive diagonal, applying the inverse rotation to f so the common component
    f @ Lam.T is unchanged.

    LQ via a QR of the transpose: top.T = q1 @ r1, so top = r1.T @ q1.T with
    r1.T lower-triangular and q1.T orthogonal. Because R is orthogonal,
    f_n @ Lam_n.T = f @ R.T @ R @ Lam.T = f @ Lam.T exactly — the common component
    is untouched. Verified numerically before this plan was written.
    """
    K = Lam.shape[1]
    top = Lam[:K, :K]
    q1, r1 = np.linalg.qr(top.T)
    L, R = r1.T, q1.T                      # L lower-triangular, R orthogonal
    signs = np.sign(np.diag(L))
    signs[signs == 0] = 1.0
    L = L * signs                          # scale columns of L
    R = (R.T * signs).T                    # and rows of R, so L @ R is unchanged
    Lam_n = Lam @ R.T
    f_n = f @ R.T
    return Lam_n, f_n
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_sampler.py -q
```

Expected: `10 passed`. `normalise` is the fiddly one — if `test_normalisation_preserves_the_common_component` fails, simplify to an explicit LQ via `scipy.linalg.rq` and keep the sign fix; the invariant that must hold is `f @ Lam.T == f_n @ Lam_n.T`.

- [ ] **Step 5: Commit**

```bash
git add code/dfm/sampler.py code/tests/test_dfm_sampler.py
git commit -m "feat(dfm): Gibbs blocks for loadings, idiosyncratic AR and factor VAR"
```

---

## Task 10: Model — `DFMConfig` and `DFMNowcaster` (arm E1)

**Files:**
- Create: `code/dfm/model.py`
- Test: `code/tests/test_dfm_model.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_model.py`:

```python
"""DFMNowcaster end-to-end on synthetic data — arm E1."""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm.model import DFMConfig, DFMNowcaster


def synthetic_panel(T=180, K=2, M=6, seed=0, ragged=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=T, freq="ME")
    Phi = np.diag([0.6, 0.4])
    Lam = rng.normal(size=(M, K))
    f = np.zeros((T, K))
    for t in range(1, T):
        f[t] = Phi @ f[t - 1] + rng.normal(size=K)
    y = f @ Lam.T + rng.normal(size=(T, M)) * 0.4
    df = pd.DataFrame(y, index=idx, columns=[f"s{i}" for i in range(M)])
    df["cpi_mom"] = f @ np.array([0.8, 0.3]) + rng.normal(size=T) * 0.2
    if ragged:
        df.iloc[-1, 1] = np.nan          # one series not yet published
        df.iloc[-1, -1] = np.nan         # target unknown for the nowcast month
    return df


@pytest.fixture(scope="module")
def cfg():
    return DFMConfig(arm="E1", k_factors=2, factor_order=1,
                     draws=120, burn=60, seed=1)


class TestFit:
    def test_fit_returns_posterior_of_expected_shape(self, cfg):
        df = synthetic_panel()
        m = DFMNowcaster(cfg).fit(df, target="cpi_mom")
        assert m.posterior["Lam"].shape == (60, df.shape[1], 2)   # draws-burn
        assert m.posterior["factors"].shape[1:] == (len(df), 2)

    def test_recovers_a_factor_that_correlates_with_the_target(self, cfg):
        df = synthetic_panel()
        m = DFMNowcaster(cfg).fit(df, target="cpi_mom")
        fmean = m.posterior["factors"].mean(axis=0)
        best = max(abs(np.corrcoef(fmean[:, k], df["cpi_mom"].values)[0, 1])
                   for k in range(2))
        assert best > 0.7


class TestNowcast:
    def test_returns_finite_mean_and_interval(self, cfg):
        df = synthetic_panel()
        m = DFMNowcaster(cfg).fit(df, target="cpi_mom")
        out = m.nowcast(df, target="cpi_mom")
        assert np.isfinite(out["mean"])
        assert out["p05"] < out["mean"] < out["p95"]

    def test_ragged_edge_does_not_break_the_nowcast(self, cfg):
        df = synthetic_panel(ragged=True)
        assert df.iloc[-1].isna().sum() >= 2
        m = DFMNowcaster(cfg).fit(df, target="cpi_mom")
        assert np.isfinite(m.nowcast(df, target="cpi_mom")["mean"])

    def test_more_information_narrows_the_interval(self, cfg):
        """A fully-observed final row should be no less certain than a ragged one."""
        full = synthetic_panel(ragged=False)
        ragged = full.copy()
        ragged.iloc[-1, :3] = np.nan
        w_full = (lambda o: o["p95"] - o["p05"])(
            DFMNowcaster(cfg).fit(full, target="cpi_mom").nowcast(full, target="cpi_mom"))
        w_rag = (lambda o: o["p95"] - o["p05"])(
            DFMNowcaster(cfg).fit(ragged, target="cpi_mom").nowcast(ragged, target="cpi_mom"))
        assert w_full <= w_rag * 1.25

    def test_seed_makes_it_reproducible(self, cfg):
        df = synthetic_panel()
        a = DFMNowcaster(cfg).fit(df, target="cpi_mom").nowcast(df, target="cpi_mom")["mean"]
        b = DFMNowcaster(cfg).fit(df, target="cpi_mom").nowcast(df, target="cpi_mom")["mean"]
        assert a == pytest.approx(b)


class TestConfig:
    def test_e1_has_no_sv_no_outliers_diagonal_phi(self):
        c = DFMConfig(arm="E1")
        assert not c.stochastic_vol and not c.outliers
        assert c.diagonal_phi and not c.block_loadings and not c.hierarchical

    def test_unknown_arm_raises(self):
        with pytest.raises(ValueError, match="unknown arm"):
            DFMConfig(arm="E9")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_model.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dfm.model'`.

- [ ] **Step 3: Write the implementation**

Create `code/dfm/model.py`:

```python
"""
dfm/model.py — the nowcaster: config, Gibbs loop, nowcast.

DFMConfig turns an arm label into a set of switches, so the arm ladder is data,
not branching code. Stage 1 implements E1 only; the later flags exist so Plans 2
and 3 flip them on without changing any call site.

  E1  diagonal Phi, no SV, no outliers, unrestricted loadings, fixed priors
  E2  + stochastic volatility            (Plan 2)
  E3  + Stock-Watson outlier mixture     (Plan 2)
  E4  + full VAR under a Minnesota prior (Plan 3)
  E5  + block-restricted, sparse loadings(Plan 3)
  E6  + hierarchical shrinkage           (Plan 3)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dfm import config as C
from dfm import sampler as S
from dfm import state_space as SS

ARMS = {
    #        SV     outlier  diagPhi  blockLam  hierarchical
    "E1": (False, False, True,  False, False),
    "E2": (True,  False, True,  False, False),
    "E3": (True,  True,  True,  False, False),
    "E4": (True,  True,  False, False, False),
    "E5": (True,  True,  False, True,  False),
    "E6": (True,  True,  False, True,  True),
}


@dataclass
class DFMConfig:
    arm: str = "E1"
    mode: str = "standalone"            # or "overlay"
    k_factors: int = C.K_FACTORS
    factor_order: int = C.FACTOR_ORDER
    draws: int = C.DRAWS_FIRST
    burn: int = C.BURN_FIRST
    seed: int = C.SEED
    loading_mask: object = None         # (M, K) bool, set by E5/E6
    stochastic_vol: bool = field(init=False)
    outliers: bool = field(init=False)
    diagonal_phi: bool = field(init=False)
    block_loadings: bool = field(init=False)
    hierarchical: bool = field(init=False)

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; expected one of {sorted(ARMS)}")
        (self.stochastic_vol, self.outliers, self.diagonal_phi,
         self.block_loadings, self.hierarchical) = ARMS[self.arm]
        if self.burn >= self.draws:
            raise ValueError("burn must be smaller than draws")
        for flag, plan in (("stochastic_vol", 2), ("outliers", 2),
                           ("hierarchical", 3)):
            if getattr(self, flag):
                raise NotImplementedError(
                    f"{self.arm} needs {flag}, which lands in Plan {plan}")


class DFMNowcaster:
    """Gibbs-sampled mixed-frequency DFM. Fit on a panel, nowcast the last row."""

    def __init__(self, cfg=None):
        self.cfg = cfg or DFMConfig()
        self.posterior = None
        self.columns = None

    def fit(self, panel, target, warm_start=None):
        """
        Run the Gibbs sampler on `panel` (DataFrame, NaN = unobserved).

        `target` names the column the nowcast is read from; it is an ordinary
        member of the measurement equation, so a NaN target row is simply the
        month being forecast.
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        y = panel.to_numpy(dtype=float)
        T, M = y.shape
        K = cfg.k_factors
        self.columns = list(panel.columns)
        self.target_pos = self.columns.index(target)

        # initialise from a PCA of the complete-case panel
        Lam, f, rho, sig = self._initialise(y, K, rng, warm_start)
        Phi = np.zeros((K, K * cfg.factor_order))
        Phi[:, :K] = np.eye(K) * 0.3
        Q_f = np.eye(K)

        keep = cfg.draws - cfg.burn
        store = dict(Lam=np.zeros((keep, M, K)),
                     Phi=np.zeros((keep, K, K * cfg.factor_order)),
                     rho=np.zeros((keep, M)), sig=np.zeros((keep, M)),
                     factors=np.zeros((keep, T, K)))

        gamma_z = np.zeros(M)                 # exogenous block enters in Plan 3
        for it in range(cfg.draws):
            state = SS.simulation_smoother(y, Lam, Q_f, Phi, rho, sig, rng)
            f = state[:, :K]
            e = state[:, 2 * K:]
            Lam = S.draw_loadings(y, f, gamma_z, sig, rng, mask=cfg.loading_mask)
            Lam, f = S.normalise(Lam, f)
            rho, sig = S.draw_idio(e, rng)
            Phi = S.draw_factor_ar(f, rng, order=cfg.factor_order,
                                   diagonal=cfg.diagonal_phi)
            if it >= cfg.burn:
                j = it - cfg.burn
                store["Lam"][j], store["Phi"][j] = Lam, Phi
                store["rho"][j], store["sig"][j] = rho, sig
                store["factors"][j] = f
        self.posterior = store
        return self

    def _initialise(self, y, K, rng, warm_start):
        """PCA start, or the previous month's posterior mean when warm-starting."""
        M = y.shape[1]
        if warm_start is not None:
            Lam = warm_start["Lam"][:M, :K].copy()
            rho = warm_start["rho"][:M].copy()
            sig = warm_start["sig"][:M].copy()
            return Lam, None, rho, sig
        filled = np.where(np.isfinite(y), y, np.nanmean(y, axis=0))
        filled = filled - filled.mean(axis=0)
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        Lam = vt[:K].T * s[:K] / np.sqrt(len(filled))
        rho = np.full(M, 0.2)
        sig = np.nanstd(y, axis=0)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0) * 0.5
        return Lam, None, rho, sig

    def nowcast(self, panel, target, row=-1):
        """Posterior predictive for `target` in row `row` (default: the last)."""
        if self.posterior is None:
            raise RuntimeError("fit() before nowcast()")
        i = self.target_pos
        f = self.posterior["factors"][:, row, :]              # (keep, K)
        lam = self.posterior["Lam"][:, i, :]                  # (keep, K)
        common = np.einsum("dk,dk->d", f, lam)
        sig = self.posterior["sig"][:, i]
        rng = np.random.default_rng(self.cfg.seed + 1)
        draws = common + rng.standard_normal(len(common)) * sig
        return dict(mean=float(draws.mean()),
                    median=float(np.median(draws)),
                    p05=float(np.percentile(draws, 5)),
                    p95=float(np.percentile(draws, 95)),
                    sd=float(draws.std()),
                    draws=draws)

    def posterior_mean(self):
        """Compact summary used to warm-start the next month's fit."""
        return {k: v.mean(axis=0) for k, v in self.posterior.items() if k != "factors"}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_model.py -q
```

Expected: `8 passed`. If `test_recovers_a_factor_that_correlates_with_the_target` fails, the normalisation rotation is the likeliest culprit — check the Task 9 invariant first.

- [ ] **Step 5: Commit**

```bash
git add code/dfm/model.py code/tests/test_dfm_model.py
git commit -m "feat(dfm): DFMConfig arm ladder and the E1 Gibbs nowcaster"
```

---

## Task 11: Backtest harness — walk-forward, purge, checkpoint, resume

**Files:**
- Create: `code/dfm/backtest.py`
- Test: `code/tests/test_dfm_backtest.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_backtest.py`:

```python
"""Walk-forward harness: causality, purge, checkpoint/resume."""
import sys, os
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import backtest as B


class FakeModel:
    """Records the training window it was handed; predicts its own row count."""
    seen = []

    def __init__(self, cfg=None):
        self.cfg = cfg

    def fit(self, panel, target, warm_start=None):
        FakeModel.seen.append((panel.index[0], panel.index[-1], len(panel)))
        self._n = len(panel)
        return self

    def nowcast(self, panel, target, row=-1):
        return dict(mean=float(self._n), p05=0.0, p95=1.0, sd=0.1, draws=np.zeros(3))

    def posterior_mean(self):
        return {"Lam": np.zeros((1, 1))}


@pytest.fixture
def panel():
    idx = pd.date_range("2012-01-31", "2018-12-31", freq="ME")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(len(idx), 4)),
                      index=idx, columns=["a", "b", "c", "cpi_mom"])
    return df


class TestCausality:
    def test_training_window_never_includes_the_test_month(self, panel):
        FakeModel.seen.clear()
        B.walk_forward(panel, target="cpi_mom", model_cls=FakeModel,
                       start_year=2016, end_year=2017, purge=0, embargo=0,
                       checkpoint_dir=None)
        for first, last, _ in FakeModel.seen:
            assert last <= pd.Timestamp("2017-12-31")

    def test_purge_shortens_the_training_window(self, panel):
        FakeModel.seen.clear()
        B.walk_forward(panel, target="cpi_mom", model_cls=FakeModel,
                       start_year=2016, end_year=2016, purge=0, embargo=0,
                       checkpoint_dir=None)
        unpurged = [n for _, _, n in FakeModel.seen]
        FakeModel.seen.clear()
        B.walk_forward(panel, target="cpi_mom", model_cls=FakeModel,
                       start_year=2016, end_year=2016, purge=12, embargo=1,
                       checkpoint_dir=None)
        purged = [n for _, _, n in FakeModel.seen]
        assert all(p < u for p, u in zip(purged, unpurged))

    def test_target_of_the_test_month_is_blanked(self, panel):
        """The model must never see the value it is asked to predict."""
        captured = {}

        class Spy(FakeModel):
            def fit(self, p, target, warm_start=None):
                captured["last_target"] = p[target].iloc[-1]
                return super().fit(p, target, warm_start)

        B.walk_forward(panel, target="cpi_mom", model_cls=Spy,
                       start_year=2016, end_year=2016, purge=0, embargo=0,
                       checkpoint_dir=None)
        assert np.isnan(captured["last_target"])


class TestOutput:
    def test_returns_one_row_per_scored_month(self, panel):
        out = B.walk_forward(panel, target="cpi_mom", model_cls=FakeModel,
                             start_year=2016, end_year=2017, purge=0, embargo=0,
                             checkpoint_dir=None)
        assert len(out) == 24
        assert set(["actual", "pred", "p05", "p95", "sd", "fit_seconds"]).issubset(out.columns)

    def test_blind_years_are_excluded(self, panel):
        out = B.walk_forward(panel, target="cpi_mom", model_cls=FakeModel,
                             start_year=2016, end_year=2017, purge=0, embargo=0,
                             checkpoint_dir=None)
        assert out.index.year.max() == 2017


class TestCheckpointing:
    def test_writes_one_checkpoint_per_month(self, panel, tmp_path):
        B.walk_forward(panel, target="cpi_mom", model_cls=FakeModel,
                       start_year=2016, end_year=2016, purge=0, embargo=0,
                       checkpoint_dir=str(tmp_path), tag="E1_test")
        files = sorted(os.listdir(tmp_path))
        assert len(files) == 12
        payload = json.load(open(os.path.join(tmp_path, files[0])))
        assert {"month", "pred", "actual"}.issubset(payload)

    def test_resume_skips_completed_months(self, panel, tmp_path):
        B.walk_forward(panel, target="cpi_mom", model_cls=FakeModel,
                       start_year=2016, end_year=2016, purge=0, embargo=0,
                       checkpoint_dir=str(tmp_path), tag="E1_test")
        FakeModel.seen.clear()
        out = B.walk_forward(panel, target="cpi_mom", model_cls=FakeModel,
                             start_year=2016, end_year=2016, purge=0, embargo=0,
                             checkpoint_dir=str(tmp_path), tag="E1_test")
        assert FakeModel.seen == [], "resume must refit nothing"
        assert len(out) == 12

    def test_metrics_by_window(self):
        idx = pd.date_range("2015-01-31", "2024-12-31", freq="ME")
        rng = np.random.default_rng(1)
        bt = pd.DataFrame({"actual": rng.normal(size=len(idx)),
                           "pred": rng.normal(size=len(idx))}, index=idx)
        m = B.metrics(bt)
        for w in ("full", "shock", "ex_shock", "pre_2020"):
            assert w in m.index
            assert np.isfinite(m.loc[w, "rmse"]) and np.isfinite(m.loc[w, "mae"])
        assert m.loc["shock", "n"] == 24
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_backtest.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dfm.backtest'`.

- [ ] **Step 3: Write the implementation**

Create `code/dfm/backtest.py`:

```python
"""
dfm/backtest.py — expanding-window walk-forward with purge, embargo and resume.

One fit per test month, matching the TVP/LGBM backtest convention in
new_factors/two_stage.py. The test month's target is blanked before the model
sees the panel, so the ragged edge in the backtest is exactly the ragged edge in
a live nowcast.

Purge/embargo reuse validation.purge_embargo — the same helper production uses,
so leakage discipline is identical, not merely similar.

Every (tag, month) result is checkpointed to JSON so a multi-hour sweep survives
interruption and resumes where it stopped.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import validation as V
from dfm import config as C

SHOCK_YEARS = (2022, 2023)


def _checkpoint_path(directory, tag, month):
    return os.path.join(directory, f"{tag}_{pd.Timestamp(month):%Y-%m}.json")


def walk_forward(panel, target, model_cls, model_cfg=None, start_year=None,
                 end_year=None, purge=None, embargo=None, checkpoint_dir=None,
                 tag="run", warm_start=True, verbose=False):
    """
    Expanding-window walk-forward over month-end rows of `panel`.

    Returns a DataFrame indexed by test month with actual, pred, interval and
    the wall-clock seconds each fit took.
    """
    start_year = start_year or C.EVAL_START
    end_year = end_year or C.EVAL_END
    purge = C.PURGE_STANDALONE if purge is None else purge
    embargo = C.EMBARGO_STANDALONE if embargo is None else embargo
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    months = [m for m in panel.index
              if start_year <= m.year <= end_year and np.isfinite(panel.loc[m, target])]
    rows, prev_post = [], None
    for m in months:
        if checkpoint_dir:
            cp = _checkpoint_path(checkpoint_dir, tag, m)
            if os.path.exists(cp):
                rows.append(json.load(open(cp)))
                continue

        train = panel.loc[:m].copy()
        actual = float(train.loc[m, target])
        train.loc[m, target] = np.nan                 # never show the model the answer
        if purge or embargo:
            hist = V.purge_embargo(train.iloc[:-1], m, purge, embargo)
            train = pd.concat([hist, train.loc[[m]]])
        if len(train) < 60:
            continue

        t0 = time.time()
        model = model_cls(model_cfg) if model_cfg is not None else model_cls()
        model.fit(train, target=target, warm_start=prev_post if warm_start else None)
        nc = model.nowcast(train, target=target)
        secs = time.time() - t0
        if warm_start:
            try:
                prev_post = model.posterior_mean()
            except Exception:
                prev_post = None

        row = dict(month=str(pd.Timestamp(m).date()), actual=actual,
                   pred=float(nc["mean"]), p05=float(nc["p05"]), p95=float(nc["p95"]),
                   sd=float(nc["sd"]), fit_seconds=round(secs, 3))
        rows.append(row)
        if checkpoint_dir:
            json.dump(row, open(_checkpoint_path(checkpoint_dir, tag, m), "w"))
        if verbose:
            print(f"  {row['month']}  pred={row['pred']:+.3f}  actual={row['actual']:+.3f}"
                  f"  ({secs:.1f}s)", flush=True)

    if not rows:
        return pd.DataFrame(columns=["actual", "pred", "p05", "p95", "sd", "fit_seconds"])
    out = pd.DataFrame(rows)
    out["month"] = pd.to_datetime(out["month"])
    return out.set_index("month").sort_index()


def _window_mask(idx, window):
    if window == "full":
        return np.ones(len(idx), dtype=bool)
    if window == "shock":
        return idx.year.isin(SHOCK_YEARS)
    if window == "ex_shock":
        return ~idx.year.isin(SHOCK_YEARS)
    if window == "pre_2020":
        return idx.year < 2020
    raise ValueError(f"unknown window {window!r}")


def metrics(bt, windows=("full", "shock", "ex_shock", "pre_2020")):
    """RMSE / MAE / n by evaluation window."""
    rows = []
    for w in windows:
        mask = _window_mask(bt.index, w)
        sub = bt[mask].dropna(subset=["actual", "pred"])
        err = sub["pred"] - sub["actual"]
        rows.append(dict(window=w, n=len(sub),
                         rmse=float(np.sqrt((err ** 2).mean())) if len(sub) else np.nan,
                         mae=float(err.abs().mean()) if len(sub) else np.nan,
                         bias=float(err.mean()) if len(sub) else np.nan))
    return pd.DataFrame(rows).set_index("window")


def dm_test(e1, e2, h=1):
    """Diebold-Mariano on squared-error differentials. Returns (stat, p)."""
    from scipy.stats import t as tdist
    d = np.asarray(e1, dtype=float) ** 2 - np.asarray(e2, dtype=float) ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dbar = d.mean()
    gamma0 = ((d - dbar) ** 2).mean()
    var = gamma0
    for lag in range(1, h):
        cov = ((d[lag:] - dbar) * (d[:-lag] - dbar)).mean()
        var += 2 * (1 - lag / h) * cov
    stat = dbar / np.sqrt(var / n) if var > 0 else np.nan
    p = 2 * (1 - tdist.cdf(abs(stat), n - 1)) if np.isfinite(stat) else np.nan
    return float(stat), float(p)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_backtest.py -q
```

Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add code/dfm/backtest.py code/tests/test_dfm_backtest.py
git commit -m "feat(dfm): walk-forward harness with purge, embargo, checkpointing and resume"
```

---

## Task 12: Benchmarks — AutoARIMA, AR(1), frozen-production replication

`code/production/model.py` exposes `nowcast()` for a single live month but no walk-forward, so the replication is composed here from the frozen constants (`targets.HEADLINE`, `production.model.LGB`) — imported, never copied by hand, so a governance change to production shows up as a changed benchmark rather than a silent divergence.

**Files:**
- Create: `code/dfm/benchmarks.py`
- Test: `code/tests/test_dfm_benchmarks.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_benchmarks.py`:

```python
"""Benchmarks read frozen production constants rather than restating them."""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import benchmarks as BM


class TestFrozenConstants:
    def test_reads_lambda_and_weights_from_targets(self):
        from targets import HEADLINE
        assert BM.PROD_LAMBDA == HEADLINE.lam
        assert BM.PROD_WEIGHTS == HEADLINE.overlay_weights

    def test_reads_lgbm_params_from_production(self):
        from production.model import LGB
        assert BM.LGB_PARAMS == LGB

    def test_expected_frozen_values(self):
        """Guards against a silent governance change going unnoticed."""
        assert BM.PROD_LAMBDA == 0.5
        assert BM.PROD_WEIGHTS == {"tvp": 0.5, "lgbm": 0.5}


class TestAR1:
    def test_ar1_backtest_shape(self):
        idx = pd.date_range("2010-01-31", "2018-12-31", freq="ME")
        rng = np.random.default_rng(0)
        y = pd.Series(np.cumsum(rng.normal(size=len(idx))) * 0.1 + 2.0, index=idx)
        bt = BM.ar1_backtest(y, start_year=2015, end_year=2018)
        assert len(bt) == 48
        assert {"actual", "pred"}.issubset(bt.columns)

    def test_ar1_is_causal(self):
        """Poisoning the future must not change any earlier prediction."""
        idx = pd.date_range("2010-01-31", "2018-12-31", freq="ME")
        rng = np.random.default_rng(1)
        y = pd.Series(rng.normal(size=len(idx)), index=idx)
        base = BM.ar1_backtest(y, start_year=2015, end_year=2016)
        y2 = y.copy()
        y2[y2.index > pd.Timestamp("2017-01-31")] = 1e6
        after = BM.ar1_backtest(y2, start_year=2015, end_year=2016)
        pd.testing.assert_frame_equal(base, after)


@pytest.mark.slow
class TestProductionReplication:
    def test_reproduces_the_documented_rel_rmse(self):
        """AA ~0.467 and production ~0.416 on 2015-2024 (README model table)."""
        res = BM.production_backtest()
        aa = np.sqrt(((res["aa_pred"] - res["actual"]) ** 2).mean())
        prod = np.sqrt(((res["prod_pred"] - res["actual"]) ** 2).mean())
        assert 0.40 < aa < 0.53, f"AA RMSE {aa:.3f} outside the documented range"
        assert prod / aa < 0.98, f"production rel-RMSE {prod/aa:.3f} shows no edge"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_benchmarks.py -q -m "not slow"
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dfm.benchmarks'`.

- [ ] **Step 3: Write the implementation**

Create `code/dfm/benchmarks.py`:

```python
"""
dfm/benchmarks.py — AutoARIMA, AR(1) and a frozen-production replication.

Production exposes nowcast() for a single live month, not a walk-forward, so the
replication is composed here from the FROZEN constants, imported rather than
restated: targets.HEADLINE carries lambda / overlay weights / pinned set, and
production.model.LGB carries the LightGBM parameters. A governance change to
production therefore changes this benchmark too, instead of silently diverging.

Everything here is READ-ONLY with respect to production code.
"""
import numpy as np
import pandas as pd

import factors as F
import uk_model_zoo as Z
import validation as V
from production.model import LGB as LGB_PARAMS
from targets import HEADLINE

from dfm import config as C

PROD_LAMBDA = HEADLINE.lam
PROD_WEIGHTS = dict(HEADLINE.overlay_weights)
PROD_PINNED = list(HEADLINE.pinned)
TARGET = HEADLINE.name


def _weighted_overlay(members):
    """
    NaN-safe weighted overlay, renormalising over the members that produced a
    finite prediction — the same rule two_stage._weighted_combo applies, so a
    dead leg degrades the overlay instead of nulling the whole forecast.
    """
    num = 0.0
    wsum = 0.0
    for name, value in members.items():
        if value is None or not np.isfinite(value):
            continue
        w = PROD_WEIGHTS[name]
        num += w * float(value)
        wsum += w
    return num / wsum if wsum > 0 else np.nan


def ar1_backtest(y, start_year=None, end_year=None):
    """Expanding-window AR(1) on a monthly series. The floor benchmark."""
    start_year = start_year or C.EVAL_START
    end_year = end_year or C.EVAL_END
    y = y.dropna()
    rows = []
    for m in y.index:
        if not (start_year <= m.year <= end_year):
            continue
        hist = y[y.index < m]
        if len(hist) < 24:
            continue
        x, t = hist.values[:-1], hist.values[1:]
        xc, tc = x - x.mean(), t - t.mean()
        phi = float(xc @ tc / (xc @ xc)) if xc @ xc > 0 else 0.0
        const = float(t.mean() - phi * x.mean())
        rows.append(dict(month=m, actual=float(y.loc[m]),
                         pred=const + phi * float(hist.iloc[-1])))
    return pd.DataFrame(rows).set_index("month") if rows else pd.DataFrame()


def load_production_matrix():
    """The frozen headline factor matrix, pub-lagged exactly as production builds it."""
    raw, status = F.build_matrix(names=PROD_PINNED + [TARGET])
    live = [n for n in PROD_PINNED if status.get(n) != "unavailable"]
    raw = raw[raw.index.year >= 1997]
    df = F.apply_publication_lags(raw, live)
    for rf in HEADLINE.reg:
        if rf in df.columns:
            observed = df[rf].dropna()
            if len(observed):
                span = df.index <= observed.index[-1]
                df.loc[span, rf] = df.loc[span, rf].fillna(0)
    return df.resample("ME").last(), live


def autoarima_backtest(df=None, start_year=None, end_year=None):
    """AutoARIMA anchor — the model production is judged against."""
    if df is None:
        df, _ = load_production_matrix()
    aa = Z.AutoARIMA().backtest(df, [], TARGET, start_year=HEADLINE.aa_start,
                                end_year=end_year or C.EVAL_END)
    lo = start_year or C.EVAL_START
    return aa[(aa.index.year >= lo) & (aa.index.year <= (end_year or C.EVAL_END))]


def production_backtest(start_year=None, end_year=None):
    """
    Walk-forward replication of AA + 0.25*TVP + 0.25*LGBM on this harness.

    RegEvent is NOT included: it is a live arithmetic layer keyed to announced cap
    resets, and the frozen backtest numbers in README/final_model.md predate it.
    Its contribution is reported separately in Plan 4.
    """
    import lightgbm as lgb

    start_year = start_year or C.EVAL_START
    end_year = end_year or C.EVAL_END
    df, live = load_production_matrix()
    aa = autoarima_backtest(df, start_year, end_year)
    resid_hist = (aa["actual"] - aa["pred"]).rename("resid")

    full_aa = Z.AutoARIMA().backtest(df, [], TARGET, start_year=HEADLINE.aa_start,
                                     end_year=end_year)
    resid_all = (full_aa["actual"] - full_aa["pred"]).rename("resid")

    rows = []
    for m in aa.index:
        d = df.copy()
        d["resid"] = resid_all.reindex(d.index)
        d.loc[d.index >= m, "resid"] = np.nan          # only past residuals are known
        try:
            tvp, _ = Z.TVP().nowcast(d, live, "resid", nowcast_date=m)
        except Exception:
            tvp = np.nan
        data = pd.DataFrame(d[live]).join(d["resid"]).dropna(subset=["resid"])
        train = V.purge_embargo(data, m, C.PURGE_OVERLAY, C.EMBARGO_OVERLAY)
        try:
            gbm = float(lgb.LGBMRegressor(**LGB_PARAMS)
                        .fit(train[live], train["resid"]).predict(d.loc[[m], live])[0])
        except Exception:
            gbm = np.nan
        overlay = _weighted_overlay({"tvp": tvp, "lgbm": gbm})
        pred = float(aa.loc[m, "pred"]) + (PROD_LAMBDA * overlay
                                           if np.isfinite(overlay) else 0.0)
        rows.append(dict(month=m, actual=float(aa.loc[m, "actual"]),
                         aa_pred=float(aa.loc[m, "pred"]),
                         tvp_resid=tvp, lgbm_resid=gbm, prod_pred=pred))
    return pd.DataFrame(rows).set_index("month")
```

- [ ] **Step 4: Run the fast tests**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_benchmarks.py -q -m "not slow"
```

Expected: `5 passed`.

- [ ] **Step 5: Run the replication once** (network; `dangerouslyDisableSandbox: true`; ~10–25 min)

The `slow` marker was already registered in `code/tests/conftest.py` in Task 2 step 5.

```bash
PYTHONPATH=code $Q -u -c "
import numpy as np
from dfm import benchmarks as BM
res = BM.production_backtest()
res.to_csv('data/dfm/production_replication.csv')
aa = np.sqrt(((res['aa_pred']-res['actual'])**2).mean())
pr = np.sqrt(((res['prod_pred']-res['actual'])**2).mean())
print(f'n={len(res)}  AA RMSE={aa:.4f}  prod RMSE={pr:.4f}  rel={pr/aa:.4f}')
"
```

Expected: AA RMSE near **0.467** and rel-RMSE near **0.89** (README model table). **If the replication is more than ~10% off those numbers, stop and reconcile before running any DFM arm** — every headline claim in this project is measured against this baseline, so a wrong baseline invalidates the whole comparison. Record the actual numbers in `docs/dfm_backtest.md` whatever they are.

- [ ] **Step 6: Commit**

```bash
git add code/dfm/benchmarks.py code/tests/test_dfm_benchmarks.py data/dfm/production_replication.csv
git commit -m "feat(dfm): AutoARIMA, AR(1) and frozen-production replication benchmarks"
```

---

## Task 13: Experiments runner and the timing probe

**Files:**
- Create: `code/dfm/experiments.py`
- Test: `code/tests/test_dfm_experiments.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_experiments.py`:

```python
"""Experiment runner: panel assembly, timing probe, results table."""
import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import experiments as E


class TestModelPanel:
    def test_target_column_is_appended_last(self, monkeypatch):
        idx = pd.date_range("2015-01-31", "2016-12-31", freq="ME")
        fake_obs = pd.DataFrame(np.random.default_rng(0).normal(size=(len(idx), 3)),
                                index=idx, columns=["a", "b", "c"])
        fake_idx = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
        monkeypatch.setattr(E.P, "asof", lambda ref, ts, start=None: fake_obs)
        monkeypatch.setattr(E.T, "cpi_index", lambda force=False: fake_idx)
        panel = E.model_panel(pd.Timestamp("2016-12-31"), pd.Timestamp("2016-12-31"))
        assert panel.columns[-1] == "cpi_mom"
        assert len(panel.columns) == 4


class TestTimingProbe:
    def test_reports_seconds_per_fit_and_a_projection(self, monkeypatch):
        calls = {"n": 0}

        class Fast:
            def __init__(self, cfg=None): pass
            def fit(self, panel, target, warm_start=None):
                calls["n"] += 1
                return self
            def nowcast(self, panel, target, row=-1):
                return dict(mean=0.0, p05=-1.0, p95=1.0, sd=0.5, draws=np.zeros(3))
            def posterior_mean(self): return {}

        monkeypatch.setattr(E, "DFMNowcaster", Fast)
        idx = pd.date_range("2010-01-31", "2016-12-31", freq="ME")
        panel = pd.DataFrame(np.random.default_rng(0).normal(size=(len(idx), 3)),
                             index=idx, columns=["a", "b", "cpi_mom"])
        rep = E.timing_probe(panel, n_months=3, arm="E1")
        assert calls["n"] == 3
        assert rep["seconds_per_fit"] > 0
        assert rep["projected_hours_full_sweep"] > 0

    def test_projection_uses_the_configured_config_count(self, monkeypatch):
        class Fast:
            def __init__(self, cfg=None): pass
            def fit(self, panel, target, warm_start=None): return self
            def nowcast(self, panel, target, row=-1):
                return dict(mean=0.0, p05=-1.0, p95=1.0, sd=0.5, draws=np.zeros(3))
            def posterior_mean(self): return {}

        monkeypatch.setattr(E, "DFMNowcaster", Fast)
        idx = pd.date_range("2010-01-31", "2016-12-31", freq="ME")
        panel = pd.DataFrame(np.random.default_rng(0).normal(size=(len(idx), 3)),
                             index=idx, columns=["a", "b", "cpi_mom"])
        a = E.timing_probe(panel, n_months=2, arm="E1", n_configs=1)
        b = E.timing_probe(panel, n_months=2, arm="E1", n_configs=12)
        assert b["projected_hours_full_sweep"] > a["projected_hours_full_sweep"] * 5


class TestResultsTable:
    def test_summarise_joins_arm_metrics_against_benchmarks(self):
        idx = pd.date_range("2015-01-31", "2024-12-31", freq="ME")
        rng = np.random.default_rng(2)
        actual = pd.Series(rng.normal(size=len(idx)), index=idx)
        bt = pd.DataFrame({"actual": actual, "pred": actual + rng.normal(size=len(idx)) * 0.5})
        bench = pd.DataFrame({"actual": actual, "pred": actual + rng.normal(size=len(idx))})
        tab = E.summarise({"E1": bt, "AutoARIMA": bench})
        assert set(tab.index) == {"E1", "AutoARIMA"}
        assert {"rmse_full", "mae_full", "rmse_shock", "rmse_ex_shock"}.issubset(tab.columns)
        assert tab.loc["E1", "rmse_full"] < tab.loc["AutoARIMA", "rmse_full"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_experiments.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'dfm.experiments'`.

- [ ] **Step 3: Write the implementation**

Create `code/dfm/experiments.py`:

```python
"""
dfm/experiments.py — assemble the model panel, probe runtime, run an arm.

Cost control is a first-class task, not an afterthought (spec §5.3): the timing
probe runs on the REAL panel before any sweep, and its projection decides whether
the documented fallbacks are needed.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from dfm import backtest as B
from dfm import benchmarks as BM
from dfm import config as C
from dfm import panel as P
from dfm import target as T
from dfm.model import DFMConfig, DFMNowcaster

N_CONFIGS_FULL = 12          # 6 arms x 2 modes
N_MONTHS_FULL = 120          # 2015-2024


def model_panel(ref_month, asof_ts, start=None):
    """Observation matrix + the CPI MoM target column, aligned on month-end."""
    obs = P.asof(ref_month, asof_ts, start=start)
    idx = T.cpi_index()
    mom = T.mom_from_index(idx).reindex(obs.index)
    out = obs.copy()
    out["cpi_mom"] = mom
    return out


def timing_probe(panel, n_months=3, arm="E1", n_configs=N_CONFIGS_FULL,
                 n_months_full=N_MONTHS_FULL, draws=None):
    """
    Time a handful of real fits and project the full sweep.

    Returns seconds_per_fit and projected_hours_full_sweep. The projection assumes
    the first fit is the expensive one and the rest are warm-started, which is how
    the sweep actually runs.
    """
    cfg = DFMConfig(arm=arm, draws=draws or C.DRAWS_WARM,
                    burn=min(C.BURN_WARM, (draws or C.DRAWS_WARM) // 4))
    months = [m for m in panel.index if np.isfinite(panel.loc[m, "cpi_mom"])][-n_months:]
    t0 = time.time()
    prev = None
    for m in months:
        train = panel.loc[:m].copy()
        train.loc[m, "cpi_mom"] = np.nan
        model = DFMNowcaster(cfg).fit(train, target="cpi_mom", warm_start=prev)
        model.nowcast(train, target="cpi_mom")
        try:
            prev = model.posterior_mean()
        except Exception:
            prev = None
    secs = (time.time() - t0) / max(len(months), 1)
    return dict(seconds_per_fit=secs, n_probe_months=len(months),
                projected_hours_full_sweep=secs * n_configs * n_months_full / 3600.0,
                arm=arm, draws=cfg.draws)


def run_arm(arm="E1", mode="standalone", panel=None, start_year=None, end_year=None,
            draws=None, burn=None, verbose=True):
    """Walk-forward one arm and write its results CSV."""
    panel = model_panel(pd.Timestamp(f"{end_year or C.EVAL_END}-12-31"),
                        pd.Timestamp(f"{end_year or C.EVAL_END}-12-31")) \
        if panel is None else panel
    cfg = DFMConfig(arm=arm, mode=mode,
                    draws=draws or C.DRAWS_WARM, burn=burn or C.BURN_WARM)
    purge = C.PURGE_STANDALONE if mode == "standalone" else C.PURGE_OVERLAY
    embargo = C.EMBARGO_STANDALONE if mode == "standalone" else C.EMBARGO_OVERLAY
    tag = f"{arm}_{mode}"
    bt = B.walk_forward(panel, target="cpi_mom", model_cls=DFMNowcaster, model_cfg=cfg,
                        start_year=start_year, end_year=end_year,
                        purge=purge, embargo=embargo,
                        checkpoint_dir=C.CHECKPOINTS, tag=tag, verbose=verbose)
    path = os.path.join(C.DATA, f"backtest_{tag}.csv")
    bt.to_csv(path)
    return bt, path


def to_yoy(bt, index=None):
    """Convert a MoM backtest to YoY space so it is comparable with production."""
    index = index if index is not None else T.cpi_index()
    rows = []
    for m, r in bt.iterrows():
        try:
            rows.append(dict(month=m,
                             actual=T.yoy_from_mom(r["actual"], index, m),
                             pred=T.yoy_from_mom(r["pred"], index, m)))
        except ValueError:
            continue
    return pd.DataFrame(rows).set_index("month")


def summarise(results):
    """name -> backtest frame  ==>  one metrics row per model."""
    rows = {}
    for name, bt in results.items():
        m = B.metrics(bt)
        rows[name] = {f"{stat}_{w}": m.loc[w, stat]
                      for w in m.index for stat in ("rmse", "mae")}
        rows[name]["n"] = int(m.loc["full", "n"])
    return pd.DataFrame(rows).T
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_experiments.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add code/dfm/experiments.py code/tests/test_dfm_experiments.py
git commit -m "feat(dfm): experiment runner, timing probe and results summariser"
```

---

## Task 14: Run the timing probe and decide the budget

**Files:** none created; this task produces a recorded measurement and possibly a config change.

- [ ] **Step 1: Run the probe on the real panel** (network; `dangerouslyDisableSandbox: true`)

```bash
PYTHONPATH=code $Q -u -c "
import json, pandas as pd
from dfm import experiments as E, config as C
panel = E.model_panel(pd.Timestamp('2024-12-31'), pd.Timestamp('2024-12-31'))
print('panel', panel.shape, panel.index[0].date(), '->', panel.index[-1].date())
rep = E.timing_probe(panel, n_months=3, arm='E1')
print(json.dumps(rep, indent=2))
open('data/dfm/timing_probe_E1.json','w').write(json.dumps(rep, indent=2))
"
```

- [ ] **Step 2: Apply the documented fallbacks if the projection exceeds 8 hours**

In fixed order, stopping as soon as the projection fits (spec §5.3):

1. Ablations to the best arm only — a Plan 4 change; note it, do nothing now.
2. Halve post-warm-start draws: set `C.DRAWS_WARM = 400`, `C.BURN_WARM = 100` in `code/dfm/config.py`.
3. Quarterly parameter refit with monthly filtering — a Plan 3 change; note it.

Whichever fallback is applied gets recorded in `docs/dfm_backtest.md` in Task 15, because it changes what the numbers mean.

- [ ] **Step 3: Commit the measurement**

```bash
git add data/dfm/timing_probe_E1.json code/dfm/config.py
git commit -m "perf(dfm): record the E1 timing probe and set the draw budget"
```

---

## Task 15: Run Stage 1 and write it up — the gate

Arm E1 is not "done" until it has a completed walk-forward, a written result, and finite predictions for ≥95% of test months (spec §4.3). Stage 2 does not start until this task passes.

**Files:**
- Create: `docs/dfm_backtest.md`
- Create: `data/dfm/stage1_summary.csv`

- [ ] **Step 1: Run the E1 walk-forward, standalone mode** (network; `dangerouslyDisableSandbox: true`; hours — resumable)

```bash
PYTHONPATH=code $Q -u -c "
import pandas as pd
from dfm import experiments as E
bt, path = E.run_arm(arm='E1', mode='standalone', verbose=True)
print('wrote', path, len(bt), 'months')
"
```

If interrupted, rerun the same command — checkpoints in `data/dfm/checkpoints/` make it resume.

- [ ] **Step 2: Score it in YoY space against the benchmarks**

```bash
PYTHONPATH=code $Q -u -c "
import numpy as np, pandas as pd
from dfm import experiments as E, backtest as B, benchmarks as BM, target as T

e1 = pd.read_csv('data/dfm/backtest_E1_standalone.csv', index_col=0, parse_dates=True)
e1_yoy = E.to_yoy(e1)
prod = pd.read_csv('data/dfm/production_replication.csv', index_col=0, parse_dates=True)

aa   = prod[['actual','aa_pred']].rename(columns={'aa_pred':'pred'})
pr   = prod[['actual','prod_pred']].rename(columns={'prod_pred':'pred'})
ar1  = BM.ar1_backtest(prod['actual'])

tab = E.summarise({'E1_standalone': e1_yoy, 'AutoARIMA': aa,
                   'Production': pr, 'AR(1)': ar1})
tab['rel_rmse_vs_aa'] = tab['rmse_full'] / tab.loc['AutoARIMA','rmse_full']
tab.to_csv('data/dfm/stage1_summary.csv')
print(tab.round(4).to_string())

common = e1_yoy.index.intersection(aa.index)
s, p = B.dm_test((e1_yoy.loc[common,'pred']-e1_yoy.loc[common,'actual']),
                 (aa.loc[common,'pred']-aa.loc[common,'actual']))
print(f'DM vs AA: stat={s:.3f} p={p:.3f}')
cov = e1['pred'].notna().mean()
print(f'coverage: {cov:.1%}')
assert cov >= 0.95, 'GATE FAILED: under 95% finite predictions'
"
```

- [ ] **Step 3: Write `docs/dfm_backtest.md`**

Create the file with this structure, filling every bracketed value from the actual output — **no bracket may survive into the commit**:

```markdown
# DFM research backtest — results log

Branch `dfm-research`. Spec:
[docs/superpowers/specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md](superpowers/specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md).

Each stage appends one section. A stage is not complete until its section is written.

## Baselines (this harness, 2015–2024)

| Model | RMSE | rel vs AA | Source |
|---|---|---|---|
| AR(1) | [x] | [x] | `dfm.benchmarks.ar1_backtest` |
| AutoARIMA | [x] | 1.000 | `dfm.benchmarks.autoarima_backtest` |
| Production replication | [x] | [x] | `dfm.benchmarks.production_backtest` |

README documents AA 0.467 and production rel-RMSE 0.89. Any divergence here is
[stated and explained].

## Release calendar coverage

| source | rows | share |
|---|---|---|
| ons_api | [x] | [x] |
| provider_doc | [x] | [x] |
| rule | [x] | [x] |

CPI bulletin: [x] real ONS dates, [x] rule fallbacks (pre-2016).

## Compute

Timing probe (E1, real panel): [x] s/fit; projected full sweep [x] h.
Fallbacks applied: [none | which ones and why].

## Stage 1 — E1, classical Stock-Watson DFM

| Window | RMSE | MAE | n |
|---|---|---|---|
| full 2015–2024 | [x] | [x] | [x] |
| shock 2022–23 | [x] | [x] | [x] |
| ex-shock | [x] | [x] | [x] |
| pre-2020 | [x] | [x] | [x] |

rel-RMSE vs AA [x]; DM vs AA stat [x], p [x]. Coverage [x]%.

**Read:** [one paragraph — did the MoM reformulation fix the 1.97 problem the prior
DFMs had? Is E1 competitive with AA, or only with AR(1)? State it plainly, including
if the answer is no.]

**Gate:** [PASS / FAIL] — [what Stage 2 is or is not cleared to build on].
```

- [ ] **Step 4: Verify no placeholders survived**

```bash
grep -n "\[x\]\|TBD\|TODO" docs/dfm_backtest.md; echo "exit=$? (1 = clean)"
```

Expected: `exit=1`.

- [ ] **Step 5: Commit**

```bash
git add docs/dfm_backtest.md data/dfm/stage1_summary.csv data/dfm/backtest_E1_standalone.csv
git commit -m "test(dfm): Stage 1 walk-forward results and the stage gate write-up"
```

- [ ] **Step 6: Run the whole DFM test suite**

```bash
PYTHONPATH=code $Q -m pytest code/tests/ -q -m "not slow and not network"
```

Expected: every test passes, including `test_headline_regression.py` and `test_dfm_isolation.py`. **If the isolation test fails, production was touched — revert that change before doing anything else.**

---

## What Plan 1 does not cover

Deliberately deferred, each to its own plan, because the spec's §4.3 gate forbids building a layer before the one below it has a written result:

- **Plan 2 — Stage 2.** E2 stochastic volatility (Kim-Shephard-Chib mixture + FFBS) and E3 Stock-Watson outlier mixture. Adds `sampler.draw_sv` and `sampler.draw_outliers`, flips the `NotImplementedError` guards in `DFMConfig`.
- **Plan 3 — Stages 3–5.** E4 full VAR under the Minnesota prior (the `draw_factor_ar(diagonal=False)` path already exists and is tested), E5 block-restricted + spike-and-slab loadings via `loading_mask`, E6 hierarchical shrinkage. Also the overlay mode wired end-to-end, and the exogenous reg block `gamma_z`.
- **Plan 4 — evaluation layer.** Bańbura-Modugno news decomposition, parameter-stability paths, the T-30 → R-1 forecast path, leave-one-block-out ablations, figures, `docs/dfm_model_card.md`, K-sensitivity, the standalone 12+1 purge sensitivity, and the May–Jul 2026 blind scoring.

---

## Self-review against the spec

| Spec section | Covered by |
|---|---|
| Non-goal / production isolation | Task 1 (`test_dfm_isolation.py`), Task 15 step 6 |
| §1.1 measurement equation | Task 8 (`design_matrix`), Task 10 |
| §1.2 state equation, diagonal Φ for E1–E3 | Task 8 (`transition_matrix`), Task 9 (`draw_factor_ar`) |
| §1.3 outlier mixture | **Plan 2** — guarded by `NotImplementedError` in `DFMConfig` |
| §1.4 loading structure, mask plumbing | Task 9 (`mask=`), Task 10 (`loading_mask`) — E5/E6 in Plan 3 |
| §1.5 hierarchical shrinkage | **Plan 3** — guarded |
| §1.6 blocks, 32 series | Task 5 |
| §1.7 availability, staleness | Task 5 step 5, `panel.staleness_report` |
| §2.1 dated release calendar | Tasks 2, 3, 5 step 6 |
| §2.2 as-of panel, daily/quarterly aggregation | Tasks 4, 6 |
| §2.3 T-30 → R-1 path | **Plan 4** (origins live in `config.ORIGIN_OFFSETS` already) |
| §2.4 leakage tests | Task 4, Task 6, Task 11 |
| §3 MoM target, YoY reconstruction, rounding floor | Task 7 |
| §4.1 arm ladder | Task 10 (`ARMS`) |
| §4.2 both modes | Task 13 (`run_arm(mode=)`); overlay wired in Plan 3 |
| §4.3 sequential gate | Task 15 |
| §4.4 ablations | **Plan 4** |
| §5.1 Gibbs blocks | Tasks 8, 9, 10 |
| §5.2 draws, warm start | Task 10 (`warm_start`), Task 11 |
| §5.3 budget, checkpointing, fallbacks | Tasks 11, 13, 14 |
| §6.1 models compared | Task 12, Task 15 |
| §6.2 RMSE, MAE, cost | Task 11 (`metrics`, `fit_seconds`); path/news/stability in Plan 4 |
| §6.3 DM, calm overshoot, error correlation | Task 11 (`dm_test`); the rest in Plan 4 |
| §6.4 purge asymmetry | Task 11, `config.PURGE_*` |
| §7 selection rule | **Plan 4** (applied once arms exist) |
| §8 module boundaries | File Structure table |
| §9 testing | Tasks 2, 4, 6, 7, 8, 9, 11 |

Known gaps carried forward deliberately: `news.py`, `stability.py`, `report.py`, `diagram.py` and the RegEvent contribution are Plan 4; the exogenous `gamma_z` block is stubbed as zeros in Task 10 and wired in Plan 3.
