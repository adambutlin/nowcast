"""
Release-calendar resolver: slug patterns, caching, rule fallback.

Network tests are marked `network` and skipped by default; the offline tests use
a monkeypatched fetcher so the logic is testable with no egress.
"""
import json
import sys, os
import urllib.error

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import release_calendar as RC


class _FakeResponse:
    """Minimal stand-in for the `http.client.HTTPResponse` context manager."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


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


class TestFetchReleasePage:
    """`_fetch_release_page` must distinguish a confirmed 404 from a transient
    failure: only the former is a legitimate `None` (used to fall back to the
    rule); everything else must raise so a bad month gets rerun, not baked in."""

    def test_200_with_valid_json_returns_parsed_timestamp(self, monkeypatch):
        body = json.dumps({"description": {"release_date": "2024-12-18T07:00:00"}}).encode()
        monkeypatch.setattr(RC.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(body))
        ts = RC._fetch_release_page("consumerpriceinflationuknovember2024")
        assert ts == pd.Timestamp("2024-12-18 07:00:00")
        assert ts.tz is None

    def test_404_returns_none(self, monkeypatch):
        def raise_404(req, timeout=None):
            raise urllib.error.HTTPError(url="http://x", code=404, msg="Not Found", hdrs=None, fp=None)

        monkeypatch.setattr(RC.urllib.request, "urlopen", raise_404)
        assert RC._fetch_release_page("nosuchslug2013") is None

    def test_non_404_http_error_raises(self, monkeypatch):
        def raise_500(req, timeout=None):
            raise urllib.error.HTTPError(url="http://x", code=500, msg="Server Error", hdrs=None, fp=None)

        monkeypatch.setattr(RC.urllib.request, "urlopen", raise_500)
        with pytest.raises(Exception):
            RC._fetch_release_page("someslug2024")

    def test_url_error_raises(self, monkeypatch):
        def raise_url_error(req, timeout=None):
            raise urllib.error.URLError("connection reset")

        monkeypatch.setattr(RC.urllib.request, "urlopen", raise_url_error)
        with pytest.raises(Exception):
            RC._fetch_release_page("someslug2024")

    def test_malformed_json_raises(self, monkeypatch):
        monkeypatch.setattr(RC.urllib.request, "urlopen",
                             lambda req, timeout=None: _FakeResponse(b"not json at all"))
        with pytest.raises(Exception):
            RC._fetch_release_page("someslug2024")

    def test_missing_release_date_raises(self, monkeypatch):
        body = json.dumps({"description": {}}).encode()
        monkeypatch.setattr(RC.urllib.request, "urlopen", lambda req, timeout=None: _FakeResponse(body))
        with pytest.raises(Exception):
            RC._fetch_release_page("someslug2024")


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

    def test_transient_failure_is_not_memoised(self, monkeypatch):
        """A raised transient error must propagate and must NOT poison the
        cache — otherwise a network blip would permanently degrade a month
        to `rule` for the rest of the process, which is a look-ahead risk."""
        def boom(slug):
            raise RuntimeError(f"ONS fetch failed for slug '{slug}': timeout")

        monkeypatch.setattr(RC, "_fetch_release_page", boom)
        RC._MEMO.clear()
        with pytest.raises(RuntimeError):
            RC.ons_release_date(pd.Timestamp("2024-11-30"))
        assert pd.Timestamp("2024-11-30").to_period("M") not in RC._MEMO

    def test_pre_2016_short_circuits_without_fetching(self, monkeypatch):
        """Reference months before the ONS release-page era are a known
        absence, not a lookup — must not fire any HTTP requests."""
        calls = []
        monkeypatch.setattr(RC, "_fetch_release_page", lambda slug: calls.append(slug))
        RC._MEMO.clear()
        assert RC.ons_release_date(pd.Timestamp("2015-06-30")) is None
        assert calls == []


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
