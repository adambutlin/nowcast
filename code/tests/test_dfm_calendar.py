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
