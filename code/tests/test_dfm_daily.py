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
