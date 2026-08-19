"""Data-freshness guards.

Two silent failures cost the July-2026 nowcast (docs/postmortem_2026_07.md §3):
a CSV drop-in that had gone stale shadowed a fresher live fetch, and a regulatory
factor's missing months were filled with 0 — asserting "no Ofgem cap reset" in the
month of a +13% reset. Both are offline-testable.
"""
import numpy as np
import pandas as pd
import pytest

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "new_factors"))

import factors as F
import two_stage as TS
from targets import Target


def _write_csv(path, pairs):
    pd.DataFrame({"date": [d for d, _ in pairs], "value": [v for _, v in pairs]}).to_csv(
        path, index=False)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(F, "REGISTRY", dict(F.REGISTRY))
    return tmp_path


def _months(start, n):
    return [str(d.date()) for d in pd.date_range(start, periods=n, freq="ME")]


class TestCsvSplicing:
    def _register(self, name, live_series, transform="level", **extra):
        F.REGISTRY[name] = dict(fetch=lambda: live_series, transform=transform,
                                pub_lag=0, candidate=True, csv=f"{name}.csv", **extra)

    def test_live_months_past_the_csv_end_are_spliced_in(self, sandbox):
        idx = _months("2026-01-31", 6)
        _write_csv(sandbox / "f.csv", list(zip(idx[:4], [1.0, 2.0, 3.0, 4.0])))
        live = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                         index=pd.to_datetime(idx))
        self._register("f", live)
        s, status = F.load_factor("f")
        assert status == "csv+live"
        assert len(s) == 6 and s.iloc[-1] == 6.0

    def test_csv_still_wins_inside_its_own_span(self, sandbox):
        idx = _months("2026-01-31", 4)
        _write_csv(sandbox / "f.csv", list(zip(idx, [10.0, 10.0, 10.0, 10.0])))
        live = pd.Series([1.0, 1.0, 1.0, 1.0, 99.0],
                         index=pd.to_datetime(_months("2026-01-31", 5)))
        self._register("f", live)
        s, status = F.load_factor("f")
        assert list(s.iloc[:4]) == [10.0] * 4        # drop-in overrides the live values
        assert s.iloc[-1] == 99.0                    # and is extended by the fresher month

    def test_splice_happens_before_the_transform(self, sandbox):
        """A yoy factor must span the join, not restart at it."""
        idx = _months("2025-01-31", 18)
        levels = pd.Series(np.arange(100.0, 118.0), index=pd.to_datetime(idx))
        _write_csv(sandbox / "f.csv", list(zip(idx[:15], levels.iloc[:15])))
        self._register("f", levels, transform="yoy")
        s, status = F.load_factor("f")
        assert status == "csv+live"
        expected = (levels.iloc[-1] / levels.iloc[-13] - 1) * 100
        assert s.iloc[-1] == pytest.approx(expected)

    def test_csv_frozen_is_never_extended(self, sandbox):
        idx = _months("2026-01-31", 6)
        _write_csv(sandbox / "t.csv", list(zip(idx[:4], [1.0, 2.0, 3.0, 4.0])))
        live = pd.Series(np.arange(1.0, 7.0), index=pd.to_datetime(idx))
        self._register("t", live, csv_frozen=True)
        s, status = F.load_factor("t")
        assert status == "csv"
        assert len(s) == 4                            # the truncation pins the nowcast month

    def test_a_failing_live_fetch_leaves_the_csv_usable(self, sandbox):
        idx = _months("2026-01-31", 3)
        _write_csv(sandbox / "f.csv", list(zip(idx, [1.0, 2.0, 3.0])))

        def boom():
            raise RuntimeError("source down")

        F.REGISTRY["f"] = dict(fetch=boom, transform="level", pub_lag=0,
                               candidate=True, csv="f.csv")
        s, status = F.load_factor("f")
        assert status == "csv" and len(s) == 3


class TestRegulatoryFillStopsAtTheSourceEnd:
    def test_zeros_inside_the_span_nan_after_it(self, monkeypatch):
        idx = pd.date_range("2026-01-31", periods=6, freq="ME")
        raw = pd.DataFrame({
            "cpi_yoy": [3.0, 3.1, 3.2, 3.0, 2.9, np.nan],
            "ofgem_cap_delta": [100.0, np.nan, np.nan, 50.0, np.nan, np.nan],
            "oil_brent": np.linspace(0.01, 0.06, 6),
        }, index=idx)
        target = Target(name="cpi_yoy", label="t", pinned=("oil_brent", "ofgem_cap_delta"),
                        overlay_weights={"tvp": 0.5, "lgbm": 0.5}, lam=0.5,
                        aa_start=2001, start=2015, end=None,
                        reg=("ofgem_cap_delta",))
        monkeypatch.setattr(TS.F, "build_matrix",
                            lambda names=None: (raw.copy(), {c: "csv" for c in raw.columns}))
        df, live, _ = TS.load_matrix(target=target)
        cap = df["ofgem_cap_delta"]
        assert cap.loc[idx[1]] == 0.0 and cap.loc[idx[2]] == 0.0   # quiet months inside the span
        assert pd.isna(cap.loc[idx[4]]) and pd.isna(cap.loc[idx[5]])  # past the source end
