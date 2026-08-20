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
