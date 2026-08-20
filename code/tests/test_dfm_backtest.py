"""Walk-forward harness: causality, purge, checkpoint/resume."""
import sys, os
import json
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import backtest as B


class FakeModel:
    """Records the training window it was handed; predicts its own row count.

    The harness standardises each fold and rescales the returned nowcast by the
    target's own mu/sd, so `pred` is NOT `len(panel)` — it is
    `len(panel) * sd + mu`. Tests therefore assert on the training windows this
    model records and on the shape of the output, never on the predicted value.
    """
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
    # Starts in 2008 so every scored month in 2016-2017 clears the harness's
    # 60-month minimum training window, with and without a 12+1 purge.
    idx = pd.date_range("2008-01-31", "2018-12-31", freq="ME")
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
        assert len(purged) == len(unpurged) == 12
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
