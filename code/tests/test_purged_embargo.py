"""
Purged + embargoed walk-forward backtest.

  validation.purge_embargo  — index-level helper that drops training rows whose
                              label-horizon overlaps a test fold (purge) plus a
                              small additional gap (embargo).
  BaseModel.backtest        — applies the helper per test fold when
                              purge_horizon/embargo > 0 (default 0 = no change).

Spec: docs/superpowers/specs/2026-06-22-purged-embargo-backtest-design.md
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "new_factors"))

import validation as V
import uk_model_zoo as Z
import two_stage as TS


def _monthly(start, n):
    return pd.date_range(start, periods=n, freq="MS")


# ─────────────────────────────────────────────────────────────────────────────
# purge_embargo unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPurgeEmbargo:
    def _train(self):
        idx = _monthly("2015-01-01", 108)          # 2015-01 .. 2023-12
        return pd.DataFrame({"x": range(len(idx))}, index=idx)

    def test_identity_when_disabled(self):
        """horizon=0, embargo=0 must leave a causal training set untouched."""
        train = self._train()
        out = V.purge_embargo(train, pd.Timestamp("2024-01-01"), horizon=0, embargo=0)
        pd.testing.assert_frame_equal(out, train)

    def test_drops_rows_inside_horizon_plus_embargo(self):
        """With horizon=12, embargo=1 the last 13 months before the fold go."""
        train = self._train()
        out = V.purge_embargo(train, pd.Timestamp("2024-01-01"), horizon=12, embargo=1)
        # cutoff = 2024-01-01 - 13 months = 2022-12-01; keep strictly earlier.
        assert pd.Timestamp("2022-11-01") in out.index          # kept
        assert pd.Timestamp("2022-12-01") not in out.index      # dropped (at cutoff)
        assert pd.Timestamp("2023-06-01") not in out.index      # dropped (inside window)
        assert out.index.max() == pd.Timestamp("2022-11-01")

    def test_embargo_extends_purge_by_one_month(self):
        """embargo widens the dropped window beyond the pure label horizon."""
        train = self._train()
        no_emb = V.purge_embargo(train, pd.Timestamp("2024-01-01"), horizon=12, embargo=0)
        with_emb = V.purge_embargo(train, pd.Timestamp("2024-01-01"), horizon=12, embargo=1)
        assert with_emb.index.max() < no_emb.index.max()
        assert no_emb.index.max() == pd.Timestamp("2022-12-01")


# ─────────────────────────────────────────────────────────────────────────────
# BaseModel.backtest integration — purge actually removes leakage
# ─────────────────────────────────────────────────────────────────────────────

class _LastValModel(Z.BaseModel):
    """Test model: predicts the most recent training target for every test row.
    Maximally sensitive to whatever sits at the train/test boundary."""
    name = "lastval"
    PRED_MIN, PRED_MAX = -1e9, 1e9

    def _fit_predict_year(self, train, test, factors, target):
        return np.full(len(test), float(train[target].iloc[-1]))


class TestBacktestPurging:
    def _df(self):
        idx = _monthly("2010-01-01", 168)          # 2010-01 .. 2023-12
        rng = np.random.default_rng(0)
        return pd.DataFrame({"cpi_yoy": rng.normal(2.0, 0.5, len(idx)),
                             "f": rng.normal(0, 1, len(idx))}, index=idx)

    def test_poisoning_purged_row_does_not_change_prediction(self):
        m = _LastValModel()
        df = self._df()
        bt1 = m.backtest(df, [], "cpi_yoy", start_year=2023, end_year=2023,
                         purge_horizon=12, embargo=1)
        df2 = df.copy()
        df2.loc[pd.Timestamp("2023-06-01"), "cpi_yoy"] += 1000.0   # inside purge window
        bt2 = m.backtest(df2, [], "cpi_yoy", start_year=2023, end_year=2023,
                         purge_horizon=12, embargo=1)
        assert np.allclose(bt1["pred"].values, bt2["pred"].values, atol=1e-9)

    def test_without_purge_boundary_row_does_change_prediction(self):
        """Sanity: the test model IS sensitive to the boundary when purge is off."""
        m = _LastValModel()
        df = self._df()
        bt1 = m.backtest(df, [], "cpi_yoy", start_year=2023, end_year=2023)
        df2 = df.copy()
        df2.loc[pd.Timestamp("2022-12-01"), "cpi_yoy"] += 1000.0   # last training row
        bt2 = m.backtest(df2, [], "cpi_yoy", start_year=2023, end_year=2023)
        assert not np.allclose(bt1["pred"].values, bt2["pred"].values, atol=1e-9)

    def test_default_backtest_unchanged_by_new_params(self):
        """Default purge_horizon=0/embargo=0 must equal a plain backtest call."""
        m = _LastValModel()
        df = self._df()
        a = m.backtest(df, [], "cpi_yoy", start_year=2023, end_year=2023)
        b = m.backtest(df, [], "cpi_yoy", start_year=2023, end_year=2023,
                       purge_horizon=0, embargo=0)
        pd.testing.assert_frame_equal(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# two_stage production wiring — members backtested with purge + embargo
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoStageWiring:
    def test_production_backtest_forwards_purge_embargo(self, monkeypatch):
        """two_stage.backtest must run every Stage-2 member with the production
        purge_horizon/embargo so the OOS metrics are leakage-honest."""
        assert TS.PURGE_HORIZON == 12 and TS.EMBARGO == 1
        idx = _monthly("2015-01-01", 120)          # 2015-01 .. 2024-12
        fake_aa = pd.DataFrame({"actual": np.linspace(2, 3, len(idx)),
                                "pred": np.linspace(2, 3, len(idx))}, index=idx)
        monkeypatch.setattr(Z.AutoARIMA, "backtest",
                            lambda self, *a, **k: fake_aa.copy())
        captured = []

        def rec(self, df, factors, target, **kw):
            captured.append(kw)
            return pd.DataFrame({"pred": np.zeros(len(idx))}, index=idx)

        for _tag, cls in TS.STAGE2:
            monkeypatch.setattr(cls, "backtest", rec)

        df = pd.DataFrame({TS.TARGET: np.linspace(2, 3, len(idx))}, index=idx)
        TS.backtest(df, live=[])

        assert len(captured) == len(TS.STAGE2)
        for kw in captured:
            assert kw.get("purge_horizon") == TS.PURGE_HORIZON
            assert kw.get("embargo") == TS.EMBARGO
