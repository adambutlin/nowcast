"""
Remediation tests for audit findings H6, C4, C1.

Written to FAIL against the pre-remediation codebase and PASS after:
  H6 — factor_health() liveness classification; budgeted (non-silent) ffill
       in BaseModel._nowcast_row.
  C4 — loud per-year backtest failures; common-sample metrics vs benchmark.
  C1 — walk-forward (causal) ensemble member selection.
"""

import numpy as np
import pandas as pd
import pytest

import factors as F
import uk_model_zoo as Z


def _monthly(start, n):
    return pd.date_range(start, periods=n, freq="ME")


# ─────────────────────────────────────────────────────────────────────────────
# H6 — factor liveness
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorHealth:
    def test_classifies_live_stale_dead(self):
        idx = _monthly("2015-01-31", 120)
        df = pd.DataFrame(index=idx, data={
            "live_fac": 1.0, "stale_fac": 1.0, "dead_fac": 1.0})
        df.loc[idx[-4]:, "stale_fac"] = np.nan   # 4m stale (budget default 1+2=3)
        df.loc[idx[-24]:, "dead_fac"] = np.nan   # 24m stale
        h = F.factor_health(df, ["live_fac", "stale_fac", "dead_fac"])
        assert h.loc["live_fac", "status"] == "LIVE"
        assert h.loc["stale_fac", "status"] == "STALE"
        assert h.loc["dead_fac", "status"] == "DEAD"
        assert h.loc["dead_fac", "months_stale"] >= 7

    def test_respects_registry_pub_lag_budget(self):
        # uk_awg has pub_lag=1 → budget 1+2=3; exactly 3 months stale is LIVE
        idx = _monthly("2015-01-31", 60)
        df = pd.DataFrame(index=idx, data={"uk_awg": 1.0})
        df.loc[idx[-3]:, "uk_awg"] = np.nan
        h = F.factor_health(df, ["uk_awg"])
        assert h.loc["uk_awg", "status"] == "LIVE"

    def test_all_nan_factor_is_dead(self):
        idx = _monthly("2015-01-31", 24)
        df = pd.DataFrame(index=idx, data={"ghost": np.nan})
        h = F.factor_health(df, ["ghost"])
        assert h.loc["ghost", "status"] == "DEAD"


class TestNowcastRowStaleness:
    def test_blocks_stale_forward_fill(self):
        """Factor dead for 10 months must NOT be silently ffilled into nowcast row."""
        idx = _monthly("2018-01-31", 60)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(index=idx, data={
            "f1": rng.standard_normal(60), "cpi_yoy": 2.0})
        df.loc[idx[-10]:, "f1"] = np.nan      # f1 last observed 10 months ago
        df.loc[idx[-1], "cpi_yoy"] = np.nan   # target unreleased in final month
        row, date = Z.BaseModel._nowcast_row(df, ["f1"], "cpi_yoy")
        assert date == idx[-1]
        assert row is None, "stale factor silently forward-filled into nowcast row"

    def test_allows_fill_within_budget(self):
        idx = _monthly("2018-01-31", 60)
        df = pd.DataFrame(index=idx, data={"f1": 1.0, "cpi_yoy": 2.0})
        df.loc[idx[-2]:, "f1"] = np.nan       # 2 months stale: within budget (3)
        df.loc[idx[-1], "cpi_yoy"] = np.nan
        row, date = Z.BaseModel._nowcast_row(df, ["f1"], "cpi_yoy")
        assert row is not None
        assert float(row["f1"].iloc[0]) == 1.0
        assert np.isnan(row["cpi_yoy"].iloc[0])   # target must stay NaN


# ─────────────────────────────────────────────────────────────────────────────
# C4 — loud failures + common-sample evaluation
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktestLoudFailures:
    def test_year_failure_is_loud(self, capsys):
        idx = _monthly("2010-01-31", 180)
        rng = np.random.default_rng(1)
        df = pd.DataFrame(index=idx, data={
            "f1": rng.standard_normal(180),
            "cpi_yoy": rng.standard_normal(180) + 2.0})

        class FailsIn2017(Z.BaseModel):
            name = "fails2017"
            def _fit_predict_year(self, train, test, factors, target):
                if test.index[0].year == 2017:
                    raise RuntimeError("boom")
                return np.zeros(len(test))

        bt = FailsIn2017().backtest(df, ["f1"], "cpi_yoy", start_year=2015)
        out = capsys.readouterr().out
        assert 2017 not in set(bt.index.year)          # year really skipped
        assert "fails2017" in out and "2017" in out    # ...but loudly


class TestCommonSampleMetrics:
    def test_benchmark_rmse_recomputed_on_common_dates(self):
        import main as NC
        idx_long = _monthly("2015-01-31", 136)   # benchmark: 136 months
        idx_short = idx_long[:112]               # model: first 112 only
        actual = pd.Series(2.0, index=idx_long)
        bench_pred = actual.copy()
        bench_pred.iloc[:112] += 0.10            # small errors on common window
        bench_pred.iloc[112:] += 5.00            # huge errors on the extra 24m
        ar1 = pd.DataFrame({"actual": actual, "pred": bench_pred})
        model = pd.DataFrame({"actual": actual.loc[idx_short],
                              "pred": actual.loc[idx_short] + 0.20})
        mdf = NC.common_sample_metrics({"AR(1)": ar1, "m": model})
        assert mdf.loc["m", "coverage"] == pytest.approx(112 / 136, abs=0.01)
        # benchmark must be re-scored on the COMMON 112 months (RMSE 0.10),
        # not the full-sample RMSE (~2.1) inflated by months the model lacks
        assert mdf.loc["m", "ar1_rmse_cs"] == pytest.approx(0.10, abs=1e-6)
        assert not mdf.loc["m", "beats_ar1"]     # 0.20 > 0.10 on common sample

    def test_benchmark_row_marked(self):
        import main as NC
        idx = _monthly("2015-01-31", 60)
        actual = pd.Series(2.0, index=idx)
        ar1 = pd.DataFrame({"actual": actual, "pred": actual + 0.3})
        mdf = NC.common_sample_metrics({"AR(1)": ar1})
        assert mdf.loc["AR(1)", "coverage"] == 1.0
        assert not mdf.loc["AR(1)", "beats_ar1"]


# ─────────────────────────────────────────────────────────────────────────────
# C1 — causal ensemble selection
# ─────────────────────────────────────────────────────────────────────────────

class TestCombineRecursive:
    def test_membership_is_causal(self):
        import main as NC
        idx = _monthly("2015-01-31", 96)                  # 2015–2022
        actual = pd.Series(2.0, index=idx)
        bench = pd.DataFrame({"actual": actual, "pred": actual + 0.50})
        good = pd.DataFrame({"actual": actual, "pred": actual + 0.10})
        # lucky_late: terrible pre-2020, perfect after. Full-sample selection
        # would admit it; causal selection must exclude it in 2018-2019.
        ll = actual.copy()
        ll[idx.year < 2020] += 3.0
        lucky = pd.DataFrame({"actual": actual, "pred": ll})
        bt = NC.combine_recursive({"good": good, "lucky": lucky, "AR(1)": bench},
                                  bench, min_hist=36)
        # 2015-2017: <36m history → equal-weight fallback over both models
        p2016 = bt[bt.index.year == 2016]["pred"].iloc[0]
        assert p2016 == pytest.approx((2.1 + 5.0) / 2, abs=1e-6)
        # 2018: history (2015-17) shows lucky loses to AR(1) → only `good` kept
        p2019 = bt[bt.index.year == 2019]["pred"].iloc[0]
        assert p2019 == pytest.approx(2.1, abs=1e-6)

    def test_falls_back_to_equal_weight_when_nothing_beats_ar1(self):
        import main as NC
        idx = _monthly("2015-01-31", 96)
        actual = pd.Series(2.0, index=idx)
        bench = pd.DataFrame({"actual": actual, "pred": actual + 0.05})  # strong AR(1)
        m1 = pd.DataFrame({"actual": actual, "pred": actual + 1.0})
        m2 = pd.DataFrame({"actual": actual, "pred": actual - 1.0})
        bt = NC.combine_recursive({"m1": m1, "m2": m2}, bench, min_hist=36)
        # no model beats AR(1) at any point → equal weight of all, never empty
        assert len(bt) == 96
        p = bt[bt.index.year == 2021]["pred"].iloc[0]
        assert p == pytest.approx(2.0, abs=1e-6)
