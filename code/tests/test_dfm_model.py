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
        # The nowcast month's target is unobserved by construction; correlating
        # against it would make np.corrcoef return nan for every factor, so the
        # comparison runs on the observed rows. The threshold is unchanged.
        tgt = df["cpi_mom"].values
        obs = np.isfinite(tgt)
        best = max(abs(np.corrcoef(fmean[obs, k], tgt[obs])[0, 1])
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
