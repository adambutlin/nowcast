"""Kalman filter and simulation smoother on synthetic data with known parameters."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import state_space as SS


def simulate(T=200, K=2, M=6, seed=0):
    """Generate factors, loadings and observations from a known DFM."""
    rng = np.random.default_rng(seed)
    Phi = np.diag([0.7, 0.5][:K])   # K-dimensional; K=2 keeps diag(0.7, 0.5)
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
