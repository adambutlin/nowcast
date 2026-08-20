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


    def test_structured_fast_path_matches_the_general_dense_core(self, sim):
        """The Z-sparsity fast path must reproduce the general filter exactly."""
        y = sim["y"].copy()
        y[30:40, 2] = np.nan
        y[77, :] = np.nan
        K, M = sim["K"], sim["M"]
        T = SS.transition_matrix(sim["Phi"], sim["rho"])
        Q = SS.state_cov(np.eye(K), sim["sig"], K, M)
        Z = SS.design_matrix(sim["Lam"], K, M)
        a0, P0 = SS._initial_state(T, Q, T.shape[0])
        fast = SS._run_filter(y, Z, T, Q, a0, P0)
        dense = SS._filter_core(y, Z, T, Q, a0, P0)
        assert abs(fast[2] - dense[2]) / abs(dense[2]) < 1e-12
        assert np.abs(fast[0] - dense[0]).max() < 1e-10
        assert np.abs(fast[1] - dense[1]).max() < 1e-10

    def test_non_canonical_design_falls_back_to_the_dense_core(self, sim):
        """A Z loading on lagged factors must not take the structured path."""
        K, M = sim["K"], sim["M"]
        Z = SS.design_matrix(sim["Lam"], K, M)
        Z[0, K] = 0.5                       # series 0 now loads on f_{t-1}
        T = SS.transition_matrix(sim["Phi"], sim["rho"])
        Q = SS.state_cov(np.eye(K), sim["sig"], K, M)
        a0, P0 = SS._initial_state(T, Q, T.shape[0])
        got = SS._run_filter(sim["y"], Z, T, Q, a0, P0)
        want = SS._filter_core(sim["y"], Z, T, Q, a0, P0)
        assert got[2] == want[2]            # bit-identical: same code path
        assert np.array_equal(got[0], want[0])

    def test_degenerate_updates_are_counted_not_silently_dropped(self, sim):
        """A collapsed forecast variance skips the update; the count says so."""
        out = SS.kalman_filter(sim["y"], sim["Lam"], np.eye(sim["K"]),
                               sim["Phi"], sim["rho"], sim["sig"])
        assert out["n_degenerate"] == 0

        # Two series with identical loadings and no idiosyncratic variance: the
        # second carries no information once the first has been applied.
        Lam = np.array([[1.0], [1.0]])
        Z = SS.design_matrix(Lam, 1, 2)
        T = SS.transition_matrix(np.array([[0.5]]), np.zeros(2))
        Q = SS.state_cov(np.eye(1), np.zeros(2), 1, 2)
        a0 = np.zeros(4)
        P0 = np.diag([1.0, 1.0, 0.0, 0.0])
        y = np.array([[1.0, 1.0]])
        assert SS._run_filter(y, Z, T, Q, a0, P0)[3] == 1        # structured path
        assert SS._filter_core(y, Z, T, Q, a0, P0)[3] == 1       # dense path


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
