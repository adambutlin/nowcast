"""Each Gibbs block must recover its own parameters from a known DGP."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import sampler as S


class TestDrawLoadings:
    def test_recovers_known_loadings(self):
        rng = np.random.default_rng(0)
        T, K, M = 400, 2, 5
        f = rng.normal(size=(T, K))
        Lam = rng.normal(size=(M, K))
        sig = np.full(M, 0.3)
        y = f @ Lam.T + rng.normal(size=(T, M)) * sig
        draws = np.stack([S.draw_loadings(y, f, np.zeros(M), sig, rng) for _ in range(50)])
        assert np.abs(draws.mean(axis=0) - Lam).max() < 0.1

    def test_respects_a_zero_mask(self):
        rng = np.random.default_rng(1)
        T, K, M = 200, 3, 4
        f = rng.normal(size=(T, K))
        y = rng.normal(size=(T, M))
        mask = np.ones((M, K), dtype=bool)
        mask[:, 2] = False                       # third factor pinned off
        got = S.draw_loadings(y, f, np.zeros(M), np.ones(M), rng, mask=mask)
        assert np.all(got[:, 2] == 0.0)

    def test_shrinks_toward_zero_with_a_tight_prior(self):
        rng = np.random.default_rng(2)
        T, K, M = 60, 2, 3
        f = rng.normal(size=(T, K))
        y = f @ np.ones((M, K)).T + rng.normal(size=(T, M))
        loose = np.stack([S.draw_loadings(y, f, np.zeros(M), np.ones(M), rng,
                                          prior_var=100.0) for _ in range(40)]).mean(0)
        tight = np.stack([S.draw_loadings(y, f, np.zeros(M), np.ones(M), rng,
                                          prior_var=0.01) for _ in range(40)]).mean(0)
        assert np.abs(tight).mean() < np.abs(loose).mean()


class TestDrawIdio:
    def test_recovers_ar_coefficient_and_variance(self):
        rng = np.random.default_rng(3)
        T, M = 600, 3
        rho_true, sig_true = np.array([0.6, 0.2, -0.4]), np.array([0.5, 1.0, 0.8])
        e = np.zeros((T, M))
        for t in range(1, T):
            e[t] = rho_true * e[t - 1] + rng.normal(size=M) * sig_true
        rho_d, sig_d = [], []
        for _ in range(60):
            r, s = S.draw_idio(e, rng)
            rho_d.append(r); sig_d.append(s)
        assert np.abs(np.mean(rho_d, axis=0) - rho_true).max() < 0.08
        assert np.abs(np.mean(sig_d, axis=0) - sig_true).max() < 0.10

    def test_rejects_nonstationary_draws(self):
        rng = np.random.default_rng(4)
        T, M = 300, 2
        e = np.cumsum(rng.normal(size=(T, M)), axis=0)     # random walk, rho -> 1
        for _ in range(30):
            rho, _ = S.draw_idio(e, rng)
            assert np.all(np.abs(rho) < 1.0)

    def test_sigma_never_falls_below_the_floor(self):
        # Near-flat prior plus all-but-zero residuals is the pathological case:
        # without a floor the drawn sig collapses to ~1e-13 and the Kalman filter
        # silently drops the series (F <= 1e-12 -> n_degenerate).
        rng = np.random.default_rng(12)
        T, M = 200, 2
        e = 1e-12 * rng.normal(size=(T, M))
        for _ in range(20):
            _, sig = S.draw_idio(e, rng, a0=1e-8, b0=1e-12)
            assert np.all(sig >= S.SIG_FLOOR)


class TestDrawFactorAR:
    def test_recovers_diagonal_ar_coefficients(self):
        rng = np.random.default_rng(5)
        T, K = 500, 2
        phi_true = np.array([0.7, -0.3])
        f = np.zeros((T, K))
        for t in range(1, T):
            f[t] = phi_true * f[t - 1] + rng.normal(size=K)
        draws = np.stack([S.draw_factor_ar(f, rng, order=1, diagonal=True) for _ in range(60)])
        est = np.array([draws[:, k, k].mean() for k in range(K)])
        assert np.abs(est - phi_true).max() < 0.08

    def test_diagonal_mode_has_zero_off_diagonals(self):
        rng = np.random.default_rng(6)
        f = np.random.default_rng(7).normal(size=(200, 3))
        Phi = S.draw_factor_ar(f, rng, order=1, diagonal=True)
        assert np.allclose(Phi - np.diag(np.diag(Phi)), 0.0)

    def test_returns_stationary_companion(self):
        rng = np.random.default_rng(8)
        f = np.cumsum(np.random.default_rng(9).normal(size=(300, 2)), axis=0)
        for _ in range(20):
            Phi = S.draw_factor_ar(f, rng, order=1, diagonal=True)
            assert np.max(np.abs(np.linalg.eigvals(Phi))) < 1.0


class TestNormalisation:
    def test_upper_block_is_lower_triangular_with_positive_diagonal(self):
        rng = np.random.default_rng(10)
        Lam = rng.normal(size=(6, 3))
        f = rng.normal(size=(100, 3))
        Lam_n, f_n = S.normalise(Lam.copy(), f.copy())
        upper = Lam_n[:3, :3]
        assert np.allclose(np.triu(upper, 1), 0.0)
        assert np.all(np.diag(upper) > 0)

    def test_normalisation_preserves_the_common_component(self):
        rng = np.random.default_rng(11)
        Lam = rng.normal(size=(6, 3))
        f = rng.normal(size=(100, 3))
        before = f @ Lam.T
        Lam_n, f_n = S.normalise(Lam.copy(), f.copy())
        assert np.allclose(before, f_n @ Lam_n.T, atol=1e-8)
