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

    def test_posterior_spread_matches_the_analytic_scale(self):
        # The recovery test above checks only the mean of the draws, and sig
        # cancels out of the posterior mean under a weak prior — so an
        # implementation dividing by sig instead of sig**2 passes it while
        # sampling from a distribution 1.8x too wide. Pin the spread too.
        rng = np.random.default_rng(13)
        T, K, M = 400, 2, 5
        f = rng.normal(size=(T, K))
        Lam = rng.normal(size=(M, K))
        sig = np.full(M, 0.3)
        y = f @ Lam.T + rng.normal(size=(T, M)) * sig
        n_draws = 400
        draws = np.stack([S.draw_loadings(y, f, np.zeros(M), sig, rng)
                          for _ in range(n_draws)])
        got = draws.std(axis=0)
        want = np.zeros((M, K))
        for i in range(M):
            V = np.linalg.inv(f.T @ f / sig[i] ** 2 + np.eye(K) / 10.0)
            want[i] = np.sqrt(np.diag(V))
        # MC error on an sd from 400 draws is ~1/sqrt(2*399) = 3.5%; 15% is ~4x
        # that, and the /sig variant is off by 83%.
        assert np.abs(got / want - 1.0).max() < 0.15


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

    def test_supplied_rho_prev_gives_the_exact_conditional_variance(self):
        # With rho_prev supplied the variance draw must be the true full
        # conditional InvGamma(a0 + n/2, b0 + SSR(rho_prev)/2), whose mean is
        # b_n/(a_n - 1). Evaluating SSR at the OLS rho instead (the rho_prev=None
        # path) understates it, so this pins which rho the block conditions on.
        rng = np.random.default_rng(14)
        T = 500
        e = np.zeros((T, 1))
        for t in range(1, T):
            e[t] = 0.5 * e[t - 1] + rng.normal() * 0.7
        rho_prev = np.array([0.2])                 # deliberately not the OLS rho
        x, y = e[:-1, 0], e[1:, 0]
        resid = y - rho_prev[0] * x
        a_n, b_n = 3.0 + len(x) / 2.0, 1.0 + 0.5 * resid @ resid
        want = b_n / (a_n - 1.0)
        got = np.mean([S.draw_idio(e, rng, rho_prev=rho_prev)[1][0] ** 2
                       for _ in range(2000)])
        assert abs(got / want - 1.0) < 0.02
        # and the OLS-profile path really is a different (smaller) variance
        prof = np.mean([S.draw_idio(e, rng)[1][0] ** 2 for _ in range(2000)])
        assert prof < got

    def test_rho_prev_length_is_checked(self):
        rng = np.random.default_rng(15)
        e = rng.normal(size=(100, 3))
        with pytest.raises(ValueError):
            S.draw_idio(e, rng, rho_prev=np.zeros(2))


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

    def test_recovers_diagonal_ar2_coefficients(self):
        # order > 1 is the configuration state_space actually documents
        # (f_t = Phi1 f_{t-1} + Phi2 f_{t-2}), so the two-lag column layout and the
        # companion branch for order > 1 both need holding down.
        rng = np.random.default_rng(16)
        T, K = 800, 2
        lag1, lag2 = np.array([0.5, 0.3]), np.array([0.25, -0.2])
        f = np.zeros((T, K))
        for t in range(2, T):
            f[t] = lag1 * f[t - 1] + lag2 * f[t - 2] + rng.normal(size=K)
        draws = np.stack([S.draw_factor_ar(f, rng, order=2, diagonal=True)
                          for _ in range(60)]).mean(0)
        assert draws.shape == (K, 2 * K)
        assert np.abs(np.diag(draws[:, :K]) - lag1).max() < 0.08
        assert np.abs(np.diag(draws[:, K:]) - lag2).max() < 0.08
        for block in (draws[:, :K], draws[:, K:]):     # still diagonal at both lags
            assert np.allclose(block - np.diag(np.diag(block)), 0.0)
        assert np.max(np.abs(np.linalg.eigvals(S._companion(draws, K, 2)))) < 1.0

    def test_full_var_recovers_ols_lag_alignment(self):
        # diagonal=False (the Minnesota path, arms E4+) is otherwise untested. A
        # swapped lag block or a transposed coefficient matrix would still look
        # plausible against the truth on one dataset, so compare against direct OLS
        # on the same lag construction: that pins the layout, not just the values.
        rng = np.random.default_rng(17)
        T, K = 800, 2
        P1 = np.array([[0.5, 0.1], [-0.2, 0.4]])
        P2 = np.array([[0.2, 0.0], [0.05, 0.15]])
        f = np.zeros((T, K))
        for t in range(2, T):
            f[t] = P1 @ f[t - 1] + P2 @ f[t - 2] + rng.normal(size=K)
        X = np.column_stack([f[1:T - 1], f[0:T - 2]])
        ols = np.linalg.lstsq(X, f[2:], rcond=None)[0].T
        draws = np.stack([S.draw_factor_ar(f, rng, order=2, diagonal=False,
                                           prior_var=10.0, minnesota_cross=10.0)
                          for _ in range(200)]).mean(0)
        assert draws.shape == (K, 2 * K)
        assert np.abs(draws - ols).max() < 0.02
        assert np.abs(draws - np.hstack([P1, P2])).max() < 0.15

    def test_minnesota_prior_shrinks_cross_lags_hardest(self):
        rng = np.random.default_rng(18)
        T, K = 200, 2
        f = np.zeros((T, K))
        for t in range(1, T):
            f[t] = np.array([0.5, 0.5]) * f[t - 1] + rng.normal(size=K)
        kw = dict(order=1, diagonal=False, prior_var=10.0)
        loose = np.stack([S.draw_factor_ar(f, rng, minnesota_cross=10.0, **kw)
                          for _ in range(60)]).mean(0)
        tight = np.stack([S.draw_factor_ar(f, rng, minnesota_cross=1e-4, **kw)
                          for _ in range(60)]).mean(0)
        off = ~np.eye(K, dtype=bool)
        assert np.abs(tight[off]).max() < np.abs(loose[off]).max()
        assert np.abs(tight[off]).max() < 0.01

    def test_shrink_fallback_is_counted_not_silent(self):
        # An explosive series gives a posterior with essentially no stationary
        # mass, so rejection must give up and the deterministic shrink must fire —
        # visibly, because those draws are biased toward zero persistence.
        rng = np.random.default_rng(19)
        T, K = 60, 2
        f = np.zeros((T, K))
        for t in range(1, T):
            f[t] = 1.3 * f[t - 1] + rng.normal(size=K)
        S.reset_shrink_fallbacks()
        Phi = S.draw_factor_ar(f, rng, order=1, diagonal=True, max_tries=5)
        assert S.SHRINK_FALLBACKS == 1
        assert np.max(np.abs(np.linalg.eigvals(Phi))) < 1.0     # still usable
        S.reset_shrink_fallbacks()
        assert S.SHRINK_FALLBACKS == 0

    def test_no_fallback_on_a_well_behaved_posterior(self):
        rng = np.random.default_rng(20)
        T, K = 400, 2
        f = np.zeros((T, K))
        for t in range(1, T):
            f[t] = np.array([0.7, -0.3]) * f[t - 1] + rng.normal(size=K)
        S.reset_shrink_fallbacks()
        for _ in range(50):
            S.draw_factor_ar(f, rng, order=1, diagonal=True)
        assert S.SHRINK_FALLBACKS == 0


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
