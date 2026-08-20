"""
dfm/state_space.py — Kalman filter and Durbin-Koopman simulation smoother.

State:   s_t = [f_t (K), f_{t-1} (K), e_t (M)]          length n = 2K + M
Trans:   f_t = Phi1 f_{t-1} + Phi2 f_{t-2} + eta_t,   e_it = rho_i e_{i,t-1} + eps_it
Obs:     y_it = lam_i' f_t + e_it                      (no separate measurement noise)

Idiosyncratic errors are carried IN the state rather than quasi-differenced away,
because idiosyncratic stochastic volatility (Stage 2) makes quasi-differencing
ill-defined. All observation noise therefore lives in the state, and the
measurement equation is exact.

The update is SEQUENTIAL (univariate): observations are applied one at a time, so
a missing value is simply a skipped update — no interpolation, no EM fill, and
cost is O(N k^2) rather than O(N^3) per period. This is what makes the ragged
edge native.

Pure numeric: no pandas, no data fetch, fully testable on synthetic data.
"""
import warnings

import numpy as np

try:
    from scipy.linalg import cho_factor, cho_solve, solve_discrete_lyapunov
    _HAVE_SCIPY = True
except ImportError:                                   # pragma: no cover
    _HAVE_SCIPY = False

try:
    from numba import njit
except ImportError:                                   # pragma: no cover
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not a else a[0]


def transition_matrix(Phi, rho):
    """
    Block companion transition for [f_t, f_{t-1}, e_t].

    Phi : (K, K) for factor_order 1, or (K, 2K) stacking [Phi1 Phi2].
    rho : (M,) idiosyncratic AR(1) coefficients.
    """
    Phi = np.atleast_2d(np.asarray(Phi, dtype=float))
    K = Phi.shape[0]
    Phi1 = Phi[:, :K]
    Phi2 = Phi[:, K:2 * K] if Phi.shape[1] >= 2 * K else np.zeros((K, K))
    M = len(rho)
    n = 2 * K + M
    T = np.zeros((n, n))
    T[:K, :K] = Phi1
    T[:K, K:2 * K] = Phi2
    T[K:2 * K, :K] = np.eye(K)
    T[2 * K:, 2 * K:] = np.diag(np.asarray(rho, dtype=float))
    return T


def state_cov(Q_f, sig, K, M):
    """Innovation covariance of the state, diag(Q_f, 0_K, sig^2)."""
    n = 2 * K + M
    Q = np.zeros((n, n))
    Q[:K, :K] = np.asarray(Q_f, dtype=float)
    Q[2 * K:, 2 * K:] = np.diag(np.asarray(sig, dtype=float) ** 2)
    return Q


def design_matrix(Lam, K, M):
    """Z with rows [lam_i, 0_K, unit_i]."""
    Z = np.zeros((M, 2 * K + M))
    Z[:, :K] = np.asarray(Lam, dtype=float)
    Z[:, 2 * K:] = np.eye(M)
    return Z


def _solve_psd(A, B):
    """
    A^-1 B for symmetric positive-definite A, via Cholesky.

    Falls back to the pseudo-inverse when the factorisation fails: A is singular
    whenever the diffuse initialisation is active or a state component is
    deterministic, and those cases must still smooth rather than raise.
    """
    if _HAVE_SCIPY:
        try:
            return cho_solve(cho_factor(A, lower=True, check_finite=False), B,
                             check_finite=False)
        except (np.linalg.LinAlgError, ValueError):
            pass
    else:                                             # pragma: no cover
        try:
            return np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.pinv(A) @ B


def _solve_lyapunov(T, Q, n):
    """Solve P = T P T' + Q for P."""
    if _HAVE_SCIPY:
        # Bartels-Stewart, O(n^3). scipy solves X - A X A' = Q, our convention.
        # It warns and perturbs when an eigenvalue pair sums to zero; the caller
        # checks the residual regardless, so the warning would only be noise on
        # a path that is about to be rejected anyway.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return solve_discrete_lyapunov(T, Q, method="bilinear")
    # O(n^6) Kronecker form, kept only as a fallback for a scipy-less env
    return np.linalg.solve(np.eye(n * n) - np.kron(T, T), Q.reshape(-1)).reshape(n, n)


def _initial_state(T, Q, n):
    """
    Stationary initialisation, with a diffuse fallback if the VAR is explosive.

    Solves P0 = T P0 T' + Q in O(n^3). The Kronecker form this replaced was
    O(n^6) -- a dense 2209x2209 solve at the Stage-1 dimension n = 2K + M = 47,
    which cost more than the Kalman filter it was initialising.

    The bilinear transform needs (T - I) invertible, so an exact unit root makes
    it fail or return nonsense rather than raise. The candidate is therefore
    accepted only if it satisfies the equation it claims to have solved; any
    failure falls back to the diffuse 1e4 * I.
    """
    a0 = np.zeros(n)
    try:
        P0 = _solve_lyapunov(T, Q, n)
    except (np.linalg.LinAlgError, ValueError):
        return a0, np.eye(n) * 1e4
    if not np.all(np.isfinite(P0)) or np.any(np.diag(P0) <= 0):
        return a0, np.eye(n) * 1e4
    resid = np.abs(P0 - (T @ P0 @ T.T + Q)).max()
    if resid > 1e-8 * max(1.0, np.abs(P0).max()):
        return a0, np.eye(n) * 1e4
    return a0, 0.5 * (P0 + P0.T)


@njit(cache=True)
def _filter_core(y, Z, T, Q, a0, P0):
    """
    Sequential univariate filter.

    Returns filtered states, covariances, loglik, and the number of updates
    skipped because the forecast variance was degenerate. A skipped update drops
    both the state correction and that observation's likelihood contribution, so
    the count is surfaced rather than left silent.
    """
    n_obs, M = y.shape
    n = a0.shape[0]
    a_filt = np.zeros((n_obs, n))
    P_filt = np.zeros((n_obs, n, n))
    a = a0.copy()
    P = P0.copy()
    loglik = 0.0
    n_degenerate = 0
    log2pi = np.log(2.0 * np.pi)
    for t in range(n_obs):
        if t > 0:
            a = T @ a
            P = T @ P @ T.T + Q
            P = 0.5 * (P + P.T)
        for i in range(M):
            yi = y[t, i]
            if np.isnan(yi):
                continue                      # missing -> skip the update entirely
            Zi = Z[i]
            PZ = P @ Zi
            F = Zi @ PZ
            if F <= 1e-12:
                n_degenerate += 1
                continue
            v = yi - Zi @ a
            a = a + PZ * (v / F)
            P = P - np.outer(PZ, PZ) / F
            P = 0.5 * (P + P.T)
            loglik += -0.5 * (log2pi + np.log(F) + v * v / F)
        a_filt[t] = a
        P_filt[t] = P
    return a_filt, P_filt, loglik, n_degenerate


@njit(cache=True)
def _filter_core_structured(y, Lam, T, Q, a0, P0, K):
    """
    Identical filter to `_filter_core`, specialised to the canonical design.

    Every row of Z is [lam_i, 0_K, unit_i], so it has K + 1 nonzeros out of
    n = 2K + M. The dense form multiplies the 2K + M - K - 1 structural zeros
    anyway; at Stage-1 dimensions that is 39 wasted columns out of 47, and the
    observation loop is ~94% of the filter's cost. This version touches only the
    nonzeros and applies the rank-1 covariance downdate in place over the upper
    triangle, mirroring it -- which makes P exactly symmetric by construction
    and removes the per-observation re-symmetrisation.

    Same arithmetic, same results to floating-point summation order; the caller
    dispatches here only when Z really has this structure.
    """
    n_obs, M = y.shape
    n = a0.shape[0]
    a_filt = np.zeros((n_obs, n))
    P_filt = np.zeros((n_obs, n, n))
    a = a0.copy()
    P = P0.copy()
    loglik = 0.0
    n_degenerate = 0
    log2pi = np.log(2.0 * np.pi)
    K2 = 2 * K
    PZ = np.empty(n)
    for t in range(n_obs):
        if t > 0:
            a = T @ a
            P = T @ P @ T.T + Q
            P = 0.5 * (P + P.T)
        for i in range(M):
            yi = y[t, i]
            if np.isnan(yi):
                continue                      # missing -> skip the update entirely
            li = Lam[i]
            ci = K2 + i                       # the unit entry for series i
            for r in range(n):                # PZ = P[:, :K] @ lam_i + P[:, ci]
                s = P[r, ci]
                for c in range(K):
                    s += P[r, c] * li[c]
                PZ[r] = s
            F = PZ[ci]
            for c in range(K):
                F += PZ[c] * li[c]
            if F <= 1e-12:
                n_degenerate += 1
                continue
            v = yi - a[ci]
            for c in range(K):
                v -= li[c] * a[c]
            g = v / F
            for r in range(n):
                a[r] += PZ[r] * g
            for r in range(n):                # P -= outer(PZ, PZ) / F, in place
                pr = PZ[r] / F
                for c in range(r, n):
                    d = P[r, c] - pr * PZ[c]
                    P[r, c] = d
                    P[c, r] = d
            loglik += -0.5 * (log2pi + np.log(F) + v * v / F)
        a_filt[t] = a
        P_filt[t] = P
    return a_filt, P_filt, loglik, n_degenerate


def _run_filter(y, Z, T, Q, a0, P0):
    """
    Run the filter, taking the structured fast path when Z permits it.

    Dispatch rather than assumption: a future Z with, say, mixed-frequency
    aggregation rows loading on lagged factors would not match the canonical
    pattern, and must fall through to the general dense core instead of being
    silently mis-filtered.
    """
    M = y.shape[1]
    n = a0.shape[0]
    K = (n - M) // 2
    if (2 * K + M == n and Z.shape == (M, n)
            and not Z[:, K:2 * K].any()
            and np.array_equal(Z[:, 2 * K:], np.eye(M))):
        Lam = np.ascontiguousarray(Z[:, :K])
        return _filter_core_structured(y, Lam, T, Q, a0, P0, K)
    return _filter_core(y, Z, T, Q, a0, P0)


def kalman_filter(y, Lam, Q_f, Phi, rho, sig):
    """Filter the panel. `y` is (T, M) with NaN for unobserved cells."""
    y = np.asarray(y, dtype=float)
    M = y.shape[1]
    K = np.atleast_2d(Lam).shape[1]
    T = transition_matrix(Phi, rho)
    Q = state_cov(Q_f, sig, K, M)
    Z = design_matrix(Lam, K, M)
    a0, P0 = _initial_state(T, Q, T.shape[0])
    a_filt, P_filt, loglik, n_degenerate = _run_filter(y, Z, T, Q, a0, P0)
    return dict(a_filt=a_filt, P_filt=P_filt, loglik=loglik, Z=Z, T=T, Q=Q,
                n_degenerate=int(n_degenerate))


def _smooth_means(y, Z, T, Q, a0, P0):
    """Disturbance-free RTS smoother; returns smoothed state means (T, n)."""
    a_filt, P_filt, _, _ = _run_filter(y, Z, T, Q, a0, P0)
    n_obs, n = a_filt.shape
    a_sm = np.zeros_like(a_filt)
    a_sm[-1] = a_filt[-1]
    for t in range(n_obs - 2, -1, -1):
        TP = T @ P_filt[t]                      # shared by P_pred and the gain
        P_pred = TP @ T.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        # G = P_filt[t] T' P_pred^-1 = (P_pred^-1 T P_filt[t])', using the
        # symmetry of both P_filt[t] and P_pred. A Cholesky solve replaces a
        # per-period SVD; only the means are needed, so the smoothed covariance
        # recursion that used to run here is gone.
        G = _solve_psd(P_pred, TP).T
        a_sm[t] = a_filt[t] + G @ (a_sm[t + 1] - T @ a_filt[t])
    return a_sm


def simulation_smoother(y, Lam, Q_f, Phi, rho, sig, rng):
    """
    Durbin-Koopman (2002) simulation smoother: one draw of the full state path.

    Simulate s+ and y+ from the model, smooth both the real and simulated data,
    and form  s_draw = s+ + E[s | y] - E[s+ | y+].  The missing-data pattern of
    y is copied onto y+ so the two smoothers condition on the same information.
    """
    y = np.asarray(y, dtype=float)
    n_obs, M = y.shape
    K = np.atleast_2d(Lam).shape[1]
    T = transition_matrix(Phi, rho)
    Q = state_cov(Q_f, sig, K, M)
    Z = design_matrix(Lam, K, M)
    n = T.shape[0]
    a0, P0 = _initial_state(T, Q, n)

    # forward simulation
    chol = np.linalg.cholesky(Q + np.eye(n) * 1e-10)
    s_plus = np.zeros((n_obs, n))
    try:
        s_plus[0] = a0 + np.linalg.cholesky(P0 + np.eye(n) * 1e-10) @ rng.standard_normal(n)
    except np.linalg.LinAlgError:
        s_plus[0] = a0 + np.sqrt(np.diag(P0)) * rng.standard_normal(n)
    for t in range(1, n_obs):
        s_plus[t] = T @ s_plus[t - 1] + chol @ rng.standard_normal(n)
    y_plus = s_plus @ Z.T
    y_plus[np.isnan(y)] = np.nan            # identical missingness pattern

    a_hat = _smooth_means(y, Z, T, Q, a0, P0)
    a_plus_hat = _smooth_means(y_plus, Z, T, Q, a0, P0)
    return s_plus + a_hat - a_plus_hat
