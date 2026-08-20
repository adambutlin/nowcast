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
import numpy as np

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


def _initial_state(T, Q, n):
    """Stationary initialisation, with a diffuse fallback if the VAR is explosive."""
    a0 = np.zeros(n)
    try:
        vecP = np.linalg.solve(np.eye(n * n) - np.kron(T, T), Q.reshape(-1))
        P0 = vecP.reshape(n, n)
        if not np.all(np.isfinite(P0)) or np.any(np.diag(P0) <= 0):
            raise np.linalg.LinAlgError
        P0 = 0.5 * (P0 + P0.T)
    except np.linalg.LinAlgError:
        P0 = np.eye(n) * 1e4
    return a0, P0


@njit(cache=True)
def _filter_core(y, Z, T, Q, a0, P0):
    """Sequential univariate filter. Returns filtered states, covs and loglik."""
    n_obs, M = y.shape
    n = a0.shape[0]
    a_filt = np.zeros((n_obs, n))
    P_filt = np.zeros((n_obs, n, n))
    a = a0.copy()
    P = P0.copy()
    loglik = 0.0
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
                continue
            v = yi - Zi @ a
            a = a + PZ * (v / F)
            P = P - np.outer(PZ, PZ) / F
            P = 0.5 * (P + P.T)
            loglik += -0.5 * (log2pi + np.log(F) + v * v / F)
        a_filt[t] = a
        P_filt[t] = P
    return a_filt, P_filt, loglik


def kalman_filter(y, Lam, Q_f, Phi, rho, sig):
    """Filter the panel. `y` is (T, M) with NaN for unobserved cells."""
    y = np.asarray(y, dtype=float)
    M = y.shape[1]
    K = np.atleast_2d(Lam).shape[1]
    T = transition_matrix(Phi, rho)
    Q = state_cov(Q_f, sig, K, M)
    Z = design_matrix(Lam, K, M)
    a0, P0 = _initial_state(T, Q, T.shape[0])
    a_filt, P_filt, loglik = _filter_core(y, Z, T, Q, a0, P0)
    return dict(a_filt=a_filt, P_filt=P_filt, loglik=loglik, Z=Z, T=T, Q=Q)


def _smooth_means(y, Z, T, Q, a0, P0):
    """Disturbance-free RTS smoother; returns smoothed state means (T, n)."""
    a_filt, P_filt, _ = _filter_core(y, Z, T, Q, a0, P0)
    n_obs, n = a_filt.shape
    a_sm = np.zeros_like(a_filt)
    a_sm[-1] = a_filt[-1]
    P_sm = P_filt[-1].copy()
    for t in range(n_obs - 2, -1, -1):
        P_pred = T @ P_filt[t] @ T.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)
        G = P_filt[t] @ T.T @ np.linalg.pinv(P_pred)
        a_sm[t] = a_filt[t] + G @ (a_sm[t + 1] - T @ a_filt[t])
        P_sm = P_filt[t] + G @ (P_sm - P_pred) @ G.T
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
