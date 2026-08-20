"""
dfm/sampler.py — Gibbs blocks for the DFM.

Stage 1 (E1) needs three blocks plus the factor draw from state_space:
  draw_loadings   Lam | f, y, sig          series-by-series Bayesian regression
  draw_idio       rho, sig | e             idiosyncratic AR(1) + variance
  draw_factor_ar  Phi | f                  factor VAR (diagonal in E1-E3)

Later stages add SV, outlier indicators and hyperpriors; those blocks land in
Plans 2 and 3 and must not change the signatures here (adding backward-compatible
optional arguments is fine; `draw_idio(..., rho_prev=)` is one such).

Every block is a conjugate draw with a weakly-informative prior, so each one is
independently testable by parameter recovery on synthetic data.
"""
import numpy as np

# Lower bound on a drawn idiosyncratic standard deviation.
#
# state_space._filter_core skips an observation whose innovation variance F is
# <= 1e-12 and counts it in `n_degenerate`. F for series i is
# lam_i' P_ff lam_i + P_ee[i], and the second term is bounded below by the
# series' own innovation variance sig_i^2, so a sig_i draw below 1e-6 can push F
# to the guard on a series whose loadings are also near zero — the series then
# vanishes from both the state update and the likelihood without any error.
# 1e-5 leaves sig^2 = 1e-10, a hundredfold clear of that guard, while still being
# ten orders of magnitude below the unit variance of the standardised panel: no
# posterior the data could support lives anywhere near it, so the floor is inert
# except in the degenerate case it exists to prevent.
SIG_FLOOR = 1e-5

# How many times draw_factor_ar exhausted its rejection attempts and fell back to
# the deterministic shrink. The shrink is NOT a draw from the truncated posterior
# (see draw_factor_ar), so a non-zero count means some draws in the chain are
# biased toward zero persistence. Read it after a run; it is never printed per
# draw. Task 10's sweep should report it alongside n_degenerate.
SHRINK_FALLBACKS = 0


def reset_shrink_fallbacks():
    """Zero the SHRINK_FALLBACKS counter (call once before a chain)."""
    global SHRINK_FALLBACKS
    SHRINK_FALLBACKS = 0


def _stationary(Phi):
    """Is the companion form of Phi stable?"""
    return np.max(np.abs(np.linalg.eigvals(Phi))) < 0.999


def draw_loadings(y, f, gamma_z, sig, rng, mask=None, prior_var=10.0):
    """
    Draw Lam (M, K) row by row from  y_i - gamma_i = lam_i' f + e_i,  e_i ~ N(0, sig_i^2).

    mask : optional (M, K) boolean; False entries are pinned to zero (E5/E6 block
           restrictions). Rows with no free entries return zeros.
    """
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    M, K = y.shape[1], f.shape[1]
    out = np.zeros((M, K))
    for i in range(M):
        free = np.ones(K, dtype=bool) if mask is None else np.asarray(mask[i], dtype=bool)
        if not free.any():
            continue
        obs = np.isfinite(y[:, i])
        if obs.sum() <= free.sum():
            continue
        X = f[obs][:, free]
        target = y[obs, i] - gamma_z[i]
        prec = X.T @ X / sig[i] ** 2 + np.eye(free.sum()) / prior_var
        V = np.linalg.inv(prec)
        m = V @ (X.T @ target / sig[i] ** 2)
        V = 0.5 * (V + V.T)
        out[i, free] = rng.multivariate_normal(m, V)
    return out


def draw_idio(e, rng, prior_var=10.0, a0=3.0, b0=1.0, max_tries=20, rho_prev=None):
    """
    Draw (rho, sig) for each idiosyncratic AR(1) series from its own residuals.

    Returns (rho (M,), sig (M,)). Non-stationary rho draws are rejected and
    retried; after `max_tries` the previous OLS estimate is clipped into (-0.99, 0.99).

    rho_prev : optional (M,) of the rho currently in the chain's state.

      supplied -> EXACT Gibbs. The residual sum of squares is evaluated at
        rho_prev, so the variance draw is the true full conditional
        p(sig^2 | rho, e) = InvGamma(a0 + n/2, b0 + SSR(rho_prev)/2), and the rho
        draw that follows is the true p(rho | sig, e). Task 10's sweep passes it.

      None -> approximate (the original behaviour, kept so existing call sites are
        unchanged). SSR is evaluated at the OLS rho instead, which profiles rho out
        rather than conditioning on it: the kernel then depends only on e, not on
        the chain's state, so it is an independence proposal used without an accept
        step. Measured at T=290 the implied sig is biased -0.14% and 1% too narrow
        — negligible, but not exact, so prefer rho_prev inside a real sweep.

    sig is floored at SIG_FLOOR — see the comment on that constant.
    """
    e = np.asarray(e, dtype=float)
    T, M = e.shape
    rho = np.zeros(M)
    sig = np.ones(M)
    if rho_prev is not None:
        rho_prev = np.asarray(rho_prev, dtype=float).reshape(-1)
        if rho_prev.shape[0] != M:
            raise ValueError(f"rho_prev has length {rho_prev.shape[0]}, expected {M}")
    for i in range(M):
        x, y = e[:-1, i], e[1:, i]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 10 or np.dot(x, x) <= 0:
            continue
        r_ols = float(np.dot(x, y) / np.dot(x, x))
        # variance, conditional on the incoming rho (exact) or on the OLS rho
        # (profile approximation) — see the docstring.
        r_cond = r_ols if rho_prev is None else float(rho_prev[i])
        resid = y - r_cond * x
        a_n = a0 + len(x) / 2.0
        b_n = b0 + 0.5 * float(resid @ resid)
        s2 = b_n / rng.gamma(a_n, 1.0)          # InvGamma(a_n, b_n) draw
        sig[i] = max(np.sqrt(s2), SIG_FLOOR)
        s2 = sig[i] ** 2                        # rho conditions on the floored sig
        # then rho | sig
        prec = np.dot(x, x) / s2 + 1.0 / prior_var
        v = 1.0 / prec
        m = v * (np.dot(x, y) / s2)
        r = m + np.sqrt(v) * rng.standard_normal()
        tries = 0
        while abs(r) >= 1.0 and tries < max_tries:
            r = m + np.sqrt(v) * rng.standard_normal()
            tries += 1
        rho[i] = r if abs(r) < 1.0 else float(np.clip(r_ols, -0.99, 0.99))
    return rho, sig


def draw_factor_ar(f, rng, order=1, diagonal=True, prior_var=1.0,
                   minnesota_cross=0.05, max_tries=200):
    """
    Draw the factor VAR coefficient matrix Phi, shape (K, K*order).

    diagonal=True  -> each factor is an independent AR(order) (arms E1-E3).
    diagonal=False -> full VAR under a Minnesota prior: own lags get `prior_var`,
                      cross lags the much tighter `minnesota_cross` (arms E4+).

    The factor innovation variance is fixed at 1 (the usual DFM scale
    identification), so it does not appear in the precision below.

    Stationarity is imposed by REJECTION: the rows are jointly resampled from the
    same Gaussian posterior until the companion form is stable, which is an exact
    draw from the posterior truncated to the stationary region (the rows are
    independent a posteriori, so redrawing all of them and accepting on the joint
    condition is valid rejection sampling).

    `max_tries` bounds the attempts so a pathological posterior can never hang a
    2000-draw chain. On exhaustion the coefficients are scaled back toward zero
    until stable and SHRINK_FALLBACKS is incremented. That shrink is a
    deterministic map, NOT a draw: it piles mass just under the boundary and biases
    persistence down, which is why it is the last resort and why it is counted.
    The default of 200 is set from the worst realistic case measured on this model
    — a near-unit-root factor at T=290 rejects ~95% of draws, where 50 attempts
    would fall back on 10% of sweeps but 200 falls back on ~1 in 10,000.
    """
    global SHRINK_FALLBACKS
    f = np.asarray(f, dtype=float)
    T, K = f.shape
    X = np.column_stack([f[order - l - 1:T - l - 1] for l in range(order)])
    Y = f[order:]
    n_coef = K * order

    # Posterior moments are fixed across attempts, so compute them once and keep a
    # Cholesky factor to sample from; rejection then costs one triangular solve and
    # one eigenvalue check per attempt.
    blocks = []
    for k in range(K):
        if diagonal:
            cols = [l * K + k for l in range(order)]
        else:
            cols = list(range(n_coef))
        Xk = X[:, cols]
        pv = np.full(len(cols), prior_var, dtype=float)
        if not diagonal:
            for j, c in enumerate(cols):
                if c % K != k:
                    pv[j] = minnesota_cross
        prec = Xk.T @ Xk + np.diag(1.0 / pv)
        V = np.linalg.inv(prec)
        V = 0.5 * (V + V.T)
        m = V @ (Xk.T @ Y[:, k])
        try:
            chol = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:           # V PSD but not PD: fall back to eigh
            w, Q = np.linalg.eigh(V)
            chol = Q @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
        blocks.append((k, cols, m, chol))

    def _sample():
        Phi = np.zeros((K, n_coef))
        for k, cols, m, chol in blocks:
            Phi[k, cols] = m + chol @ rng.standard_normal(len(cols))
        return Phi

    for _ in range(max(1, int(max_tries))):
        Phi = _sample()
        if _stationary(_companion(Phi, K, order)):
            return Phi

    SHRINK_FALLBACKS += 1
    tries = 0
    while not _stationary(_companion(Phi, K, order)) and tries < 200:
        Phi *= 0.95
        tries += 1
    return Phi


def _companion(Phi, K, order):
    if order == 1:
        return Phi[:, :K]
    C = np.zeros((K * order, K * order))
    C[:K, :] = Phi
    C[K:, :-K] = np.eye(K * (order - 1))
    return C


def normalise(Lam, f):
    """
    Identify the factors: make the top K x K block of Lam lower-triangular with a
    positive diagonal, applying the inverse rotation to f so the common component
    f @ Lam.T is unchanged.

    LQ via a QR of the transpose: top.T = q1 @ r1, so top = r1.T @ q1.T with
    r1.T lower-triangular and q1.T orthogonal. Because R is orthogonal,
    f_n @ Lam_n.T = f @ R.T @ R @ Lam.T = f @ Lam.T exactly — the common component
    is untouched. Verified numerically before this plan was written.

    CAVEAT for whoever chooses which series head the panel: this assumes the top
    K x K block has full rank. If it is rank deficient the rotation still runs and
    the common component is still preserved, but the corresponding diagonal entry
    comes out at machine epsilon (~1e-16) and its sign is then positive only by
    floating-point luck — the positive-diagonal identification silently stops
    holding. Put K series that genuinely load on distinct factors in the first K
    rows.
    """
    K = Lam.shape[1]
    top = Lam[:K, :K]
    q1, r1 = np.linalg.qr(top.T)
    L, R = r1.T, q1.T                      # L lower-triangular, R orthogonal
    signs = np.sign(np.diag(L))
    signs[signs == 0] = 1.0
    L = L * signs                          # scale columns of L
    R = (R.T * signs).T                    # and rows of R, so L @ R is unchanged
    Lam_n = Lam @ R.T
    f_n = f @ R.T
    return Lam_n, f_n
