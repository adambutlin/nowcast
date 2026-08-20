"""
dfm/sampler.py — Gibbs blocks for the DFM.

Stage 1 (E1) needs three blocks plus the factor draw from state_space:
  draw_loadings   Lam | f, y, sig          series-by-series Bayesian regression
  draw_idio       rho, sig | e             idiosyncratic AR(1) + variance
  draw_factor_ar  Phi | f                  factor VAR (diagonal in E1-E3)

Later stages add SV, outlier indicators and hyperpriors; those blocks land in
Plans 2 and 3 and must not change the signatures here.

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


def draw_idio(e, rng, prior_var=10.0, a0=3.0, b0=1.0, max_tries=20):
    """
    Draw (rho, sig) for each idiosyncratic AR(1) series from its own residuals.

    Returns (rho (M,), sig (M,)). Non-stationary rho draws are rejected and
    retried; after `max_tries` the previous OLS estimate is clipped into (-0.99, 0.99).

    Note on exactness: the variance is drawn conditional on the OLS rho, then rho
    is drawn conditional on that variance. The rho draw is the exact conditional
    p(rho | sig, e); the sig draw is a profile approximation to p(sig | e), since
    it evaluates the residual sum of squares at the OLS rho instead of integrating
    over rho. The kernel is therefore an approximate, not an exact, Gibbs step; the
    error in the implied posterior for sig^2 is O(1/T) and slightly downward
    (SSR is minimised at the OLS rho). Making it exact needs the current rho passed
    in, which is a signature change deferred to a later plan.

    sig is floored at SIG_FLOOR — see the comment on that constant.
    """
    e = np.asarray(e, dtype=float)
    T, M = e.shape
    rho = np.zeros(M)
    sig = np.ones(M)
    for i in range(M):
        x, y = e[:-1, i], e[1:, i]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 10 or np.dot(x, x) <= 0:
            continue
        # variance first, conditional on an OLS rho
        r_ols = float(np.dot(x, y) / np.dot(x, x))
        resid = y - r_ols * x
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
                   minnesota_cross=0.05, max_tries=50):
    """
    Draw the factor VAR coefficient matrix Phi, shape (K, K*order).

    diagonal=True  -> each factor is an independent AR(order) (arms E1-E3).
    diagonal=False -> full VAR under a Minnesota prior: own lags get `prior_var`,
                      cross lags the much tighter `minnesota_cross` (arms E4+).
    Non-stationary draws are rejected; on repeated failure the coefficients are
    scaled back toward zero until the companion form is stable.

    The factor innovation variance is fixed at 1 (the usual DFM scale
    identification), so it does not appear in the precision below.
    """
    f = np.asarray(f, dtype=float)
    T, K = f.shape
    X = np.column_stack([f[order - l - 1:T - l - 1] for l in range(order)])
    Y = f[order:]
    n_coef = K * order
    Phi = np.zeros((K, n_coef))
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
        m = V @ (Xk.T @ Y[:, k])
        V = 0.5 * (V + V.T)
        Phi[k, cols] = rng.multivariate_normal(m, V)

    comp = _companion(Phi, K, order)
    tries = 0
    while not _stationary(comp) and tries < max_tries:
        Phi *= 0.95
        comp = _companion(Phi, K, order)
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
