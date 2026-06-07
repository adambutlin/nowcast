import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import yfinance as yf
from dbnomics import fetch_series
import matplotlib.pyplot as plt


# ============================================================
# 1) DATA
# ============================================================

class UKInflationNowcaster:
    def __init__(self, start_date="2015-01-01"):
        self.start = start_date
        self.mu = None
        self.sigma = None

    def get_ons_cpi(self):
        cpi_series = fetch_series("ONS", "MM23", "D7G7.M")
        cpi = cpi_series[["original_period", "value"]].copy()
        cpi["original_period"] = pd.to_datetime(cpi["original_period"])
        cpi = (
            cpi.set_index("original_period")
               .rename(columns={"value": "CPI_YoY"})
               .sort_index()
               .resample("M")
               .last()
        )
        return cpi

    def fetch_data(self):
        tickers = {
            "Oil": "BZ=F",
            "GBPUSD": "GBPUSD=X",
        }

        raw = yf.download(
            list(tickers.values()),
            start=self.start,
            auto_adjust=True,
            progress=False,
        )

        prices = raw["Close"].rename(columns={v: k for k, v in tickers.items()}).sort_index()

        monthly_prices = prices.resample("M").last()
        monthly_rets = np.log(monthly_prices).diff()

        cpi = self.get_ons_cpi()

        df = monthly_rets.join(cpi, how="outer").sort_index()

        self.mu = df.mean()
        self.sigma = df.std()
        z = (df - self.mu) / self.sigma
        return z


nowcaster = UKInflationNowcaster()
df_final = nowcaster.fetch_data()
df_final = df_final[["Oil", "GBPUSD", "CPI_YoY"]].copy()

# Keep missing CPI if present; usually after join it is monthly aligned, but this handles gaps
Y_np = df_final.to_numpy(dtype=float)
T, K = Y_np.shape

# PyTensor-friendly constant
Y = pt.as_tensor_variable(Y_np)


# ============================================================
# 2) MARGINALIZED HMM LOG-LIKELIHOOD
# ============================================================

def mvn_logp_diag(y_t, mu_t, sigma):
    """
    Diagonal-Gaussian log density for one time t, handling NaNs in y_t.
    Missing entries contribute zero to the log-likelihood.
    """
    is_obs = ~pt.isnan(y_t)
    y_filled = pt.switch(is_obs, y_t, 0.0)
    mu_filled = pt.switch(is_obs, mu_t, 0.0)

    logp_each = (
        -0.5 * pt.log(2.0 * np.pi)
        - pt.log(sigma)
        - 0.5 * ((y_filled - mu_filled) / sigma) ** 2
    )
    logp_each = pt.switch(is_obs, logp_each, 0.0)
    return pt.sum(logp_each)


def forward_loglike(
    Y,
    pi0,
    P,
    phi,
    q,
    loadings,
    sigma_obs,
    sigma_f0,
):
    """
    Marginalized log-likelihood of a 2-state Markov-switching dynamic factor model.

    Model:
        s_t in {0,1} follows a Markov chain with transition matrix P
        f_t | f_{t-1}, s_t ~ N(phi[s_t] * f_{t-1}, q[s_t]^2)
        y_t | f_t, s_t ~ N(loadings[s_t] * f_t, diag(sigma_obs^2))

    To keep this NUTS-friendly, we integrate out:
        - the full discrete state path s_{1:T}
        - the continuous factor path f_{1:T}
    using a Gaussian-mixture forward recursion.

    Key idea:
        conditional on each regime, the filtered factor remains Gaussian,
        so at each t we only carry:
            w_t(j)    = regime probability mass (unnormalized, in log space)
            m_t(j)    = filtered mean of factor in regime j
            C_t(j)    = filtered variance of factor in regime j
    """

    n_states = 2
    K = Y.shape[1]

    # --- t = 0 initialization ---
    # prior for factor under each regime at t=0
    m0 = pt.zeros((n_states,))
    C0 = pt.ones((n_states,)) * sigma_f0**2

    def initial_regime_logp(j, y0, pi0, loadings, sigma_obs, m0, C0):
        # predictive observation under regime j integrating out f_0
        # y0 ~ N(0, L_j L_j' * C0[j] + diag(sigma_obs^2))
        # use Kalman update formulas for scalar latent factor

        Lj = loadings[j]  # shape (K,)
        R = sigma_obs**2
        a = m0[j]
        Qf = C0[j]

        # predictive mean and covariance in scalar-factor form
        mu_y = Lj * a
        F = R + (Lj**2) * Qf

        # diagonal approximation in observation space conditional on scalar factor integrated by update
        # use a sequential scalar update instead of full covariance inversion
        logp = pt.as_tensor_variable(0.0)
        m = a
        C = Qf

        for k in range(K):
            yk = y0[k]
            is_obs = ~pt.isnan(yk)

            lk = Lj[k]
            rk = R[k]

            fk = lk**2 * C + rk
            vk = yk - lk * m
            kk = C * lk / fk

            logp_k = -0.5 * (
                pt.log(2.0 * np.pi)
                + pt.log(fk)
                + (vk**2) / fk
            )

            m = pt.switch(is_obs, m + kk * vk, m)
            C = pt.switch(is_obs, C - kk * lk * C, C)
            logp = pt.switch(is_obs, logp + logp_k, logp)

        return pt.log(pi0[j]) + logp, m, C

    init_vals = [initial_regime_logp(j, Y[0], pi0, loadings, sigma_obs, m0, C0) for j in range(n_states)]
    logw0 = pt.stack([v[0] for v in init_vals])   # shape (2,)
    m_filt0 = pt.stack([v[1] for v in init_vals]) # shape (2,)
    C_filt0 = pt.stack([v[2] for v in init_vals]) # shape (2,)

    def step(y_t, prev_logw, prev_m, prev_C, P, phi, q, loadings, sigma_obs):
        """
        One forward-filter step.

        For each new regime j:
            sum over previous regime i
            transition i->j
            predict factor from i to j
            update with y_t
        """
        n_states = 2
        K = y_t.shape[0]

        new_logw = []
        new_m = []
        new_C = []

        for j in range(n_states):
            candidate_logw = []
            candidate_m = []
            candidate_C = []

            Lj = loadings[j]
            R = sigma_obs**2

            for i in range(n_states):
                # Predict factor under new regime j from filtered state in previous regime i
                a = phi[j] * prev_m[i]
                Qf = (phi[j] ** 2) * prev_C[i] + q[j] ** 2

                # Sequential Kalman update over observed variables
                m = a
                C = Qf
                ll = pt.as_tensor_variable(0.0)

                for k in range(K):
                    yk = y_t[k]
                    is_obs = ~pt.isnan(yk)

                    lk = Lj[k]
                    rk = R[k]

                    fk = lk**2 * C + rk
                    vk = yk - lk * m
                    kk = C * lk / fk

                    logp_k = -0.5 * (
                        pt.log(2.0 * np.pi)
                        + pt.log(fk)
                        + (vk**2) / fk
                    )

                    m = pt.switch(is_obs, m + kk * vk, m)
                    C = pt.switch(is_obs, C - kk * lk * C, C)
                    ll = pt.switch(is_obs, ll + logp_k, ll)

                candidate_logw.append(prev_logw[i] + pt.log(P[i, j]) + ll)
                candidate_m.append(m)
                candidate_C.append(C)

            candidate_logw = pt.stack(candidate_logw)  # from i=0,1 into current j
            candidate_m = pt.stack(candidate_m)
            candidate_C = pt.stack(candidate_C)

            # mixture collapse by moment matching within regime j
            norm_j = pm.math.logsumexp(candidate_logw)
            w_norm = pt.exp(candidate_logw - norm_j)

            m_j = pt.sum(w_norm * candidate_m)
            second_moment_j = pt.sum(w_norm * (candidate_C + candidate_m**2))
            C_j = second_moment_j - m_j**2

            new_logw.append(norm_j)
            new_m.append(m_j)
            new_C.append(pt.maximum(C_j, 1e-8))

        new_logw = pt.stack(new_logw)
        new_m = pt.stack(new_m)
        new_C = pt.stack(new_C)

        return new_logw, new_m, new_C

    outputs, _ = pytensor.scan(
        fn=step,
        sequences=[Y[1:]],
        outputs_info=[logw0, m_filt0, C_filt0],
        non_sequences=[P, phi, q, loadings, sigma_obs],
    )

    logw_seq, m_seq, C_seq = outputs

    # total log-likelihood:
    # log p(y_1:T) = logsumexp(final regime log weights)
    final_logw = logw_seq[-1]
    total_ll = pm.math.logsumexp(final_logw)

    return total_ll, (logw0, m_filt0, C_filt0, logw_seq, m_seq, C_seq)


# ============================================================
# 3) PYMC MODEL
# ============================================================

with pm.Model() as ms_dfm:

    # Transition probabilities
    p11 = pm.Beta("p11", alpha=20, beta=2)
    p22 = pm.Beta("p22", alpha=20, beta=2)

    P = pt.stack([
        pt.stack([p11, 1.0 - p11]),
        pt.stack([1.0 - p22, p22]),
    ])

    # Initial regime probabilities from stationary distribution
    denom = 2.0 - p11 - p22
    pi0 = pt.stack([
        (1.0 - p22) / denom,
        (1.0 - p11) / denom,
    ])

    # Regime-specific AR(1) factor dynamics
    phi_raw = pm.Normal("phi_raw", mu=0.0, sigma=1.0, shape=2)
    phi = pm.Deterministic("phi", 0.98 * pt.tanh(phi_raw))

    q = pm.Exponential("q", 1.0, shape=2)

    # Factor prior std
    sigma_f0 = pm.Exponential("sigma_f0", 1.0)

    # Regime-specific loadings
    # Identification: Oil loading positive in both regimes
    lambda_oil = pm.HalfNormal("lambda_oil", sigma=1.0, shape=2)

    # Other loadings can switch sign
    lambda_gbp = pm.Normal("lambda_gbp", mu=0.0, sigma=1.0, shape=2)
    lambda_cpi = pm.Normal("lambda_cpi", mu=0.8, sigma=0.5, shape=2)

    loadings = pt.stack(
        [lambda_oil, lambda_gbp, lambda_cpi],
        axis=1
    )  # shape (2 states, 3 vars)

    # Idiosyncratic observation noise
    sigma_obs = pm.Exponential("sigma_obs", 1.0, shape=K)

    # Marginalized forward likelihood
    total_ll, aux = forward_loglike(
        Y=Y,
        pi0=pi0,
        P=P,
        phi=phi,
        q=q,
        loadings=loadings,
        sigma_obs=sigma_obs,
        sigma_f0=sigma_f0,
    )

    pm.Potential("ms_dfm_loglike", total_ll)

    trace = pm.sample(
        draws=1000,
        tune=1000,
        target_accept=0.95,
        chains=4,
        cores=4,
    )

print("Marginalized Markov-switching DFM sampling complete.")