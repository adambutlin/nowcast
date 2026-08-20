"""
dfm/model.py — the nowcaster: config, Gibbs loop, nowcast.

DFMConfig turns an arm label into a set of switches, so the arm ladder is data,
not branching code. Stage 1 implements E1 only; the later flags exist so Plans 2
and 3 flip them on without changing any call site.

  E1  diagonal Phi, no SV, no outliers, unrestricted loadings, fixed priors
  E2  + stochastic volatility            (Plan 2)
  E3  + Stock-Watson outlier mixture     (Plan 2)
  E4  + full VAR under a Minnesota prior (Plan 3)
  E5  + block-restricted, sparse loadings(Plan 3)
  E6  + hierarchical shrinkage           (Plan 3)

The panel handed to `fit` is taken as-is: no centring, no scaling. Standardising
is the caller's job (dfm.panel.standardize, which computes its moments causally
on the training rows only) because only the caller knows where the training
window ends. Standardising here would either leak the evaluation rows into the
moments or need the cut point passed down anyway.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dfm import config as C
from dfm import sampler as S
from dfm import state_space as SS

ARMS = {
    #        SV     outlier  diagPhi  blockLam  hierarchical
    "E1": (False, False, True,  False, False),
    "E2": (True,  False, True,  False, False),
    "E3": (True,  True,  True,  False, False),
    "E4": (True,  True,  False, False, False),
    "E5": (True,  True,  False, True,  False),
    "E6": (True,  True,  False, True,  True),
}

# Flags whose machinery does not exist in Stage 1. Every arm above E1 sets at
# least one, so the guard is exhaustive by construction rather than by luck:
# E2/E3 via stochastic_vol, E4 via stochastic_vol, E5/E6 via stochastic_vol and
# block_loadings. `diagonal_phi=False` is deliberately NOT here — the full
# Minnesota VAR is already implemented in sampler.draw_factor_ar.
UNIMPLEMENTED = (("stochastic_vol", 2), ("outliers", 2),
                 ("block_loadings", 3), ("hierarchical", 3))


@dataclass
class DFMConfig:
    arm: str = "E1"
    mode: str = "standalone"            # or "overlay"
    k_factors: int = C.K_FACTORS
    factor_order: int = C.FACTOR_ORDER
    draws: int = C.DRAWS_FIRST
    burn: int = C.BURN_FIRST
    seed: int = C.SEED
    loading_mask: object = None         # (M, K) bool, set by E5/E6
    stochastic_vol: bool = field(init=False)
    outliers: bool = field(init=False)
    diagonal_phi: bool = field(init=False)
    block_loadings: bool = field(init=False)
    hierarchical: bool = field(init=False)

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; expected one of {sorted(ARMS)}")
        (self.stochastic_vol, self.outliers, self.diagonal_phi,
         self.block_loadings, self.hierarchical) = ARMS[self.arm]
        if self.burn >= self.draws:
            raise ValueError("burn must be smaller than draws")
        for flag, plan in UNIMPLEMENTED:
            if getattr(self, flag):
                raise NotImplementedError(
                    f"{self.arm} needs {flag}, which lands in Plan {plan}")


class DFMNowcaster:
    """Gibbs-sampled mixed-frequency DFM. Fit on a panel, nowcast the last row."""

    def __init__(self, cfg=None):
        self.cfg = cfg or DFMConfig()
        self.posterior = None
        self.columns = None
        self.index = None
        self.target = None
        self.target_pos = None
        self.diagnostics = None

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, panel, target, warm_start=None):
        """
        Run the Gibbs sampler on `panel` (DataFrame, NaN = unobserved).

        `target` names the column the nowcast is read from; it is an ordinary
        member of the measurement equation, so a NaN target row is simply the
        month being forecast.
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        y = panel.to_numpy(dtype=float)
        T, M = y.shape
        K = cfg.k_factors
        self.columns = list(panel.columns)
        self.index = panel.index
        if target not in self.columns:
            raise KeyError(f"target {target!r} is not a column of the panel")
        self.target = target
        self.target_pos = self.columns.index(target)

        # SHRINK_FALLBACKS is module-global in dfm.sampler. Zero it once per
        # chain (not per sweep) so the count recorded below is this chain's.
        S.reset_shrink_fallbacks()

        # initialise from a PCA of the complete-case panel, or from a previous fit
        Lam, rho, sig, Phi = self._initialise(y, K, warm_start)
        Q_f = np.eye(K)

        keep = cfg.draws - cfg.burn
        store = dict(Lam=np.zeros((keep, M, K)),
                     Phi=np.zeros((keep, K, K * cfg.factor_order)),
                     rho=np.zeros((keep, M)), sig=np.zeros((keep, M)),
                     factors=np.zeros((keep, T, K)))

        gamma_z = np.zeros(M)                 # exogenous block enters in Plan 3
        for it in range(cfg.draws):
            state = SS.simulation_smoother(y, Lam, Q_f, Phi, rho, sig, rng)
            f = state[:, :K]
            e = state[:, 2 * K:]
            Lam = S.draw_loadings(y, f, gamma_z, sig, rng, mask=cfg.loading_mask)
            Lam, f = S.normalise(Lam, f)
            # rho_prev makes the variance draw the true full conditional
            # p(sig^2 | rho, e) rather than a profile approximation; without it
            # sig carries a small but measurable downward bias.
            rho, sig = S.draw_idio(e, rng, rho_prev=rho)
            Phi = S.draw_factor_ar(f, rng, order=cfg.factor_order,
                                   diagonal=cfg.diagonal_phi)
            if it >= cfg.burn:
                j = it - cfg.burn
                store["Lam"][j], store["Phi"][j] = Lam, Phi
                store["rho"][j], store["sig"][j] = rho, sig
                store["factors"][j] = f
        self.posterior = store
        self.diagnostics = self._diagnose(y, store)
        return self

    def _diagnose(self, y, store):
        """
        Chain diagnostics that would otherwise be invisible.

        `shrink_fallbacks` is exact for the chain: draw_factor_ar increments the
        module counter whenever rejection sampling gave up and fell back to the
        deterministic shrink, which biases persistence toward zero.

        `n_degenerate` is NOT a chain total. The sweep only calls
        simulation_smoother, which does not return the count, so this is one
        extra kalman_filter pass at the posterior mean — enough to reveal a panel
        or a parameter region where observations are silently being dropped, but
        it does not see a degenerate update that occurred at some individual draw.
        One filter against 2 * draws smoother passes is ~0.05% overhead.
        """
        Lam = store["Lam"].mean(axis=0)
        Phi = store["Phi"].mean(axis=0)
        rho = store["rho"].mean(axis=0)
        sig = store["sig"].mean(axis=0)
        kf = SS.kalman_filter(y, Lam, np.eye(Lam.shape[1]), Phi, rho, sig)
        return dict(shrink_fallbacks=int(S.SHRINK_FALLBACKS),
                    n_degenerate_at_posterior_mean=int(kf["n_degenerate"]),
                    loglik_at_posterior_mean=float(kf["loglik"]),
                    draws=self.cfg.draws, burn=self.cfg.burn)

    # ── initialisation ───────────────────────────────────────────────────────
    def _cold_start(self, y, K):
        """PCA of the mean-filled panel; AR(1) and scale defaults."""
        M = y.shape[1]
        filled = np.where(np.isfinite(y), y, np.nanmean(y, axis=0))
        filled = filled - filled.mean(axis=0)
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        Lam = np.zeros((M, K))
        kk = min(K, vt.shape[0])
        Lam[:, :kk] = vt[:kk].T * s[:kk] / np.sqrt(len(filled))
        rho = np.full(M, 0.2)
        sig = np.nanstd(y, axis=0)
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0) * 0.5
        return Lam, rho, np.maximum(sig, S.SIG_FLOOR)

    def _initialise(self, y, K, warm_start):
        """
        Starting values for the chain: (Lam, rho, sig, Phi).

        The cold start always runs, so a warm start only has to overwrite what it
        can actually supply — a series the previous fit did not carry keeps its
        PCA value instead of inheriting someone else's loadings.

        Alignment is BY COLUMN NAME when the warm start carries `columns`
        (posterior_mean does). The walk-forward panel grows rows every month and
        may grow columns too; a positional slice would then hand series i the
        loadings of whatever sat at position i last month, which is a silent
        wrong answer rather than an error. Positional is still accepted for a
        hand-built warm start, but only when the widths match exactly.
        """
        M = y.shape[1]
        Lam, rho, sig = self._cold_start(y, K)
        Phi = np.zeros((K, K * self.cfg.factor_order))
        Phi[:, :K] = np.eye(K) * 0.3
        if warm_start is None:
            return Lam, rho, sig, Phi

        W_lam = np.asarray(warm_start["Lam"], dtype=float)
        W_rho = np.asarray(warm_start["rho"], dtype=float).reshape(-1)
        W_sig = np.asarray(warm_start["sig"], dtype=float).reshape(-1)
        prev = warm_start.get("columns")
        kk = min(K, W_lam.shape[1])
        if prev is not None:
            if not (W_lam.shape[0] == len(prev) == W_rho.shape[0] == W_sig.shape[0]):
                raise ValueError("warm_start arrays disagree with its column list")
            pos = {c: j for j, c in enumerate(prev)}
            for i, name in enumerate(self.columns):
                j = pos.get(name)
                if j is None:
                    continue                      # new series: keep the cold start
                Lam[i, :kk] = W_lam[j, :kk]
                rho[i] = W_rho[j]
                sig[i] = max(float(W_sig[j]), S.SIG_FLOOR)
        else:
            if not (W_lam.shape[0] == M == W_rho.shape[0] == W_sig.shape[0]):
                raise ValueError(
                    f"warm_start has {W_lam.shape[0]} series but the panel has {M}; "
                    "pass a warm start carrying 'columns' so it can be aligned by name")
            Lam[:, :kk] = W_lam[:, :kk]
            rho[:] = W_rho
            sig[:] = np.maximum(W_sig, S.SIG_FLOOR)

        # Phi too, when it is the right shape and stationary — the whole point of
        # a warm start is a shorter burn-in, and persistence is the slowest block
        # to settle. A mean of stationary draws is not guaranteed stationary, so
        # it is checked rather than assumed.
        W_phi = warm_start.get("Phi")
        if W_phi is not None:
            W_phi = np.asarray(W_phi, dtype=float)
            if W_phi.shape == Phi.shape and S._stationary(
                    S._companion(W_phi, K, self.cfg.factor_order)):
                Phi = W_phi.copy()
        return Lam, rho, sig, Phi

    # ── nowcast ──────────────────────────────────────────────────────────────
    def nowcast(self, panel, target, row=-1):
        """
        Posterior predictive for `target` in row `row` (default: the last).

        `row` indexes the FITTED panel, and `posterior["factors"]` has one row per
        panel row, so row -1 is the last month of the panel handed to `fit` — the
        nowcast month, whose target the caller blanked. The panel and target are
        re-checked against the fit for exactly this reason: reading a stale row is
        silent and produces a plausible number (docs/postmortem_2026_07.md), so
        the returned dict also names the month it refers to.
        """
        if self.posterior is None:
            raise RuntimeError("fit() before nowcast()")
        if target != self.target:
            raise ValueError(f"fitted on target {self.target!r}, asked for {target!r}")
        T = self.posterior["factors"].shape[1]
        if panel is not None:
            if list(panel.columns) != self.columns:
                raise ValueError("panel columns differ from the fitted panel")
            if len(panel) != T:
                raise ValueError(
                    f"panel has {len(panel)} rows but the fit used {T}; refit first")
        i = self.target_pos
        f = self.posterior["factors"][:, row, :]              # (keep, K)
        lam = self.posterior["Lam"][:, i, :]                  # (keep, K)
        common = np.einsum("dk,dk->d", f, lam)
        sig = self.posterior["sig"][:, i]
        rng = np.random.default_rng(self.cfg.seed + 1)
        draws = common + rng.standard_normal(len(common)) * sig
        month = None if self.index is None else pd.Timestamp(self.index[row])
        return dict(mean=float(draws.mean()),
                    median=float(np.median(draws)),
                    p05=float(np.percentile(draws, 5)),
                    p95=float(np.percentile(draws, 95)),
                    sd=float(draws.std()),
                    month=month,
                    draws=draws)

    def posterior_mean(self):
        """
        Compact summary used to warm-start the next month's fit.

        Carries `columns` so the next fit can align by name; see _initialise.
        """
        out = {k: v.mean(axis=0) for k, v in self.posterior.items() if k != "factors"}
        out["columns"] = list(self.columns)
        out["k_factors"] = self.cfg.k_factors
        return out
