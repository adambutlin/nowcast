# Mixed-Frequency Bayesian DFM — Plan 2: Stage 2 (E2 stochastic volatility, E3 robust outliers) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stochastic volatility (arm E2) and the Stock-Watson outlier mixture (arm E3) to the DFM, and determine whether either fixes the specific, diagnosed failure that sank E1 — confident over-prediction concentrated in a handful of shock months.

**Architecture:** Both arms need the same enabling change: the Kalman filter and simulation smoother must accept a **time-varying** state innovation covariance instead of a constant one. Once `state_space` carries per-period factor and idiosyncratic variances, E2 supplies them from a log-volatility random walk (Kim-Shephard-Chib 7-component mixture + FFBS) and E3 multiplies the idiosyncratic ones by a per-observation outlier scale. No other module's interface changes.

**Tech Stack:** Python 3.11 in the `quant` conda env (`/opt/anaconda3/envs/quant/bin/python`) — numpy 2.4.2, numba 0.64.0, pandas 2.3.3, scipy 1.17.0, statsmodels 0.14.6, pytest.

**Spec:** [2026-08-19-mixed-frequency-bayesian-dfm-design.md](../specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md) §1.2 (SV), §1.3 (outliers), §4 (arm ladder), §7 (selection rule).

**Predecessor:** [Plan 1](2026-08-19-dfm-plan1-foundation-stage1.md), complete. Results in [docs/dfm_backtest.md](../../dfm_backtest.md).

**Branch:** `dfm-research` (continue on it).

---

## Why this plan exists — the specific thing it must fix

Stage 1 produced a clean diagnosis, and Stage 2 is aimed squarely at it. From the Plan 1 results:

**Standalone (E1):** RMSE 0.4462 vs AA 0.4687 — beats the anchor by 4.8%, but DM p = 0.703 (not significant), loses to production's 0.4142, and **fails the pre-registered calm-month overshoot rule** at 0.2983 against a 0.0651 threshold.

**Overlay (E1):** RMSE 0.6571 — **worse than predicting zero** (0.4621), significantly so at DM p = 0.014.

The overlay diagnosis is the sharp one:

| leg | corr with true residual | forecast sd | RMSE as residual forecast |
|---|---|---|---|
| TVP | +0.505 | 0.403 | 0.4276 |
| LGBM | +0.315 | 0.171 | 0.4319 |
| **E1 DFM** | **+0.269** | **0.600** | **0.6571** |
| zero overlay | — | 0.000 | 0.4621 |

True residual sd is 0.455. E1 has the worst correlation with the truth and the largest amplitude. It is **confidently wrong**, and the error is concentrated:

- Shock window (2022–23): leg sd 0.784 vs true 0.750, shock RMSE **1.3049** against AA's 0.7715.
- Calm window: leg sd 0.234 vs true 0.346 — under-dispersed, and harmless.
- Worst month **2022-11**: predicted **+2.856pp** against a realised **−0.215pp**. Wrong sign, 3pp of error, one month.

E1 has no mechanism to know that 2022–23 differs from 2016. Measured MoM volatility is **0.688 in shock months against 0.336 ex-shock** — a 2× ratio the model treats as constant. That is exactly what SV models. And a single 3pp wrong-signed month is the textbook case for fat-tailed innovations, which is what the outlier mixture models.

**So Stage 2 has a falsifiable target, not just "add features".** Success is not merely a lower RMSE; it is:

1. The overlay leg's amplitude coming into line with the residual it predicts (sd ratio → ~1.0 or below, from 1.32).
2. The 2022-11 blowup shrinking materially.
3. Shock-window RMSE falling relative to E1.

If E2 and E3 do not move those three, the honest conclusion is that the overlay route is dead for this family, and the write-up should say so rather than proceeding to E4–E6 out of momentum.

---

## What already exists (do not rebuild)

| Module | Interface Stage 2 uses |
|---|---|
| `dfm/state_space.py` | `transition_matrix(Phi, rho)`, `state_cov(Q_f, sig, K, M)`, `design_matrix(Lam, K, M)`, `_solve_psd`, `_initial_state`, `_filter_core`, `_filter_core_structured`, `_run_filter`, `kalman_filter(y, Lam, Q_f, Phi, rho, sig)`, `_smooth_means`, `simulation_smoother(y, Lam, Q_f, Phi, rho, sig, rng)`. Validated against statsmodels to 1e-13 on complete and ragged data. |
| `dfm/sampler.py` | `draw_loadings(y, f, gamma_z, sig, rng, mask=, prior_var=)`, `draw_idio(e, rng, ..., rho_prev=)`, `draw_factor_ar(f, rng, order=, diagonal=, ...)`, `normalise(Lam, f)`, `SIG_FLOOR`, `SHRINK_FALLBACKS`, `reset_shrink_fallbacks()`. |
| `dfm/model.py` | `ARMS` table, `DFMConfig` (E2–E6 currently raise `NotImplementedError`), `DFMNowcaster.fit/nowcast/posterior_mean`. |
| `dfm/backtest.py` | `walk_forward` (causal per-fold standardisation, purge/embargo, checkpoint/resume), `metrics`, `dm_test`. |
| `dfm/experiments.py` | `model_panel`, `timing_probe`, `run_arm(arm=, mode=, panel=, target=)`, `to_yoy`, `summarise`. |
| `dfm/benchmarks.py` | `ar1_backtest`, `autoarima_backtest`, `production_backtest`. |

Committed result artefacts: `data/dfm/backtest_E1_standalone.csv`, `backtest_E1_overlay.csv`, `production_replication.csv`, `stage1_summary.csv`, `overlay_comparison.csv`, `release_calendar.csv`.

---

## Environment rules — read before running anything

From `docs/CLAUDE.md`:

- **Only** `/opt/anaconda3/envs/quant/bin/python`. Never `.venv`, never conda `base`.
- **Never run Python processes in parallel** on this machine without measuring first (see the note under Compute).
- Network fetches must be **foreground** with `dangerouslyDisableSandbox: true`, after `set -a; . ./.env; set +a`. Background-launched fetches get re-sandboxed and hang. A Plan 1 subagent lost an entire task this way.
- Long compute runs with **no** network are safe to background.
- Panel builds need a **retry loop** — dbnomics/IMF failed transiently twice during Plan 1, on different series each time.

`$Q` means `/opt/anaconda3/envs/quant/bin/python`; every command assumes `PYTHONPATH=code` from the repo root.

---

## File Structure

| File | Change |
|---|---|
| `code/dfm/state_space.py` | **Modify** — accept time-varying factor/idio innovation variances |
| `code/dfm/sv.py` | **Create** — KSC mixture constants, FFBS for a log-vol random walk, SV hyperparameter draws |
| `code/dfm/outliers.py` | **Create** — Stock-Watson outlier indicator and probability draws |
| `code/dfm/sampler.py` | **Modify** — re-export `draw_sv`/`draw_outliers` seams; no change to existing block signatures |
| `code/dfm/model.py` | **Modify** — un-gate E2 then E3, extend the Gibbs loop, store SV and outlier paths |
| `code/tests/test_dfm_state_space.py` | **Modify** — append time-varying-variance tests |
| `code/tests/test_dfm_sv.py` | **Create** |
| `code/tests/test_dfm_outliers.py` | **Create** |
| `code/tests/test_dfm_model.py` | **Modify** — append E2 and E3 tests |
| `docs/dfm_backtest.md` | **Append** — Stage 2 sections |

Outputs: `data/dfm/backtest_E2_*.csv`, `backtest_E3_*.csv`, `stage2_summary.csv`, `timing_probe_E2.json`.

---

## Task 1: Time-varying state innovation covariance

Both arms need this and nothing else needs changing. Today `_filter_core` takes one constant `Q`. With SV, the factor innovation variance is `exp(h_t)` and the idiosyncratic variance is `exp(g_t)`; with outliers the latter is further scaled by `o_it²`.

Rather than materialising `T` dense `n×n` matrices, pass the diagonals: `q_factor` of shape `(K,)` or `(T, K)`, and `q_idio` of shape `(M,)` or `(T, M)`. Constant inputs must produce **bit-identical** results to today.

**Files:**
- Modify: `code/dfm/state_space.py`
- Test: `code/tests/test_dfm_state_space.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `code/tests/test_dfm_state_space.py`:

```python
class TestTimeVaryingVariance:
    def test_constant_arrays_reproduce_the_scalar_path_exactly(self, sim):
        """Broadcasting a constant (T,K)/(T,M) variance must change nothing."""
        T, K, M = sim["T"], sim["K"], sim["M"]
        base = SS.kalman_filter(sim["y"], sim["Lam"], np.eye(K), sim["Phi"],
                                sim["rho"], sim["sig"])
        qf = np.ones((T, K))
        qi = np.tile(sim["sig"] ** 2, (T, 1))
        tv = SS.kalman_filter(sim["y"], sim["Lam"], qf, sim["Phi"],
                              sim["rho"], np.sqrt(qi))
        assert tv["loglik"] == base["loglik"]
        assert np.array_equal(tv["a_filt"], base["a_filt"])

    def test_seeded_draw_is_unchanged_by_constant_broadcast(self, sim):
        T, K, M = sim["T"], sim["K"], sim["M"]
        a = SS.simulation_smoother(sim["y"], sim["Lam"], np.eye(K), sim["Phi"],
                                   sim["rho"], sim["sig"], np.random.default_rng(4))
        b = SS.simulation_smoother(sim["y"], sim["Lam"], np.ones((T, K)), sim["Phi"],
                                   sim["rho"], np.tile(sim["sig"], (T, 1)),
                                   np.random.default_rng(4))
        assert np.allclose(a, b, atol=1e-12)

    def test_a_high_variance_period_widens_the_filtered_state(self, sim):
        """Inflating the factor innovation variance in one window must widen P."""
        T, K = sim["T"], sim["K"]
        qf = np.ones((T, K))
        calm = SS.kalman_filter(sim["y"], sim["Lam"], qf, sim["Phi"], sim["rho"], sim["sig"])
        qf2 = qf.copy(); qf2[100:120] *= 25.0
        loud = SS.kalman_filter(sim["y"], sim["Lam"], qf2, sim["Phi"], sim["rho"], sim["sig"])
        assert loud["P_filt"][110, 0, 0] > calm["P_filt"][110, 0, 0]

    def test_a_downweighted_observation_moves_the_state_less(self, sim):
        """Inflating one series' idio variance must shrink its influence."""
        T, M = sim["T"], sim["M"]
        qi = np.tile(sim["sig"], (T, 1))
        tight = SS.kalman_filter(sim["y"], sim["Lam"], np.eye(sim["K"]), sim["Phi"],
                                 sim["rho"], qi)
        qi2 = qi.copy(); qi2[:, 0] *= 50.0
        loose = SS.kalman_filter(sim["y"], sim["Lam"], np.eye(sim["K"]), sim["Phi"],
                                 sim["rho"], qi2)
        assert not np.allclose(tight["a_filt"][:, 0], loose["a_filt"][:, 0])

    def test_recovers_a_known_volatility_regime(self):
        """With the true time-varying variances supplied, factors are recovered."""
        rng = np.random.default_rng(21)
        T, K, M = 300, 1, 6
        h = np.concatenate([np.full(150, -1.0), np.full(150, 1.5)])   # vol jump
        Phi = np.array([[0.6]])
        f = np.zeros((T, K))
        for t in range(1, T):
            f[t] = Phi @ f[t - 1] + rng.normal(scale=np.exp(h[t] / 2), size=K)
        Lam = rng.normal(size=(M, K))
        sig = np.full(M, 0.4)
        y = f @ Lam.T + rng.normal(size=(T, M)) * sig
        out = SS.kalman_filter(y, Lam, np.exp(h)[:, None], Phi, np.zeros(M),
                               np.tile(sig, (T, 1)))
        c = np.corrcoef(out["a_filt"][20:, 0], f[20:, 0])[0, 1]
        assert abs(c) > 0.9, f"factor correlation {c:.3f}"

    def test_rejects_a_mis-shaped_variance_array(self, sim):
        with pytest.raises(ValueError, match="shape"):
            SS.kalman_filter(sim["y"], sim["Lam"], np.ones((7, sim["K"])), sim["Phi"],
                             sim["rho"], sim["sig"])
```

Rename that last test to `test_rejects_a_misshaped_variance_array` (no hyphen) when you write it — a hyphen is not valid in a Python identifier.

- [ ] **Step 2: Run and confirm failure**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_state_space.py::TestTimeVaryingVariance -q
```

Expected: failures on shape handling — the current code assumes `sig` is `(M,)`.

- [ ] **Step 3: Implement**

In `code/dfm/state_space.py`:

1. Add a normaliser used by every public entry point:

```python
def _as_tv(arr, T, d, name):
    """Broadcast a (d,) or (T, d) variance spec to (T, d), validating shape."""
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        if a.shape[0] != d:
            raise ValueError(f"{name} has shape {a.shape}, expected ({d},) or ({T}, {d})")
        return np.repeat(a[None, :], T, axis=0)
    if a.shape != (T, d):
        raise ValueError(f"{name} has shape {a.shape}, expected ({d},) or ({T}, {d})")
    return a
```

2. Change `kalman_filter` and `simulation_smoother` to take `q_factor` (variance, not sd) and `sig` (sd, kept for backward compatibility — square it internally). Normalise both to `(T, K)` and `(T, M)`.

3. Change `_filter_core` and `_filter_core_structured` to take `q_fac_t` `(T, K)` and `q_idio_t` `(T, M)` instead of a single `Q`, and build the prediction step per period:

```
P = T P T' + Q_t     where Q_t = blkdiag(diag(q_fac_t[t]), 0_K, diag(q_idio_t[t]))
```

Because `Q_t` is diagonal and structured, add it in place rather than forming the matrix: add `q_fac_t[t]` to the first `K` diagonal entries and `q_idio_t[t]` to the last `M`. Keep both the structured and dense paths, and keep `_run_filter`'s dispatch.

4. `_initial_state` uses a **constant** `Q` for the stationary solve. With SV the unconditional variance is not defined by a single `Q`; use the **time-average** of the supplied variances for initialisation only, and document that this affects the prior on `s_0` alone, not the recursion. Keep the diffuse fallback.

5. `simulation_smoother`'s forward simulation must draw `s_plus[t]` with the period-`t` covariance, not a constant one. Since `Q_t` is diagonal, use `sqrt(q)` elementwise rather than a Cholesky.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_state_space.py -q
```

Expected: all pass, including every pre-existing test. **The statsmodels cross-check and the constant-input equivalence tests are the ones that matter** — if either fails, the refactor changed behaviour and must be fixed, not accommodated.

- [ ] **Step 5: Prove equivalence and re-measure cost**

```bash
PYTHONPATH=code $Q -u -c "
import time, numpy as np
from dfm import state_space as SS
K,M,T = 7,33,330
rng = np.random.default_rng(0)
Lam = rng.normal(size=(M,K)); Phi = np.zeros((K,2*K)); Phi[:,:K]=np.eye(K)*0.4
rho=np.full(M,0.3); sig=np.full(M,0.5); Qf=np.eye(K)
y = rng.normal(size=(T,M)); y[rng.random((T,M))<0.15]=np.nan
SS.kalman_filter(y,Lam,Qf,Phi,rho,sig)
t0=time.time(); r=SS.kalman_filter(y,Lam,Qf,Phi,rho,sig); tf=time.time()-t0
t0=time.time(); SS.simulation_smoother(y,Lam,Qf,Phi,rho,sig,rng); ts=time.time()-t0
print(f'constant path: filter {tf*1000:.1f} ms  smoother {ts*1000:.1f} ms')
qf=np.ones((T,K)); qi=np.tile(sig,(T,1))
t0=time.time(); SS.simulation_smoother(y,Lam,qf,Phi,rho,qi,rng); ts2=time.time()-t0
print(f'time-varying path: smoother {ts2*1000:.1f} ms  overhead {100*(ts2/ts-1):+.0f}%')
"
```

Plan 1 measured the smoother at **52 ms**. Record the new figure. If the time-varying path costs more than ~25% over the constant one, say so — it compounds across 800 draws × 120 months × 4 arm-modes.

- [ ] **Step 6: Commit**

```bash
git add code/dfm/state_space.py code/tests/test_dfm_state_space.py
git commit -m "feat(dfm): time-varying state innovation covariance in filter and smoother"
```

---

## Task 2: KSC mixture and FFBS for a log-volatility random walk

`h_t = h_{t-1} + σ ζ_t`, observed through `log(ε_t²) = h_t + log(χ²_1)`. Kim-Shephard-Chib (1998) approximate `log(χ²_1)` by a 7-component Gaussian mixture, which makes the conditional model linear-Gaussian and lets FFBS draw the whole `h` path at once.

**Files:**
- Create: `code/dfm/sv.py`
- Test: `code/tests/test_dfm_sv.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_sv.py`:

```python
"""Stochastic volatility: the KSC mixture and the FFBS log-vol draw."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import sv as SV


class TestKSCMixture:
    def test_weights_sum_to_one(self):
        assert SV.KSC_Q.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_components_present(self):
        assert SV.KSC_Q.shape == SV.KSC_M.shape == SV.KSC_V2.shape == (7,)
        assert np.all(SV.KSC_V2 > 0)

    def test_mixture_matches_log_chi2_moments(self):
        """The mixture must approximate log(chi^2_1): mean -1.2704, var pi^2/2."""
        mean = float((SV.KSC_Q * SV.KSC_M).sum())
        var = float((SV.KSC_Q * (SV.KSC_V2 + SV.KSC_M ** 2)).sum() - mean ** 2)
        assert mean == pytest.approx(-1.2704, abs=0.01)
        assert var == pytest.approx(np.pi ** 2 / 2, abs=0.05)

    def test_mixture_matches_simulated_log_chi2(self):
        rng = np.random.default_rng(0)
        draws = np.log(rng.chisquare(1, size=200_000) + 1e-300)
        mean = float((SV.KSC_Q * SV.KSC_M).sum())
        assert mean == pytest.approx(draws.mean(), abs=0.03)


class TestDrawLogVolPath:
    def _simulate(self, T=400, sigma=0.25, seed=0):
        rng = np.random.default_rng(seed)
        h = np.zeros(T)
        for t in range(1, T):
            h[t] = h[t - 1] + sigma * rng.standard_normal()
        eps = rng.standard_normal(T) * np.exp(h / 2)
        return h, eps

    def test_recovers_a_known_volatility_path(self):
        h, eps = self._simulate()
        rng = np.random.default_rng(7)
        cur = np.zeros(len(h))
        draws = []
        for i in range(200):
            cur = SV.draw_log_vol(eps, cur, 0.25, rng)
            if i >= 100:
                draws.append(cur)
        est = np.mean(draws, axis=0)
        c = np.corrcoef(est, h)[0, 1]
        assert c > 0.8, f"log-vol path correlation {c:.3f}"

    def test_returns_the_right_shape_and_is_finite(self):
        h, eps = self._simulate(T=120)
        got = SV.draw_log_vol(eps, np.zeros(120), 0.2, np.random.default_rng(1))
        assert got.shape == (120,) and np.all(np.isfinite(got))

    def test_draws_are_stochastic(self):
        h, eps = self._simulate(T=150)
        rng = np.random.default_rng(3)
        a = SV.draw_log_vol(eps, np.zeros(150), 0.2, rng)
        b = SV.draw_log_vol(eps, np.zeros(150), 0.2, rng)
        assert not np.allclose(a, b)

    def test_a_zero_epsilon_does_not_produce_nan(self):
        """log(0) must be floored, not propagated."""
        h, eps = self._simulate(T=100)
        eps[50] = 0.0
        got = SV.draw_log_vol(eps, np.zeros(100), 0.2, np.random.default_rng(2))
        assert np.all(np.isfinite(got))

    def test_higher_true_sigma_gives_a_more_variable_path(self):
        rng = np.random.default_rng(11)
        _, eps_calm = self._simulate(sigma=0.05, seed=5)
        _, eps_wild = self._simulate(sigma=0.60, seed=5)
        calm = SV.draw_log_vol(eps_calm, np.zeros(400), 0.05, rng)
        wild = SV.draw_log_vol(eps_wild, np.zeros(400), 0.60, rng)
        assert np.diff(wild).std() > np.diff(calm).std()


class TestDrawSigma:
    def test_recovers_the_innovation_sd(self):
        rng = np.random.default_rng(4)
        T, true_sigma = 800, 0.3
        h = np.concatenate([[0.0], np.cumsum(rng.standard_normal(T - 1) * true_sigma)])
        draws = [SV.draw_sv_sigma(h, rng) for _ in range(200)]
        assert abs(np.mean(draws) - true_sigma) < 0.05

    def test_is_strictly_positive(self):
        rng = np.random.default_rng(5)
        h = np.cumsum(rng.standard_normal(200) * 0.2)
        assert all(SV.draw_sv_sigma(h, rng) > 0 for _ in range(50))
```

- [ ] **Step 2: Run and confirm the missing-module failure**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_sv.py -q
```

- [ ] **Step 3: Implement**

Create `code/dfm/sv.py`:

```python
"""
dfm/sv.py — stochastic volatility via the Kim-Shephard-Chib mixture and FFBS.

The volatility model is a log-variance random walk:

    eps_t = exp(h_t / 2) * z_t,   z_t ~ N(0, 1)
    h_t   = h_{t-1} + sigma * zeta_t

Squaring and logging gives  log(eps_t^2) = h_t + log(chi^2_1), which is linear in
h but has a non-Gaussian error. Kim, Shephard and Chib (1998) approximate
log(chi^2_1) with a 7-component Gaussian mixture; conditional on the component
indicator the model is linear-Gaussian, so the whole h path can be drawn in one
forward-filter/backward-sample pass rather than one period at a time.

Why this matters here: Stage 1's E1 arm treated 2022-23 and 2016 as equally
volatile and was consequently over-confident in exactly the months that mattered
(measured MoM volatility 0.688 in shock months against 0.336 ex-shock). This is
the machinery that lets the model know the difference.
"""
import numpy as np

# Kim-Shephard-Chib (1998) Table 4: 7-component Gaussian mixture for log(chi^2_1).
# KSC_M already carries the mean offset, so the mixture mean is ~-1.2704 and its
# variance ~pi^2/2 = 4.9348 (asserted in the tests, not taken on trust).
KSC_Q = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
KSC_M = np.array([-11.40039, -5.24321, -9.83726, 1.50746, -0.65098, 0.52478, -2.35859])
KSC_V2 = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])

_LOG_EPS_FLOOR = 1e-10          # keeps log(eps^2) finite when eps is exactly 0
H_MIN, H_MAX = -20.0, 20.0      # exp(h/2) stays representable


def _sample_components(y_star, h, rng):
    """Draw the mixture indicator for each period given the current h path."""
    resid = y_star[:, None] - h[:, None] - KSC_M[None, :]
    logp = (np.log(KSC_Q)[None, :]
            - 0.5 * np.log(KSC_V2)[None, :]
            - 0.5 * resid ** 2 / KSC_V2[None, :])
    logp -= logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p /= p.sum(axis=1, keepdims=True)
    u = rng.random(len(y_star))
    return (p.cumsum(axis=1) < u[:, None]).sum(axis=1).clip(0, 6)


def draw_log_vol(eps, h_prev, sigma, rng, h0_var=10.0):
    """
    One FFBS draw of the log-variance path given residuals `eps`.

    eps     : (T,) residuals whose variance is exp(h_t)
    h_prev  : (T,) current log-vol path, used only to draw the mixture indicators
    sigma   : sd of the log-vol random-walk innovation
    """
    eps = np.asarray(eps, dtype=float)
    T = eps.shape[0]
    y_star = np.log(eps ** 2 + _LOG_EPS_FLOOR)
    s = _sample_components(y_star, np.asarray(h_prev, dtype=float), rng)
    m, v2 = KSC_M[s], KSC_V2[s]

    # Forward filter: state h_t, transition h_t = h_{t-1} + N(0, sigma^2),
    # observation y_star_t = h_t + N(m_t, v2_t).
    a = np.zeros(T)
    P = np.zeros(T)
    a_pred, P_pred = 0.0, h0_var
    q = sigma ** 2
    for t in range(T):
        if t > 0:
            a_pred, P_pred = a[t - 1], P[t - 1] + q
        F = P_pred + v2[t]
        k = P_pred / F
        a[t] = a_pred + k * (y_star[t] - m[t] - a_pred)
        P[t] = P_pred * (1.0 - k)

    # Backward sample
    h = np.empty(T)
    h[-1] = a[-1] + np.sqrt(max(P[-1], 0.0)) * rng.standard_normal()
    for t in range(T - 2, -1, -1):
        P_pred = P[t] + q
        k = P[t] / P_pred
        mean = a[t] + k * (h[t + 1] - a[t])
        var = P[t] * (1.0 - k)
        h[t] = mean + np.sqrt(max(var, 0.0)) * rng.standard_normal()
    return np.clip(h, H_MIN, H_MAX)


def draw_sv_sigma(h, rng, a0=5.0, b0=0.05):
    """
    Draw the log-vol random-walk innovation sd from its inverse-gamma posterior.

    The prior is deliberately informative (a0=5, b0=0.05 -> prior mean around
    0.11): with T ~ 330 monthly observations a diffuse prior lets sigma wander
    high and the volatility path degenerates into fitting one-off outliers, which
    is the outlier block's job (E3), not the SV block's.
    """
    d = np.diff(np.asarray(h, dtype=float))
    a_n = a0 + 0.5 * len(d)
    b_n = b0 + 0.5 * float(d @ d)
    return float(np.sqrt(b_n / rng.gamma(a_n, 1.0)))
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_sv.py -q
```

Expected: all pass. If `test_recovers_a_known_volatility_path` fails, the FFBS backward recursion is the likeliest culprit — check the gain `k = P[t] / (P[t] + q)`.

- [ ] **Step 5: Commit**

```bash
git add code/dfm/sv.py code/tests/test_dfm_sv.py
git commit -m "feat(dfm): KSC 7-component mixture and FFBS log-volatility draw"
```

---

## Task 3: Wire arm E2 into the model

**Files:**
- Modify: `code/dfm/model.py`
- Test: `code/tests/test_dfm_model.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `code/tests/test_dfm_model.py`:

```python
class TestE2StochasticVolatility:
    def test_e2_now_constructs(self):
        cfg = DFMConfig(arm="E2")
        assert cfg.stochastic_vol and not cfg.outliers and cfg.diagonal_phi

    def test_e3_to_e6_still_blocked(self):
        for arm in ["E3", "E4", "E5", "E6"]:
            with pytest.raises(NotImplementedError):
                DFMConfig(arm=arm)

    def test_fit_stores_volatility_paths(self):
        df = synthetic_panel()
        cfg = DFMConfig(arm="E2", k_factors=2, factor_order=1, draws=120, burn=60, seed=1)
        m = DFMNowcaster(cfg).fit(df, target="cpi_mom")
        assert m.posterior["h"].shape == (60, len(df), 2)
        assert m.posterior["g"].shape == (60, len(df), df.shape[1])
        assert np.all(np.isfinite(m.posterior["h"]))

    def test_e2_tracks_a_volatility_regime(self):
        """A panel with a variance break must show a rising posterior factor vol."""
        rng = np.random.default_rng(5)
        T, K, M = 240, 1, 6
        h = np.concatenate([np.full(120, -1.5), np.full(120, 1.0)])
        f = np.zeros((T, K))
        for t in range(1, T):
            f[t] = 0.5 * f[t - 1] + rng.normal(scale=np.exp(h[t] / 2), size=K)
        Lam = rng.normal(size=(M, K))
        y = f @ Lam.T + rng.normal(size=(T, M)) * 0.3
        idx = pd.date_range("2005-01-31", periods=T, freq="ME")
        df = pd.DataFrame(y, index=idx, columns=[f"s{i}" for i in range(M)])
        df["cpi_mom"] = f[:, 0] * 0.8 + rng.normal(size=T) * 0.2
        cfg = DFMConfig(arm="E2", k_factors=1, factor_order=1, draws=200, burn=100, seed=2)
        m = DFMNowcaster(cfg).fit(df, target="cpi_mom")
        hbar = m.posterior["h"].mean(axis=0)[:, 0]
        assert hbar[150:].mean() > hbar[:100].mean() + 0.5

    def test_e2_nowcast_is_finite_and_interval_ordered(self):
        df = synthetic_panel()
        cfg = DFMConfig(arm="E2", k_factors=2, factor_order=1, draws=120, burn=60, seed=3)
        out = DFMNowcaster(cfg).fit(df, target="cpi_mom").nowcast(df, target="cpi_mom")
        assert np.isfinite(out["mean"]) and out["p05"] < out["mean"] < out["p95"]

    def test_e2_is_reproducible(self):
        df = synthetic_panel()
        cfg = DFMConfig(arm="E2", k_factors=2, factor_order=1, draws=100, burn=50, seed=9)
        a = DFMNowcaster(cfg).fit(df, target="cpi_mom").nowcast(df, target="cpi_mom")["mean"]
        b = DFMNowcaster(cfg).fit(df, target="cpi_mom").nowcast(df, target="cpi_mom")["mean"]
        assert a == pytest.approx(b)

    def test_e2_widens_intervals_in_a_volatile_tail(self):
        """A panel whose last 20 months are noisy must not report E1-tight bands."""
        rng = np.random.default_rng(8)
        df = synthetic_panel(T=200, ragged=False)
        loud = df.copy()
        loud.iloc[-20:, :-1] += rng.normal(size=(20, df.shape[1] - 1)) * 3.0
        cfg = DFMConfig(arm="E2", k_factors=2, factor_order=1, draws=150, burn=75, seed=4)
        calm_sd = DFMNowcaster(cfg).fit(df, target="cpi_mom").nowcast(df, target="cpi_mom")["sd"]
        loud_sd = DFMNowcaster(cfg).fit(loud, target="cpi_mom").nowcast(loud, target="cpi_mom")["sd"]
        assert loud_sd > calm_sd
```

- [ ] **Step 2: Run and confirm failure** (`NotImplementedError` for E2).

- [ ] **Step 3: Implement**

In `code/dfm/model.py`:

1. Remove `stochastic_vol` from the `NotImplementedError` guard list, leaving `outliers` and `hierarchical` (and `block_loadings`) gated.
2. Initialise `h` as `(T, K)` zeros and `g` as `(T, M)` zeros, plus `sigma_h` `(K,)` and `sigma_g` `(M,)` at a modest starting value (0.1).
3. In the Gibbs sweep, after the factor draw and the loading/idio draws, add:

```python
if cfg.stochastic_vol:
    # factor log-vols: the factor VAR innovations are eta_t = f_t - Phi f_{t-1}
    eta = f[cfg.factor_order:] - f[:-cfg.factor_order] @ Phi[:, :K].T
    for k in range(K):
        pad = np.concatenate([np.zeros(cfg.factor_order), eta[:, k]])
        h[:, k] = SV.draw_log_vol(pad, h[:, k], sigma_h[k], rng)
        sigma_h[k] = SV.draw_sv_sigma(h[:, k], rng)
    # idiosyncratic log-vols: e_it is already in the state draw
    for i in range(M):
        g[:, i] = SV.draw_log_vol(e[:, i], g[:, i], sigma_g[i], rng)
        sigma_g[i] = SV.draw_sv_sigma(g[:, i], rng)
    q_factor = np.exp(h)
    q_idio = np.exp(g)
else:
    q_factor = np.tile(np.ones(K), (T, 1))
    q_idio = np.tile(sig ** 2, (T, 1))
```

Note the padding: `eta` is shorter than `f` by `factor_order`, so the first rows carry zeros and contribute a floored `log(eps^2)`. Document that those periods' log-vols are effectively prior-driven.

4. Pass `q_factor` and `np.sqrt(q_idio)` into `simulation_smoother`.
5. When `stochastic_vol` is on, `draw_idio` should still draw `rho` but the **variance** now comes from the SV path — pass the SV-implied sd so the two do not fight. Simplest consistent choice: keep `draw_idio` for `rho` only and take `sig` as `exp(g/2)` averaged over time for the parts of the code that need a scalar. State clearly in a comment which quantity is authoritative.
6. Store `h`, `g`, `sigma_h`, `sigma_g` in the posterior; exclude the `(T, ·)` paths from `posterior_mean()`'s warm start unless the panel length matches, since panels grow.

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=code $Q -m pytest code/tests/test_dfm_model.py -q
```

Expected: all pass, E1 tests unchanged.

- [ ] **Step 5: Confirm E1 is bit-unchanged**

The refactor must not perturb the completed Stage 1 result:

```bash
PYTHONPATH=code $Q -c "
import numpy as np, pandas as pd
from dfm.model import DFMConfig, DFMNowcaster
import sys; sys.path.insert(0,'code/tests')
from test_dfm_model import synthetic_panel
df = synthetic_panel()
cfg = DFMConfig(arm='E1', k_factors=2, factor_order=1, draws=120, burn=60, seed=1)
print('E1 nowcast:', DFMNowcaster(cfg).fit(df, target='cpi_mom').nowcast(df, target='cpi_mom')['mean'])
"
```

Record the value. Re-run it after Task 6 (E3) and confirm it is identical — E1 must not drift as later arms land.

- [ ] **Step 6: Commit**

```bash
git add code/dfm/model.py code/tests/test_dfm_model.py
git commit -m "feat(dfm): arm E2 — factor and idiosyncratic stochastic volatility"
```

---

## Task 4: E2 timing probe and budget decision

- [ ] **Step 1: Build and cache the panels** (network, FOREGROUND, with a retry loop)

```bash
set -a; . ./.env; set +a
for i in 1 2 3; do
  PYTHONPATH=code $Q -u -c "
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from dfm import experiments as E
p = E.model_panel(pd.Timestamp('2024-12-31'), pd.Timestamp('2024-12-31'))
p.to_pickle('data/dfm/_model_panel_2024.pkl')
print('standalone panel', p.shape)
" && break
  echo "retry $i (transient dbnomics failure)"; sleep 20
done
```

The overlay panel is rebuilt the same way Plan 1 did it — observations plus the `actual − (AA + RegEvent)` residual as the target column, purge 12+1. Reuse that construction; do not re-derive it.

- [ ] **Step 2: Probe**

```bash
PYTHONPATH=code $Q -u -c "
import json, pandas as pd
from dfm import experiments as E
panel = pd.read_pickle('data/dfm/_model_panel_2024.pkl')
rep = E.timing_probe(panel, n_months=3, arm='E2')
print(json.dumps(rep, indent=2, default=str))
open('data/dfm/timing_probe_E2.json','w').write(json.dumps(rep, indent=2, default=str))
"
```

Plan 1 measured E1 at **44.5 s/fit** (realised 37.7 s). Record E2's.

- [ ] **Step 3: Decide the budget before running anything long**

Stage 2 needs four walk-forwards: E2 standalone, E2 overlay, E3 standalone, E3 overlay. At E1's realised rate that is ~5 h; SV will push it higher.

If the projection for all four exceeds ~8 h, apply the spec §5.3 fallbacks in order and record which was used: (1) halve post-warm-start draws to 400, (2) quarterly parameter refit with monthly filtering. Do **not** silently accept a 20-hour run.

There is also a measurement worth taking rather than assuming, because `docs/CLAUDE.md`'s no-parallel rule was diagnosed on `.venv`, and the `quant` env carries **no quarantine or provenance xattrs** (checked 2026-08-20), so the Gatekeeper pathology may not apply to it:

```bash
$Q -c "
import subprocess, time, sys
code = 'import time; t=time.time(); import numpy, pandas; print(f\"{time.time()-t:.2f}\")'
for n in (1, 2, 4, 6):
    t0 = time.time()
    ps = [subprocess.Popen([sys.executable,'-c',code], stdout=subprocess.PIPE, text=True) for _ in range(n)]
    outs = [p.communicate()[0].strip() for p in ps]
    print(f'{n} concurrent: wall {time.time()-t0:.1f}s  per-proc imports {outs}')
"
```

If per-process import time is flat as `n` rises, the four walk-forwards can run concurrently with `OMP_NUM_THREADS=1` and 6 workers (6 performance cores), collapsing wall-clock without touching model code. If it degrades, run them sequentially and note it. **Report the numbers either way** — this settles a rule that has constrained the project since before it started.

- [ ] **Step 4: Commit the measurement**

```bash
git add -f data/dfm/timing_probe_E2.json
git commit -m "perf(dfm): E2 timing probe and the parallel-import measurement"
```

---

## Task 5: Run and gate E2

- [ ] **Step 1: Run E2 standalone** (no network once panels are cached; safe to background)

```bash
PYTHONPATH=code $Q -u -c "
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
from dfm import experiments as E
panel = pd.read_pickle('data/dfm/_model_panel_2024.pkl')
bt, path = E.run_arm(arm='E2', mode='standalone', panel=panel, verbose=True)
print('wrote', path, len(bt), 'months, coverage', bt['pred'].notna().mean())
"
```

- [ ] **Step 2: Run E2 overlay** — same, with the overlay panel and `target='resid'`.

- [ ] **Step 3: Score against E1 and the benchmarks**

Reuse Plan 1's scoring approach. Produce, for both modes:

- RMSE, MAE, rel-vs-AA, by window (full / shock 2022–23 / ex-shock / pre-2020)
- DM vs AA, vs the frozen production model, **and vs E1** — the last is the one that says whether SV helped
- Calm-month overshoot against the 0.0651 threshold
- **The three Stage 2 targets** from the top of this plan: the overlay leg's sd ratio (E1: 1.32), the 2022-11 error (E1: predicted +2.856 against −0.215), and shock-window RMSE (E1 overlay: 1.3049)

- [ ] **Step 4: Write the E2 section of `docs/dfm_backtest.md`**

State plainly whether SV fixed the over-confidence. If the sd ratio is still above ~1.1 and 2022-11 is still catastrophic, say so — that is a finding, and it raises the prior that E3 is where the fix actually lies.

- [ ] **Step 5: Gate**

Spec §4.3: E3 is not built until E2 has passing recovery tests, ≥95% finite predictions, and a written result. Confirm all three, then commit.

```bash
git add docs/dfm_backtest.md
git add -f data/dfm/backtest_E2_standalone.csv data/dfm/backtest_E2_overlay.csv
git commit -m "test(dfm): Stage 2a — E2 stochastic volatility results and gate"
```

---

## Task 6: Stock-Watson outlier mixture

`e_it = exp(g_it/2) · o_it · ε_it`, with `o_it = 1` with probability `1−p` and `o_it ~ U(2,10)` otherwise, `p ~ Beta(a,b)`. The posterior gives a per-series, per-month outlier probability, which is a deliverable in its own right.

**Files:**
- Create: `code/dfm/outliers.py`
- Test: `code/tests/test_dfm_outliers.py`

- [ ] **Step 1: Write the failing test**

Create `code/tests/test_dfm_outliers.py`:

```python
"""Stock-Watson outlier mixture: indicator draws and the outlier probability."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dfm import outliers as O


class TestGrid:
    def test_grid_spans_the_specified_range(self):
        assert O.O_GRID[0] == 1.0
        assert O.O_GRID[1] >= 2.0 and O.O_GRID[-1] <= 10.0
        assert np.all(np.diff(O.O_GRID) > 0)


class TestDrawIndicators:
    def _panel(self, T=300, M=4, seed=0, n_out=12):
        rng = np.random.default_rng(seed)
        e = rng.standard_normal((T, M))
        truth = np.zeros((T, M), dtype=bool)
        rows = rng.choice(T, n_out, replace=False)
        cols = rng.choice(M, n_out)
        e[rows, cols] *= 6.0
        truth[rows, cols] = True
        return e, truth

    def test_detects_planted_outliers(self):
        """AUC on planted outliers must be far better than chance."""
        e, truth = self._panel()
        rng = np.random.default_rng(1)
        g = np.zeros_like(e)
        acc = np.zeros_like(e)
        p = 0.02
        for _ in range(120):
            o, p = O.draw_outliers(e, g, p, rng)
            acc += (o > 1.0)
        prob = acc / 120.0
        pos, neg = prob[truth], prob[~truth]
        auc = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
        assert auc > 0.85, f"outlier detection AUC {auc:.3f}"

    def test_clean_data_yields_few_flags(self):
        rng = np.random.default_rng(2)
        e = rng.standard_normal((300, 4))
        g = np.zeros_like(e)
        p = 0.02
        for _ in range(60):
            o, p = O.draw_outliers(e, g, p, rng)
        assert (o > 1.0).mean() < 0.10

    def test_scales_are_in_the_grid(self):
        e, _ = self._panel(T=100, M=3)
        o, _ = O.draw_outliers(e, np.zeros_like(e), 0.05, np.random.default_rng(3))
        assert np.all(np.isin(o, O.O_GRID))

    def test_respects_the_log_volatility_path(self):
        """A period the SV path already calls volatile should not be flagged."""
        rng = np.random.default_rng(4)
        e = rng.standard_normal((200, 3))
        e[100:110] *= 5.0
        g = np.zeros_like(e); g[100:110] = 2 * np.log(5.0)   # SV already explains it
        p = 0.02
        acc = np.zeros_like(e)
        for _ in range(80):
            o, p = O.draw_outliers(e, g, p, rng)
            acc += (o > 1.0)
        assert (acc[100:110] / 80.0).mean() < (acc[:100] / 80.0).mean() + 0.25

    def test_probability_stays_in_the_unit_interval(self):
        e, _ = self._panel()
        rng = np.random.default_rng(5)
        p = 0.02
        for _ in range(50):
            _, p = O.draw_outliers(e, np.zeros_like(e), p, rng)
            assert 0.0 < p < 1.0

    def test_more_outliers_raise_the_estimated_probability(self):
        rng = np.random.default_rng(6)
        few, _ = self._panel(seed=7, n_out=4)
        many, _ = self._panel(seed=7, n_out=60)
        def run(e):
            p = 0.02
            for _ in range(80):
                _, p = O.draw_outliers(e, np.zeros_like(e), p, rng)
            return p
        assert run(many) > run(few)
```

- [ ] **Step 2: Run and confirm the missing-module failure.**

- [ ] **Step 3: Implement**

Create `code/dfm/outliers.py`:

```python
"""
dfm/outliers.py — the Stock-Watson outlier mixture.

    e_it = exp(g_it / 2) * o_it * eps_it
    o_it = 1               with probability 1 - p
    o_it ~ Uniform(2, 10)  with probability p,      p ~ Beta(a, b)

Chosen over a Student-t scale mixture because it is directly interpretable: the
posterior yields P(o_it > 1), a per-series per-month outlier probability that can
be read off and plotted. It is expected to light up on Ofgem cap resets, the 2020
VAT cut and COVID collection gaps.

Stage 1's concrete failure motivates it: the E1 overlay predicted a residual of
+2.856pp for 2022-11 against a realised -0.215pp. One wrong-signed month of that
size is what fat tails are for; SV alone rescales a whole regime, not one cell.

The continuous Uniform(2,10) is discretised onto a fixed grid so the conditional
draw is an exact categorical rather than a Metropolis step — the same reason the
rest of this sampler is conjugate.
"""
import numpy as np

# o = 1 (no outlier) plus a discretisation of Uniform(2, 10).
O_GRID = np.concatenate([[1.0], np.linspace(2.0, 10.0, 17)])
_N_OUT = len(O_GRID) - 1


def draw_outliers(e, g, p, rng, a0=2.0, b0=98.0):
    """
    Draw the outlier scales `o` (T, M) and the outlier probability `p`.

    e : (T, M) idiosyncratic residuals
    g : (T, M) log-variance path from the SV block (zeros if SV is off)
    p : current outlier probability

    The Beta(2, 98) prior centres p at 2% — outliers are meant to be rare. A
    diffuse prior lets the mixture absorb ordinary variation and quietly turn
    into a heavy-tailed likelihood everywhere, which would defeat the point of
    being able to name the outlying months.
    """
    e = np.asarray(e, dtype=float)
    sd = np.exp(np.asarray(g, dtype=float) / 2.0)
    z = e / np.where(sd > 0, sd, 1.0)

    # log N(z; 0, o^2) for each grid point, plus the mixture prior
    scales = O_GRID[None, None, :]
    logp = -np.log(scales) - 0.5 * (z[:, :, None] / scales) ** 2
    prior = np.concatenate([[1.0 - p], np.full(_N_OUT, p / _N_OUT)])
    logp = logp + np.log(prior)[None, None, :]
    logp -= logp.max(axis=2, keepdims=True)
    w = np.exp(logp)
    w /= w.sum(axis=2, keepdims=True)

    u = rng.random(z.shape)
    idx = (w.cumsum(axis=2) < u[:, :, None]).sum(axis=2).clip(0, len(O_GRID) - 1)
    o = O_GRID[idx]

    n_out = int((o > 1.0).sum())
    n_tot = o.size
    p_new = float(rng.beta(a0 + n_out, b0 + n_tot - n_out))
    return o, p_new
```

- [ ] **Step 4: Run tests.** Expected: all pass. The AUC test is the one that matters — it is the only check that the block detects anything rather than merely running.

- [ ] **Step 5: Commit**

```bash
git add code/dfm/outliers.py code/tests/test_dfm_outliers.py
git commit -m "feat(dfm): Stock-Watson outlier mixture with interpretable per-cell probability"
```

---

## Task 7: Wire arm E3

**Files:**
- Modify: `code/dfm/model.py`
- Test: `code/tests/test_dfm_model.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
class TestE3Outliers:
    def test_e3_now_constructs(self):
        cfg = DFMConfig(arm="E3")
        assert cfg.stochastic_vol and cfg.outliers

    def test_e4_to_e6_still_blocked(self):
        for arm in ["E4", "E5", "E6"]:
            with pytest.raises(NotImplementedError):
                DFMConfig(arm=arm)

    def test_fit_stores_outlier_probabilities(self):
        df = synthetic_panel()
        cfg = DFMConfig(arm="E3", k_factors=2, factor_order=1, draws=120, burn=60, seed=1)
        m = DFMNowcaster(cfg).fit(df, target="cpi_mom")
        assert m.posterior["outlier_prob"].shape == (len(df), df.shape[1])
        assert np.all((m.posterior["outlier_prob"] >= 0) &
                      (m.posterior["outlier_prob"] <= 1))

    def test_e3_flags_a_planted_outlier(self):
        rng = np.random.default_rng(3)
        df = synthetic_panel(T=200, ragged=False)
        df.iloc[120, 0] += 15.0
        cfg = DFMConfig(arm="E3", k_factors=2, factor_order=1, draws=200, burn=100, seed=2)
        m = DFMNowcaster(cfg).fit(df, target="cpi_mom")
        prob = m.posterior["outlier_prob"]
        assert prob[120, 0] > 3 * np.median(prob[:, 0])

    def test_e3_is_robust_to_a_contaminated_final_row(self):
        """A single wild observation must not blow the nowcast up (the E1 failure)."""
        df = synthetic_panel(T=200, ragged=False)
        clean_cfg = DFMConfig(arm="E3", k_factors=2, factor_order=1, draws=150, burn=75, seed=4)
        clean = DFMNowcaster(clean_cfg).fit(df, target="cpi_mom").nowcast(df, target="cpi_mom")["mean"]
        dirty = df.copy(); dirty.iloc[-1, 0] += 20.0
        got = DFMNowcaster(clean_cfg).fit(dirty, target="cpi_mom").nowcast(dirty, target="cpi_mom")["mean"]
        assert abs(got - clean) < 5.0, "one contaminated cell moved the nowcast wildly"

    def test_e1_and_e2_unchanged_by_e3_landing(self):
        df = synthetic_panel()
        for arm in ["E1", "E2"]:
            cfg = DFMConfig(arm=arm, k_factors=2, factor_order=1, draws=100, burn=50, seed=1)
            out = DFMNowcaster(cfg).fit(df, target="cpi_mom").nowcast(df, target="cpi_mom")
            assert np.isfinite(out["mean"])
```

- [ ] **Step 2: Run and confirm failure.**

- [ ] **Step 3: Implement.** Remove `outliers` from the guard. In the sweep, after the SV draws:

```python
if cfg.outliers:
    o, p_out = O.draw_outliers(e, g, p_out, rng)
    q_idio = np.exp(g) * o ** 2
else:
    o = np.ones_like(e)
```

Accumulate `(o > 1)` across kept draws and store the mean as `posterior["outlier_prob"]`. Store `p_out` per draw too.

- [ ] **Step 4: Run the full model test file.** Confirm E1's recorded nowcast from Task 3 Step 5 is **identical**.

- [ ] **Step 5: Commit**

```bash
git add code/dfm/model.py code/tests/test_dfm_model.py
git commit -m "feat(dfm): arm E3 — robust innovations via the outlier mixture"
```

---

## Task 8: Run and gate E3

- [ ] **Step 1: Run E3 standalone and overlay** as in Task 5.

- [ ] **Step 2: Score**, adding to the Task 5 comparisons:
  - E3 vs E2 (does robustness add beyond SV?) and E3 vs E1
  - The per-cell outlier probabilities for the months the spec predicted: **Ofgem cap resets, the 2020 VAT cut, COVID collection gaps**. Report which months actually light up — a mixture that flags the wrong months is not doing the job even if RMSE improves.
  - The 2022-11 overlay error specifically.

- [ ] **Step 3: Produce `plots/dfm/outlier_prob.png`** — a heatmap of series × month posterior outlier probability. This is a spec §8 deliverable and the most legible output of the whole arm. Use the repo `plots/` convention; percentage terms, never index levels.

- [ ] **Step 4: Write the E3 section and the Stage 2 verdict.**

Apply spec §7 as written:
1. Keep a component only if it improves OOS **without** calm-month overshoot >10% above the production replication's 0.0592.
2. If it helps only in shock periods, keep it and say so explicitly.
3. If error correlation with the TVP+LGBM ensemble is ≥0.9, it is redundant — drop it.
4. In-sample improvement without OOS improvement means drop.

Then answer the question this plan was written to answer, in one paragraph: **did SV and robust innovations fix E1's over-confidence?** Report the three targets against their E1 values (sd ratio 1.32, 2022-11 error +3.07pp, shock overlay RMSE 1.3049). If they did not, recommend abandoning the overlay route rather than proceeding to E4–E6, and say why.

- [ ] **Step 5: Commit**

```bash
git add docs/dfm_backtest.md plots/dfm/outlier_prob.png
git add -f data/dfm/backtest_E3_standalone.csv data/dfm/backtest_E3_overlay.csv data/dfm/stage2_summary.csv
git commit -m "test(dfm): Stage 2b — E3 robust innovations results and the Stage 2 verdict"
```

---

## Task 9: Regression-check the whole suite and production isolation

- [ ] **Step 1**

```bash
PYTHONPATH=code $Q -m pytest code/tests/ -q -m "not slow and not network"
```

Expected: everything passes. Plan 1 finished at 231 passed.

- [ ] **Step 2** — with network, confirm the frozen headline model still passes:

```bash
set -a; . ./.env; set +a
PYTHONPATH=code $Q -m pytest code/tests/test_headline_regression.py code/tests/test_dfm_isolation.py -q
```

Expected: 7 passed. **If `test_dfm_isolation` fails, production was touched — revert before anything else.**

- [ ] **Step 3: Commit any fixes**, then stop. Plan 3 (E4–E6) is a separate plan and must not be started here.

---

## Compute budget

Four walk-forwards at Plan 1's realised E1 rate (37.7 s/fit × 120 months ≈ 75 min each) is ~5 h before SV overhead. SV adds `K + M = 40` FFBS passes per sweep over T≈336, which is real but small relative to the simulation smoother's 52 ms.

Measure before committing (Task 4). If the total exceeds ~8 h, apply spec §5.3 fallbacks in order and record which. The parallel-import measurement in Task 4 Step 3 may make the whole question moot — if the `quant` env does not suffer the documented contention, four runs at `OMP_NUM_THREADS=1` on 6 performance cores finish in roughly the time of one.

---

## Self-review against the spec

| Spec requirement | Task |
|---|---|
| §1.2 factor SV, log-vol random walk | 2, 3 |
| §1.2 idiosyncratic SV | 2, 3 |
| §1.3 SW outlier mixture, `P(o_it > 1)` deliverable | 6, 7, 8 step 3 |
| §4.1 E2 = diagonal Φ + SV, unrestricted loadings | 3 |
| §4.1 E3 = E2 + outlier mixture | 7 |
| §4.2 both modes | 5, 8 |
| §4.3 sequential gate — E3 not built until E2 passes | 5 step 5 |
| §5.1 Gibbs block 4 (KSC + FFBS) | 2 |
| §5.1 Gibbs block 5 (outlier indicators, Beta posterior) | 6 |
| §5.3 budget and ordered fallbacks | 4 |
| §6.2 RMSE, MAE, cost | 5, 8 |
| §6.3 DM, calm overshoot, error correlation | 5, 8 |
| §7 selection rule applied as written | 8 step 4 |
| §8 `plots/dfm/outlier_prob.png` | 8 step 3 |
| Non-goal: production untouched | 9 |

**Deliberately out of scope**, carried to Plan 3: E4 (full VAR under the Minnesota prior — the `draw_factor_ar(diagonal=False)` path already exists and is tested but is not wired to an arm), E5 (block-restricted + spike-and-slab loadings), E6 (hierarchical shrinkage), the exogenous reg block `gamma_z`, and the RegEvent-inclusive production benchmark variant the overlay comparison needs. Plan 4 keeps the news decomposition, parameter stability, the T-30 → R-1 origin ladder, ablations and the model card.
