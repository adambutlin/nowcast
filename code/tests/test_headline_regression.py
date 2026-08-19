"""Headline path invariants — the frozen SPEC, not a data-dependent number.

This file originally (2026-07-23) pinned the June-2026 nowcast byte-for-byte
(aa 2.726 / forecast 2.683) to prove that parameterizing the pipeline by Target
left the HEADLINE path untouched. Those goldens are not reproducible and were
never a stable lock: the model is a live nowcaster, so both the target month and
the factor matrix move with every ONS/market release. Re-running the *pre-fix*
code against a target series frozen at May-2026 now yields 2.711, not 2.683 —
purely from revised/extended factor data.

What is genuinely invariant is the frozen spec (λ=0.5, equal TVP/LGBM weights,
9 pinned factors) and the arithmetic that assembles the forecast from its legs.
That is what is locked here. Slow integration test — fetches live factors.
"""
import math

import pandas as pd
import pytest

import production.model as M
import targets as T


@pytest.fixture(scope="module")
def headline():
    return M.nowcast(target=T.HEADLINE)


@pytest.fixture(scope="module")
def core():
    return M.nowcast(target=T.CORE)


def _first_unreleased(target):
    """The month the nowcast must target: first month after the last actual."""
    import two_stage as TS
    df, _, _ = TS.load_matrix(target=target)
    known = df[target.name].dropna()
    return (known.index[-1] + pd.offsets.MonthEnd(1)).date()


def test_headline_spec_is_frozen(headline):
    assert headline["target"] == "cpi_yoy"
    assert headline["lam"] == 0.5
    assert T.HEADLINE.overlay_weights == {"tvp": 0.5, "lgbm": 0.5}
    assert headline["n_pinned"] == 9
    assert headline["n_live"] == 8                       # gscp is the expected drop
    assert set(headline["missing_factors"]) <= M.OPTIONAL_PINNED
    assert headline["degraded"] is False


def test_headline_forecast_is_the_shrunk_overlay(headline):
    """forecast = AA + λ·(0.5·TVP + 0.5·LGBM) + AdminStep, from the reported legs."""
    expected = (headline["aa"] + 0.5 * (0.5 * headline["tvp_resid"] +
                                        0.5 * headline["lgbm_resid"])
                + headline["admin_step"])
    assert headline["forecast"] == pytest.approx(expected, abs=1e-3)
    assert headline["tvp_contribution"] == pytest.approx(0.25 * headline["tvp_resid"], abs=1e-3)
    assert headline["lgbm_contribution"] == pytest.approx(0.25 * headline["lgbm_resid"], abs=1e-3)


def test_headline_targets_the_first_unreleased_month(headline):
    assert headline["nowcast_date"] == str(_first_unreleased(T.HEADLINE))


def test_admin_layer_is_headline_only(headline, core):
    """Core CPI excludes energy by construction, so a cap reset cannot touch it."""
    assert T.HEADLINE.admin_layer is True and T.CORE.admin_layer is False
    assert "AdminStep" in headline["spec"] and "AdminStep" not in core["spec"]
    assert core["admin_step"] == 0.0
    if headline["admin_step"]:                           # a reset month
        d = headline["admin_detail"]
        assert d["reason"] == "applied" and d["cap_pct"] != 0


def test_core_targets_the_same_month_and_is_finite(core, headline):
    assert core["target"] == "uk_core_cpi"
    assert core["nowcast_date"] == str(_first_unreleased(T.CORE))
    assert core["nowcast_date"] == headline["nowcast_date"]
    assert math.isfinite(core["forecast"])
    if core["n_pinned"] == 0:                            # AA-only fallback
        assert core["forecast"] == core["aa"]
