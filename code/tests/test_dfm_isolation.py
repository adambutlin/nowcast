"""
Production-isolation gate for the DFM research branch.

The frozen production model must be byte-identical to `main` on this branch.
Spec: docs/superpowers/specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md (Non-goal)
"""
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FROZEN_PATHS = [
    "code/production",
    "code/factors.py",
    "code/targets.py",
    "code/new_factors/two_stage.py",
    "code/uk_model_zoo.py",
]


def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args],
                          capture_output=True, text=True, timeout=60)


def test_frozen_production_untouched():
    """No diff against main on any frozen production path."""
    if _git("rev-parse", "--verify", "main").returncode != 0:
        pytest.skip("no local main branch to diff against")
    out = _git("diff", "main", "--stat", "--", *FROZEN_PATHS)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        "frozen production paths differ from main:\n" + out.stdout)


def test_dfm_package_imports():
    """The dfm package imports without pulling in production side effects."""
    import dfm
    from dfm import config

    assert dfm.__version__
    assert config.K_FACTORS == 7
    assert config.EVAL_START == 2015 and config.EVAL_END == 2024
