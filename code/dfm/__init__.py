"""
dfm — mixed-frequency Bayesian dynamic factor model for UK CPI (research branch).

This package is a RESEARCH ALTERNATIVE to the frozen production model. It imports
code/production, code/factors.py, code/targets.py and code/uk_model_zoo.py
READ-ONLY, for benchmarking, and must never modify them
(enforced by code/tests/test_dfm_isolation.py).

Spec: docs/superpowers/specs/2026-08-19-mixed-frequency-bayesian-dfm-design.md
"""

__version__ = "0.1.0"
