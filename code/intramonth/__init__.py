"""Intramonth regime-dependent CPI nowcasting system.

Layers (Part C):
  1 baseline persistence (AutoARIMA)
  2 factor residual    (BVAR)
  3 regime-aware TVP
  4 intramonth MIDAS (high-frequency)
  5 latent regime detector (HMM) — gate + scenario posteriors

See config.py for the switchable configuration.
"""
