"""
dfm/config.py — single source of truth for paths, windows and model dimensions.

Everything downstream reads from here so the whole pipeline changes behaviour by
editing this file, never by hard-coding in a module.

Importing this module CREATES its output directories (data/dfm, plots/dfm and the
daily/checkpoint caches). That is deliberate — every entry point needs them — but
it means the import fails on a read-only filesystem rather than degrading.
"""
import os

_THIS = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(_THIS)
ROOT = os.path.dirname(CODE)

DATA = os.path.join(ROOT, "data", "dfm")
PLOTS = os.path.join(ROOT, "plots", "dfm")
DAILY_CACHE = os.path.join(DATA, "daily")
CHECKPOINTS = os.path.join(DATA, "checkpoints")
for _d in (DATA, PLOTS, DAILY_CACHE, CHECKPOINTS):
    os.makedirs(_d, exist_ok=True)

CALENDAR_CSV = os.path.join(DATA, "release_calendar.csv")

# ── model dimensions ─────────────────────────────────────────────────────────
K_FACTORS = 7          # 1 pervasive `global` + 6 block factors; fixed across all arms
FACTOR_ORDER = 2       # lags in the factor VAR
IDIO_ORDER = 1         # idiosyncratic AR(1), carried in the state vector

# ── walk-forward windows (match production exactly) ──────────────────────────
PANEL_START = 1997     # earliest panel month
EVAL_START = 2015      # first scored year
EVAL_END = 2024        # last scored year; 2025+ is blind
BLIND_FROM = 2025      # never used to fit or tune
HOLDOUT_MONTHS = ["2026-05", "2026-06", "2026-07"]   # live scored months, held out

# Purge + embargo. The overlay label (cpi_yoy residual) is a 12-month difference;
# the standalone label (CPI MoM) is a 1-month difference. See spec §6.4.
PURGE_OVERLAY, EMBARGO_OVERLAY = 12, 1
PURGE_STANDALONE, EMBARGO_STANDALONE = 1, 1

# ── MCMC ─────────────────────────────────────────────────────────────────────
DRAWS_FIRST, BURN_FIRST = 2000, 1000     # first walk-forward fit
DRAWS_WARM, BURN_WARM = 800, 200         # warm-started fits thereafter
SEED = 20260819

# ── forecast path origins ────────────────────────────────────────────────────
# (label, days relative to reference month-end). Negative = before month-end,
# positive = after, inside the release window. R-1's offset is None because it is
# resolved per month from the actual ONS release date (dfm.release_calendar), not
# from a fixed day count — the two cannot be a pair of parallel lists without
# inviting a zip() that silently drops it.
ORIGINS = [("T-30", -30), ("T-21", -21), ("T-14", -14), ("T-7", -7),
           ("T", 0), ("T+5", 5), ("T+10", 10), ("R-1", None)]
ORIGIN_LABELS = [label for label, _ in ORIGINS]
