"""
rates/config.py — paths, constants, surprise/units conventions.

All rate MOVES are signed so that POSITIVE = hawkish (yields up).
All rate LEVELS are in percent; moves are reported in basis points (bp).
Surprises are in the unit of the nowcast (UK CPI is quoted YoY %), so a
my_surprise of +0.2 means "my nowcast is 0.2pp above the consensus anchor".
"""

import os

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(os.path.dirname(_THIS))          # repo root
DATA  = os.path.join(ROOT, "data")
PLOTS = os.path.join(ROOT, "plots")

# ── source artifacts (reused from the CPI pipeline) ──────────────────────────
BACKTEST_CSV = os.path.join(DATA, "nowcast_cpi_backtest.csv")  # my causal nowcast preds
MY_MODEL     = "Combined-Static"   # which backtest model is "my nowcast" (override in build)
MY_MODEL_FALLBACK = "AR(1)"        # if MY_MODEL absent for a month

# ── CSV drop-in adapters (real data; pipeline runs with NaN columns if absent)─
UCL_CSV       = os.path.join(DATA, "ucl_nowcast.csv")        # [date, value]  YoY %
CONSENSUS_CSV = os.path.join(DATA, "economist_consensus.csv")# [date, value]  YoY %
MKT_IMPLIED_CSV = os.path.join(DATA, "market_implied_cpi.csv")# [date, value] YoY %
RATES_DAILY_CSV = os.path.join(DATA, "uk_rates_daily.csv")   # [date, ois_1y, gilt_2y, gilt_5y, gilt_10y]  percent
MPC_DATES_CSV   = os.path.join(DATA, "mpc_dates.csv")        # [date]  MPC decision dates
BUDGET_CSV      = os.path.join(DATA, "budget_event.csv")     # [date, value]  existing
BANKRATE_CSV    = os.path.join(DATA, "mpc_rate_change.csv")  # [date, value]  monthly bp change, existing

# ── output ───────────────────────────────────────────────────────────────────
PANEL_CSV   = os.path.join(DATA, "rates_event_panel.csv")
GATE1_CSV   = os.path.join(DATA, "rates_gate1.csv")
GATE2_CSV   = os.path.join(DATA, "rates_gate2.csv")
MVP_CSV     = os.path.join(DATA, "rates_mvp_backtest.csv")
SIGNAL_CSV  = os.path.join(DATA, "rates_signal_backtest.csv")

# ── rate columns ─────────────────────────────────────────────────────────────
RATE_COLS = ["ois_1y", "gilt_2y", "gilt_5y", "gilt_10y"]
MOVE_COLS = {"ois_1y": "boe_1y_ois_move", "gilt_2y": "uk_2y_gilt_move",
             "gilt_5y": "uk_5y_gilt_move", "gilt_10y": "uk_10y_gilt_move"}
PRIMARY_MOVE = "uk_2y_gilt_move"   # primary front-end target (Part D)

# ── regimes / events ─────────────────────────────────────────────────────────
# LDI / Truss gilt crisis: non-CPI repricing — flagged for exclusion in Gate 2.
LDI_WINDOW = ("2022-09-19", "2022-10-31")

# Gate-2 pass thresholds (a skeptical-PM bar). In-sample HAC t alone
# false-positives on ~140 pts x several regressors, so the verdict ALSO
# requires out-of-sample evidence (walk-forward corr + non-negative OOS R^2).
GATE2_T_THRESHOLD   = 2.0     # HAC t-stat on my_surprise (in-sample, necessary not sufficient)
GATE2_INCR_R2_MIN   = 0.02    # incremental in-sample R^2 from adding my_surprise
GATE2_OOS_CORR_MIN  = 0.20    # walk-forward corr(pred, realized) — the real discriminator
GATE2_OOS_R2_MIN    = 0.0     # walk-forward OOS R^2 must beat the mean forecast
