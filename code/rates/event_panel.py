"""
rates/event_panel.py — Deliverable 1.

build_event_panel() assembles one row per UK CPI release with predictors known
on the release eve (T-1) and outcomes measured release-day. Output schema:

  index: ref_month (month-end Timestamp)
  release_date, actual_cpi_mom, my_nowcast, ucl_nowcast, economist_consensus,
  market_implied_expectation, baseline_expectation,
  my_surprise, ucl_surprise, market_surprise, actual_surprise,
  boe_1y_ois_move, uk_2y_gilt_move, uk_5y_gilt_move, uk_10y_gilt_move,
  mpc_regime, days_to_mpc, budget_event, ldi_event

`actual_cpi_mom` carries whatever unit the nowcast model emits (the shipped CPI
model is YoY %); the column name follows the requested schema. Surprises are in
that same unit.
"""

import numpy as np
import pandas as pd

from . import config as C
from . import sources as S


# ─────────────────────────────────────────────────────────────────────────────
# reference-month -> CPI release date
# ─────────────────────────────────────────────────────────────────────────────

def _third_wednesday(year, month):
    d = pd.Timestamp(year, month, 1)
    # first Wednesday
    first_wed = d + pd.offsets.Week(weekday=2) if d.weekday() != 2 else d
    return first_wed + pd.Timedelta(weeks=2)


def cpi_release_date(ref_month):
    """ONS publishes CPI for reference month M ~3rd Wednesday of M+1.
    Override per-month with data/cpi_release_dates.csv [ref_month, release_date]."""
    nxt = pd.Timestamp(ref_month) + pd.offsets.MonthBegin(1)   # first day of M+1
    return _third_wednesday(nxt.year, nxt.month)


def _anchor(panel):
    """Choose the common baseline expectation against which surprises are
    measured. Precedence:
      consensus -> market_implied -> ucl_self (my edge vs UCL) -> naive_rw.
    naive_rw uses last published CPI (actual.shift(1), known at T-1) so the
    pipeline yields a runnable Gate 2 even with no public-forecast data; the
    verdict is labelled with the anchor mode so it is never mistaken for the
    true UCL-incremental test. Returns (baseline Series, mode str)."""
    if panel["economist_consensus"].notna().any():
        return panel["economist_consensus"], "consensus"
    if panel["market_implied_expectation"].notna().any():
        return panel["market_implied_expectation"], "market_implied"
    if panel["ucl_nowcast"].notna().any():
        return panel["ucl_nowcast"], "ucl_self"
    return panel["actual_cpi_mom"].shift(1), "naive_rw"


# ─────────────────────────────────────────────────────────────────────────────
# main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_event_panel(my_model=None, save=True):
    pred, actual = S.my_nowcast(model=my_model)
    ucl   = S.ucl_nowcast()
    cons  = S.economist_consensus()
    mkt   = S.market_implied()

    ref_months = pred.dropna().index
    if len(ref_months) == 0:
        raise RuntimeError("no my_nowcast available — data/nowcast_cpi_backtest.csv missing/empty")

    panel = pd.DataFrame(index=ref_months)
    panel.index.name = "ref_month"
    panel["release_date"] = [cpi_release_date(m) for m in ref_months]
    panel["actual_cpi_mom"]            = actual.reindex(ref_months)
    panel["my_nowcast"]                = pred.reindex(ref_months)
    panel["ucl_nowcast"]               = ucl.reindex(ref_months)
    panel["economist_consensus"]       = cons.reindex(ref_months)
    panel["market_implied_expectation"]= mkt.reindex(ref_months)

    baseline, mode = _anchor(panel)
    panel["baseline_expectation"] = baseline
    panel.attrs["anchor_mode"] = mode

    panel["my_surprise"]     = panel["my_nowcast"]  - panel["baseline_expectation"]
    panel["ucl_surprise"]    = panel["ucl_nowcast"] - panel["baseline_expectation"]
    panel["market_surprise"] = panel["market_implied_expectation"] - panel["baseline_expectation"]
    panel["actual_surprise"] = panel["actual_cpi_mom"] - panel["baseline_expectation"]  # diagnostic

    # ── outcomes: release-day signed rate moves ──────────────────────────────
    rates = S.daily_rates()
    moves = S.rate_moves(rates, panel["release_date"].tolist())
    moves.index.name = "release_date"
    # map release_date -> ref_month for the join
    rd2rm = {pd.Timestamp(rd): rm for rm, rd in panel["release_date"].items()}
    if len(moves):
        moves = moves.rename(index=rd2rm)
        for col in C.MOVE_COLS.values():
            panel[col] = moves[col].reindex(panel.index) if col in moves.columns else np.nan
    else:
        for col in C.MOVE_COLS.values():
            panel[col] = np.nan

    # ── context / event flags ────────────────────────────────────────────────
    rds = panel["release_date"].tolist()
    meetings = S.mpc_dates()
    panel["mpc_regime"]   = S.mpc_regime(rds).reindex(rds).values
    panel["days_to_mpc"]  = S.days_to_next_mpc(rds, meetings).reindex(rds).values
    panel["budget_event"] = S.budget_flag(rds).reindex(rds).values
    panel["ldi_event"]    = S.ldi_flag(rds).reindex(rds).values

    panel = panel.sort_values("release_date")
    if save:
        panel.to_csv(C.PANEL_CSV)
    return panel


SCHEMA = [
    "release_date", "actual_cpi_mom", "my_nowcast", "ucl_nowcast",
    "economist_consensus", "market_implied_expectation", "baseline_expectation",
    "my_surprise", "ucl_surprise", "market_surprise", "actual_surprise",
    "boe_1y_ois_move", "uk_2y_gilt_move", "uk_5y_gilt_move", "uk_10y_gilt_move",
    "mpc_regime", "days_to_mpc", "budget_event", "ldi_event",
]
