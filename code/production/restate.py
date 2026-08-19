"""Re-run past reference months through the CURRENT pipeline.

    PYTHONPATH=code python code/production/restate.py 2026-04 2026-05 2026-06

For each month it truncates the target series to the previous month — which is the
only thing that pins the nowcast month — and runs `nowcast()` exactly as production
would, then prints the result beside the realised print and the as-run scorecard
value.

WHAT THIS IS NOT: a vintage replay. Factor values are today's, so any ONS/market
revision since the original run is baked in, and CSV drop-ins carry whatever is in
them now. The regulatory-event layer is honest by construction (its beta is fitted
on resets strictly before the target month), and the overlays only ever train on
residuals up to the truncation, but a restated number is still "what the current
code makes of today's data", not "what we would have printed at the time". The
live scorecard therefore keeps its as-run values; this is for judging changes.
"""
import os
import sys

import pandas as pd

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE)
sys.path.insert(0, os.path.join(_CODE, "new_factors"))

import two_stage as TS                                    # noqa: E402
from production import model as M                         # noqa: E402
from targets import HEADLINE, CORE                        # noqa: E402

_ROOT = os.path.dirname(_CODE)
SCORECARDS = {"cpi_yoy": ("live_scorecard.csv", "final_production"),
              "uk_core_cpi": ("live_scorecard_core.csv", "core_final")}


def _as_run(target_name, month):
    """The value production actually printed for `month`, if it was ever run."""
    fname, col = SCORECARDS[target_name]
    path = os.path.join(_ROOT, "data", fname)
    if not os.path.exists(path):
        return None
    sc = pd.read_csv(path)
    row = sc[sc["release_month"] == month]
    if row.empty or pd.isna(row[col].iloc[0]):
        return None
    return float(row[col].iloc[0])


def _actual(df, target, month):
    """The realised print, taken from the target series itself (not the scorecard,
    which only covers months production was actually run for)."""
    nd = pd.Timestamp(month) + pd.offsets.MonthEnd(0)
    v = df[target.name].get(nd)
    return None if v is None or pd.isna(v) else float(v)


def restate(month, target, matrix=None):
    """Nowcast `month` (YYYY-MM) with the current pipeline. Returns the nowcast dict."""
    df, live, _ = matrix if matrix is not None else TS.load_matrix(target=target)
    nd = pd.Timestamp(month) + pd.offsets.MonthEnd(0)
    df = df.copy()
    df.loc[df.index >= nd, target.name] = float("nan")     # target month unreleased
    nc = M.nowcast(df, live, target)
    if nc["nowcast_date"] != str(nd.date()):
        raise RuntimeError(f"expected to target {nd.date()}, got {nc['nowcast_date']}")
    return nc


def main(months):
    for target in (HEADLINE, CORE):
        matrix = TS.load_matrix(target=target)
        print(f"\n=== {target.label} — restated with the current pipeline ===")
        print(f"{'month':<9} {'AA':>7} {'TVP':>7} {'LGBM':>7} {'RegEvent':>9} "
              f"{'restated':>9} {'as-run':>8} {'actual':>7} {'err(new)':>9} {'err(old)':>9}")
        rows = []
        for m in months:
            nc = restate(m, target, matrix)
            old, actual = _as_run(target.name, m), _actual(matrix[0], target, m)
            err_new = None if actual is None else nc["forecast"] - actual
            err_old = None if (actual is None or old is None) else old - actual
            fmt = lambda v, w=8: (f"{v:>{w}.3f}" if isinstance(v, float) else f"{'—':>{w}}")
            print(f"{m:<9} {nc['aa']:>7.3f} {nc['tvp_contribution']:>+7.3f} "
                  f"{nc['lgbm_contribution']:>+7.3f} {nc['reg_event']:>+9.3f} "
                  f"{nc['forecast']:>9.3f} {fmt(old)} "
                  f"{fmt(actual, 7)} {fmt(err_new, 9)} {fmt(err_old, 9)}")
            rows.append((err_new, err_old))
        scored = [(n, o) for n, o in rows if n is not None]
        if scored:
            rmse = lambda v: (sum(e * e for e in v) / len(v)) ** 0.5
            new = rmse([n for n, _ in scored])
            both = [(n, o) for n, o in scored if o is not None]
            line = f"  RMSE restated {new:.3f} over {len(scored)} month(s)"
            if both:
                line += (f"; on the {len(both)} also run live: restated "
                         f"{rmse([n for n, _ in both]):.3f} vs as-run {rmse([o for _, o in both]):.3f}")
            print(line)


if __name__ == "__main__":
    args = sys.argv[1:] or ["2026-04", "2026-05", "2026-06"]
    main(args)
