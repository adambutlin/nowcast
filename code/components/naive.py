"""Phase 1 — the component-naive benchmark, against aggregate AutoARIMA.

    PYTHONPATH=code:code/components python code/components/naive.py

The question this answers, and the ONLY question it answers:

    Does mechanically correct component aggregation, plus explicit motor-fuel and
    household-energy treatment, add value beyond aggregate AutoARIMA BEFORE any
    sophisticated component nowcasting is introduced?

Nothing here is fitted beyond two single-regressor pass-throughs estimated on prior
months only. If this beats AA, the win belongs to the ARCHITECTURE — correct
aggregation plus two observable mechanisms — not to any modelling. If it loses, the
architecture does not pay and no amount of component cleverness later should be
credited to it.

PRE-REGISTERED RULES (fixed before any result was seen)

  household_energy  DETERMINISTIC. Capped tariffs are fixed within an Ofgem quarter,
                    so component MoM is ~0 except in reset months. In a reset month
                    MoM = beta_hh * cap_pct_change, with beta_hh fitted through the
                    origin on PRIOR resets only. NO kappa: the basket weight now does
                    the job the 0.75 haircut was doing at aggregate level.
  motor_fuel        OBSERVED. The DESNZ monthly-mean pump price for month T is known
                    at month-end T. MoM = a + b * pump_MoM(T), (a,b) on prior months.
  food_alc_tob      NAIVE, two variants, both reported:
  core                MA12  recursive 12-month mean of own MoM (the Cleveland food rule)
                      SEAS  trailing mean of the same calendar month, prior 5 years

STANDPOINT is month-end T with information <= T, matching production. CPI(T-1)
publishes mid-month T, so component MoM history through T-1 and the index level
Index(T-1) are both legitimately available. Weights are set each January.

AGGREGATION  MoM(T) = sum_i (w_i/1000) * MoM_i(T)
             Index(T) = Index(T-1) * (1 + MoM(T)/100)      [Index(T-1) published]
             YoY(T) = 100 * (Index(T)/Index(T-12) - 1)     [Index(T-12) published]

Because both index levels are published, the ONLY quantity forecast is this month's
component movements. Base effects are arithmetic.

REPORTING follows the evaluation framework: raw RMSE, the identity-reconciliation
floor from basket.py, and performance net of that floor. Errors from reconstruction
and from forecasting are approximately independent, so the floor-adjusted model error
is sqrt(RMSE^2 - floor^2) — what the component forecasts would score against a
hypothetical perfectly-measured basket.

THE LADDER. A single benchmark number would not be interpretable, because two
different things could produce it: working in MoM with a seasonal rule and chaining
correctly, or genuinely decomposing the basket. The ladder separates them, and each
rung differs from the one above by exactly one change:

    A   aggregate AutoARIMA                          the incumbent anchor
    B0  headline MoM seasonal-naive, NO components   isolates MoM + seasonality + chaining
    B1  components, all seasonal-naive               adds decomposition, nothing else
    B3  B1 + deterministic Ofgem household energy    adds mechanism 1
    B2  B3 + observed pump prices for motor fuel     adds mechanism 2  [the Phase 1 system]

Reading B1 against B0 answers "does decomposition pay?" on its own. Reading B2 and B3
against B1 answers "do the observable mechanisms pay?".

Out: data/components/naive_results.csv
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402

import uk_model_zoo as Z                                  # noqa: E402
import factors as F                                       # noqa: E402
from components.basket import basket, PARTITION           # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "components")
os.makedirs(_OUT, exist_ok=True)

EVAL_START, EVAL_END = 2015, 2024
RESET_MONTHS = (1, 4, 7, 10)
MIN_PRIOR_RESETS = 4          # resets needed before beta_hh is trusted
FLOOR_4WAY = 0.05758          # pp, from basket.check() — pre-registered


# ── pre-registered naive rules ───────────────────────────────────────────────
def dm_stat(e_aa, e_fc, h=1):
    """Diebold-Mariano, squared-error loss. d = e_aa^2 - e_fc^2; >0 => fc better.

    Inlined rather than imported from new_factors.core_validate, which pulls in
    two_stage and with it the whole production factor stack — this package must not
    depend on the frozen production path.
    """
    from scipy import stats
    d = np.asarray(e_aa) ** 2 - np.asarray(e_fc) ** 2
    n = len(d)
    var = np.var(d, ddof=0) / n
    if var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var)
    return dm, 1 - stats.norm.cdf(dm)


def naive_ma12(mom, t):
    """Recursive 12-month mean of own MoM, using data strictly before t."""
    h = mom[mom.index < t].dropna()
    return float(h.iloc[-12:].mean()) if len(h) >= 12 else np.nan


def naive_seasonal(mom, t, years=5):
    """Mean of the same calendar month over the prior `years`, strictly before t."""
    h = mom[(mom.index < t) & (mom.index.month == t.month)].dropna()
    return float(h.iloc[-years:].mean()) if len(h) >= 3 else np.nan


def fit_passthrough(y, x, before, through_origin=False, min_n=12):
    """Least squares of y on x using only observations strictly before `before`."""
    d = pd.DataFrame({"y": y, "x": x}).dropna()
    d = d[d.index < before]
    if len(d) < min_n:
        return None
    if through_origin:
        b = float((d.x * d.y).sum() / (d.x ** 2).sum()) if (d.x ** 2).sum() > 0 else 0.0
        return 0.0, b
    b, a = np.polyfit(d.x.values, d.y.values, 1)
    return float(a), float(b)


def pump_monthly():
    """DESNZ monthly-mean pump price — the registry series, in levels."""
    s, status = F.load_factor("uk_pump_price")
    if status == "unavailable":
        raise RuntimeError("uk_pump_price unavailable")
    return s          # registry transform is logret; recovered as MoM% below


def _rules(mom, wts, comps, cap, pump, seasonal=True):
    """Return the per-rung component-MoM rule set. Each rung adds ONE change."""
    naive = naive_seasonal if seasonal else naive_ma12

    def all_naive(t):
        return {c: naive(mom[c], t) for c in comps}

    def ofgem(t, m):
        """Deterministic household energy: flat within an Ofgem quarter, step on reset.

        No kappa. The 0.75 haircut in production/reg_events.py exists because the step
        is applied at AGGREGATE level, where nothing tells the layer what share of the
        basket it moves. Here the basket weight carries that information explicitly, so
        the haircut has nothing left to do.
        """
        if t.month in RESET_MONTHS and t in cap.index:
            if len(cap[cap.index < t]) >= MIN_PRIOR_RESETS:
                f = fit_passthrough(mom["household_energy"], cap, t,
                                    through_origin=True, min_n=MIN_PRIOR_RESETS)
                if f:
                    m["household_energy"] = f[1] * float(cap.loc[t])
                    return m
            return m
        m["household_energy"] = 0.0        # capped tariffs do not move mid-quarter
        return m

    def fuel(t, m):
        """Motor fuel from the observed month-T pump price (DESNZ, known at month end)."""
        f = fit_passthrough(mom["motor_fuel"], pump, t)
        if f and t in pump.index and pd.notna(pump.loc[t]):
            m["motor_fuel"] = f[0] + f[1] * float(pump.loc[t])
        return m

    return dict(
        B1=lambda t: all_naive(t),
        B3=lambda t: ofgem(t, all_naive(t)),
        B2=lambda t: fuel(t, ofgem(t, all_naive(t))))


def main(seasonal=True):
    idx, wts, _ = basket(start_year=2000)
    comps = list(PARTITION)
    mom = idx.pct_change() * 100.0

    from production import reg_events as RE
    cap = RE.cap_pct_change()
    pump_lr, _ = F.load_factor("uk_pump_price")
    pump = (np.exp(pump_lr) - 1.0) * 100.0

    df_aa, _ = F.build_matrix(names=["cpi_yoy"])
    df_aa = df_aa[df_aa.index.year >= 1997].resample("ME").last()
    aa = Z.AutoARIMA().backtest(df_aa, [], "cpi_yoy", start_year=2001, end_year=None)
    months = aa.index[(aa.index.year >= EVAL_START)
                      & (aa.index.year <= EVAL_END)].intersection(idx.index)

    def chain(mom_hat, t):
        """Published Index(T-1) and Index(T-12) mean only this month's move is forecast."""
        prev, back = idx["headline"].shift(1).loc[t], idx["headline"].shift(12).loc[t]
        if pd.isna(mom_hat) or pd.isna(prev) or pd.isna(back):
            return np.nan
        return 100.0 * ((prev * (1 + mom_hat / 100.0)) / back - 1.0)

    def aggregate(m, t):
        if m is None or any(pd.isna(v) for v in m.values()):
            return np.nan
        return sum(float(wts.loc[t, c]) / 1000.0 * m[c] for c in comps)

    rules = _rules(mom, wts, comps, cap, pump, seasonal)
    naive = naive_seasonal if seasonal else naive_ma12
    preds = {"A": pd.Series({t: aa["pred"].loc[t] for t in months}),
             "B0": pd.Series({t: chain(naive(mom["headline"], t), t) for t in months})}
    for tag, rule in rules.items():
        preds[tag] = pd.Series({t: chain(aggregate(rule(t), t), t) for t in months})

    label = {"A": "aggregate AutoARIMA",
             "B0": "headline MoM naive, NO components",
             "B1": "components, all naive",
             "B3": "B1 + deterministic Ofgem energy",
             "B2": "B3 + observed pump prices  [Phase 1]"}
    print("=== PHASE 1: does correct aggregation + observable mechanisms beat AA? ===")
    print(f"  partition : {comps}")
    print(f"  naive rule: {'seasonal (same month, prior 5y)' if seasonal else 'MA12'}")
    print(f"  window    : {months.min().date()} … {months.max().date()}  (n={len(months)})")
    print(f"  floor     : {FLOOR_4WAY:.5f} pp, pre-registered\n")
    print(f"  {'':<4}{'benchmark':<38}{'n':>5}{'RMSE':>9}{'rel':>8}"
          f"{'floor-adj':>11}{'DM':>8}{'p':>7}")
    rows = []
    e_aa = None
    for tag in ("A", "B0", "B1", "B3", "B2"):
        P = preds[tag].dropna()
        c = P.index.intersection(aa.index)
        a = aa["actual"].reindex(c)
        e = a - P.reindex(c)
        if tag == "A":
            e_aa = e
        rmse = float(np.sqrt((e ** 2).mean()))
        rmse_a = float(np.sqrt((e_aa ** 2).mean()))
        adj = float(np.sqrt(max(rmse ** 2 - FLOOR_4WAY ** 2, 0.0)))
        dm, p = (np.nan, np.nan) if tag == "A" else dm_stat(
            e_aa.reindex(c).values, e.values)
        ds = f"{dm:>8.3f}" if np.isfinite(dm) else f"{'—':>8}"
        ps = f"{p:>7.3f}" if np.isfinite(p) else f"{'—':>7}"
        print(f"  {tag:<4}{label[tag]:<38}{len(c):>5}{rmse:>9.4f}"
              f"{rmse/rmse_a:>8.4f}{adj:>11.4f}{ds}{ps}")
        rows.append(dict(rung=tag, label=label[tag], n=len(c), rmse=rmse,
                         rel_aa=rmse / rmse_a, floor=FLOOR_4WAY, rmse_floor_adj=adj,
                         dm_vs_aa=dm, p_beats_aa=p))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(_OUT, "naive_results.csv"), index=False)

    g = {r["rung"]: r["rmse"] for r in rows}
    print(f"\n  ATTRIBUTION (change in RMSE, each rung vs the one above):")
    print(f"    MoM + seasonality + chaining   A  -> B0   {g['B0']-g['A']:+.4f}")
    print(f"    decomposition itself           B0 -> B1   {g['B1']-g['B0']:+.4f}")
    print(f"    deterministic Ofgem energy     B1 -> B3   {g['B3']-g['B1']:+.4f}")
    print(f"    observed pump prices           B3 -> B2   {g['B2']-g['B3']:+.4f}")
    print(f"\n  production reference: RMSE 0.4312, rel 0.9201 "
          f"(AA + 0.25*TVP + 0.25*LGBM + RegEvent)")
    print(f"  wrote data/components/naive_results.csv")
    return R


if __name__ == "__main__":
    main(seasonal=True)
