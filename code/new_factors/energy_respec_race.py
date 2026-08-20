"""Does re-specifying the energy factors survive a walk-forward?

    PYTHONPATH=code:code/new_factors python code/new_factors/energy_respec_race.py

`energy_passthrough.py` shows the production energy spec is the worst available
construction: as a contemporaneous 1-month log return, `oil_brent` and `gas_eu` correlate
−0.08…0.00 with headline CPI YoY, while the SAME series as a 12-month log change lagged
7–13 months correlate 0.38–0.63 (HAC p 0.004–0.04, hump-shaped lag profiles). That is the
energy→cost→wage transmission arriving over quarters, which a lag-0 monthly return cannot
represent.

Two reasons that finding cannot be shipped as-is:

  1. SELECTION. Those lags are argmaxes over a 5×4×25×5 grid on the full sample. Re-testing
     the argmax on the sample that produced it is circular.
  2. ANCHOR OVERLAP. The overlay does not model CPI, it models the AutoARIMA RESIDUAL. CPI
     YoY is strongly autocorrelated, so a lagged energy change can correlate with CPI YoY
     while carrying nothing AA has not already extracted from CPI's own history.

This script addresses both: lags come from a PRE-SPECIFIED round grid (3/6/9/12 months) with
no argmax anywhere, every spec is scored on the AA residual by walk-forward, and the winners
are then run through the same purged+embargoed pinned-set race as `energy_fx_race.py` so the
numbers are comparable to the ship decision that rejected the contemporaneous specs.

Out: data/new_factors/energy_respec_{univ,arms}.csv
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
from sklearn.linear_model import Ridge                    # noqa: E402

import factors as F                                       # noqa: E402
import uk_model_zoo as Z                                  # noqa: E402
import two_stage as TS                                    # noqa: E402
from ablation import walk_forward, _score                 # noqa: E402
from core_validate import dm_stat                         # noqa: E402
from energy_passthrough import energy_panel               # noqa: E402
from targets import HEADLINE as H                         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "new_factors")
END = 2024

# Pre-specified, NOT fitted. Round quarters spanning the transmission window the
# pass-through study found (peaks 7-13m); no argmax is taken over this grid before
# the walk-forward, so the walk-forward result is not conditioned on the answer.
WINDOWS = [6, 12]
LAGS = [3, 6, 9, 12]
LEGS = ["brent_gbp", "ttf_gbp", "pump_gbp"]


def respec_panel():
    """Energy legs as w-month log changes at lag L, month-end.

    Lagging by L months uses information L months OLD, so unlike the lag-0 spec these
    columns cannot be optimistic about data availability — they are strictly more
    conservative than the pub_lag=0 the registry already claims for these series.
    """
    e = energy_panel()
    cols = {}
    for leg in LEGS:
        lvl = np.log(e[leg].replace(0, np.nan))
        for w in WINDOWS:
            chg = lvl - lvl.shift(w)
            for L in LAGS:
                cols[f"{leg}_{w}m_l{L}"] = chg.shift(L)
    return pd.DataFrame(cols).sort_index()


def univ_rel(df, aa, resid, months, col):
    """Walk-forward single-factor ridge on the AA residual — the factor_race metric."""
    d = pd.DataFrame({"resid": resid.reindex(months), "x": df[col].reindex(months),
                      "actual": aa["actual"].reindex(months),
                      "aa": aa["pred"].reindex(months)}).dropna()
    ef, ea = [], []
    for yr in range(H.start + 2, END + 1):
        tr, te = d[d.index.year < yr], d[d.index.year == yr]
        if len(tr) < 24 or not len(te):
            continue
        m = Ridge(alpha=1.0).fit(tr[["x"]], tr["resid"])
        p = te["aa"] + m.predict(te[["x"]])
        ef += list(te["actual"] - p)
        ea += list(te["actual"] - te["aa"])
    ef, ea = np.array(ef), np.array(ea)
    if not len(ef):
        return np.nan, np.nan, 0
    rel = float(np.sqrt((ef ** 2).mean()) / np.sqrt((ea ** 2).mean()))
    return rel, float(d["resid"].corr(d["x"])), len(ef)


def main():
    print("Building re-specified energy panel …")
    rs = respec_panel()
    df, live, status = TS.load_matrix(target=H)
    inc = [f for f in H.pinned if f in live]

    # Join the re-specified columns onto the production matrix. They are derived from
    # month-end market prices, so no publication lag applies beyond the explicit lag L.
    df = df.join(rs.reindex(df.index), how="left")

    aa = Z.AutoARIMA().backtest(df, [], H.name, start_year=H.aa_start, end_year=None)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    months = aa.index[(aa.index.year >= H.start) & (aa.index.year <= END)]
    actual, aap = aa["actual"].reindex(months), aa["pred"].reindex(months)
    rmse_aa = float(np.sqrt(np.mean((actual - aap).dropna() ** 2)))

    print(f"  incumbent  : {inc}")
    print(f"  window     : {months.min().date()} … {months.max().date()} (n={len(months)})")
    print(f"  AA-only    : RMSE {rmse_aa:.4f}\n")

    # ── A. univariate walk-forward on the AA residual, full pre-specified grid ──
    print("A. Univariate rel-RMSE on the AA residual (walk-forward; <1 = helps)")
    rows = []
    for col in ["oil_brent", "gas_eu"] + list(rs.columns):
        rel, c, n = univ_rel(df, aa, resid, months, col)
        rows.append(dict(spec=col, univ_rel_rmse=rel, corr_resid=c, n=n,
                         incumbent=col in ("oil_brent", "gas_eu")))
    U = pd.DataFrame(rows).sort_values("univ_rel_rmse")
    U.to_csv(os.path.join(_OUT, "energy_respec_univ.csv"), index=False)
    print(f"  {'spec':<22}{'rel-RMSE':>10}{'corr_resid':>12}{'n':>6}")
    for _, r in U.iterrows():
        tag = "  <- INCUMBENT SPEC" if r.incumbent else ""
        print(f"  {r.spec:<22}{r.univ_rel_rmse:>10.4f}{r.corr_resid:>12.3f}{int(r.n):>6}{tag}")

    # ── B. pinned-set arms, purged+embargoed, same gate as energy_fx_race ──────
    # Arms use FIXED lags chosen a priori (6 and 12 months = the two halves of the
    # transmission window), never the univariate winner, so B does not inherit A's
    # selection.
    no_energy = [f for f in inc if f not in ("oil_brent", "gas_eu")]
    arms = {
        "base": inc,
        "respec6":     no_energy + ["brent_gbp_12m_l6", "ttf_gbp_12m_l6"],
        "respec12":    no_energy + ["brent_gbp_12m_l12", "ttf_gbp_12m_l12"],
        "respec6+12":  no_energy + ["brent_gbp_12m_l6", "ttf_gbp_12m_l6",
                                    "brent_gbp_12m_l12", "ttf_gbp_12m_l12"],
        "base+respec": inc + ["brent_gbp_12m_l6", "ttf_gbp_12m_l6",
                              "brent_gbp_12m_l12", "ttf_gbp_12m_l12"],
        "respec+pump": no_energy + ["brent_gbp_12m_l6", "ttf_gbp_12m_l6",
                                    "brent_gbp_12m_l12", "ttf_gbp_12m_l12",
                                    "pump_gbp_12m_l6"],
    }
    print(f"\nB. Pinned-set arms (purged {TS.PURGE_HORIZON}+{TS.EMBARGO}, "
          f"production spec AA + 0.5·(0.5·TVP + 0.5·LGBM))")
    err, out = {}, []
    for tag, facs in arms.items():
        print(f"  walk-forward [{tag}] ({len(facs)} factors) …", flush=True)
        fc, tvp, lgbm = walk_forward(df, facs, H, aa, resid, months)
        r = _score(actual, fc, aap)
        err[tag] = pd.Series(r["err"], index=r["index"])
        out.append(dict(arm=tag, k=len(facs), n=r["n"], rmse=r["rmse"], mae=r["mae"],
                        rel_rmse=r["rel"], tvp_gaps=int(tvp.isna().sum()),
                        factors=" ".join(facs)))
    A = pd.DataFrame(out)
    eb = err["base"]
    dms, ps = [], []
    for tag in A.arm:
        if tag == "base":
            dms.append(np.nan); ps.append(np.nan); continue
        ix = eb.dropna().index.intersection(err[tag].dropna().index)
        dm, p = dm_stat(eb.reindex(ix).values, err[tag].reindex(ix).values)
        dms.append(dm); ps.append(p)
    A["dm_vs_base"], A["p_beats_base"] = dms, ps
    A.to_csv(os.path.join(_OUT, "energy_respec_arms.csv"), index=False)

    print(f"\n  {'arm':<13}{'k':>3}{'n':>5}{'RMSE':>9}{'rel-RMSE':>10}{'DM':>8}{'p':>7}")
    print(f"  {'AA-only':<13}{0:>3}{len(months):>5}{rmse_aa:>9.4f}{1.0:>10.4f}{'—':>8}{'—':>7}")
    for _, r in A.iterrows():
        dm = f"{r.dm_vs_base:>8.3f}" if np.isfinite(r.dm_vs_base) else f"{'—':>8}"
        p = f"{r.p_beats_base:>7.3f}" if np.isfinite(r.p_beats_base) else f"{'—':>7}"
        print(f"  {r.arm:<13}{int(r.k):>3}{int(r.n):>5}{r.rmse:>9.4f}{r.rel_rmse:>10.4f}{dm}{p}")

    print(f"\n  Shock window 2021-2023:")
    sh = eb.index[(eb.index.year >= 2021) & (eb.index.year <= 2023)]
    for tag in arms:
        s = err[tag].reindex(sh).dropna()
        print(f"    {tag:<13} RMSE {float(np.sqrt((s ** 2).mean())):.4f}")

    print(f"\n  wrote data/new_factors/energy_respec_{{univ,arms}}.csv")


if __name__ == "__main__":
    main()
