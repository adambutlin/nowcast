"""FX and inflation-expectations factors: `gbpusd` and `uk_be5`.

    PYTHONPATH=code:code/new_factors python code/new_factors/fx_expectations.py

Both series are registered `candidate=False` — documented in `factors.py` as "core factor,
always kept" — and neither is in `targets.HEADLINE.pinned`. `docs/energy_fx_race.md` showed
that flag is dead metadata read only by the legacy `main.py` path, and that the headline
pinned set has carried no FX and no expectations factor since the `d25f185` rewrite. Both
nonetheless out-rank most pinned factors in `data/new_factors/factor_race.csv`
(`uk_be5` 8/50 at rel 0.982, `gbpusd` 12/50 at 0.987).

This script asks whether either deserves pinning, and finds first that one of them does not
measure what its name says.

DEFECT IN `uk_be5`. `factors._uk_breakeven` computes

    be = _boe_spot_5y("glcinflationmonthedata.zip") - _boe_spot_5y("glcrealmonthedata.zip")

but `glcinflationmonthedata.zip` is the Bank of England's **implied inflation** curve, which
already *is* nominal minus real — verified here to 0.0 max abs difference against
`glcnominalmonthedata.zip` minus the real curve. The registry factor therefore evaluates to
`nominal - 2*real`: the real yield is subtracted twice. It correlates 0.208 with the true 5y
breakeven and 0.746 with the NEGATED real yield. `uk_be5` is a real-rate factor wearing an
inflation-expectations label, and whatever rank it earned in the factor race it earned as one.

HORIZON. Even fixed, a 5-year breakeven is the wrong tenor for a one-month nowcast — it is
the most anchored point on the curve. The BoE file carries the whole curve, so this pulls
1/2/3/5/10-year breakevens and lets the horizon be an empirical question rather than an
inherited default.

RPI CAVEAT. UK index-linked gilts are RPI-linked, so these breakevens price **RPI**, not CPI.
The RPI-CPI wedge (formula effect plus housing) averages roughly 0.8-1.0pp and is persistent,
so as a LEVEL regressor the wedge is absorbed by the intercept — but the series should never
be read as a direct market forecast of CPI.

FX SPECIFICATION. `gbpusd` is registered as a contemporaneous 1-month log return.
Exchange-rate pass-through to consumer prices is slow — import prices first, consumer prices
over roughly a year or two — so the same mis-specification `docs/energy_passthrough.md` found
in the energy factors is likely here. Bilateral USD is also the wrong weighting for UK import
costs; `gbp_eer` (broad effective rate) is tested alongside it.

Out: data/new_factors/fx_exp_{surface,univ,arms}.csv
"""
import io
import os
import sys
import warnings
import zipfile
import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import requests                                           # noqa: E402
from sklearn.linear_model import Ridge                    # noqa: E402

import factors as F                                       # noqa: E402
import uk_model_zoo as Z                                  # noqa: E402
import two_stage as TS                                    # noqa: E402
from ablation import walk_forward, _score                 # noqa: E402
from core_validate import dm_stat                         # noqa: E402
from energy_passthrough import hac_corr, profile_shape, raw_series   # noqa: E402
from targets import HEADLINE as H                         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "new_factors")
END = 2024
MAX_LAG = 24
# The BoE implied-inflation spot curve starts at 2.5 years — there is no 1y or 2y point,
# so the shortest market-implied UK breakeven available for free is 2.5y. (The registry's
# `uk_infl_swap_1y` covers the short end but needs an LSEG Workspace session.)
TENORS = [2.5, 3.0, 5.0, 10.0]
FX_WINDOWS = [1, 3, 6, 12]
FX_LAGS = [0, 3, 6, 9, 12, 18]


def _tname(t):
    """Column label for a maturity: y2_5 for 2.5 years, y5 for 5."""
    return "y" + (f"{t:g}".replace(".", "_"))


def boe_curve(zip_url, tenors):
    """BoE spot curve at several maturities. Generalises `factors._boe_spot_5y`,
    which hardcodes the 5y column."""
    r = requests.get(zip_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    frames = []
    for fname in sorted(z.namelist()):
        xl = pd.ExcelFile(z.open(fname))
        if "4. spot curve" not in xl.sheet_names:
            continue
        df = xl.parse("4. spot curve", header=None)
        mats = df.iloc[3, 1:].values.astype(float)
        data = df.iloc[5:].copy()
        data.columns = ["date"] + list(mats)
        data = data[data["date"].apply(
            lambda x: isinstance(x, (pd.Timestamp, datetime.datetime)))]
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date").apply(pd.to_numeric, errors="coerce")
        cols = {_tname(t): min(mats, key=lambda x: abs(x - t)) for t in tenors}
        picked = data[list(cols.values())].copy()
        picked.columns = list(cols.keys())
        frames.append(picked)
    if not frames:
        raise RuntimeError(f"no spot curve sheets in {zip_url}")
    return pd.concat(frames).sort_index().resample("ME").last()


def factor_panel():
    """Fixed breakevens by tenor, the buggy registry construction, and FX legs."""
    B = F.BOE_BASE
    infl = boe_curve(B + "glcinflationmonthedata.zip", TENORS)
    real = boe_curve(B + "glcrealmonthedata.zip", TENORS)
    nom = boe_curve(B + "glcnominalmonthedata.zip", TENORS)

    # Sanity: the inflation curve must equal nominal - real, or the premise is wrong.
    chk = float((infl["y5"] - (nom["y5"] - real["y5"])).abs().max())
    print(f"  check  BoE inflation curve == nominal - real : max abs diff {chk:.6f}")

    cols = {}
    for t in TENORS:
        cols["be" + _tname(t)[1:]] = infl[_tname(t)]        # TRUE breakeven, RPI-linked
    cols["be5_registry_bug"] = infl["y5"] - real["y5"]      # what uk_be5 actually computes
    cols["real5"] = real["y5"]
    # Anchoring: a wide long-minus-short spread means expectations are drifting at the
    # far end, which is the classic de-anchoring signature.
    cols["be_slope_10y_2_5y"] = infl["y10"] - infl["y2_5"]

    fx = {}
    fx["gbpusd"] = raw_series("gbpusd")                     # USD per GBP
    try:
        fx["gbp_eer"] = raw_series("gbp_eer")               # BIS real broad effective rate
    except Exception as e:
        print(f"  [warn] gbp_eer unavailable: {e}")
    for name, s in fx.items():
        lvl = np.log(s.replace(0, np.nan))
        for w in FX_WINDOWS:
            chg = lvl - lvl.shift(w)
            for L in FX_LAGS:
                cols[f"{name}_{w}m_l{L}"] = chg.shift(L)
    return pd.DataFrame(cols).sort_index()


def univ_rel(df, aa, resid, months, col):
    """Walk-forward single-factor ridge on the AA residual — the factor_race metric."""
    d = pd.DataFrame({"resid": resid.reindex(months), "x": df[col].reindex(months),
                      "actual": aa["actual"].reindex(months),
                      "aa": aa["pred"].reindex(months)}).dropna()
    if len(d) < 48:
        return np.nan, np.nan, 0
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
    return (float(np.sqrt((ef ** 2).mean()) / np.sqrt((ea ** 2).mean())),
            float(d["resid"].corr(d["x"])), len(ef))


def main():
    print("Building BoE curve and FX panel …")
    P = factor_panel()

    df, live, status = TS.load_matrix(target=H)
    inc = [f for f in H.pinned if f in live]
    df = df.join(P.reindex(df.index), how="left")

    # The registry's own uk_be5 column, for a like-for-like comparison.
    be_reg, be_status = F.load_factor("uk_be5")
    df["uk_be5_registry"] = be_reg.reindex(df.index)

    aa = Z.AutoARIMA().backtest(df, [], H.name, start_year=H.aa_start, end_year=None)
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    months = aa.index[(aa.index.year >= H.start) & (aa.index.year <= END)]
    actual, aap = aa["actual"].reindex(months), aa["pred"].reindex(months)
    rmse_aa = float(np.sqrt(np.mean((actual - aap).dropna() ** 2)))
    print(f"  incumbent : {inc}")
    print(f"  window    : {months.min().date()} … {months.max().date()} (n={len(months)})")
    print(f"  AA-only   : RMSE {rmse_aa:.4f}   uk_be5 status={be_status}\n")

    # ── A. the defect, quantified ────────────────────────────────────────────
    d = P[["be5", "be5_registry_bug", "real5"]].dropna()
    print("A. uk_be5 construction")
    print(f"  corr(registry uk_be5, TRUE 5y breakeven) = {d.be5_registry_bug.corr(d.be5):+.3f}")
    print(f"  corr(registry uk_be5, NEGATED real yield) = {d.be5_registry_bug.corr(-d.real5):+.3f}")
    print(f"  mean level: registry {d.be5_registry_bug.mean():.2f}  true {d.be5.mean():.2f}")

    # ── B. lead-lag vs CPI, fixed breakevens by tenor ────────────────────────
    print(f"\nB. Lead-lag vs headline CPI YoY (HAC 12+k, lags 0…{MAX_LAG})")
    tgt = df["cpi_yoy"]
    rows = []
    be_cols = ["be" + _tname(t)[1:] for t in TENORS] + ["be5_registry_bug", "real5",
                                                        "be_slope_10y_2_5y"]
    print(f"  {'series':<20}{'peak lag':>9}{'corr':>8}{'HAC t':>8}{'HAC p':>8}{'n':>6}  shape")
    for c in be_cols:
        rs = []
        for k in range(MAX_LAG + 1):
            r, t, p, n = hac_corr(P[c].shift(k), tgt, nw_lag=12 + k)
            rs.append((r, t, p, n))
            rows.append(dict(series=c, lag=k, corr=r, hac_t=t, hac_p=p, n=n))
        sh = profile_shape([x[0] for x in rs])
        k = int(np.nanargmax([abs(x[0]) if x[0] == x[0] else -1 for x in rs]))
        r, t, p, n = rs[k]
        print(f"  {c:<20}{k:>9}{r:>8.3f}{t:>8.2f}{p:>8.3f}{n:>6}  {sh}")
    pd.DataFrame(rows).to_csv(os.path.join(_OUT, "fx_exp_surface.csv"), index=False)

    # ── C. walk-forward univariate on the AA residual ────────────────────────
    print(f"\nC. Univariate rel-RMSE on the AA residual (walk-forward; <1 = helps)")
    cand = be_cols + [c for c in P.columns if c.startswith(("gbpusd_", "gbp_eer_"))]
    urows = []
    for c in cand + ["uk_be5_registry", "gbpusd"]:
        if c not in df.columns:
            continue
        rel, cc, n = univ_rel(df, aa, resid, months, c)
        urows.append(dict(spec=c, univ_rel_rmse=rel, corr_resid=cc, n=n))
    U = pd.DataFrame(urows).dropna(subset=["univ_rel_rmse"]).sort_values("univ_rel_rmse")
    U.to_csv(os.path.join(_OUT, "fx_exp_univ.csv"), index=False)
    print(f"  {'spec':<22}{'rel-RMSE':>10}{'corr_resid':>12}{'n':>6}")
    for _, r in U.head(18).iterrows():
        print(f"  {r.spec:<22}{r.univ_rel_rmse:>10.4f}{r.corr_resid:>12.3f}{int(r.n):>6}")
    print("  …")
    for _, r in U[U.spec.isin(["uk_be5_registry", "gbpusd", "be5_registry_bug"])].iterrows():
        print(f"  {r.spec:<22}{r.univ_rel_rmse:>10.4f}{r.corr_resid:>12.3f}{int(r.n):>6}  <- as registered")

    # ── D. pinned-set arms ───────────────────────────────────────────────────
    # Specs are named a priori: the SHORTEST breakeven available (be2_5) and
    # the FX construction the pass-through literature implies (12m change, 12m lag), not the
    # univariate argmax — so D does not inherit C's selection.
    arms = {
        "base": inc,
        "+be_short": inc + ["be2_5"],
        "+be5_true": inc + ["be5"],
        "+be5_registry": inc + ["uk_be5_registry"],
        "+fx_slow": inc + ["gbpusd_12m_l12"],
        "+be_short+fx_slow": inc + ["be2_5", "gbpusd_12m_l12"],
    }
    print(f"\nD. Pinned-set arms (purged {TS.PURGE_HORIZON}+{TS.EMBARGO})")
    err, out = {}, []
    for tag, facs in arms.items():
        print(f"  walk-forward [{tag}] ({len(facs)}) …", flush=True)
        fc, tvp, _ = walk_forward(df, facs, H, aa, resid, months)
        r = _score(actual, fc, aap)
        err[tag] = pd.Series(r["err"], index=r["index"])
        out.append(dict(arm=tag, k=len(facs), n=r["n"], rmse=r["rmse"],
                        rel_rmse=r["rel"], factors=" ".join(facs)))
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
    A.to_csv(os.path.join(_OUT, "fx_exp_arms.csv"), index=False)
    print(f"\n  {'arm':<15}{'k':>3}{'n':>5}{'RMSE':>9}{'rel':>9}{'DM':>8}{'p':>7}")
    print(f"  {'AA-only':<15}{0:>3}{len(months):>5}{rmse_aa:>9.4f}{1.0:>9.4f}{'—':>8}{'—':>7}")
    for _, r in A.iterrows():
        ds = f"{r.dm_vs_base:>8.3f}" if np.isfinite(r.dm_vs_base) else f"{'—':>8}"
        ps_ = f"{r.p_beats_base:>7.3f}" if np.isfinite(r.p_beats_base) else f"{'—':>7}"
        print(f"  {r.arm:<15}{int(r.k):>3}{int(r.n):>5}{r.rmse:>9.4f}{r.rel_rmse:>9.4f}{ds}{ps_}")

    sh = eb.index[(eb.index.year >= 2021) & (eb.index.year <= 2023)]
    print(f"\n  Shock window 2021-2023:")
    for tag in arms:
        s = err[tag].reindex(sh).dropna()
        print(f"    {tag:<15} RMSE {float(np.sqrt((s ** 2).mean())):.4f}")

    print(f"\n  wrote data/new_factors/fx_exp_{{surface,univ,arms}}.csv")


if __name__ == "__main__":
    main()
