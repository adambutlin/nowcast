"""Can a Knotek-Zaman DMS structure forecast UK petrol pump prices?

    PYTHONPATH=code:code/new_factors python code/new_factors/pump_dms.py

The Cleveland Fed's inflation nowcast (Knotek & Zaman 2017; FRB Cleveland WP 24-06)
nowcasts headline CPI from disaggregates, and the disaggregate that does the work is
GASOLINE: they nowcast monthly retail gasoline inflation from weekly retail gasoline
prices plus daily oil, then feed it into the headline aggregate. Their words: the
high-frequency gasoline leg is "crucial to obtaining accurate nowcasts of headline
inflation."

The UK analogue is DESNZ weekly road-fuel prices, already in the registry as
`uk_pump_price` (monthly mean; corr 0.993 with CPI 07.2.2 MoM). This script asks
whether the DMS *structure* — deterministic switching on the information set — buys
anything for the UK pump series, and whether the UK tax wedge changes the answer.

DMS in one line: one model per information state, switched on deterministically by
what has actually been published. Here the state is simply how many of the month's
weekly DESNZ readings exist yet (s = 1,2,3,4), so the model that runs in week 1 is
not the model that runs in week 4.

THE UK TWIST — the tax wedge. A US gasoline price is mostly product cost. A UK pump
price is
        pump = (wholesale + duty) x (1 + VAT)
with duty a FIXED pence/litre (57.95p 2011-2022, 52.95p since, rising 1p on
2026-09-01) and VAT ad valorem. Duty is roughly 40% of the pump price, so it damps
proportional pass-through: a 10% crude move moves the pump price ~3%, and the damping
factor itself drifts as the wholesale share moves. A log-log regression of pump on
crude therefore has a structurally time-varying slope. Modelling the WHOLESALE
component instead removes the wedge analytically and should restore a stable
elasticity. DESNZ publishes duty and VAT in the same CSV, so the wedge is observed,
not assumed.

Out: data/new_factors/pump_dms.csv
"""
import io
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
import requests                                           # noqa: E402
from sklearn.linear_model import Ridge                    # noqa: E402

import factors as F                                       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "new_factors")
os.makedirs(_OUT, exist_ok=True)

LITRES_PER_BBL = 158.987291
EVAL_START, EVAL_END = 2011, 2026     # 2011+: VAT settled at 20%, duty schedule clean


def desnz_weekly():
    """Weekly ULSP pump price, duty and VAT from the DESNZ road-fuel CSVs.

    factors._desnz_pump_price collapses to a monthly mean and averages petrol with
    diesel; the DMS state here is defined on the WEEKLY grid and the question is about
    petrol, so this reads the same source at native frequency and keeps ULSP alone.
    """
    html = requests.get(F.DESNZ_FUEL_PAGE, headers={"User-Agent": "Mozilla/5.0"},
                        timeout=30).text
    urls = list(dict.fromkeys(re.findall(r'https://[^"\']+\.csv', html)))
    frames = []
    for u in urls:
        r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = [c.strip().lstrip("﻿") for c in d.columns]
        date_col = d.columns[0]
        def pick(kind):
            m = [c for c in d.columns if "ULSP" in c and kind in c]
            return m[0] if m else None
        p, du, va = pick("Pump price"), pick("Duty rate"), pick("VAT")
        if not all([p, du, va]):
            continue
        d[date_col] = pd.to_datetime(d[date_col], dayfirst=True, errors="coerce")
        d = d.dropna(subset=[date_col]).set_index(date_col)
        frames.append(pd.DataFrame({
            "pump": pd.to_numeric(d[p], errors="coerce"),
            "duty": pd.to_numeric(d[du], errors="coerce"),
            "vat": pd.to_numeric(d[va], errors="coerce"),
        }))
    w = pd.concat(frames).sort_index()
    w = w[~w.index.duplicated(keep="last")].dropna()
    # VAT is published as a percentage (20.0) in some vintages and a rate (0.2) in
    # others; normalise to a rate so the wedge algebra is vintage-independent.
    w["vat"] = np.where(w["vat"] > 1.0, w["vat"] / 100.0, w["vat"])
    w["wholesale"] = w["pump"] / (1.0 + w["vat"]) - w["duty"]
    return w


def brent_pence_litre():
    """Brent in pence per litre — the sterling cost of the raw input, daily."""
    usd_bbl = F._fred("DCOILBRENTEU").rename("brent")       # month-end in factors; refetch daily
    from fredapi import Fred
    fred = Fred(api_key=F.FRED_KEY)
    b = fred.get_series("DCOILBRENTEU", observation_start="2000-01-01")
    fx = fred.get_series("DEXUSUK", observation_start="2000-01-01")   # USD per GBP
    b.index, fx.index = pd.to_datetime(b.index), pd.to_datetime(fx.index)
    d = pd.DataFrame({"usd_bbl": b, "usd_gbp": fx}).ffill().dropna()
    return (d["usd_bbl"] / LITRES_PER_BBL / d["usd_gbp"] * 100.0).rename("brent_ppl")


def build_states(w, brent):
    """One row per (month, information state s = weeks observed so far).

    Each row carries what is KNOWN at that point — the partial mean of the month's
    published weeks, the last published week, and Brent through that week — and the
    target, the month's eventual full mean. That is the DMS information set s(t)
    made explicit.
    """
    rows = []
    w = w.copy()
    w["ym"] = w.index.to_period("M")
    for ym, g in w.groupby("ym"):
        g = g.sort_index()
        full_mean = g["pump"].mean()
        full_ws = g["wholesale"].mean()
        prev = w[w["ym"] == ym - 1]
        if not len(prev):
            continue
        for s in range(1, len(g) + 1):
            obs = g.iloc[:s]
            asof = obs.index[-1]
            bl = brent.loc[:asof]
            if not len(bl):
                continue
            # Brent momentum between the last observed pump week and the 4 weeks before
            b_now = float(bl.iloc[-1])
            b_lag = float(bl.loc[:asof - pd.Timedelta(days=28)].iloc[-1]) if len(
                bl.loc[:asof - pd.Timedelta(days=28)]) else b_now
            rows.append(dict(
                ym=ym.to_timestamp("M"), s=s, n_weeks=len(g),
                partial_mean=obs["pump"].mean(), last_week=obs["pump"].iloc[-1],
                partial_ws=obs["wholesale"].mean(), last_ws=obs["wholesale"].iloc[-1],
                prev_mean=prev["pump"].mean(), prev_ws=prev["wholesale"].mean(),
                duty=obs["duty"].iloc[-1], vat=obs["vat"].iloc[-1],
                brent_ppl=b_now, brent_chg=np.log(b_now / b_lag) if b_lag > 0 else 0.0,
                target=full_mean, target_ws=full_ws))
    return pd.DataFrame(rows)


def walk_forward(df, feats, target_col, reconstruct=None):
    """Expanding-window walk-forward by year, fitting one model PER STATE s.

    Fitting per state is the whole point of DMS: in week 1 the partial mean carries
    little of the month and oil should get weight; by week 4 the month is nearly
    determined and the partial mean should dominate. A single pooled model cannot
    express both.
    """
    out = []
    for yr in range(EVAL_START + 3, EVAL_END + 1):
        tr = df[df.ym.dt.year < yr]
        te = df[df.ym.dt.year == yr]
        if len(tr) < 60 or not len(te):
            continue
        for s in sorted(te.s.unique()):
            trs, tes = tr[tr.s == s], te[te.s == s]
            if len(trs) < 24 or not len(tes):
                continue
            m = Ridge(alpha=1.0).fit(trs[feats], trs[target_col])
            pred = m.predict(tes[feats])
            if reconstruct is not None:
                pred = reconstruct(pred, tes)
            out.append(pd.DataFrame({"ym": tes.ym.values, "s": s,
                                     "pred": pred, "actual": tes.target.values}))
    return pd.concat(out) if out else pd.DataFrame()


def score(res, label):
    d = res.dropna()
    by = []
    for s in sorted(d.s.unique()):
        x = d[d.s == s]
        by.append((s, len(x), float(np.sqrt(((x.actual - x.pred) ** 2).mean()))))
    tot = float(np.sqrt(((d.actual - d.pred) ** 2).mean()))
    return dict(model=label, n=len(d), rmse_all=tot,
                **{f"rmse_s{s}": r for s, n, r in by})


def main():
    print("Fetching DESNZ weekly road-fuel prices and daily Brent/FX …")
    w = desnz_weekly()
    brent = brent_pence_litre()
    print(f"  weekly ULSP : n={len(w)}  {w.index[0].date()} … {w.index[-1].date()}")
    print(f"  brent p/l   : n={len(brent)}  {brent.index[0].date()} … {brent.index[-1].date()}")
    print(f"  duty now    : {w['duty'].iloc[-1]:.2f} p/l   VAT {w['vat'].iloc[-1]:.0%}")
    ws_share = (w["wholesale"] / w["pump"]).iloc[-1]
    print(f"  wholesale share of pump: {ws_share:.1%}  "
          f"(tax wedge {1 - ws_share:.1%}) — this is the pass-through damper\n")

    df = build_states(w, brent)
    df = df[(df.ym.dt.year >= EVAL_START) & (df.ym.dt.year <= EVAL_END)]
    print(f"  state rows: {len(df)}  months {df.ym.min().date()} … {df.ym.max().date()}")
    print(f"  weeks per month: {df.groupby('ym').n_weeks.first().value_counts().to_dict()}\n")

    results = []

    # (a) RW — the month's mean equals the last published week. The benchmark to beat.
    rw = df.assign(pred=df.last_week)[["ym", "s", "pred", "target"]].rename(
        columns={"target": "actual"})
    rw = rw[rw.ym.dt.year >= EVAL_START + 3]
    results.append(score(rw, "RW (last published week)"))

    # (b) partial mean alone — no model, no oil
    pm = df.assign(pred=df.partial_mean)[["ym", "s", "pred", "target"]].rename(
        columns={"target": "actual"})
    pm = pm[pm.ym.dt.year >= EVAL_START + 3]
    results.append(score(pm, "partial mean (no oil)"))

    # (c) DMS in PUMP space — partial info + oil, one model per state
    results.append(score(walk_forward(
        df, ["partial_mean", "last_week", "prev_mean", "brent_chg"], "target"),
        "DMS, pump space"))

    # (d) DMS in WHOLESALE space — same, but the tax wedge is removed before fitting
    #     and reinstated afterwards, so the fitted elasticity is not fighting duty.
    def back_to_pump(pred_ws, te):
        return (pred_ws + te["duty"].values) * (1.0 + te["vat"].values)
    results.append(score(walk_forward(
        df, ["partial_ws", "last_ws", "prev_ws", "brent_chg"], "target_ws",
        reconstruct=back_to_pump), "DMS, wholesale space"))

    R = pd.DataFrame(results)
    R.to_csv(os.path.join(_OUT, "pump_dms.csv"), index=False)
    cols = ["model", "n", "rmse_all"] + [c for c in R.columns if c.startswith("rmse_s")]
    print("  Nowcast of the month's MEAN pump price, RMSE in pence/litre")
    print(f"  (s = weeks of the month already published; walk-forward, "
          f"{EVAL_START + 3}-{EVAL_END})\n")
    hdr = f"  {'model':<26}{'n':>5}{'all':>8}" + "".join(
        f"{c.replace('rmse_', ''):>8}" for c in cols[3:])
    print(hdr)
    for _, r in R.iterrows():
        line = f"  {r.model:<26}{int(r.n):>5}{r.rmse_all:>8.3f}"
        for c in cols[3:]:
            v = r.get(c)
            line += f"{v:>8.3f}" if pd.notna(v) else f"{'—':>8}"
        print(line)

    # Elasticity stability: the reason wholesale space should win.
    print("\n  Pass-through elasticity to Brent, by era (log-log, weekly MoM):")
    m = w.resample("ME").mean().dropna()
    m["b"] = brent.resample("ME").mean().reindex(m.index)
    m = m.dropna()
    for lo, hi in [(2011, 2015), (2015, 2019), (2019, 2023), (2023, 2027)]:
        sub = m[(m.index.year >= lo) & (m.index.year < hi)]
        if len(sub) < 24:
            continue
        dp = np.log(sub["pump"]).diff()
        dw = np.log(sub["wholesale"]).diff()
        db = np.log(sub["b"]).diff()
        d = pd.DataFrame({"dp": dp, "dw": dw, "db": db}).dropna()
        bp = np.polyfit(d.db, d.dp, 1)[0]
        bw = np.polyfit(d.db, d.dw, 1)[0]
        print(f"    {lo}-{hi-1}  pump-on-brent {bp:.3f}   wholesale-on-brent {bw:.3f}"
              f"   (n={len(d)})")

    print(f"\n  wrote data/new_factors/pump_dms.csv")


if __name__ == "__main__":
    main()
