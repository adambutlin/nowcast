"""
THE valid experiment: PRODUCTION model (two_stage: AutoARIMA + Z.MIDAS + BVAR + TVP on
PINNED incl uk_ppi_input/deep_sea_freight, weights 0.375/0.25/0.375, sample 2015-2024)
under TRUE as-of truncation across horizons T-30..R-1.

Only the within-reference-month varying inputs are truncated as-of (M_end - h days):
  - daily financials oil_brent, gas_eu (monthly logret recomputed vs prior full month-end)
  - MIDAS daily mean (Brent/GBP/VIX/TTF) -> partial-month mean (cache injection)
Monthly cost factors (imf/ppi/gdp/freight/ofgem/mpc) DO NOT change intra-month-M (their M
value is unpublished; available value fixed) -> left at production. AutoARIMA is the common
baseline (does not use intramonth financials) -> edge = rmse(AA) - rmse(full) is clean and
AA-vintage-independent. Post-month-end (h<=0) == full production (proven frozen).

Out: data/timing/prod_asof/{forecast_evolution,edge_by_horizon,edge_decomposition}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/production_asof.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
import uk_model_zoo as Z, two_stage as TS

_ROOT = os.path.dirname(_CODE)
_OUT = os.path.join(_ROOT, "data", "timing", "prod_asof")
os.makedirs(_OUT, exist_ok=True)
HORIZONS = [30, 21, 14, 10, 7, 5, 2, 1, 0, -5, -10, -17]   # h days before T; <=0 = post-month-end
LABEL = {30:"T-30",21:"T-21",14:"T-14",10:"T-10",7:"T-7",5:"T-5",2:"T-2",1:"T-1",
         0:"T",-5:"T+5",-10:"T+10",-17:"R-1"}
W = TS.WEIGHTS
WIN = {"full": lambda i: i.year>=2015, "2022_23": lambda i: i.year in (2022,2023),
       "ex_shock": lambda i: i.year not in (2022,2023)}


def _daily():
    d = pd.read_csv(os.path.join(_ROOT,"data","intramonth","hf_daily.csv"),
                    parse_dates=["Date"]).set_index("Date").sort_index()
    return d


def _asof_mask(idx, h):
    me = idx + pd.offsets.MonthEnd(0)
    return (me - idx).days >= h             # keep days at least h before that month's end


def asof_monthly(daily_col, h, how):
    df = daily_col.dropna().to_frame("v")
    df = df[_asof_mask(df.index, h)]
    g = df.groupby(pd.Grouper(freq="ME"))["v"]
    return (g.last() if how=="last" else g.mean())


def asof_logret(daily_col, h):
    """monthly logret with month-M level truncated as-of (vs prior FULL month-end)."""
    asof_last = asof_monthly(daily_col, h, "last")
    full_last = daily_col.resample("ME").last()
    prev = full_last.shift(1).reindex(asof_last.index)
    return np.log(asof_last / prev)


def build_mm(daily, h):
    """MIDAS monthly-mean matrix at horizon h (cols brent_ma/gbpusd_ma/vix_ma/gas_ma)."""
    cols = {"brent":"brent_ma","gbp":"gbpusd_ma","vix":"vix_ma","gas":"gas_ma"}
    out = {}
    for src,name in cols.items():
        if src in daily.columns:
            out[name] = asof_monthly(daily[src], h, "mean")
    return pd.DataFrame(out)


def run_horizon(df0, live, daily, h):
    df = df0.copy()
    if h > 0:                                # pre-month-end: truncate financials as-of
        for f, src in [("oil_brent","brent"), ("gas_eu","gas")]:
            if f in df.columns and src in daily.columns:
                df[f] = asof_logret(daily[src], h).reindex(df.index)
        Z._MIDAS_CACHE.clear(); Z._MIDAS_CACHE["mm"] = build_mm(daily, h)
    else:                                    # h<=0: full month = production
        Z._MIDAS_CACHE.clear()
    bt = TS.backtest(df, live)
    Z._MIDAS_CACHE.clear()
    return bt


def rmse(e): e=pd.Series(e).dropna(); return float(np.sqrt((e**2).mean())) if len(e) else np.nan
def mae(e): e=pd.Series(e).dropna(); return float(e.abs().mean()) if len(e) else np.nan


def main():
    df0, live, status = TS.load_matrix()
    daily = _daily()
    print("live factors:", live)
    evo_rows, edge_rows, dec_rows = [], [], []
    aa_ref = None
    for h in HORIZONS:
        bt = run_horizon(df0, live, daily, h)
        a = bt["actual"]; aa = bt["aa_pred"]; full = bt["forecast"]
        if aa_ref is None: aa_ref = aa
        for d in bt.index:
            evo_rows.append(dict(date=d, horizon=LABEL[h], h=h, actual=a[d],
                                 aa=aa[d], full=full[d],
                                 bvar=bt.get("bvar_pred",pd.Series()).get(d,np.nan),
                                 tvp=bt.get("tvp_pred",pd.Series()).get(d,np.nan),
                                 midas=bt.get("midas_pred",pd.Series()).get(d,np.nan)))
        rec = dict(horizon=LABEL[h], h=h)
        for w,fn in WIN.items():
            m = bt.index.map(lambda x: fn(x)).values.astype(bool)
            rec[f"rmse_AA_{w}"]=rmse(a[m]-aa[m]); rec[f"rmse_full_{w}"]=rmse(a[m]-full[m])
            rec[f"edge_{w}"]=rmse(a[m]-aa[m])-rmse(a[m]-full[m])
        rec["mae_full"]=mae(a-full); rec["corr_full"]=float(np.corrcoef(a,full)[0,1])
        edge_rows.append(rec)
        # member overlay contributions (mean |member_pred - AA|) + AA+member edge
        ov = {}
        for tag in ["bvar","tvp","midas"]:
            if f"{tag}_pred" in bt:
                ov[f"contrib_{tag}"]=float((bt[f"{tag}_pred"]-aa).abs().mean())
                ov[f"edge_AA+{tag}"]=rmse(a-aa)-rmse(a-bt[f"{tag}_pred"])
        dec_rows.append(dict(horizon=LABEL[h], **ov))
        print(f"  {LABEL[h]:5} edge_full={rec['edge_full']:+.4f} rel={rec['rmse_full_full']/rec['rmse_AA_full']:.3f} "
              f"contrib b/t/m={ov.get('contrib_bvar',0):.3f}/{ov.get('contrib_tvp',0):.3f}/{ov.get('contrib_midas',0):.3f}")

    pd.DataFrame(evo_rows).to_csv(os.path.join(_OUT,"forecast_evolution.csv"),index=False)
    edge=pd.DataFrame(edge_rows).set_index("horizon"); edge.to_csv(os.path.join(_OUT,"edge_by_horizon.csv"))
    dec=pd.DataFrame(dec_rows).set_index("horizon"); dec.to_csv(os.path.join(_OUT,"edge_decomposition.csv"))
    pd.options.display.width=220
    print("\n=== EDGE BY HORIZON (production model, as-of) ===")
    print(edge[["rmse_AA_full","rmse_full_full","edge_full","edge_ex_shock","edge_2022_23"]].round(4).to_string())
    print("\n=== DECOMPOSITION (mean |overlay|, AA+member edge) ===")
    print(dec.round(4).to_string())


if __name__ == "__main__":
    main()
