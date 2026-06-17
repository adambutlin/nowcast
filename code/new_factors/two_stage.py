"""
Two-stage UK CPI-YoY forecaster — FINAL fixed architecture.

  Stage 1: AutoARIMA           (level / trend / seasonality / base-effect arithmetic)
  Stage 2: fixed equal-weight ensemble of {BVAR, TVP, MIDAS} on the AA residual
           resid_t = CPI_t - AutoARIMA_t ; each member predicts resid; combine 1/3 each.
  Forecast_t = AutoARIMA_t + mean_{m in {BVAR,TVP,MIDAS}} ( resid_pred_m,t )

  NO detector. NO switching. NO HMM. NO regime weights. Weights are fixed 1/3 each
  (renormalised over members available on a given date — e.g. MIDAS warm-up months).

Walk-forward (expanding window) backtest + live nowcast for the first unreleased month.
Outputs: data/new_factors/backtest.csv, metrics.csv, nowcast.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/new_factors/two_stage.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import factors as F, uk_model_zoo as Z

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "new_factors")
os.makedirs(_OUT, exist_ok=True)

TARGET = "cpi_yoy"
PINNED = ["oil_brent", "gas_eu", "uk_quarterly_gdp", "imf_all_commodity",
          "global_supply_chain_pressure", "mpc_rate_change",
          "ofgem_cap_delta",
          "uk_ppi_input", "deep_sea_freight"]   # mpc_vote_split+budget_event dropped (SHAP dead weight)
REG = ["mpc_rate_change", "ofgem_cap_delta"]
STAGE2 = [("bvar", Z.BVAR), ("tvp", Z.TVP), ("midas", Z.MIDAS)]   # TVP reinstated; weight via WEIGHTS
WEIGHTS = {"bvar": 0.375, "tvp": 0.25, "midas": 0.375}   # TVP locked 0.25; BVAR/MIDAS via alloc_sweep.py
AA_START, START, END, TRAIN_FROM = 2001, 2015, 2024, 1997


def _weighted_combo(members, weights):
    """Row-wise weighted mean over available members; weights renormalised per row."""
    cols = [c for c in members.columns if c in weights]
    w = pd.Series({c: weights[c] for c in cols})
    M = members[cols]
    wmat = M.notna().astype(float) * w.values
    denom = wmat.sum(axis=1).replace(0, np.nan)
    return (M.fillna(0) * w.values).sum(axis=1) / denom


def load_matrix(pinned=None):
    pinned = PINNED if pinned is None else pinned
    raw, status = F.build_matrix(names=pinned + [TARGET])
    live = [n for n in pinned if status.get(n) != "unavailable"]
    raw = raw[raw.index.year >= TRAIN_FROM]
    df = F.apply_publication_lags(raw, live)
    for rf in REG:
        if rf in df.columns:
            df[rf] = df[rf].fillna(0)
    return df.resample("ME").last(), live, status


def backtest(df, live):
    """Walk-forward reconstruction. Returns per-month frame with AA + members + combo."""
    aa = Z.AutoARIMA().backtest(df, [], TARGET, start_year=AA_START, end_year=END)
    aa = aa[(aa.index.year >= START) & (aa.index.year <= END)]
    out = pd.DataFrame({"actual": aa["actual"], "aa_pred": aa["pred"]})
    df = df.copy()
    df["resid"] = (aa["actual"] - aa["pred"]).reindex(df.index)
    member_recon = {}
    for tag, cls in STAGE2:
        bt = cls().backtest(df, live, "resid", start_year=START, end_year=END)
        if bt is not None and len(bt):
            recon = (aa["pred"].reindex(bt.index) + bt["pred"]).rename(tag)
            member_recon[tag] = recon
            out[f"{tag}_pred"] = recon.reindex(out.index)
    members = pd.DataFrame(member_recon).reindex(out.index)
    out["stage2_pred"] = _weighted_combo(members, WEIGHTS)   # WEIGHTS, NaN-safe renorm
    out["n_members"] = members.notna().sum(axis=1)
    out["forecast"] = out["stage2_pred"].fillna(out["aa_pred"])
    out["aa_err"] = out["actual"] - out["aa_pred"]
    out["err"] = out["actual"] - out["forecast"]
    return out.dropna(subset=["actual", "forecast"])


def metrics(bt):
    def m(e):
        e = e.dropna().values
        return np.sqrt((e**2).mean()), np.abs(e).mean()
    wins = {"full": bt.index.year >= START,
            "2022_23": bt.index.year.isin([2022, 2023]),
            "ex_shock": ~bt.index.year.isin([2022, 2023]),
            "pre_2020": bt.index.year <= 2019}
    rows = []
    for w, msk in wins.items():
        s = bt[msk]
        r_aa, m_aa = m(s["aa_err"]); r_f, m_f = m(s["err"])
        dir_acc = float((np.sign(s["forecast"].diff()) == np.sign(s["actual"].diff())).mean())
        rows.append(dict(window=w, n=len(s), rmse_AA=r_aa, rmse_2stage=r_f,
                         rel_rmse=r_f / r_aa, mae_AA=m_aa, mae_2stage=m_f,
                         dir_acc_2stage=dir_acc))
    return pd.DataFrame(rows).set_index("window")


def nowcast(df, live):
    """Live forecast for the first unreleased CPI month."""
    aa_pred, ndate = Z.AutoARIMA().nowcast(df, [], TARGET)
    d = df.copy()
    # residual target for stage-2: fit members on (CPI - AA) over history.
    aa_hist = Z.AutoARIMA().backtest(df, [], TARGET, start_year=AA_START, end_year=END)
    d["resid"] = (aa_hist["actual"] - aa_hist["pred"]).reindex(d.index)
    contrib = {}
    for tag, cls in STAGE2:
        rp, _ = cls().nowcast(d, live, "resid")
        if np.isfinite(rp):
            contrib[tag] = float(rp)
    wsum = sum(WEIGHTS[t] for t in contrib) or 1.0
    overlay = sum(WEIGHTS[t] * v for t, v in contrib.items()) / wsum
    fc = float(aa_pred + overlay) if np.isfinite(aa_pred) else np.nan
    return dict(nowcast_date=str(pd.Timestamp(ndate).date()) if ndate is not None else None,
                aa_pred=float(aa_pred), stage2_overlay=overlay, forecast=fc,
                members=contrib)


def main():
    print("Loading factor matrix …")
    df, live, status = load_matrix()
    for n in PINNED:
        if status.get(n) == "unavailable":
            print(f"  [DROP] {n} unavailable")
    print("live factors:", live)

    print("\nWalk-forward backtest (Stage1=AutoARIMA, Stage2=mean[BVAR,TVP,MIDAS]) …")
    bt = backtest(df, live)
    bt.to_csv(os.path.join(_OUT, "backtest.csv"))
    met = metrics(bt)
    met.to_csv(os.path.join(_OUT, "metrics.csv"))
    pd.options.display.width = 200
    print("\n=== METRICS (rel_rmse<1 => 2-stage beats AutoARIMA) ===")
    print(met.round(4).to_string())

    print("\nLive nowcast …")
    nc = nowcast(df, live)
    pd.Series(nc).to_csv(os.path.join(_OUT, "nowcast.csv"))
    print(f"  date           {nc['nowcast_date']}")
    print(f"  AutoARIMA      {nc['aa_pred']:.3f}")
    print(f"  Stage-2 overlay {nc['stage2_overlay']:+.3f}  members={ {k: round(v,3) for k,v in nc['members'].items()} }")
    print(f"  FORECAST       {nc['forecast']:.3f}")
    print("\nwritten:", os.path.join(_OUT, "backtest.csv / metrics.csv / nowcast.csv"))


if __name__ == "__main__":
    main()
