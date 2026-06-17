"""
PART B — Construct the HelpfulStage2 target.

For every walk-forward CPI-YoY forecast (expanding window, 2015-2024):
  AA_pred_t     = AutoARIMA (Stage 1) prediction
  Stage2_pred_t = AA_pred_t + residual-model prediction of (CPI - AA)
                  for each of BVAR / UCM / TVP / MIDAS, plus an equal-weight combo.

Define:
  AA_err_t      = actual_t - AA_pred_t
  Stage2_err_t  = actual_t - Stage2_pred_t (combo)
  HelpfulStage2_t = 1[ |Stage2_err_t| < |AA_err_t| ]
  SkillGain_t     = |AA_err_t| - |Stage2_err_t|     (>0 => Stage-2 helped)

Also attach causal high-frequency observables for month t (within-month daily
financials, known at nowcast time — same information MIDAS uses):
  brent/gas/fx/vol momentum + realized vol, rates momentum, Ofgem/MPC event flags.

Output: data/reg_detect/targets.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python code/reg_detect/build_targets.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import factors as F, uk_model_zoo as Z

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "reg_detect")
os.makedirs(_OUT, exist_ok=True)

TARGET = "cpi_yoy"
PINNED = ["oil_brent", "gas_eu", "uk_quarterly_gdp", "imf_all_commodity",
          "global_supply_chain_pressure"]
REG = ["mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]
PINNED = PINNED + REG
AA_START, START, END, TRAIN_FROM = 2001, 2015, 2024, 1997
STAGE2 = [("bvar", Z.BVAR), ("ucm", Z.UCM), ("tvp", Z.TVP), ("midas", Z.MIDAS)]


def hf_observables():
    """Monthly causal observables from within-month daily financials + factor events."""
    hf = pd.read_csv(os.path.join(_ROOT, "data", "intramonth", "hf_daily.csv"),
                     parse_dates=["Date"]).set_index("Date").sort_index()
    out = {}
    g = hf.groupby(pd.Grouper(freq="ME"))
    for col, tag in [("brent", "brent"), ("gas", "gas"), ("gbp", "fx"), ("vix", "vix")]:
        s = hf[col]
        dl = np.log(s).diff()                       # daily log return
        mom = g[col].apply(lambda x: np.log(x.dropna().iloc[-1] / x.dropna().iloc[0])
                           if x.dropna().shape[0] > 1 else np.nan)   # month logret
        rv = dl.groupby(pd.Grouper(freq="ME")).std() * np.sqrt(21)   # realized vol
        lvl = g[col].mean()
        out[f"{tag}_mom"] = mom
        out[f"{tag}_absmom"] = mom.abs()
        out[f"{tag}_rv"] = rv
        if tag == "vix":
            out["vix_lvl"] = lvl
            out["vix_chg"] = lvl.diff()
    obs = pd.DataFrame(out)
    # Rates momentum from 5y breakeven + MPC; Ofgem month flag — via factor matrix
    raw, status = F.build_matrix(names=["uk_be5", "mpc_rate_change", "ofgem_cap_delta"])
    raw = raw.resample("ME").last()
    if "uk_be5" in raw:
        obs["rates_mom"] = raw["uk_be5"].diff().reindex(obs.index)
    for c in ["mpc_rate_change", "ofgem_cap_delta"]:
        if c in raw:
            obs[c] = raw[c].reindex(obs.index).fillna(0.0)
    obs["ofgem_flag"] = (obs.get("ofgem_cap_delta", 0).abs() > 1e-9).astype(float)
    return obs


def main():
    print("Fetching factors …")
    raw, status = F.build_matrix(names=PINNED + [TARGET])
    live = [n for n in PINNED if status.get(n) != "unavailable"]
    raw = raw[raw.index.year >= TRAIN_FROM]
    df = F.apply_publication_lags(raw, live)
    for rf in REG:
        if rf in df.columns:
            df[rf] = df[rf].fillna(0)
    df = df.resample("ME").last()

    print("AutoARIMA Stage-1 backtest …")
    aa = Z.AutoARIMA().backtest(df, [], TARGET, start_year=AA_START, end_year=END)
    aa = aa[(aa.index.year >= START) & (aa.index.year <= END)]
    base = pd.DataFrame({"actual": aa["actual"], "aa_pred": aa["pred"]})
    base["aa_err"] = base["actual"] - base["aa_pred"]
    df["resid"] = (aa["actual"] - aa["pred"]).reindex(df.index)

    recon_preds = {}
    for tag, cls in STAGE2:
        print(f"Stage-2 {tag} backtest …")
        try:
            bt = cls().backtest(df, live, "resid", start_year=START, end_year=END)
        except Exception as e:
            print(f"  [WARN] {tag} failed: {str(e)[:80]}")
            bt = None
        if bt is not None and len(bt):
            p = (aa["pred"].reindex(bt.index) + bt["pred"]).rename(f"{tag}_pred")
            recon_preds[tag] = p
            base[f"{tag}_pred"] = p.reindex(base.index)
            base[f"{tag}_err"] = base["actual"] - base[f"{tag}_pred"]

    # Equal-weight Stage-2 combo over models available each date
    pred_cols = [f"{t}_pred" for t, _ in STAGE2 if f"{t}_pred" in base]
    base["stage2_pred"] = base[pred_cols].mean(axis=1)
    base["stage2_err"] = base["actual"] - base["stage2_pred"]

    # Targets
    base["helpful"] = (base["stage2_err"].abs() < base["aa_err"].abs()).astype(int)
    base["skillgain"] = base["aa_err"].abs() - base["stage2_err"].abs()
    # per-model helpful flags
    for tag, _ in STAGE2:
        if f"{tag}_err" in base:
            base[f"helpful_{tag}"] = (base[f"{tag}_err"].abs() < base["aa_err"].abs()).astype(int)

    obs = hf_observables()
    out = base.join(obs, how="left")
    out = out[out["stage2_err"].notna() & out["aa_err"].notna()]
    out.index.name = "date"
    out.to_csv(os.path.join(_OUT, "targets.csv"))

    n = len(out)
    print("\n" + "=" * 70)
    print(f"TARGETS BUILT  n={n}  ({out.index.min().date()} … {out.index.max().date()})")
    print(f"  base rate HelpfulStage2 (combo) = {out['helpful'].mean():.3f}")
    for tag, _ in STAGE2:
        c = f"helpful_{tag}"
        if c in out:
            print(f"  base rate helpful_{tag:6} = {out[c].mean():.3f}   "
                  f"meanSkill(±)= {(out['aa_err'].abs()-out[f'{tag}_err'].abs()).mean():+.4f}")
    print(f"  mean SkillGain (combo) = {out['skillgain'].mean():+.4f}  "
          f"(>0 => Stage-2 helps on average)")
    print(f"  RMSE AA   = {np.sqrt((out['aa_err']**2).mean()):.4f}")
    print(f"  RMSE S2   = {np.sqrt((out['stage2_err']**2).mean()):.4f}")
    print("  by regime window  helpful-rate / meanSkill:")
    for w, m in {"full": out.index.year >= 2015,
                 "2022_23": out.index.year.isin([2022, 2023]),
                 "ex_shock": ~out.index.year.isin([2022, 2023]),
                 "pre_2020": out.index.year <= 2019}.items():
        s = out[m]
        print(f"    {w:9} n={len(s):3}  helpful={s['helpful'].mean():.3f}  "
              f"skill={s['skillgain'].mean():+.4f}")
    print("written:", os.path.join(_OUT, "targets.csv"))


if __name__ == "__main__":
    main()
