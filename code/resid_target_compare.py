"""
Does TVP+BVAR forecast the CORE or SERVICES CPI residual (over AutoARIMA) better than
the HEADLINE residual?  Residual framework, walk-forward, same pinned factors per target.

For each target: AutoARIMA baseline (univariate, walk-forward) → residual r = CPI − AA;
BVAR and TVP fit r on the pinned factors; final = AA + residual_model. Report each
target's AutoARIMA RMSE, BVAR/TVP/combo RMSE, and the % improvement of the TVP+BVAR
combo over AutoARIMA — the higher the improvement, the more predictable that target's
residual is from the factors.

Run:  set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python code/resid_target_compare.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import factors as F, uk_model_zoo as Z

PINNED = ["oil_brent", "gas_eu", "uk_quarterly_gdp", "imf_all_commodity",
          "mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]
REG = ["mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]
AA_START, START, END, TRAIN_FROM = 2001, 2016, 2024, 1997


def core_yoy():
    idx = F._fred("GBRCPICORMINMEI")           # UK core CPI index (OECD, ex food+energy)
    return idx.resample("ME").last().pct_change(12).mul(100).dropna()


def target_series(name):
    if name == "headline":
        s, _ = F.load_factor("cpi_yoy")
    elif name == "services":
        s, _ = F.load_factor("uk_services_cpi")
    elif name == "core":
        s = core_yoy()
    return s.dropna().resample("ME").last()


def _rmse(bt):
    return float(np.sqrt(((bt["actual"] - bt["pred"]) ** 2).mean())) if bt is not None and len(bt) else np.nan


def run_target(name, fac, live):
    y = target_series(name)
    df = fac.copy()
    df["target"] = y.reindex(df.index)
    end = min(END, int(y.index[-1].year))      # core ends 2025 → still allow ≤2024
    # AutoARIMA baseline
    aa = Z.AutoARIMA().backtest(df, [], "target", start_year=AA_START, end_year=end)
    aa_test = aa[(aa.index.year >= START) & (aa.index.year <= end)]
    base_rmse = _rmse(aa_test)
    df["resid"] = (aa["actual"] - aa["pred"]).reindex(df.index)
    # residual models
    out = {"target": name, "n": len(aa_test), "end": end, "AutoARIMA": base_rmse,
           "resid_std": float(df["resid"].std())}
    recon = {}
    for nm, m in [("BVAR", Z.BVAR()), ("TVP", Z.TVP())]:
        bt = m.backtest(df, live, "resid", start_year=START, end_year=end)
        if bt is not None and len(bt):
            r = bt.copy()
            r["pred"] = aa["pred"].reindex(bt.index) + bt["pred"]
            r["actual"] = aa["pred"].reindex(bt.index) + bt["actual"]
            recon[nm] = r
            out[nm] = _rmse(r)
        else:
            out[nm] = np.nan
    # TVP+BVAR equal-weight combo on common dates
    if "BVAR" in recon and "TVP" in recon:
        common = recon["BVAR"].index.intersection(recon["TVP"].index)
        cp = 0.5 * recon["BVAR"].loc[common, "pred"] + 0.5 * recon["TVP"].loc[common, "pred"]
        ca = recon["BVAR"].loc[common, "actual"]
        out["TVP+BVAR"] = float(np.sqrt(((ca - cp) ** 2).mean()))
    else:
        out["TVP+BVAR"] = np.nan
    out["improve_vs_AA_%"] = (100 * (base_rmse - out["TVP+BVAR"]) / base_rmse
                              if np.isfinite(out["TVP+BVAR"]) else np.nan)
    return out


def main():
    print("Fetching pinned factors …")
    raw, status = F.build_matrix(names=PINNED + ["cpi_yoy"])
    live = [n for n in PINNED if status.get(n) != "unavailable"]
    raw = raw[raw.index.year >= TRAIN_FROM]
    fac = F.apply_publication_lags(raw, live)
    for rf in REG:
        if rf in fac.columns:
            fac[rf] = fac[rf].fillna(0)
    fac = fac.resample("ME").last()

    rows = [run_target(t, fac, live) for t in ["headline", "core", "services"]]
    df = pd.DataFrame(rows).set_index("target")
    cols = ["n", "end", "resid_std", "AutoARIMA", "BVAR", "TVP", "TVP+BVAR", "improve_vs_AA_%"]
    print("\n" + "=" * 78)
    print("RESIDUAL PREDICTABILITY BY TARGET — TVP+BVAR on (CPI − AutoARIMA), 2016-2024")
    print("=" * 78)
    print(df[cols].round(4).to_string())
    print("\n(improve_vs_AA_% > 0 means TVP+BVAR beats AutoARIMA baseline on that target)")
    best = df["improve_vs_AA_%"].astype(float).idxmax()
    print(f"\nMost residual-predictable target: {best} "
          f"(combo beats AutoARIMA by {df.loc[best,'improve_vs_AA_%']:.1f}%)")
    df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "resid_target_compare.csv"))


if __name__ == "__main__":
    main()
