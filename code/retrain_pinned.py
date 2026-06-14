"""
Residual-inflation retrain (user spec 2026-06-13).

  AutoARIMA = persistence baseline. Residual r_t = CPI_t − AutoARIMA_forecast_t
  (causal walk-forward forecast). The other models (TVP, DFM, MIDAS, BVAR,
  HuberNet) are trained to predict r_t from the exogenous factors. Final forecast
  = AutoARIMA_t + residual_model_t, so a residual model's backtest RMSE on r IS
  the final CPI RMSE, and the baseline ("predict residual = 0") RMSE = AutoARIMA's
  own CPI RMSE. A residual model earns its place iff it beats that baseline.

Factors (pinned): oil_brent (brent crude), gas_eu, uk_quarterly_gdp,
imf_all_commodity, global_supply_chain_pressure (NYFed GSCPI; skipped if the
NY Fed file is HTML-walled).

Run:  FRED_API_KEY=... .venv/bin/python code/retrain_pinned.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import factors as F, uk_model_zoo as Z

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA  = os.path.join(_ROOT, "data"); _PLOTS = os.path.join(_ROOT, "plots")
TARGET = "cpi_yoy"; RESID = "cpi_resid"
PINNED = ["oil_brent", "gas_eu", "uk_quarterly_gdp",
          "imf_all_commodity", "global_supply_chain_pressure"]
REG = ["mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]  # reg-event factors
PINNED = PINNED + REG
START, END, TRAIN_FROM, AA_START = 2015, 2024, 1997, 2001

print("Fetching pinned factors + target …")
df_raw, status = F.build_matrix(names=PINNED + [TARGET])
for n in PINNED:
    print(f"  {n:30} {status.get(n)}")
live = [n for n in PINNED if status.get(n) != "unavailable"]
if status.get(TARGET) == "unavailable":
    sys.exit("target cpi_yoy unavailable.")
print(f"  available factors: {live}")

df_raw = df_raw[df_raw.index.year >= TRAIN_FROM]
df = F.apply_publication_lags(df_raw, live)
# reg-event factors: 0 in periods before their data starts (no event = 0)
for _f in REG:
    if _f in df.columns:
        df[_f] = df[_f].fillna(0)

# ── Stage 1: causal AutoARIMA forecast over the whole sample → residual ──────
print(f"\nStage 1: AutoARIMA walk-forward forecast ({AA_START}-{END}) …")
aa = Z.AutoARIMA()
aa_bt = aa.backtest(df, [], TARGET, start_year=AA_START, end_year=END)   # univariate
aa_f = aa_bt["pred"]                                  # causal 1-step AutoARIMA forecast
actual = aa_bt["actual"]
df[RESID] = (actual - aa_f).reindex(df.index)         # residual where AA forecast exists
print(f"  residual defined {df[RESID].dropna().index.min().date()} → "
      f"{df[RESID].dropna().index.max().date()}  (std={df[RESID].std():.3f})")

# ── Benchmarks: AR(1), AR(2), AutoARIMA standalone CPI RMSE on test window ───
def ar_backtest(series, p, start, end):
    """Walk-forward OLS AR(p): yearly refit on data<yr, 1-step with realized lags."""
    s = series.dropna(); rows = []
    for yr in range(start, end + 1):
        tr = s[s.index.year < yr]; te = s[s.index.year == yr]
        if len(tr) < 30 or len(te) == 0:
            continue
        y = tr.values; n = len(y)
        X = np.column_stack([np.ones(n - p)] + [y[p - l:n - l] for l in range(1, p + 1)])
        beta = np.linalg.lstsq(X, y[p:], rcond=None)[0]
        hist = list(y)
        for idx in te.index:
            xr = np.array([1.0] + [hist[-l] for l in range(1, p + 1)])
            rows.append((idx, float(s.loc[idx]), float(xr @ beta)))
            hist.append(float(s.loc[idx]))
    return pd.DataFrame(rows, columns=["date", "actual", "pred"]).set_index("date")

aa_test = aa_bt[(aa_bt.index.year >= START) & (aa_bt.index.year <= END)]
base_rmse = float(np.sqrt(((aa_test["actual"] - aa_test["pred"])**2).mean()))
def _rmse(bt): return float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean()))
ar1_bt = ar_backtest(df[TARGET], 1, START, END)
ar2_bt = ar_backtest(df[TARGET], 2, START, END)
ar1_rmse, ar2_rmse = _rmse(ar1_bt), _rmse(ar2_bt)
print(f"  BENCHMARKS CPI RMSE {START}-{END}: "
      f"AR(1)={ar1_rmse:.4f}  AR(2)={ar2_rmse:.4f}  AutoARIMA={base_rmse:.4f}")

# ── Stage 2: residual models on the factors ─────────────────────────────────
resid_models = [Z.BVAR(), Z.DFM(), Z.UCM(), Z.HMM(), Z.TVP(), Z.HuberNet()]
print(f"\nStage 2: training {len(resid_models)} residual models on {len(live)} factors …")
def final_rmse(bt):   # residual-RMSE == final-CPI-RMSE by construction
    return float(np.sqrt(((bt["actual"] - bt["pred"])**2).mean())) if bt is not None and len(bt) else np.nan
bt_dict = {}
rows = [dict(model="AR(1) [bench]",      cpi_rmse=ar1_rmse, n=len(ar1_bt), beats_AA=ar1_rmse < base_rmse),
        dict(model="AR(2) [bench]",      cpi_rmse=ar2_rmse, n=len(ar2_bt), beats_AA=ar2_rmse < base_rmse),
        dict(model="AutoARIMA [bench]",  cpi_rmse=base_rmse, n=len(aa_test), beats_AA=False)]
for m in resid_models:
    try:
        bt = m.backtest(df, live, RESID, start_year=START, end_year=END)
        bt = bt if (bt is not None and len(bt)) else None
    except Exception as e:
        bt = None; print(f"  {m.name} ERROR {str(e)[:50]}")
    bt_dict[m.name] = bt
    r = final_rmse(bt)
    rows.append(dict(model=m.name, cpi_rmse=r, n=(len(bt) if bt is not None else 0),
                     beats_AA=(r < base_rmse) if np.isfinite(r) else False))

# Combined-Dynamic over residual models beating baseline (inverse-RMSE weighting)
import main as NC
beaters = {n: b for n, b in bt_dict.items()
           if b is not None and final_rmse(b) < base_rmse}
cd = NC.combine_dynamic(beaters, window=12) if beaters else None
if cd is not None and len(cd):
    bt_dict["Combined-Dynamic"] = cd
    rows.append(dict(model="Combined-Dynamic", cpi_rmse=final_rmse(cd),
                     n=len(cd), beats_AA=final_rmse(cd) < base_rmse))

mdf = pd.DataFrame(rows).set_index("model").sort_values("cpi_rmse")
print("\n" + "="*60 + "\nRESIDUAL CPI RMSE — 3 benchmarks + residual models (final = AutoARIMA + residual)\n" + "="*60)
print(mdf.round(4).to_string())
print(f"\n  (baseline = AutoARIMA alone = {base_rmse:.4f}; a model earns its place iff cpi_rmse < baseline)")

# ── Shapley screen: which factors predict the RESIDUAL ──────────────────────
print("\n" + "="*60 + "\nSHAPLEY SCREEN — factors predicting the RESIDUAL (pre-2015)\n" + "="*60)
try:
    import shap; from lightgbm import LGBMRegressor
    screen = df[df.index.year < START][live + [RESID]].dropna()
    if len(screen) >= 30:
        X, y = screen[live], screen[RESID]
        lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=4,
                             min_child_samples=30, reg_alpha=2, reg_lambda=2,
                             random_state=42, verbose=-1).fit(X, y)
        sv = shap.TreeExplainer(lgbm).shap_values(X)
        imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
        for k, v in imp.items(): print(f"  {k:30} {v:.5f}")
    else:
        print(f"  insufficient pre-{START} residual obs ({len(screen)})")
except Exception as e:
    print("  SHAP failed:", str(e)[:60])

# ── save + plot (final = AutoARIMA + residual) ──────────────────────────────
out = []
for n, b in bt_dict.items():
    if b is not None and len(b):
        fb = b.copy()
        fb["pred"]   = aa_f.reindex(b.index) + b["pred"]      # reconstruct CPI forecast
        fb["actual"] = aa_f.reindex(b.index) + b["actual"]    # = true CPI
        fb["model"] = n; out.append(fb)
aa_out = aa_test.copy(); aa_out["model"] = "AutoARIMA"; out.append(aa_out)
pd.concat(out).reset_index().to_csv(os.path.join(_DATA, "nowcast_cpi_backtest.csv"), index=False)
mdf.to_csv(os.path.join(_DATA, "nowcast_cpi_metrics.csv"))
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(_PLOTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(aa_test.index, aa_test["actual"], "k-", lw=2, label="actual CPI YoY")
    ax.plot(aa_test.index, aa_test["pred"], "--", color="grey", lw=1.2,
            label=f"AutoARIMA baseline ({base_rmse:.3f})")
    for n in ["Combined-Dynamic", "BVAR", "DFM", "UCM", "HMM", "TVP", "HuberNet"]:
        b = bt_dict.get(n)
        if b is not None and len(b):
            ax.plot(b.index, aa_f.reindex(b.index) + b["pred"], lw=1, alpha=0.8,
                    label=f"{n} ({final_rmse(b):.3f})")
    ax.set_title("UK CPI YoY — residual framework (AutoARIMA + factor residual model)")
    ax.legend(fontsize=8, ncol=2); ax.set_ylabel("CPI YoY %"); ax.grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(_PLOTS, "retrain_pinned.png")
    fig.savefig(p, dpi=120); plt.close(fig); print(f"\nSaved plot → {p}")
except Exception as e:
    print("  plot failed:", str(e)[:60])
print("Saved backtest → data/nowcast_cpi_backtest.csv  | metrics → data/nowcast_cpi_metrics.csv")
