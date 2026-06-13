"""
Focused retrain on the user-pinned factor set (mirrors `main.py --factors` but
fetches only the needed series). Retrains the 5 operational models + AR(1) +
Combined-Dynamic, prints RMSEs, reruns the Shapley factor screen, saves the
backtest, and plots actual vs forecasts.

Run:  FRED_API_KEY=... .venv/bin/python code/retrain_pinned.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import factors as F, uk_model_zoo as Z
import main as NC

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA  = os.path.join(_ROOT, "data"); _PLOTS = os.path.join(_ROOT, "plots")
TARGET = "cpi_yoy"
PINNED = ["uk_house_prices", "gas_eu", "uk_quarterly_gdp",
          "imf_all_commodity", "global_supply_chain_pressure"]
START, END, TRAIN_FROM = 2015, 2024, 1997

print("Fetching pinned factors + target …")
df_raw, status = F.build_matrix(names=PINNED + [TARGET])
for n in PINNED:
    print(f"  {n:30} {status.get(n)}")
live = [n for n in PINNED if status.get(n) != "unavailable"]
if status.get(TARGET) == "unavailable":
    sys.exit("target cpi_yoy unavailable (dbnomics). Need data/cpi_yoy.csv.")
print(f"  available factors: {live}")

df_raw = df_raw[df_raw.index.year >= TRAIN_FROM]
df = F.apply_publication_lags(df_raw, live)
df["cpi_3m_chg"] = df[TARGET].shift(1).diff(3)
live_facs = live + ["cpi_3m_chg"]

# ── retrain ───────────────────────────────────────────────────────────────
models = Z.all_models()
print(f"\nRetraining {len(models)} models on {len(live_facs)} factors ({START}-{END}) …")
bt_dict = {}
for m in models:
    try:
        bt = m.backtest(df, live_facs, TARGET, start_year=START, end_year=END)
        bt_dict[m.name] = bt if len(bt) else None
    except Exception as e:
        bt_dict[m.name] = None; print(f"  {m.name} ERROR {str(e)[:50]}")
bt_ar1 = NC.ar1_backtest(df, TARGET, start_year=START, end_year=END)
bt_dict["AR(1)"] = bt_ar1
ar1_rmse = float(np.sqrt(((bt_ar1["actual"]-bt_ar1["pred"])**2).mean()))
beaters = {n: b for n, b in bt_dict.items()
           if n != "AR(1)" and b is not None and
           float(np.sqrt(((b["actual"]-b["pred"])**2).mean())) < ar1_rmse}
_cd = NC.combine_dynamic(beaters, window=12)
bt_dict["Combined-Dynamic"] = _cd if (_cd is not None and len(_cd)) else None

def rmse(b): return float(np.sqrt(((b["actual"]-b["pred"])**2).mean())) if b is not None and len(b) else np.nan
rows = [dict(model=n, rmse=rmse(b), n=(len(b) if b is not None else 0),
            beats_AR1=(rmse(b) < ar1_rmse) if b is not None else False)
        for n, b in bt_dict.items()]
mdf = pd.DataFrame(rows).set_index("model").sort_values("rmse")
print("\n" + "="*52 + "\nMODEL RMSE (pinned factors, common 2015-2024)\n" + "="*52)
print(mdf.round(4).to_string())

# ── Shapley factor screen (pre-START, same as main) ─────────────────────────
print("\n" + "="*52 + "\nSHAPLEY FACTOR SCREEN (pre-2015 fit, mean |SHAP|)\n" + "="*52)
try:
    import shap; from lightgbm import LGBMRegressor
    screen = df[df.index.year < START][live + ["cpi_3m_chg", TARGET]].dropna()
    X, y = screen[live + ["cpi_3m_chg"]], screen[TARGET]
    if len(screen) >= 30:
        lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=4,
                             min_child_samples=30, reg_alpha=2, reg_lambda=2,
                             random_state=42, verbose=-1).fit(X, y)
        sv = shap.TreeExplainer(lgbm).shap_values(X)
        imp = pd.Series(np.abs(sv).mean(0), index=X.columns).sort_values(ascending=False)
        for k, v in imp.items(): print(f"  {k:30} {v:.5f}")
    else:
        print(f"  insufficient pre-{START} obs ({len(screen)}) to screen")
except Exception as e:
    print("  SHAP failed:", str(e)[:60])

# ── save + plot ─────────────────────────────────────────────────────────────
out = []
for n, b in bt_dict.items():
    if b is not None and len(b):
        bb = b.copy(); bb["model"] = n; out.append(bb)
pd.concat(out).reset_index().to_csv(os.path.join(_DATA, "nowcast_cpi_backtest.csv"), index=False)
mdf.to_csv(os.path.join(_DATA, "nowcast_cpi_metrics.csv"))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(_PLOTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    act = bt_dict["AR(1)"][["actual"]].copy()
    ax.plot(act.index, act["actual"], "k-", lw=2, label="actual CPI YoY")
    for n in ["Combined-Dynamic", "ElasticNet", "AutoARIMA", "UCM", "TVP", "MIDAS"]:
        b = bt_dict.get(n)
        if b is not None and len(b):
            ax.plot(b.index, b["pred"], lw=1, alpha=0.8, label=f"{n} ({rmse(b):.3f})")
    ax.set_title("UK CPI YoY — retrain on pinned factors (5 models, 2015-2024)")
    ax.legend(fontsize=8, ncol=2); ax.set_ylabel("CPI YoY %"); ax.grid(alpha=0.3)
    fig.tight_layout(); p = os.path.join(_PLOTS, "retrain_pinned.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"\nSaved plot → {p}")
except Exception as e:
    print("  plot failed:", str(e)[:60])
print(f"Saved backtest → data/nowcast_cpi_backtest.csv  | metrics → data/nowcast_cpi_metrics.csv")
