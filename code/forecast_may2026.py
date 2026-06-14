"""
May-2026 UK CPI nowcast — AutoARIMA + (TVP+BVAR)/2 residual model.
Plus Exercise-1 plot (all 6 AA-residual models) and scenario fan chart.

Deliverables:
  plots/ex1_aa_residual_models.png   — Exercise 1: all 6 models, AA-residual reconstructed CPI
  plots/may2026_fan.png              — May 2026 fan chart + error bars + oil-shock scenarios
  data/may2026_forecast.csv          — point forecast, bands, scenario table

Production model: final_CPI = AutoARIMA_forecast + 0.5*(TVP_resid + BVAR_resid)
  Combo OOS RMSE (2015-2024) = 0.4168 (beats AutoARIMA 0.4687 in 4/4 robustness windows).

Run:  FRED_API_KEY=... .venv/bin/python code/forecast_may2026.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import factors as F, uk_model_zoo as Z

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA  = os.path.join(_ROOT, "data"); _PLOTS = os.path.join(_ROOT, "plots")
TARGET = "cpi_yoy"; RESID = "cpi_resid"
PINNED = ["oil_brent", "gas_eu", "uk_quarterly_gdp", "imf_all_commodity"]
REG    = ["mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]
PINNED = PINNED + REG
ENERGY = ["oil_brent", "gas_eu", "imf_all_commodity"]   # oil-shock scenario levers
AA_START, TRAIN_FROM = 2001, 1997
COMBO_RMSE = 0.4168   # TVP+BVAR full-sample OOS RMSE from sweep_residual_regime.py

# ── data ────────────────────────────────────────────────────────────────────
print("Fetching factors + target …")
df_raw, status = F.build_matrix(names=PINNED + [TARGET])
live = [n for n in PINNED if status.get(n) != "unavailable"]
df_raw = df_raw[df_raw.index.year >= TRAIN_FROM]
df = F.apply_publication_lags(df_raw, live)
for _f in REG:
    if _f in df.columns:
        df[_f] = df[_f].fillna(0)

cpi = df[TARGET].dropna()
last_actual_date = cpi.index[-1]
nowcast_date = (last_actual_date + pd.offsets.MonthEnd(1))
print(f"  last CPI actual: {last_actual_date.date()} = {cpi.iloc[-1]:.1f}%")
print(f"  nowcast target:  {nowcast_date.date()} (May 2026 print, released next week)")

# ── AutoARIMA walk-forward → residual through last actual ───────────────────
print("\nAutoARIMA walk-forward (2001 → last actual) …")
aa = Z.AutoARIMA()
END_YEAR = last_actual_date.year
aa_bt = aa.backtest(df, [], TARGET, start_year=AA_START, end_year=END_YEAR)
aa_f, actual = aa_bt["pred"], aa_bt["actual"]
df[RESID] = (actual - aa_f).reindex(df.index)
print(f"  residual std={df[RESID].std():.3f}, defined through {df[RESID].dropna().index[-1].date()}")

# ── AutoARIMA point forecast for nowcast month (fit on CPI ≤ last actual) ────
import statsmodels.api as sm
y_tr = cpi.copy()
best_bic, best_fit, best_order = np.inf, None, (1, 0, 0)
for p in range(1, 4):
    for q in range(0, 3):
        try:
            fit = sm.tsa.ARIMA(y_tr, order=(p, 0, q)).fit(method="statespace")
            if fit.bic < best_bic:
                best_bic, best_fit, best_order = fit.bic, fit, (p, 0, q)
        except Exception:
            pass
aa_now = float(np.asarray(best_fit.forecast(steps=1))[0])
print(f"  AutoARIMA order={best_order}, May-2026 point forecast = {aa_now:.3f}%")

# ── residual predictor via model nowcast() on a (possibly perturbed) df ──────
# TVP reads CONTEMPORANEOUS factors (factor_t → resid_t) — its nowcast() uses the
# final Kalman beta on the nowcast-month row. BVAR reads LAGGED factors
# (factor_{t-1..3} → resid_t), so a one-month contemporaneous shock leaves BVAR
# unchanged; a sustained *regime* (energy elevated across the feed window) moves it.
# Scenario therefore perturbs the energy feed window (last FEED months + nowcast
# month) so BOTH channels respond — this is the "oil-shock regime CONTINUES" framing.
FEED = 3   # BVAR lag depth

def predict_residual(df_scen):
    """Fit TVP, BVAR on known residual in df_scen; nowcast the residual for May."""
    preds = {}
    for nm, m in [("TVP", Z.TVP()), ("BVAR", Z.BVAR())]:
        try:
            r, _ = m.nowcast(df_scen, live, RESID)
            preds[nm] = float(r)
        except Exception as e:
            print(f"    {nm} nowcast failed: {str(e)[:50]}"); preds[nm] = np.nan
    vals = [v for v in [preds.get("TVP"), preds.get("BVAR")] if np.isfinite(v)]
    return preds.get("TVP"), preds.get("BVAR"), (np.mean(vals) if vals else np.nan)

def make_scenario_df(mult):
    """Return df copy with energy factors shifted mult·σ over the feed window+nowcast."""
    d = df.copy()
    # ensure the nowcast month exists as a row (factors known, RESID NaN)
    if nowcast_date not in d.index:
        d.loc[nowcast_date] = np.nan
        d = d.sort_index()
        for f in live:                      # carry last-known factor into nowcast row
            if not np.isfinite(d.loc[nowcast_date, f]):
                d.loc[nowcast_date, f] = float(df[f].dropna().iloc[-1])
    feed_dates = d.index[-(FEED + 1):]       # last FEED months + nowcast month
    for f in ENERGY:
        if f in live:
            sd = float(df[f].dropna().iloc[-24:].std())
            d.loc[feed_dates, f] = d.loc[feed_dates, f] + mult * sd
    return d

# ── BASE nowcast (realized factors) ─────────────────────────────────────────
df_base = make_scenario_df(0.0)
tvp_b, bvar_b, resid_b = predict_residual(df_base)
cpi_base = aa_now + resid_b
print(f"\nBASE nowcast: AutoARIMA {aa_now:.3f} + resid {resid_b:+.3f} "
      f"(TVP {tvp_b:+.3f}, BVAR {bvar_b:+.3f}) = {cpi_base:.3f}%")

# ── SCENARIOS: oil-shock regime vs normalisation (±1σ energy over feed window) ─
scenarios = {}
for label, mult in [("oil_shock_continues", +1.0), ("base", 0.0), ("normalisation", -1.0)]:
    d = make_scenario_df(mult)
    tvp_s, bvar_s, resid_s = predict_residual(d)
    scenarios[label] = dict(cpi=aa_now + resid_s, resid=resid_s, tvp=tvp_s, bvar=bvar_s)
    sign = "+" if mult > 0 else ("-" if mult < 0 else "=")
    print(f"  {label:22} energy {sign}1σ → resid {resid_s:+.3f} "
          f"(TVP {tvp_s:+.3f}, BVAR {bvar_s:+.3f}) → CPI {aa_now+resid_s:.3f}%")

# ── save forecast table ─────────────────────────────────────────────────────
band1, band2 = COMBO_RMSE, 2*COMBO_RMSE
rows = [dict(item="point (base)", cpi=cpi_base, lo1=cpi_base-band1, hi1=cpi_base+band1,
             lo2=cpi_base-band2, hi2=cpi_base+band2),
        dict(item="autoarima_only", cpi=aa_now, lo1=np.nan, hi1=np.nan, lo2=np.nan, hi2=np.nan)]
for label, s in scenarios.items():
    rows.append(dict(item=label, cpi=s["cpi"], lo1=s["cpi"]-band1, hi1=s["cpi"]+band1,
                     lo2=s["cpi"]-band2, hi2=s["cpi"]+band2))
fc_df = pd.DataFrame(rows)
fc_df.to_csv(os.path.join(_DATA, "may2026_forecast.csv"), index=False)
print("\n" + fc_df.round(3).to_string(index=False))

# ── PLOT 1: Exercise-1 AA-residual models ───────────────────────────────────
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs(_PLOTS, exist_ok=True)
bt = pd.read_csv(os.path.join(_DATA, "nowcast_cpi_backtest.csv"), parse_dates=["date"])
aa_rows = bt[bt["model"] == "AutoARIMA"]
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(aa_rows["date"], aa_rows["actual"], color="black", ls="-", lw=2.8,
        label="actual CPI YoY", zorder=10)
ax.plot(aa_rows["date"], aa_rows["pred"], color="0.4", ls="--", lw=1.4,
        label="AutoARIMA baseline (0.469)")
RMSE = {"BVAR":0.459,"DFM":0.613,"UCM":0.462,"HMM":0.457,"TVP":0.475,"HuberNet":0.468}
# Colour-blind safe: distinguish by LINESTYLE + MARKER, not colour.
STYLE = {
    "HMM":      dict(ls="-",            marker="o", me=(0, 9)),
    "BVAR":     dict(ls="--",           marker="s", me=(3, 9)),
    "UCM":      dict(ls="-.",           marker="^", me=(6, 9)),
    "HuberNet": dict(ls=":",            marker="v", me=(1, 9)),
    "TVP":      dict(ls=(0,(3,1,1,1)),  marker="D", me=(4, 9)),
    "DFM":      dict(ls=(0,(5,1)),      marker="x", me=(7, 9)),
}
for nm in ["HMM","BVAR","UCM","HuberNet","TVP","DFM"]:
    sub = bt[bt["model"] == nm]
    if len(sub):
        s = STYLE[nm]
        ax.plot(sub["date"], sub["pred"], color="0.25", lw=1.0, alpha=0.9,
                ls=s["ls"], marker=s["marker"], markersize=4.5, markevery=s["me"],
                markerfacecolor="white", markeredgecolor="0.15",
                label=f"{nm} ({RMSE[nm]:.3f})")
ax.set_title("Exercise 1 — UK CPI YoY: AutoARIMA + factor-residual models (2015-2024 walk-forward)")
ax.set_ylabel("CPI YoY %"); ax.legend(fontsize=8, ncol=2, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); p1 = os.path.join(_PLOTS, "ex1_aa_residual_models.png")
fig.savefig(p1, dpi=130); plt.close(fig); print(f"\nSaved {p1}")

# ── PLOT 2: May-2026 fan chart + scenarios (colour-blind safe) ──────────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios":[2.4,1]})
# Left: CPI history + nowcast fan — greyscale fill, black lines, distinct markers
hist = cpi.iloc[-18:]
axL.plot(hist.index, hist.values, color="black", ls="-", marker="o", ms=4, lw=1.8,
         label="actual CPI YoY")
nd = nowcast_date
axL.fill_between([last_actual_date, nd], [cpi.iloc[-1], cpi_base-band2],
                 [cpi.iloc[-1], cpi_base+band2], color="0.80", alpha=0.7, label="±2σ (0.83)")
axL.fill_between([last_actual_date, nd], [cpi.iloc[-1], cpi_base-band1],
                 [cpi.iloc[-1], cpi_base+band1], color="0.60", alpha=0.7, label="±1σ (0.42)")
axL.plot([last_actual_date, nd], [cpi.iloc[-1], cpi_base], color="black", ls="--",
         marker="o", lw=1.8, ms=7, label=f"AutoARIMA+TVP+BVAR ({cpi_base:.2f}%)")
axL.plot([last_actual_date, nd], [cpi.iloc[-1], aa_now], color="0.45", ls=":",
         marker="s", ms=5, lw=1.4, label=f"AutoARIMA only ({aa_now:.2f}%)")
# scenario markers — distinct shapes, all black
sc_off = pd.Timedelta(days=6)
axL.plot(nd+sc_off, scenarios["oil_shock_continues"]["cpi"], marker="^", color="black",
         ms=11, ls="none", markerfacecolor="black",
         label=f"oil-shock ({scenarios['oil_shock_continues']['cpi']:.2f}%)")
axL.plot(nd+sc_off, scenarios["normalisation"]["cpi"], marker="v", color="black",
         ms=11, ls="none", markerfacecolor="white", markeredgewidth=1.6,
         label=f"normalisation ({scenarios['normalisation']['cpi']:.2f}%)")
axL.axhline(2.0, color="0.3", ls=(0,(1,1)), lw=0.9, alpha=0.8)
axL.text(hist.index[0], 2.05, "BoE 2% target", color="0.3", fontsize=8)
axL.set_title(f"UK CPI YoY — May 2026 nowcast (release {nd.strftime('%b %Y')})")
axL.set_ylabel("CPI YoY %"); axL.legend(fontsize=8, loc="upper right"); axL.grid(alpha=0.3)

# Right: scenario bars — distinguish by HATCH, not colour
labels  = ["normalisation","base","oil_shock_continues"]
vals    = [scenarios[l]["cpi"] for l in labels]
hatches = ["//", "", "\\\\"]
for i,(l,v,h) in enumerate(zip(labels,vals,hatches)):
    axR.barh(i, v, facecolor="0.85", edgecolor="black", hatch=h, lw=1.1,
             xerr=band1, error_kw=dict(ecolor="black", capsize=4, lw=1))
    axR.text(v+band1+0.03, i, f"{v:.2f}%", va="center", fontsize=9)
axR.set_yticks(range(3))
axR.set_yticklabels(["normalise\n(−1σ energy)","base","oil-shock\n(+1σ energy)"], fontsize=8)
axR.axvline(aa_now, color="0.4", ls=":", lw=1.2)
axR.text(aa_now, 2.62, f"AA {aa_now:.2f}", color="0.3", fontsize=8, ha="center")
axR.set_xlabel("CPI YoY %"); axR.set_title("Scenario × ±1σ band"); axR.grid(alpha=0.3, axis="x")
fig.tight_layout(); p2 = os.path.join(_PLOTS, "may2026_fan.png")
fig.savefig(p2, dpi=130); plt.close(fig); print(f"Saved {p2}")

# ── final summary ───────────────────────────────────────────────────────────
print("\n" + "="*64)
print(f"MAY 2026 UK CPI NOWCAST — AutoARIMA + (TVP+BVAR)/2 residual")
print("="*64)
print(f"  Point forecast:     {cpi_base:.2f}%   (±1σ {cpi_base-band1:.2f}–{cpi_base+band1:.2f},  "
      f"±2σ {cpi_base-band2:.2f}–{cpi_base+band2:.2f})")
print(f"  AutoARIMA only:     {aa_now:.2f}%")
print(f"  Oil-shock regime:   {scenarios['oil_shock_continues']['cpi']:.2f}%")
print(f"  Normalisation:      {scenarios['normalisation']['cpi']:.2f}%")
print(f"  Last actual (Apr):  {cpi.iloc[-1]:.1f}%")
