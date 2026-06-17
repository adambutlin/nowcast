"""
ObservableShock label + detection comparison + switched architecture.

1) ObservableShock_t: causal OR-of-exceedances over expanding 80th-pctile thresholds
   (no lookahead) across event channels:
     energy     : |brent_mom|, brent_rv, |gas_mom|, gas_rv
     commodity  : |imf_all_commodity logret|
     shipping   : |deep_sea_freight logret|  (GSCPI fallback)  [may be absent -> documented]
     regulatory : ofgem_flag | |mpc_rate_change|>0 | budget_event
     weather    : ABSENT as a clean series -> documented; gas_rv used as cold-snap proxy
2) Detection: do TVP / HMM latent regimes detect ObservableShock vs a simple observable?
   HMM  = 2-state Markov-switching (switching variance) on AA residual -> P(high-var state)
   TVP  = stochastic-vol proxy: expanding-standardised rolling resid vol -> P(high-vol)
   simple = brent_rv expanding-percentile (one observable)
   Report walk-forward AUC / balanced-acc / agreement vs ObservableShock.
3) Switched architecture: AA+BVAR in normal months, AA+MIDAS in ObservableShock months.
   Compare vs fixed AA+BVAR, equal-weight Stage-2 combo, AA-only. RMSE/MAE + DM by window.

Out: data/reg_detect/observable_shock.csv, shock_detection.csv, switched_architecture.csv
Run: PYTHONPATH=code .venv/bin/python -u code/reg_detect/observable_shock.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import factors as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DIR = os.path.join(_ROOT, "data", "reg_detect")
Q = 0.85          # exceedance quantile (per channel, causal)
MIN_HIST = 18     # months before a causal threshold is defined


def expanding_exceed(s, q=Q, min_hist=MIN_HIST):
    """1 where |s_t| exceeds the expanding q-quantile of |s| over data strictly before t."""
    a = s.abs()
    out = pd.Series(0.0, index=s.index)
    for i in range(len(a)):
        if i < min_hist or not np.isfinite(a.iloc[i]):
            continue
        thr = a.iloc[:i].quantile(q)
        out.iloc[i] = float(a.iloc[i] > thr)
    return out


def load_extra():
    """commodity + shipping monthly logret; documents availability."""
    avail = {}
    raw, status = F.build_matrix(names=["imf_all_commodity", "deep_sea_freight",
                                        "global_supply_chain_pressure", "budget_event"])
    raw = raw.resample("ME").last()
    out = pd.DataFrame(index=raw.index)
    if status.get("imf_all_commodity") != "unavailable" and "imf_all_commodity" in raw:
        out["commodity_mom"] = raw["imf_all_commodity"]; avail["commodity"] = "imf_all_commodity"
    ship = None
    for c in ["deep_sea_freight", "global_supply_chain_pressure"]:
        if status.get(c) != "unavailable" and c in raw and raw[c].notna().sum() > 30:
            out["shipping_mom"] = raw[c]; avail["shipping"] = c; ship = c; break
    if ship is None:
        avail["shipping"] = "ABSENT"
    out["budget_event"] = raw["budget_event"].fillna(0) if "budget_event" in raw else 0.0
    avail["weather"] = "ABSENT (no UK HDD/temperature series ingested; gas_rv used as cold-snap proxy)"
    return out, avail, status


def build_label(df):
    extra, avail, status = load_extra()
    d = df.join(extra, how="left")
    chan = {}
    # energy: a real directional move in oil or gas (RV/gas_rv reserved for weather proxy)
    en = (expanding_exceed(d["brent_mom"]) + expanding_exceed(d["gas_mom"]))
    chan["energy"] = (en > 0).astype(int)
    # commodity
    chan["commodity"] = expanding_exceed(d["commodity_mom"]).astype(int) if "commodity_mom" in d \
        else pd.Series(0, index=d.index)
    # shipping
    chan["shipping"] = expanding_exceed(d["shipping_mom"]).astype(int) if "shipping_mom" in d \
        else pd.Series(0, index=d.index)
    # regulatory
    reg = ((d.get("ofgem_flag", 0) > 0) | (d.get("mpc_rate_change", 0).abs() > 1e-9) |
           (d.get("budget_event", 0).abs() > 1e-9)).astype(int)
    chan["regulatory"] = reg
    # weather proxy (cold-snap gas vol) — flagged separately, NOT in core label
    chan["weather_proxy"] = expanding_exceed(d["gas_rv"]).astype(int)
    lab = pd.DataFrame(chan)
    lab["ObservableShock"] = ((lab["energy"] + lab["commodity"] + lab["shipping"] +
                               lab["regulatory"]) > 0).astype(int)
    lab.index = d.index
    return lab, avail


# ── Detectors of ObservableShock ─────────────────────────────────────────────
def hmm_highvar_prob(resid):
    """walk-forward P(high-variance regime) from 2-state Markov-switching on AA resid."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    out = pd.Series(np.nan, index=resid.index)
    yrs = sorted(resid.index.year.unique())
    for y in yrs:
        tr = resid[resid.index.year < y]; te = resid[resid.index.year == y]
        if len(tr) < 24 or len(te) == 0:
            continue
        try:
            res = MarkovRegression(tr.values, k_regimes=2, trend="c",
                                   switching_variance=True).fit(disp=False)
            sig2 = [res.params[res.model.param_names.index(f"sigma2[{i}]")] for i in range(2)]
            hi = int(np.argmax(sig2))
            smp = np.asarray(res.smoothed_marginal_probabilities)
            out.loc[te.index] = smp[-1, hi]
        except Exception:
            continue
    return out


def tvp_highvol_prob(resid, win=6):
    """TVP/stochastic-vol proxy: rolling resid vol, expanding-standardised -> logistic prob."""
    rv = resid.rolling(win, min_periods=3).std()
    out = pd.Series(np.nan, index=resid.index)
    for i in range(len(rv)):
        if i < MIN_HIST or not np.isfinite(rv.iloc[i]):
            continue
        hist = rv.iloc[:i].dropna()
        if len(hist) < 6:
            continue
        z = (rv.iloc[i] - hist.mean()) / (hist.std() + 1e-9)
        out.iloc[i] = 1.0 / (1.0 + np.exp(-z))
    return out


def simple_obs_prob(df):
    """single observable: brent_rv expanding-percentile rank."""
    a = df["brent_rv"]
    out = pd.Series(np.nan, index=a.index)
    for i in range(len(a)):
        if i < MIN_HIST or not np.isfinite(a.iloc[i]):
            continue
        hist = a.iloc[:i].dropna()
        out.iloc[i] = (hist < a.iloc[i]).mean()
    return out


def auc_safe(y, p):
    m = y.notna() & p.notna()
    if m.sum() < 10 or y[m].nunique() < 2:
        return np.nan
    return float(roc_auc_score(y[m], p[m]))


def bal_acc(y, p, thr=0.5):
    m = y.notna() & p.notna()
    if m.sum() < 5 or y[m].nunique() < 2:
        return np.nan
    yhat = (p[m] >= thr).astype(int); yy = y[m]
    tpr = ((yhat == 1) & (yy == 1)).sum() / max((yy == 1).sum(), 1)
    tnr = ((yhat == 0) & (yy == 0)).sum() / max((yy == 0).sum(), 1)
    return float(0.5 * (tpr + tnr))


# ── Switched architecture ────────────────────────────────────────────────────
def perf(err):
    e = np.asarray(err, float); e = e[np.isfinite(e)]
    return dict(rmse=float(np.sqrt((e**2).mean())), mae=float(np.abs(e).mean()), n=len(e))


def dm_test(e1, e2):
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    d = e1**2 - e2**2; d = d[np.isfinite(d)]; n = len(d)
    if n < 8 or np.allclose(d, 0):
        return np.nan, np.nan
    dbar = d.mean(); dd = d - dbar; g0 = (dd @ dd) / n; var = g0
    L = max(1, int(round(n ** (1/3))))
    for k in range(1, L + 1):
        var += 2 * (1 - k / (L + 1)) * ((dd[k:] @ dd[:-k]) / n)
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / n)
    from scipy.stats import t as tdist
    return float(stat), float(2 * (1 - tdist.cdf(abs(stat), df=n - 1)))


WINDOWS = {"full": lambda i: i.year >= 2018,
           "2022_23": lambda i: i.year.isin([2022, 2023]),
           "ex_shock": lambda i: (i.year >= 2018) & ~i.year.isin([2022, 2023]),
           "pre_2020": lambda i: (i.year >= 2018) & (i.year <= 2019)}


def main():
    df = pd.read_csv(os.path.join(_DIR, "targets.csv"), parse_dates=["date"]).set_index("date")
    lab, avail = build_label(df)
    out = df.join(lab)
    out.to_csv(os.path.join(_DIR, "observable_shock.csv"))
    print("channel availability:", avail)
    print(f"ObservableShock base rate (full) = {lab['ObservableShock'].mean():.3f}  "
          f"n_shock={int(lab['ObservableShock'].sum())}/{len(lab)}")
    for w, fn in WINDOWS.items():
        m = fn(lab.index)
        print(f"  {w:9} shock_rate={lab.loc[m,'ObservableShock'].mean():.3f}  "
              f"helpful_midas={out.loc[m,'helpful_midas'].mean():.3f}  "
              f"helpful_bvar={out.loc[m,'helpful_bvar'].mean():.3f}")

    # ---- Detection comparison (eval 2018+) ----
    resid = df["aa_err"]      # = actual - aa_pred = AA residual realised
    y = lab["ObservableShock"]
    det = {"HMM_highvar": hmm_highvar_prob(resid),
           "TVP_highvol": tvp_highvol_prob(resid),
           "simple_brent_rv": simple_obs_prob(df)}
    ev = y.index.year >= 2018
    rows = []
    for name, p in det.items():
        rows.append(dict(detector=name, auc=auc_safe(y[ev], p[ev]),
                         bal_acc=bal_acc(y[ev], p[ev]),
                         auc_exshock=auc_safe(y[ev & ~y.index.year.isin([2022, 2023])],
                                              p[ev & ~y.index.year.isin([2022, 2023])])))
    det_tbl = pd.DataFrame(rows).set_index("detector")
    det_tbl.to_csv(os.path.join(_DIR, "shock_detection.csv"))
    print("\n=== DETECTION of ObservableShock (latent vs simple observable) ===")
    print(det_tbl.round(3).to_string())

    # ---- Switched architecture ----
    shock = lab["ObservableShock"].astype(bool)
    bvar_p = df["bvar_pred"]; midas_p = df["midas_pred"]
    # switched: BVAR normal, MIDAS shock; fallback to BVAR if MIDAS missing
    sw_pred = bvar_p.copy()
    use_midas = shock & midas_p.notna()
    sw_pred[use_midas] = midas_p[use_midas]
    err_sw = df["actual"] - sw_pred
    systems = {
        "AA_only": df["aa_err"],
        "AA+BVAR_fixed": df["bvar_err"],
        "fixed_combo": df["stage2_err"],
        "switched_BVAR_MIDAS": err_sw,
    }
    comp = []
    for w, fn in WINDOWS.items():
        m = fn(df.index)
        if m.sum() < 5:
            continue
        row = dict(window=w, n=int(m.sum()), n_shock=int(shock[m].sum()))
        for nm, e in systems.items():
            row[f"rmse_{nm}"] = perf(e[m])["rmse"]
        # DM: switched vs each benchmark (stat<0 => switched better)
        for bench in ["AA+BVAR_fixed", "fixed_combo"]:
            s, p = dm_test(err_sw[m], systems[bench][m])
            row[f"dm_sw_vs_{bench}"] = s; row[f"p_sw_vs_{bench}"] = p
        comp.append(row)
    comp = pd.DataFrame(comp).set_index("window")
    comp.to_csv(os.path.join(_DIR, "switched_architecture.csv"))
    pd.options.display.width = 220
    print("\n=== SWITCHED ARCHITECTURE (RMSE; DM stat<0 => switched beats benchmark) ===")
    cols = ["n", "n_shock", "rmse_AA_only", "rmse_AA+BVAR_fixed", "rmse_fixed_combo",
            "rmse_switched_BVAR_MIDAS", "dm_sw_vs_AA+BVAR_fixed", "p_sw_vs_AA+BVAR_fixed",
            "dm_sw_vs_fixed_combo", "p_sw_vs_fixed_combo"]
    print(comp[cols].round(4).to_string())
    print("\nwritten observable_shock.csv, shock_detection.csv, switched_architecture.csv")


if __name__ == "__main__":
    main()
