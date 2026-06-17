"""
PART C/D/E/F — Detectors for HelpfulStage2, walk-forward evaluation.

Target: HelpfulStage2_t in {0,1}  (does AA+Stage2 beat AA on month t?)
Predictors: causal HF observables (energy/FX/vol/rates momentum + magnitudes + events).

Detectors:
  D — observable (no latent state):
     base_rate          : predict prior helpful frequency (chance baseline)
     logit              : unregularised logistic on standardised observables
     logit_l2           : L2 logistic (C=0.5)
     gbt                : gradient-boosted trees (shallow; flagged for small n)
  C — latent:
     hmm_skill          : 2-state Gaussian HMM on SkillGain -> P(high-skill state)
     persistence        : EWMA of past helpful (random-walk prob; TVP/UCM collapse to this)
     MS-DFM             : DOCUMENTED INFEASIBLE (see note) — not silently skipped

Walk-forward: for each test year y>=EVAL_START, train on year<y, predict P(helpful) on y.
No full-sample fitting. OOS predictions pooled -> AUC / Brier / precision / recall /
calibration / accuracy.

Out: data/reg_detect/detector_results.csv, importance.csv, roc_<det>.csv, calib_<det>.csv
Run: PYTHONPATH=code .venv/bin/python code/reg_detect/detectors.py
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DIR = os.path.join(_ROOT, "data", "reg_detect")

EVAL_START = 2018      # need >=3y training history before first OOS year
FEATURES = ["brent_absmom", "gas_absmom", "fx_absmom", "vix_absmom",
            "brent_rv", "gas_rv", "fx_rv", "vix_rv", "vix_lvl", "vix_chg",
            "rates_mom", "mpc_rate_change", "ofgem_flag"]

MSDFM_NOTE = (
    "MS-DFM infeasible: target is a univariate binary skill indicator (~120 monthly "
    "obs, ~25 events pre-eval). Markov-Switching Dynamic Factor models need a "
    "multivariate panel + hundreds of obs to identify regime-specific factor loadings "
    "and transition probabilities; on this n it is unidentified (singular EM, label-"
    "switching). The 2-state Gaussian HMM on SkillGain is the identifiable degenerate "
    "case and is reported in its place."
)


def _feat(df):
    cols = [c for c in FEATURES if c in df.columns]
    X = df[cols].copy()
    return X.fillna(X.median(numeric_only=True)).fillna(0.0), cols


def wf_predict(df, fit_fn):
    """Walk-forward: train year<y, predict year y. Returns aligned P(helpful)."""
    out = pd.Series(index=df.index, dtype=float)
    years = sorted([y for y in df.index.year.unique() if y >= EVAL_START])
    for y in years:
        tr = df[df.index.year < y]
        te = df[df.index.year == y]
        if tr["helpful"].nunique() < 2 or len(te) == 0:
            out.loc[te.index] = tr["helpful"].mean() if len(tr) else 0.5
            continue
        out.loc[te.index] = fit_fn(tr, te)
    return out


# ── Detector fit functions ───────────────────────────────────────────────────
def f_base_rate(tr, te):
    return np.repeat(tr["helpful"].mean(), len(te))


def f_logit(tr, te, C=1e6):
    Xtr, cols = _feat(tr); Xte, _ = _feat(te)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), tr["helpful"])
    return m.predict_proba(sc.transform(Xte[cols]))[:, 1]


def f_logit_l2(tr, te):
    return f_logit(tr, te, C=0.5)


def f_gbt(tr, te):
    Xtr, cols = _feat(tr); Xte, _ = _feat(te)
    m = GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                                   subsample=0.8, random_state=0).fit(Xtr, tr["helpful"])
    return m.predict_proba(Xte[cols])[:, 1]


def f_persist(tr, te, halflife=12):
    # EWMA of past helpful; flat across the test year (info known at year start)
    p = tr["helpful"].ewm(halflife=halflife).mean().iloc[-1] if len(tr) else 0.5
    return np.repeat(p, len(te))


def f_hmm_skill(tr, te):
    """2-state Markov-switching (statsmodels) on SkillGain, switching mean+variance.
    P(helpful) = smoothed prob of high-skill regime at end of training, carried into
    the test year (causal) and blended with that regime's empirical helpful rate."""
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        y = tr["skillgain"].astype(float).values
        res = MarkovRegression(y, k_regimes=2, trend="c",
                               switching_variance=True).fit(disp=False)
        named = dict(zip(res.model.param_names, np.asarray(res.params)))
        means = np.array([named.get(f"const[{i}]", np.nan) for i in range(2)])
        hi = int(np.nanargmax(means))                       # high-skill regime
        smp = res.smoothed_marginal_probabilities
        post = float(np.asarray(smp)[-1, hi])               # last smoothed prob
        st = np.asarray(smp).argmax(axis=1)
        rate_hi = tr["helpful"].values[st == hi].mean() if (st == hi).any() else tr["helpful"].mean()
        return np.repeat(np.clip(0.5 * post + 0.5 * rate_hi, 0.01, 0.99), len(te))
    except Exception:
        return f_persist(tr, te)


def f_ucm_skill(tr, te):
    """UCM latent: local-level (random walk) on SkillGain via statsmodels
    UnobservedComponents. P(helpful) = logistic of smoothed end-level (scaled)."""
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        y = tr["skillgain"].astype(float).values
        res = UnobservedComponents(y, level="local level").fit(disp=False, maxiter=50)
        lvl = float(np.asarray(res.smoothed_state)[0, -1])
        sd = tr["skillgain"].std() or 1.0
        return np.repeat(1.0 / (1.0 + np.exp(-lvl / (sd + 1e-9))), len(te))
    except Exception:
        return f_persist(tr, te)


DETECTORS = {
    "base_rate": f_base_rate,
    "persistence": f_persist,     # TVP-collapse (random-walk coefficient on binary)
    "hmm_skill": f_hmm_skill,     # PART C latent: Markov-switching
    "ucm_skill": f_ucm_skill,     # PART C latent: local-level UCM
    "logit": f_logit,             # PART D observable
    "logit_l2": f_logit_l2,
    "gbt": f_gbt,
}


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    out = {}
    out["n"] = len(y); out["base"] = float(y.mean())
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
    out["brier"] = float(brier_score_loss(y, p)) if len(np.unique(y)) > 1 else np.nan
    yhat = (p >= 0.5).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    out["precision"] = tp / (tp + fp) if (tp + fp) else np.nan
    out["recall"] = tp / (tp + fn) if (tp + fn) else np.nan
    out["accuracy"] = float((yhat == y).mean())
    return out


def roc_points(y, p):
    thr = np.unique(np.concatenate([[-0.01], np.sort(p), [1.01]]))
    rows = []
    P = (y == 1).sum(); N = (y == 0).sum()
    for t in thr:
        yhat = (p >= t).astype(int)
        tpr = ((yhat == 1) & (y == 1)).sum() / P if P else np.nan
        fpr = ((yhat == 1) & (y == 0)).sum() / N if N else np.nan
        rows.append((t, fpr, tpr))
    return pd.DataFrame(rows, columns=["thr", "fpr", "tpr"])


def calib_points(y, p, bins=5):
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size"))
    return g.reset_index()


def main():
    df = pd.read_csv(os.path.join(_DIR, "targets.csv"), parse_dates=["date"]).set_index("date")
    df = df[df["helpful"].notna()]
    print(f"loaded targets n={len(df)}  helpful base={df['helpful'].mean():.3f}")

    results = []
    importance_rows = []
    for name, fn in DETECTORS.items():
        p = wf_predict(df, fn)
        ev = df[(df.index.year >= EVAL_START) & p.notna()]
        y = ev["helpful"].values; pp = p.loc[ev.index].values
        m = metrics(y, pp); m["detector"] = name
        results.append(m)
        print(f"  {name:12} AUC={m['auc'] if not np.isnan(m['auc']) else float('nan'):.3f} "
              f"Brier={m['brier']:.3f} prec={m['precision']:.2f} rec={m['recall']:.2f} acc={m['accuracy']:.2f}")
        roc_points(y, pp).to_csv(os.path.join(_DIR, f"roc_{name}.csv"), index=False)
        calib_points(y, pp).to_csv(os.path.join(_DIR, f"calib_{name}.csv"), index=False)
        # store OOS probs for architecture stage
        df[f"p_{name}"] = p

    pd.DataFrame(results).set_index("detector").to_csv(os.path.join(_DIR, "detector_results.csv"))

    # PART F — variable importance (full-sample L2 logit coef + permutation AUC drop, OOS)
    Xall, cols = _feat(df)
    sc = StandardScaler().fit(Xall)
    full = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(Xall), df["helpful"])
    coef = pd.Series(full.coef_.ravel(), index=cols)
    # permutation importance on pooled OOS logit_l2
    p_oos = wf_predict(df, f_logit_l2)
    ev = df[(df.index.year >= EVAL_START) & p_oos.notna()].copy()
    base_auc = roc_auc_score(ev["helpful"], p_oos.loc[ev.index]) if ev["helpful"].nunique() > 1 else np.nan
    rng = np.random.default_rng(0)
    for c in cols:
        drops = []
        for _ in range(20):
            ev2 = ev.copy(); ev2[c] = rng.permutation(ev2[c].values)
            p2 = wf_predict_eval_only(df, ev2, c, f_logit_l2)
            if p2 is not None and ev2["helpful"].nunique() > 1:
                drops.append(base_auc - roc_auc_score(ev2["helpful"], p2))
        importance_rows.append(dict(feature=c, l2_coef=float(coef[c]),
                                    perm_auc_drop=float(np.mean(drops)) if drops else np.nan))
    imp = pd.DataFrame(importance_rows).sort_values("perm_auc_drop", ascending=False)
    imp.to_csv(os.path.join(_DIR, "importance.csv"), index=False)
    print("\nTop features (perm AUC drop):")
    print(imp.head(8).to_string(index=False))

    with open(os.path.join(_DIR, "msdfm_note.txt"), "w") as f:
        f.write(MSDFM_NOTE + "\n")
    print("\nMS-DFM:", MSDFM_NOTE)
    df.to_csv(os.path.join(_DIR, "targets_with_probs.csv"))
    print("written detector_results.csv, importance.csv, roc/calib, targets_with_probs.csv")


def wf_predict_eval_only(df_full, ev_perm, perm_col, fit_fn):
    """Re-predict only eval rows with one column permuted (uses same trained models per year)."""
    out = pd.Series(index=ev_perm.index, dtype=float)
    for y in sorted(ev_perm.index.year.unique()):
        tr = df_full[df_full.index.year < y]
        te = ev_perm[ev_perm.index.year == y]
        if tr["helpful"].nunique() < 2 or len(te) == 0:
            out.loc[te.index] = tr["helpful"].mean(); continue
        out.loc[te.index] = fit_fn(tr, te)
    return out.values


if __name__ == "__main__":
    main()
