"""
PART G/H — Architecture test + hostile review.

Three systems on CPI-YoY forecast error:
  A : AA only                         err = aa_err
  B : AA + Stage2 always              err = stage2_err
  C : AA + Stage2 only when gate>=thr err = gated

Gate = walk-forward detector P(helpful) (default: best observable detector).
Compare RMSE/MAE/OOS-corr across windows; Diebold-Mariano (HAC) on squared error.

Hostile review windows: full / 2022_23 / ex_shock / ex_covid / pre_2020.
Q: does gating beat both A and B? is the gain only 2022/23? survive pre-2020?

Out: data/reg_detect/architecture_comparison.csv, robustness.csv
Run: PYTHONPATH=code .venv/bin/python code/reg_detect/architecture.py
"""
import os, sys, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DIR = os.path.join(_ROOT, "data", "reg_detect")
EVAL_START = 2018


def dm_test(e1, e2, h=1):
    """Diebold-Mariano on squared-error loss diff d=e1^2-e2^2. HAC (Newey-West) var.
    Positive stat => system1 worse (loss1>loss2)."""
    e1 = np.asarray(e1, float); e2 = np.asarray(e2, float)
    d = e1**2 - e2**2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8 or np.allclose(d, 0):
        return np.nan, np.nan
    dbar = d.mean(); dd = d - dbar
    gamma0 = (dd @ dd) / n
    var = gamma0
    L = max(1, int(round(n ** (1/3))))
    for k in range(1, L + 1):
        gk = (dd[k:] @ dd[:-k]) / n
        var += 2 * (1 - k / (L + 1)) * gk
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / n)
    from scipy.stats import t as tdist
    pval = 2 * (1 - tdist.cdf(abs(stat), df=n - 1))
    return float(stat), float(pval)


def perf(err):
    err = np.asarray(err, float); err = err[np.isfinite(err)]
    return dict(rmse=float(np.sqrt((err**2).mean())), mae=float(np.abs(err).mean()), n=len(err))


WINDOWS = {
    "full":     lambda idx: idx.year >= EVAL_START,
    "2022_23":  lambda idx: idx.year.isin([2022, 2023]),
    "ex_shock": lambda idx: (idx.year >= EVAL_START) & ~idx.year.isin([2022, 2023]),
    "ex_covid": lambda idx: (idx.year >= EVAL_START) & ~idx.year.isin([2020, 2021]),
    "pre_2020": lambda idx: (idx.year >= EVAL_START) & (idx.year <= 2019),
}


def build_systems(df, gate_col, thr=0.5):
    actual = df["actual"]
    errA = df["aa_err"]
    errB = df["stage2_err"]
    use = (df[gate_col] >= thr)
    predC = np.where(use, df["stage2_pred"], df["aa_pred"])
    errC = actual - predC
    return errA, errB, pd.Series(errC, index=df.index), use


def main():
    df = pd.read_csv(os.path.join(_DIR, "targets_with_probs.csv"),
                     parse_dates=["date"]).set_index("date")
    df = df[(df.index.year >= EVAL_START) & df["aa_err"].notna() & df["stage2_err"].notna()]
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    # pick best observable gate by OOS AUC from detector_results
    det = pd.read_csv(os.path.join(_DIR, "detector_results.csv")).set_index("detector")
    obs_dets = [d for d in ["logit_l2", "logit", "gbt"] if f"p_{d}" in df.columns]
    gate = max(obs_dets, key=lambda d: det.loc[d, "auc"] if d in det.index else -1) if obs_dets else prob_cols[0][2:]
    gate_col = f"p_{gate}"
    print(f"gate detector = {gate}")

    # threshold sweep on full window to pick C threshold (then test all windows)
    rows = []
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        errA, errB, errC, use = build_systems(df, gate_col, thr)
        pA, pC = perf(errA), perf(errC)
        rows.append(dict(thr=thr, n_gated_on=int(use.sum()),
                         rmseA=pA["rmse"], rmseC=pC["rmse"], gain=pA["rmse"] - pC["rmse"]))
    thr_tbl = pd.DataFrame(rows)
    print("\nthreshold sweep (full):"); print(thr_tbl.round(4).to_string(index=False))
    best_thr = float(thr_tbl.sort_values("rmseC").iloc[0]["thr"])
    print(f"chosen C threshold = {best_thr}")

    # architecture comparison across windows
    comp = []
    rob = []
    for w, fn in WINDOWS.items():
        m = fn(df.index); sub = df[m]
        if len(sub) < 5:
            continue
        errA, errB, errC, use = build_systems(sub, gate_col, best_thr)
        pA, pB, pC = perf(errA), perf(errB), perf(errC)
        cA = np.corrcoef(sub["actual"], sub["aa_pred"])[0, 1]
        cB = np.corrcoef(sub["actual"], sub["stage2_pred"])[0, 1]
        cC = np.corrcoef(sub["actual"], sub["actual"] - errC)[0, 1]
        sCA, pCA = dm_test(errC, errA)   # C vs A : negative stat => C better
        sCB, pCB = dm_test(errC, errB)   # C vs B
        sBA, pBA = dm_test(errB, errA)   # B vs A
        comp.append(dict(window=w, n=len(sub), n_gated_on=int(use.sum()),
                         rmseA=pA["rmse"], rmseB=pB["rmse"], rmseC=pC["rmse"],
                         maeA=pA["mae"], maeB=pB["mae"], maeC=pC["mae"],
                         corrA=cA, corrB=cB, corrC=cC,
                         dm_CvA=sCA, p_CvA=pCA, dm_CvB=sCB, p_CvB=pCB,
                         dm_BvA=sBA, p_BvA=pBA))
        rob.append(dict(window=w, n=len(sub), helpful_rate=float(sub["helpful"].mean()),
                        meanSkillGain=float(sub["skillgain"].mean()),
                        rmse_gain_C_vs_A=pA["rmse"] - pC["rmse"],
                        rmse_gain_B_vs_A=pA["rmse"] - pB["rmse"]))
    comp = pd.DataFrame(comp).set_index("window")
    rob = pd.DataFrame(rob).set_index("window")
    comp.to_csv(os.path.join(_DIR, "architecture_comparison.csv"))
    rob.to_csv(os.path.join(_DIR, "robustness.csv"))
    pd.options.display.width = 200
    print("\n=== ARCHITECTURE COMPARISON (RMSE; DM stat<0 => C beats benchmark) ===")
    print(comp[["n", "n_gated_on", "rmseA", "rmseB", "rmseC",
                "dm_CvA", "p_CvA", "dm_CvB", "p_CvB", "dm_BvA", "p_BvA"]].round(4).to_string())
    print("\n=== ROBUSTNESS (where does Stage-2 / gating help?) ===")
    print(rob.round(4).to_string())
    print("\nwritten architecture_comparison.csv, robustness.csv")


if __name__ == "__main__":
    main()
