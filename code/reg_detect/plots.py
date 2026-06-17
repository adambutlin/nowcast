"""PART J — plots: ROC curves, calibration, skill-gain. Out: plots/reg_detect/."""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DIR = os.path.join(_ROOT, "data", "reg_detect")
_PLOT = os.path.join(_ROOT, "plots", "reg_detect")
os.makedirs(_PLOT, exist_ok=True)
DETS = ["logit_l2", "logit", "gbt", "hmm_skill", "ucm_skill", "persistence"]


def roc():
    plt.figure(figsize=(6, 6))
    res = pd.read_csv(os.path.join(_DIR, "detector_results.csv")).set_index("detector")
    for d in DETS:
        f = os.path.join(_DIR, f"roc_{d}.csv")
        if not os.path.exists(f):
            continue
        r = pd.read_csv(f).sort_values("fpr")
        auc = res.loc[d, "auc"] if d in res.index else np.nan
        plt.plot(r["fpr"], r["tpr"], label=f"{d} (AUC={auc:.2f})", lw=1.6)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("HelpfulStage2 detectors — OOS ROC (2018-24)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(_PLOT, "roc_curves.png"), dpi=110); plt.close()


def calib():
    plt.figure(figsize=(6, 6))
    for d in DETS:
        f = os.path.join(_DIR, f"calib_{d}.csv")
        if not os.path.exists(f):
            continue
        c = pd.read_csv(f)
        plt.plot(c["p_mean"], c["y_rate"], "o-", label=d, ms=4, lw=1.2)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    plt.xlabel("mean predicted P(helpful)"); plt.ylabel("observed helpful rate")
    plt.title("Calibration (OOS)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(_PLOT, "calibration.png"), dpi=110); plt.close()


def skillgain():
    t = pd.read_csv(os.path.join(_DIR, "targets.csv"), parse_dates=["date"]).set_index("date")
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    col = ["#c0392b" if v < 0 else "#27ae60" for v in t["skillgain"]]
    ax[0].bar(t.index, t["skillgain"], width=20, color=col)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("SkillGain_t = |AA_err| - |Stage2_err|  (green=Stage-2 helped)")
    ax[0].set_ylabel("skill gain (pp)")
    ax[0].axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"),
                  color="orange", alpha=0.12, label="2022/23 shock")
    ax[0].legend(fontsize=8)
    ax[1].plot(t.index, t["aa_err"].abs(), label="|AA err|", lw=1.2)
    ax[1].plot(t.index, t["stage2_err"].abs(), label="|Stage2 err|", lw=1.2)
    ax[1].axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31"), color="orange", alpha=0.12)
    ax[1].set_ylabel("abs error (pp)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(_PLOT, "skill_gain.png"), dpi=110); plt.close()


if __name__ == "__main__":
    roc(); calib(); skillgain()
    print("plots ->", _PLOT, sorted(os.listdir(_PLOT)))
