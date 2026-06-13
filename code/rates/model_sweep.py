"""
rates/model_sweep.py — Parts B-G. Sweep every model in nowcast_cpi_backtest.csv
through the EXISTING guarded Stage 1, with gap = Model_i - AutoARIMA.

No new models are fit: this reuses the already-computed walk-forward preds and
the unmodified stage1.stage1_test (same walk-forward, HAC, OOS, guard).
"""

import os
import numpy as np
import pandas as pd

from . import config as C
from . import event_panel as EP
from . import stage1 as S1
from . import consensus as CN

OUT_DATA  = os.path.join(C.DATA, "model_sweep")
OUT_PLOTS = os.path.join(C.PLOTS, "model_sweep")
BASELINE  = "AutoARIMA"


def _load_preds():
    bt = pd.read_csv(C.BACKTEST_CSV, parse_dates=["date"])
    w = bt.pivot_table(index="date", columns="model", values="pred", aggfunc="first").sort_index()
    actual = (bt.pivot_table(index="date", columns="model", values="actual", aggfunc="first")
                .bfill(axis=1).iloc[:, 0].sort_index())
    return w, actual


def _panel_for(model_col, w, actual, baseline_col=BASELINE):
    """Build a minimal Stage-1 panel: gap = model - baseline, surprise = actual - baseline."""
    base = w[baseline_col]
    idx = w.index
    p = pd.DataFrame(index=idx)
    p.index.name = "ref_month"
    p["actual_cpi_mom"]        = actual.reindex(idx)
    p["my_nowcast"]            = w[model_col].reindex(idx)
    p["baseline_expectation"]  = base.reindex(idx)
    p["my_surprise"]           = p["my_nowcast"] - p["baseline_expectation"]
    p["actual_surprise"]       = p["actual_cpi_mom"] - p["baseline_expectation"]
    rds = [EP.cpi_release_date(m) for m in idx]
    lo, hi = pd.Timestamp(C.LDI_WINDOW[0]), pd.Timestamp(C.LDI_WINDOW[1])
    p["ldi_event"] = [int(lo <= pd.Timestamp(rd) <= hi) for rd in rds]
    p.attrs["anchor_mode"] = "autoarima"
    return p.dropna(subset=["my_surprise", "actual_surprise"])


def _row(name, r):
    return dict(model=name, verdict=r.get("verdict"), n=r.get("n"),
                b=r.get("b"), t_HAC=r.get("t_b_HAC"),
                oos_corr=r.get("oos_corr"), oos_r2=r.get("oos_r2"),
                sign_hit=r.get("oos_sign_hit"),
                mechanical=r.get("mechanical_identity"),
                gap_level_corr=r.get("gap_vs_forecast_level_corr"))


def sweep(save=True):
    os.makedirs(OUT_DATA, exist_ok=True); os.makedirs(OUT_PLOTS, exist_ok=True)
    w, actual = _load_preds()
    # every model except the baseline itself
    cols = [c for c in w.columns if c != BASELINE]

    # ── Part B: full-sample Stage 1 per model ────────────────────────────────
    rows, robust_rows = [], []
    for m in cols:
        p = _panel_for(m, w, actual)
        if len(p) < 30:
            rows.append(dict(model=m, verdict="TOO_SMALL", n=len(p))); continue
        r = S1.stage1_test(p, plot=False)
        rows.append(_row(m, r))
        # ── Part C: robustness subperiods ────────────────────────────────────
        rob = CN.robustness(p, lambda pn: S1.stage1_test(pn, plot=False))
        for win, rr in rob.iterrows():
            robust_rows.append(dict(model=m, window=win, verdict=rr.get("verdict"),
                                    n=rr.get("n"), b=rr.get("b"), t_HAC=rr.get("t_b_HAC"),
                                    oos_corr=rr.get("oos_corr")))

    sweep_df = (pd.DataFrame(rows).set_index("model")
                .sort_values("oos_r2", ascending=False, na_position="last"))
    robust_df = pd.DataFrame(robust_rows)

    # ── Part D: attribution vs Combined-Static gap ───────────────────────────
    attrib = _attribution(w, actual)

    if save:
        sweep_df.to_csv(os.path.join(OUT_DATA, "stage1_sweep.csv"))
        robust_df.to_csv(os.path.join(OUT_DATA, "stage1_robustness.csv"), index=False)
        attrib.to_csv(os.path.join(OUT_DATA, "attribution.csv"))
        _plots(sweep_df, w, actual)
    return sweep_df, robust_df, attrib


def _members(w, actual):
    """Reconstruct Combined-Static membership: models beating AR(1) on common sample."""
    common = w.dropna(subset=["AR(1)"]).index
    def rmse(col):
        d = (actual.reindex(common) - w[col].reindex(common)).dropna()
        return np.sqrt((d**2).mean()) if len(d) else np.inf
    ar1 = rmse("AR(1)")
    exclude = {"AR(1)", "Combined-Static", "Combined-Dynamic", "Combined-Absolute"}
    return [c for c in w.columns if c not in exclude and rmse(c) < ar1]


def _attribution(w, actual):
    """Corr of each member gap with the ensemble gap + leave-one-out Stage-1 OOS R²."""
    members = _members(w, actual)
    ens_gap = (w["Combined-Static"] - w[BASELINE])
    rows = []
    for m in members:
        gap = (w[m] - w[BASELINE])
        c = gap.corr(ens_gap)
        # leave-one-out ensemble (mean of remaining members), Stage 1 OOS R²
        loo_cols = [x for x in members if x != m]
        loo = w[loo_cols].mean(axis=1)
        p = _panel_for_series(loo, w, actual)
        r = S1.stage1_test(p, plot=False) if len(p) >= 30 else {}
        rows.append(dict(member=m, corr_gap_with_ensemble=round(float(c), 3),
                         loo_oos_r2=r.get("oos_r2"), loo_t=r.get("t_b_HAC")))
    return pd.DataFrame(rows).set_index("member")


def _panel_for_series(series, w, actual, baseline_col=BASELINE):
    p = pd.DataFrame(index=w.index); p.index.name = "ref_month"
    p["actual_cpi_mom"] = actual.reindex(w.index)
    p["my_nowcast"] = series.reindex(w.index)
    p["baseline_expectation"] = w[baseline_col].reindex(w.index)
    p["my_surprise"] = p["my_nowcast"] - p["baseline_expectation"]
    p["actual_surprise"] = p["actual_cpi_mom"] - p["baseline_expectation"]
    rds = [EP.cpi_release_date(m) for m in w.index]
    lo, hi = pd.Timestamp(C.LDI_WINDOW[0]), pd.Timestamp(C.LDI_WINDOW[1])
    p["ldi_event"] = [int(lo <= pd.Timestamp(rd) <= hi) for rd in rds]
    p.attrs["anchor_mode"] = "autoarima"
    return p.dropna(subset=["my_surprise", "actual_surprise"])


def _plots(sweep_df, w, actual):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = sweep_df["oos_r2"].dropna().sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(d.index, d.values)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("Stage 1 OOS R²  (gap = model − AutoARIMA)")
        ax.set_title("Model-level incremental-information sweep")
        fig.tight_layout(); fig.savefig(os.path.join(OUT_PLOTS, "oos_r2_by_model.png"), dpi=110)
        plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    s, r, a = sweep()
    print(s.to_string())
