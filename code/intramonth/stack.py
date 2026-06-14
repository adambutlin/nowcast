"""
intramonth/stack.py — multilayer, switchable model stack (Part C).

Layer 1  baseline   : AutoARIMA (univariate persistence)            -> baseline CPI path
Layer 2  factor     : BVAR on monthly factors + HF as-of features   -> residual
Layer 3  regime_tvp : TVP  on monthly factors + HF as-of features   -> residual
Layer 4  intramonth : MIDAS — regularized regression on HF as-of    -> residual
                      features only (U-MIDAS on partial-month aggregates)
Layer 5  regime     : HMM detector (regime.py) — consumed by weights/scenarios

Residual framework (config.RESIDUAL_FRAMEWORK): layers 2-4 predict r = target - baseline,
so reconstructed forecast = baseline + residual_model, and each model's CPI RMSE = its
residual RMSE. Every model is run independently and through the same walk-forward, and
each returns: a forecast path, an uncertainty estimate (rolling RMSE), and a contribution
score (mean |residual prediction| = how far it moves off the baseline).

Switchable: STACK maps layer -> zoo class name; change config/env to swap a layer.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uk_model_zoo as Z
from intramonth import config as C, panel as P

RESID = "intra_resid"

# layer -> feature policy ("all" = monthly+HF, "hf" = HF only, "none" = univariate)
_FEATURE_POLICY = {"baseline": "none", "factor": "all", "regime_tvp": "all",
                   "intramonth": "hf"}


def _zoo_class(name):
    """Resolve a model name to a uk_model_zoo class. 'MIDAS' -> U-MIDAS (ElasticNet on HF)."""
    alias = {"MIDAS": "ElasticNet"}     # U-MIDAS = regularized reg on HF aggregates
    cls = getattr(Z, alias.get(name, name), None)
    if cls is None:
        raise ValueError(f"model {name!r} not in uk_model_zoo")
    return cls


def _feats(layer, meta):
    pol = _FEATURE_POLICY[layer]
    if pol == "none":
        return []
    if pol == "hf":
        return [c for c in meta["hf_cols"] if c != "hf_coverage"]
    return P.factor_columns(meta)       # monthly + HF


def _rolling_rmse(err, window=24):
    """Causal rolling RMSE of a reconstructed-error series."""
    return float(np.sqrt((err.dropna().tail(window) ** 2).mean())) if len(err.dropna()) else np.nan


class ModelStack:
    """Runs all configured layers walk-forward at one origin and returns predictions."""

    def __init__(self, panel, meta, start_year=None, end_year=None, stack=None):
        self.panel = panel.copy()
        self.meta = meta
        self.target = meta["target"]
        self.start_year = start_year or C.TRAIN_FROM + 4
        self.end_year = end_year
        self.stack = stack or C.STACK

    # ── Layer 1: baseline ────────────────────────────────────────────────────
    def baseline(self):
        aa = _zoo_class(self.stack["baseline"])()
        sub = self.panel[[self.target]].dropna()
        bt = aa.backtest(self.panel, [], self.target,
                         start_year=C.AA_START, end_year=self.end_year)
        return bt   # columns actual, pred (CPI level), index=date

    # ── Layers 2-4: residual models ──────────────────────────────────────────
    def run(self):
        return self.run_with_baseline(self.baseline())

    def run_with_baseline(self, aa_bt):
        """Run residual layers given a precomputed (origin-invariant) baseline."""
        out = {"target": self.target, "origin_k": self.meta["origin_k"]}
        self._baseline_bt = aa_bt
        out["baseline"] = aa_bt
        aa_pred = aa_bt["pred"]
        # residual = target - baseline (causal, defined where baseline exists)
        resid = (aa_bt["actual"] - aa_bt["pred"]).reindex(self.panel.index)
        self.panel[RESID] = resid if C.RESIDUAL_FRAMEWORK else self.panel[self.target]
        tgt = RESID if C.RESIDUAL_FRAMEWORK else self.target

        models = {}
        for layer in ["factor", "regime_tvp", "intramonth"]:
            name = self.stack[layer]
            cls = _zoo_class(name)
            feats = _feats(layer, self.meta)
            try:
                bt = cls().backtest(self.panel, feats, tgt,
                                    start_year=self.start_year, end_year=self.end_year)
                bt = bt if (bt is not None and len(bt)) else None
            except Exception as e:
                print(f"  [stack] {layer}/{name} failed: {str(e)[:60]}")
                bt = None
            if bt is None:
                models[layer] = None
                continue
            # reconstruct CPI forecast = baseline + residual (or direct)
            recon = bt.copy()
            if C.RESIDUAL_FRAMEWORK:
                recon["cpi_pred"] = aa_pred.reindex(bt.index) + bt["pred"]
                recon["cpi_actual"] = aa_pred.reindex(bt.index) + bt["actual"]
            else:
                recon["cpi_pred"] = bt["pred"]; recon["cpi_actual"] = bt["actual"]
            recon["err"] = recon["cpi_actual"] - recon["cpi_pred"]
            recon["resid_pred"] = bt["pred"]
            models[layer] = dict(name=name, bt=recon,
                                 rmse=float(np.sqrt((recon["err"] ** 2).mean())),
                                 contribution=float(recon["resid_pred"].abs().mean()))
        out["models"] = models

        # baseline as its own "layer" for comparison (residual=0)
        aa_err = aa_bt["actual"] - aa_bt["pred"]
        out["baseline_rmse"] = float(np.sqrt((aa_err.reindex(
            aa_bt[aa_bt.index.year >= self.start_year].index) ** 2).mean()))
        out["resid_std"] = float(resid.std())
        return out

    # ── Live nowcast at this origin (one forward month) ──────────────────────
    def nowcast(self):
        """Per-model point nowcast for the next unreleased month at this origin."""
        ext, nd = P.extend_to_nowcast(self.panel, self.meta, self.meta["origin_k"])
        # baseline forecast for nd
        aa = _zoo_class(self.stack["baseline"])()
        aa_now, _ = aa.nowcast(ext, [], self.target)
        resid_known = (self.baseline()["actual"] - self.baseline()["pred"])
        ext[RESID] = resid_known.reindex(ext.index)
        preds = {"baseline": aa_now, "_nowcast_date": nd}
        for layer in ["factor", "regime_tvp", "intramonth"]:
            cls = _zoo_class(self.stack[layer]); feats = _feats(layer, self.meta)
            try:
                r, _ = cls().nowcast(ext, feats, RESID if C.RESIDUAL_FRAMEWORK else self.target)
                preds[layer] = (aa_now + r) if (C.RESIDUAL_FRAMEWORK and np.isfinite(r)) else r
                preds[layer + "_resid"] = r
            except Exception:
                preds[layer] = np.nan; preds[layer + "_resid"] = np.nan
        return preds


if __name__ == "__main__":
    pan, meta = P.build_panel(C.DEFAULT_TARGET, k=1)
    st = ModelStack(pan, meta, end_year=2024)
    res = st.run()
    print(f"target={res['target']} baseline_rmse={res['baseline_rmse']:.3f} "
          f"resid_std={res['resid_std']:.3f}")
    for layer, m in res["models"].items():
        if m:
            print(f"  {layer:11} {m['name']:10} rmse={m['rmse']:.3f} contrib={m['contribution']:.3f}")
