"""
FROZEN PRODUCTION MODEL — UK CPI YoY nowcast.  (2026-06-19)

    Forecast = AA + lambda * Overlay,   Overlay = 0.5*TVP + 0.5*LGBM,   lambda = 0.5
    => Forecast = AutoARIMA + 0.25*TVP_resid + 0.25*LGBM_resid

GOVERNANCE: this is the single canonical production forecaster. The research phase is over;
do not modify weights/members/lambda here without a governance decision (see docs/final_model.md).
- Statistical optimum lambda ≈ 0.8 (overlay-shrinkage + Bayesian reliability audits).
- Production lambda = 0.5 (chosen): the overlay is informative but ~79% noise (R²≈0.21) and
  misfires in calm months (May-2026: AA 2.71 / actual 2.80 / experimental λ=1 → 3.111).
  λ=0.5 retains most historical edge (rel ~0.89) while halving calm-month magnitude risk.

Members (all from uk_model_zoo, factors from factors.py PINNED set):
  AA   = AutoARIMA            (anchor: persistence/seasonality/base-effects; ~96% of the level)
  TVP  = time-varying-param   (shock pass-through overlay; the diversifier)
  LGBM = LightGBM on AA resid (cost-pressure / PPI overlay; stable nonlinear)
NOT regime-switching / detector / HMM / release-day-updating / latent-state. (See governance.)

Run (prints current nowcast):
  set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/production/model.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
import lightgbm as lgb
import uk_model_zoo as Z, two_stage as TS

# ── FROZEN SPEC ──────────────────────────────────────────────────────────────
LAMBDA = 0.5
OVERLAY_WEIGHTS = {"tvp": 0.5, "lgbm": 0.5}          # within the overlay
AA_START, END = TS.AA_START, 2024                     # AA walk-forward / residual-train span
LGB = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
           min_child_samples=12, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=0, verbose=-1)
SPEC = "AA + 0.5*(0.5*TVP + 0.5*LGBM) = AA + 0.25*TVP + 0.25*LGBM"
# PINNED factors documented to always drop (not a real degradation when missing).
# global_supply_chain_pressure: ONS/NY-Fed xlsx fails engine detection — known, expected.
OPTIONAL_PINNED = {"global_supply_chain_pressure"}


def _residual_history(df):
    aa = Z.AutoARIMA().backtest(df, [], TS.TARGET, start_year=AA_START, end_year=END)
    return (aa["actual"] - aa["pred"]).rename("resid")


def nowcast(df=None, live=None):
    """Return the frozen production nowcast for the first unreleased CPI month.
    df/live default to the production factor matrix (two_stage.load_matrix).

    A PINNED factor that fails both its CSV drop-in and live fetch is silently
    dropped from `live` by two_stage.load_matrix, which degrades the TVP/LGBM
    overlay WITHOUT raising. We surface that here: the returned dict carries
    `missing_factors` / `n_live` / `degraded`, and any degradation is printed
    LOUDLY to stderr (the module-level warnings.filterwarnings("ignore") would
    otherwise swallow a warnings.warn). Forecast is still produced — caller decides."""
    if df is None:
        df, live, _ = TS.load_matrix()
    missing = [n for n in TS.PINNED if n not in (live or [])]
    # Only an UNEXPECTED drop counts as degradation; OPTIONAL_PINNED is known-to-drop.
    missing_required = [n for n in missing if n not in OPTIONAL_PINNED]
    if missing_required:
        print(f"[production.nowcast] ⚠ DEGRADED: {len(missing_required)} unexpected PINNED "
              f"factor(s) unavailable — overlay trained on a reduced set. Missing: {missing_required}",
              file=sys.stderr, flush=True)
    aa_pred, nd = Z.AutoARIMA().nowcast(df, [], TS.TARGET)
    resid = _residual_history(df).reindex(df.index)
    d = df.copy(); d["resid"] = resid
    tvp_resid, _ = Z.TVP().nowcast(d, live, "resid")
    data = df[live].join(resid).dropna(subset=["resid"])
    lgbm_resid = float(lgb.LGBMRegressor(**LGB).fit(data[live], data["resid"])
                       .predict(df.loc[[nd], live])[0])
    overlay = OVERLAY_WEIGHTS["tvp"] * float(tvp_resid) + OVERLAY_WEIGHTS["lgbm"] * lgbm_resid
    forecast = float(aa_pred) + LAMBDA * overlay
    return dict(nowcast_date=str(pd.Timestamp(nd).date()), spec=SPEC, lam=LAMBDA,
                aa=round(float(aa_pred), 3), tvp_resid=round(float(tvp_resid), 3),
                lgbm_resid=round(lgbm_resid, 3), overlay=round(overlay, 3),
                tvp_contribution=round(LAMBDA*OVERLAY_WEIGHTS["tvp"]*float(tvp_resid), 3),
                lgbm_contribution=round(LAMBDA*OVERLAY_WEIGHTS["lgbm"]*lgbm_resid, 3),
                forecast=round(forecast, 3),
                n_pinned=len(TS.PINNED), n_live=len(live or []),
                live_factors=list(live or []), missing_factors=missing,
                missing_required=missing_required, degraded=bool(missing_required))


def main():
    nc = nowcast()
    print("=== FROZEN PRODUCTION NOWCAST ===")
    print(f"  spec        : {nc['spec']}  (lambda={nc['lam']})")
    print(f"  month       : {nc['nowcast_date']}")
    print(f"  AutoARIMA   : {nc['aa']}")
    print(f"  + TVP   ({LAMBDA*OVERLAY_WEIGHTS['tvp']:.2f}) : {nc['tvp_contribution']:+.3f}  (resid {nc['tvp_resid']})")
    print(f"  + LGBM  ({LAMBDA*OVERLAY_WEIGHTS['lgbm']:.2f}) : {nc['lgbm_contribution']:+.3f}  (resid {nc['lgbm_resid']})")
    print(f"  FORECAST    : {nc['forecast']}")
    cov = f"{nc['n_live']}/{nc['n_pinned']} PINNED factors live"
    if nc["degraded"]:
        print(f"  ⚠ DEGRADED  : {cov} — unexpected missing {nc['missing_required']}")
    elif nc["missing_factors"]:
        print(f"  factors     : {cov} (expected-drop: {nc['missing_factors']})")
    else:
        print(f"  factors     : {cov}")


if __name__ == "__main__":
    main()
