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

DUAL OUTPUT (2026-07-23): the pipeline is parameterized by `target` (targets.HEADLINE / targets.CORE).
main() prints BOTH the headline CPI nowcast and the Core CPI nowcast. Headline params/output are
unchanged (regression-locked, tests/test_headline_regression.py). Core uses its own pinned set +
calibration; when a target has no pinned overlay factors it prints AA-only.

Members (all from uk_model_zoo, factors from factors.py PINNED set):
  AA   = AutoARIMA            (anchor: persistence/seasonality/base-effects; ~96% of the level)
  TVP  = time-varying-param   (shock pass-through overlay; the diversifier)
  LGBM = LightGBM on AA resid (cost-pressure / PPI overlay; stable nonlinear)
NOT regime-switching / detector / HMM / release-day-updating / latent-state. (See governance.)

Run — ONLY the `quant` conda env, never .venv/base (slow imports); prints headline + core:
  set -a; . ./.env; set +a; PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python -u code/production/model.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
import lightgbm as lgb
import uk_model_zoo as Z, two_stage as TS
import validation as V
from targets import HEADLINE, CORE, Target

# ── FROZEN SPEC ──────────────────────────────────────────────────────────────
# λ / overlay weights / pinned set / backtest window are now carried by the Target
# (targets.py). HEADLINE mirrors the frozen values byte-for-byte. Shared, target-agnostic:
LGB = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, max_depth=3,
           min_child_samples=12, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
           random_state=0, verbose=-1)
# PINNED factors documented to always drop (not a real degradation when missing).
# global_supply_chain_pressure: ONS/NY-Fed xlsx fails engine detection — known, expected.
OPTIONAL_PINNED = {"global_supply_chain_pressure"}


def _spec(t: Target) -> str:
    if not t.pinned:
        return f"{t.name} = AutoARIMA (AA-only)"
    w = t.overlay_weights
    return (f"AA + {t.lam}*({w['tvp']}*TVP + {w['lgbm']}*LGBM) "
            f"= AA + {t.lam*w['tvp']:g}*TVP + {t.lam*w['lgbm']:g}*LGBM")


def _residual_history(df, target: Target = HEADLINE):
    aa = Z.AutoARIMA().backtest(df, [], target.name, start_year=target.aa_start, end_year=target.end)
    return (aa["actual"] - aa["pred"]).rename("resid")


def _lgbm_resid(df, live, resid, nd):
    """LGBM overlay on PURGED + EMBARGOED residual history, predicting month nd.

    The residual target (cpi_yoy, a 12-month difference) is autocorrelated:
    residuals within TS.PURGE_HORIZON months of nd share its YoY window. Purge that
    label horizon + a TS.EMBARGO-month gap so the most-recent overlapping residuals
    cannot leak regime-shift information into the overlay via autocorrelation.
    See validation.purge_embargo / docs/superpowers/specs."""
    data = pd.DataFrame(df[live]).join(resid.rename("resid")).dropna(subset=["resid"])
    train = V.purge_embargo(data, pd.Timestamp(nd), TS.PURGE_HORIZON, TS.EMBARGO)
    return float(lgb.LGBMRegressor(**LGB).fit(train[live], train["resid"])
                 .predict(df.loc[[nd], live])[0])


def nowcast(df=None, live=None, target: Target = HEADLINE):
    """Return the production nowcast for `target`'s first unreleased month.
    df/live default to the target's production factor matrix (two_stage.load_matrix).

    A PINNED factor that fails both its CSV drop-in and live fetch is silently
    dropped from `live` by two_stage.load_matrix, which degrades the TVP/LGBM
    overlay WITHOUT raising. We surface that here: the returned dict carries
    `missing_factors` / `n_live` / `degraded`, and any degradation is printed
    LOUDLY to stderr. When `target.pinned` is empty the nowcast is AA-only."""
    if df is None:
        df, live, _ = TS.load_matrix(target=target)
    missing = [n for n in target.pinned if n not in (live or [])]
    # Only an UNEXPECTED drop counts as degradation; OPTIONAL_PINNED is known-to-drop.
    missing_required = [n for n in missing if n not in OPTIONAL_PINNED]
    if missing_required:
        print(f"[production.nowcast:{target.name}] ⚠ DEGRADED: {len(missing_required)} unexpected "
              f"PINNED factor(s) unavailable — overlay trained on a reduced set. "
              f"Missing: {missing_required}", file=sys.stderr, flush=True)
    aa_pred, nd = Z.AutoARIMA().nowcast(df, [], target.name)

    base = dict(nowcast_date=str(pd.Timestamp(nd).date()), label=target.label,
                target=target.name, spec=_spec(target), lam=target.lam,
                aa=round(float(aa_pred), 3), n_pinned=len(target.pinned), pinned_n=len(target.pinned),
                n_live=len(live or []), live_factors=list(live or []),
                missing_factors=missing, missing_required=missing_required,
                degraded=bool(missing_required))

    if not target.pinned:                       # AA-only (e.g. core before an overlay is shipped)
        base.update(forecast=round(float(aa_pred), 3), tvp_resid=0.0, lgbm_resid=0.0, overlay=0.0,
                    tvp_contribution=0.0, lgbm_contribution=0.0)
        return base

    resid_full = _residual_history(df, target).reindex(df.index)
    # Overlays train on residual history extended to the latest actual, but EMBARGOED:
    # blank the trailing PURGE_HORIZON+EMBARGO months before nd so neither overlay trains
    # into the YoY-autocorrelated window. See validation.embargo_series.
    resid = V.embargo_series(resid_full, nd, TS.PURGE_HORIZON, TS.EMBARGO)
    d = df.copy(); d["resid"] = resid
    tvp_resid, _ = Z.TVP().nowcast(d, live, "resid")
    lgbm_resid = _lgbm_resid(df, live, resid, nd)
    ow = target.overlay_weights
    overlay = ow["tvp"] * float(tvp_resid) + ow["lgbm"] * lgbm_resid
    forecast = float(aa_pred) + target.lam * overlay
    base.update(tvp_resid=round(float(tvp_resid), 3), lgbm_resid=round(lgbm_resid, 3),
                overlay=round(overlay, 3),
                tvp_contribution=round(target.lam * ow["tvp"] * float(tvp_resid), 3),
                lgbm_contribution=round(target.lam * ow["lgbm"] * lgbm_resid, 3),
                forecast=round(forecast, 3))
    return base


def _print_block(nc):
    print(f"\n  {nc['label']}")
    print(f"    spec      : {nc['spec']}  (lambda={nc['lam']})")
    print(f"    month     : {nc['nowcast_date']}")
    print(f"    AutoARIMA : {nc['aa']}")
    if nc["n_pinned"]:
        print(f"    + TVP     : {nc['tvp_contribution']:+.3f}  (resid {nc['tvp_resid']})")
        print(f"    + LGBM    : {nc['lgbm_contribution']:+.3f}  (resid {nc['lgbm_resid']})")
        print(f"    FORECAST  : {nc['forecast']}")
        cov = f"{nc['n_live']}/{nc['n_pinned']} PINNED factors live"
        if nc["degraded"]:
            print(f"    ⚠ DEGRADED: {cov} — unexpected missing {nc['missing_required']}")
        elif nc["missing_factors"]:
            print(f"    factors   : {cov} (expected-drop: {nc['missing_factors']})")
        else:
            print(f"    factors   : {cov}")
    else:
        print(f"    FORECAST  : {nc['forecast']}   [AA-only — no overlay shipped for this target]")


def main():
    print("=== PRODUCTION NOWCAST (headline + core) ===")
    for tgt in (HEADLINE, CORE):
        try:
            _print_block(nowcast(target=tgt))
        except Exception as e:
            print(f"\n  {tgt.label}: FAILED — {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
