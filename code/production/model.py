"""
FROZEN PRODUCTION MODEL — UK CPI YoY nowcast.  (2026-06-19)

    Forecast = AA + lambda * Overlay + RegEvent,   Overlay = 0.5*TVP + 0.5*LGBM,  lambda = 0.5
    => Forecast = AutoARIMA + 0.25*TVP_resid + 0.25*LGBM_resid + RegEvent

RegEvent (added 2026-08-19, headline only) is the announced Ofgem price-cap reset priced
into CPI arithmetically — see production/reg_events.py. It is zero in the ~3 months out
of 4 with no reset, and core CPI is ex-energy so it never applies there.

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
import factors as F
from production import reg_events as REG
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
    reg = " + RegEvent" if t.reg_event_layer else ""
    return (f"AA + {t.lam}*({w['tvp']}*TVP + {w['lgbm']}*LGBM){reg} "
            f"= AA + {t.lam*w['tvp']:g}*TVP + {t.lam*w['lgbm']:g}*LGBM{reg}")


def _residual_history(df, target: Target = HEADLINE):
    aa = Z.AutoARIMA().backtest(df, [], target.name, start_year=target.aa_start, end_year=target.end)
    return (aa["actual"] - aa["pred"]).rename("resid")


def _tvp_resid(df, live, resid, nd):
    """TVP overlay on the residual history, predicting month nd.

    Left to infer its own target month, BaseModel._nowcast_row takes the first
    month after the last non-NaN residual. That is correct for an unembargoed
    residual column and WRONG for an embargoed one (it would point the TVP at
    the first embargoed month and feed it THAT month's factor row — a 13-month
    stale forecast presented as the current-month overlay; fixed 2026-08-19, see
    docs/postmortem_2026_07.md §3). Pin the month explicitly either way, exactly
    as _lgbm_resid pins its prediction row."""
    d = df.copy()
    d["resid"] = resid
    pred, _ = Z.TVP().nowcast(d, live, "resid", nowcast_date=nd)
    return float(pred)


def _lgbm_resid(df, live, resid, nd, purge_horizon=0, embargo=0):
    """LGBM overlay on the residual history, predicting month nd.

    `purge_horizon`/`embargo` drop training rows within that many months of nd.
    They default to OFF: in a live nowcast every residual up to the last released
    CPI month is genuinely observable, so purging them discards real information
    (governance decision 2026-08-19; two_stage.py has always documented the purge
    as backtest-only). A WALK-FORWARD BACKTEST must pass TS.PURGE_HORIZON /
    TS.EMBARGO, because there the residual target (cpi_yoy, a 12-month difference)
    is autocorrelated: training rows within 12 months of a test month share its
    YoY window and leak regime-shift information into the overlay.
    See validation.purge_embargo / docs/superpowers/specs."""
    data = pd.DataFrame(df[live]).join(resid.rename("resid")).dropna(subset=["resid"])
    train = V.purge_embargo(data, pd.Timestamp(nd), purge_horizon, embargo)
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

    # Freshness guard. The overlays paper over a dead source — LightGBM routes NaN
    # silently, the TVP forward-fills within its budget — so staleness has to be
    # surfaced here; this is what hid a missing uk_ppi_input and a zeroed Ofgem cap
    # step in July-2026 (docs/postmortem_2026_07.md §3). Two tiers:
    #   event/regulatory series must cover nd exactly — "no cap reset this month" is a
    #     recorded 0, never an absence, so a gap is a stale source, not a quiet month;
    #   everything else gets factors.factor_health's pub_lag + grace budget, which is
    #     what keeps a quarterly series (uk_quarterly_gdp) from flagging mid-quarter.
    stale = [f for f in target.reg
             if f in (live or []) and f in df.columns and pd.isna(df.loc[nd, f])]
    paced = [f for f in (live or []) if f not in target.reg and f in df.columns]
    if paced:
        health = F.factor_health(df, paced, asof=nd)
        stale += [f for f, r in health.iterrows() if r["status"] != "LIVE"]
    if stale:
        print(f"[production.nowcast:{target.name}] ⚠ STALE: no observation at "
              f"{pd.Timestamp(nd).date()} for {stale} — their source ends earlier; the "
              f"overlay is guessing for these factors.", file=sys.stderr, flush=True)

    base = dict(nowcast_date=str(pd.Timestamp(nd).date()), label=target.label,
                target=target.name, spec=_spec(target), lam=target.lam,
                aa=round(float(aa_pred), 3), n_pinned=len(target.pinned), pinned_n=len(target.pinned),
                n_live=len(live or []), live_factors=list(live or []),
                missing_factors=missing, missing_required=missing_required,
                degraded=bool(missing_required),
                stale_factors=stale, stale=bool(stale))

    if not target.pinned:                       # AA-only (e.g. core before an overlay is shipped)
        base.update(forecast=round(float(aa_pred), 3), tvp_resid=0.0, lgbm_resid=0.0, overlay=0.0,
                    tvp_contribution=0.0, lgbm_contribution=0.0)
        return base

    # Live path: overlays train on the FULL residual history, up to the last released
    # CPI month. The purge/embargo window is a backtest device (it stops a test month's
    # YoY window leaking into training); applied here it would only discard residuals a
    # real-time analyst can legitimately see. Governance decision 2026-08-19 — see
    # docs/final_model.md §11. two_stage.backtest still purges.
    resid = _residual_history(df, target).reindex(df.index)
    tvp_resid = _tvp_resid(df, live, resid, nd)
    lgbm_resid = _lgbm_resid(df, live, resid, nd)
    ow = target.overlay_weights
    # A leg can legitimately fail to produce a number: BaseModel._nowcast_row blocks
    # the TVP outright when any PINNED factor is past its ffill budget, which happens
    # routinely to the quarterly uk_quarterly_gdp late in a quarter. Weighting NaN into
    # the sum turned that into a NaN FORECAST (found on the August-2026 roll). Drop the
    # dead legs and renormalise over what survives, as two_stage._weighted_combo does
    # for the backtest; if both legs die the nowcast falls back to AA (+ RegEvent).
    legs = {"tvp": float(tvp_resid), "lgbm": float(lgbm_resid)}
    usable = {k: v for k, v in legs.items() if np.isfinite(v)}
    dead = sorted(set(legs) - set(usable))
    wsum = sum(ow[k] for k in usable)
    overlay = sum(ow[k] * v for k, v in usable.items()) / wsum if wsum else 0.0
    if dead:
        print(f"[production.nowcast:{target.name}] ⚠ OVERLAY LEG DEAD: {dead} returned no "
              f"value for {pd.Timestamp(nd).date()} — overlay renormalised over "
              f"{sorted(usable) or 'nothing (AA only)'}.", file=sys.stderr, flush=True)
    # Regulatory-event layer: a cap reset is announced arithmetic, not something
    # the overlay can learn from 23 nonzero months. Zero in non-reset months.
    step, step_detail = (REG.reg_event_step(nd) if target.reg_event_layer else (0.0, dict(reason="off")))
    forecast = float(aa_pred) + target.lam * overlay + step
    base.update(tvp_resid=round(legs["tvp"], 3), lgbm_resid=round(legs["lgbm"], 3),
                overlay=round(overlay, 3), dead_legs=dead,
                tvp_contribution=round(target.lam * ow["tvp"] * legs["tvp"] / wsum
                                       if wsum and "tvp" in usable else 0.0, 3),
                lgbm_contribution=round(target.lam * ow["lgbm"] * legs["lgbm"] / wsum
                                        if wsum and "lgbm" in usable else 0.0, 3),
                reg_event=round(float(step), 3), reg_event_detail=step_detail,
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
        if nc.get("reg_event"):
            d = nc["reg_event_detail"]
            print(f"    + RegEvent: {nc['reg_event']:+.3f}  (Ofgem cap {d['cap_pct']:+.2f}%, "
                  f"beta {d['beta']}, kappa {d['kappa']})")
        print(f"    FORECAST  : {nc['forecast']}")
        cov = f"{nc['n_live']}/{nc['n_pinned']} PINNED factors live"
        if nc["degraded"]:
            print(f"    ⚠ DEGRADED: {cov} — unexpected missing {nc['missing_required']}")
        elif nc["missing_factors"]:
            print(f"    factors   : {cov} (expected-drop: {nc['missing_factors']})")
        else:
            print(f"    factors   : {cov}")
        if nc.get("stale_factors"):
            print(f"    ⚠ STALE   : no {nc['nowcast_date']} observation for "
                  f"{nc['stale_factors']}")
    else:
        print(f"    FORECAST  : {nc['forecast']}   [AA-only — no overlay shipped for this target]")


_RUNS_DIR = os.path.join(os.path.dirname(_CODE), "data", "production", "runs")


def _git_sha():
    try:
        import subprocess
        return subprocess.run(["git", "-C", _CODE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def write_manifest(blocks, path=None):
    """Persist everything a post-mortem needs about one run.

    Reconstructing what a past nowcast actually saw is otherwise impossible: the
    factor matrix is fetched live and is revised and extended continuously, so a
    replay months later reads different inputs (docs/postmortem_2026_07.md).
    One JSON per target-month: the nowcast dict, the factor row the overlays were
    given, each factor's source status and last observation, and the code version.
    """
    import json
    os.makedirs(_RUNS_DIR, exist_ok=True)
    written = []
    for b in blocks:
        nc, df, status = b["nowcast"], b["matrix"], b["status"]
        nd = pd.Timestamp(nc["nowcast_date"])
        cols = [c for c in df.columns if c != nc["target"]]
        row = df.loc[nd, cols] if nd in df.index else pd.Series(dtype=float)
        payload = dict(
            run_utc=pd.Timestamp.utcnow().isoformat(),
            git_sha=_git_sha(),
            nowcast=nc,
            lgb_params=LGB,
            factor_row={c: (None if pd.isna(v) else float(v)) for c, v in row.items()},
            factor_status={c: status.get(c) for c in cols},
            factor_last_obs={c: (str(df[c].dropna().index[-1].date())
                                 if df[c].notna().any() else None) for c in cols},
        )
        out = path or os.path.join(_RUNS_DIR, f"{nc['target']}_{nd:%Y-%m}.json")
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        written.append(out)
    return written


def main():
    print("=== PRODUCTION NOWCAST (headline + core) ===")
    try:                       # keep the Ofgem cap series current before anything reads it
        added = REG.refresh_cap_csv()
        if added:
            print(f"  [ofgem] cap series extended: {added}")
    except Exception as e:
        print(f"  [ofgem] cap refresh failed ({type(e).__name__}: {e}) — using the CSV as-is")
    blocks = []
    for tgt in (HEADLINE, CORE):
        try:
            df, live, status = TS.load_matrix(target=tgt)
            nc = nowcast(df, live, tgt)
            _print_block(nc)
            blocks.append(dict(nowcast=nc, matrix=df, status=status))
        except Exception as e:
            print(f"\n  {tgt.label}: FAILED — {type(e).__name__}: {e}")
    for out in write_manifest(blocks):
        print(f"\n  manifest  : {os.path.relpath(out, os.path.dirname(_CODE))}")


if __name__ == "__main__":
    main()
