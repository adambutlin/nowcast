"""
rates/run_production.py — Part J. Production workflow, MODEL-switchable.

  python -m rates.run_production                       # uses config.MODEL (default HuberNet)
  RATES_MODEL=TVP python -m rates.run_production
  python -m rates.run_production --model Combined-Dynamic --target uk_2y_gilt_move --compare

Steps: load forecast -> build signal -> identify regime -> forecast repricing
-> position -> risk controls -> trade recommendation -> attribution.
"""

import os
import argparse
import pandas as pd

from . import config as C
from . import event_panel as EP
from . import regime as R
from . import prod_signal as PS
from . import production as P


def _build_panel(model):
    return EP.build_event_panel(my_model=model, save=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL, help=f"one of {C.MODELS}")
    ap.add_argument("--target", default=C.TARGET_PRIMARY)
    ap.add_argument("--compare", action="store_true", help="run all MODELS side by side")
    args = ap.parse_args()
    os.makedirs(C.PROD_DIR, exist_ok=True)

    print(f"=== PRODUCTION PIPELINE  model={args.model}  target={args.target} ===")

    # 1. load model forecast + 2/3. build signal + regime
    panel = _build_panel(args.model)
    panel = PS.build_signals(panel)
    print(f"[1-3] panel n={len(panel)}  anchor={panel.attrs.get('anchor_mode')}")
    print("      regime summary:")
    print(R.regime_summary(panel).round(3).to_string())

    # 4. forecast repricing
    bt, met = P.forecast_repricing(panel, target=args.target)
    if bt is None or len(bt) == 0:
        print("[4] repricing produced no rows — stop."); return
    print(f"[4] repricing OOS R2={met.get('oos_r2'):.4f}  sign_hit={met.get('sign_hit'):.3f}  n={met.get('n')}")

    # 5/6. positions + risk
    pos = P.build_positions(panel, bt, target=args.target)

    # 7. trade recommendation (latest event)
    last = pos.dropna(subset=["pred_move"]).iloc[-1]
    rec = dict(ref_month=str(pos.dropna(subset=["pred_move"]).index[-1].date()),
               model=args.model, target=args.target,
               regime=last.get("regime"), confidence=round(float(last.get("confidence", 0)), 3),
               pred_move_bp=round(float(last["pred_move"]), 2),
               position=round(float(last["position"]), 3),
               tradeable=bool(last.get("tradeable", False)),
               reason=last.get("reason", ""))
    print("\n[7] TRADE RECOMMENDATION:")
    for k, v in rec.items():
        print(f"      {k:14}: {v}")
    pd.Series(rec).to_csv(C.TRADE_REC_CSV)

    # 8. backtest metrics + attribution
    met_bt = P.backtest_metrics(pos)
    print("\n[8] PRODUCTION BACKTEST:")
    for k, v in met_bt.items():
        print(f"      {k:14}: {v}")
    attr = P.attribution(pos)
    print("\n      PnL by regime:")
    print(attr["by_regime"].round(2).to_string())
    print("\n      PnL by policy regime:")
    print(attr["by_policy"].round(2).to_string())

    pos.to_csv(C.PROD_BT_CSV)
    attr["by_regime"].to_csv(C.ATTRIB_CSV)

    if args.compare:
        print("\n=== MODEL COMPARISON (same pipeline, switched MODEL) ===")
        comp = P.run_model_comparison(_build_panel, target=args.target)
        print(comp.round(3).to_string())
        comp.to_csv(os.path.join(C.PROD_DIR, "model_comparison.csv"))


if __name__ == "__main__":
    main()
