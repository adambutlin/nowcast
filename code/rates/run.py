"""
rates/run.py — orchestrator. Build panel -> Gate 1 -> Gate 2 -> (PASS only)
MVP -> trading signal. Downstream stages are HARD-GATED on Gate 2 PASS.

Usage:
  .venv/bin/python -m rates.run                 # real data (needs daily rates; UCL optional)
  .venv/bin/python -m rates.run --synthetic     # demo on planted incremental signal
  .venv/bin/python -m rates.run --target uk_2y_gilt_move
"""

import argparse
import pandas as pd

from . import config as C
from . import event_panel as EP
from . import gates as G
from . import mvp as M
from . import signal as SIG
from . import synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="run on a synthetic panel with planted incremental signal")
    ap.add_argument("--null", action="store_true",
                    help="synthetic panel with NO incremental signal (gate must FAIL)")
    ap.add_argument("--target", default=C.PRIMARY_MOVE)
    ap.add_argument("--my-model", default=None, help="which backtest model = my nowcast")
    args = ap.parse_args()

    if args.synthetic or args.null:
        panel = synth.make_synthetic_panel(incremental=not args.null)
        print(f"[synthetic panel] incremental={not args.null}  n={len(panel)}")
    else:
        panel = EP.build_event_panel(my_model=args.my_model, save=True)
        print(f"[event panel] n={len(panel)}  anchor={panel.attrs.get('anchor_mode')}  -> {C.PANEL_CSV}")

    print("\n=== GATE 1 — forecast accuracy ===")
    g1 = G.gate1_accuracy(panel)
    print(g1.round(4).to_string())

    print("\n=== GATE 2 — incremental rates information ===")
    g2 = G.gate2_incremental(panel, target=args.target)
    for k, v in g2.items():
        print(f"  {k:22}: {v}")
    g1.to_csv(C.GATE1_CSV); pd.Series(g2).to_csv(C.GATE2_CSV)

    if g2.get("verdict") != "PASS":
        print(f"\nGate 2 = {g2.get('verdict')}. Downstream MVP/signal NOT built "
              f"(hard gate). Drop UCL/consensus + rates data and rerun.")
        return

    print("\n=== MVP — walk-forward repricing model ===")
    bt, mm = M.walk_forward_mvp(panel, target=args.target)
    print("  " + "  ".join(f"{k}={v}" for k, v in mm.items()))

    print("\n=== TRADING SIGNAL — event backtest ===")
    trades, sm = SIG.backtest_signal(bt, panel, target=args.target)
    print("  " + "  ".join(f"{k}={v}" for k, v in sm.items()))


if __name__ == "__main__":
    main()
