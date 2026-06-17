"""
End-to-end RMSE comparison: does adding uk_ppi_input + deep_sea_freight to the
two-stage (AA + [BVAR,TVP,MIDAS]) factor set improve the CPI-YoY nowcast?

Runs the full walk-forward backtest twice (baseline PINNED vs PINNED+extras) and the
live nowcast for both. Out: data/new_factors/compare_factors.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/new_factors/compare_factors.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import two_stage as TS

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data", "new_factors")
EXTRAS = ["uk_ppi_input", "deep_sea_freight"]
SETS = {"baseline": TS.PINNED, "plus_ppi_freight": TS.PINNED + EXTRAS}


def run(pinned):
    df, live, status = TS.load_matrix(pinned)
    bt = TS.backtest(df, live)
    met = TS.metrics(bt)
    nc = TS.nowcast(df, live)
    return met, nc, live


def main():
    results = {}
    for name, pinned in SETS.items():
        print(f"\n### {name}  ({len(pinned)} pinned) ###")
        met, nc, live = run(pinned)
        results[name] = (met, nc)
        print("live factors:", live)
        print(met[["n", "rmse_AA", "rmse_2stage", "rel_rmse"]].round(4).to_string())
        print(f"nowcast {nc['nowcast_date']}: AA {nc['aa_pred']:.3f} + "
              f"{nc['stage2_overlay']:+.3f} = {nc['forecast']:.3f}  members={ {k:round(v,3) for k,v in nc['members'].items()} }")

    # delta table
    base, plus = results["baseline"][0], results["plus_ppi_freight"][0]
    delta = pd.DataFrame({
        "rmse_2stage_base": base["rmse_2stage"],
        "rmse_2stage_plus": plus["rmse_2stage"],
        "rmse_delta": plus["rmse_2stage"] - base["rmse_2stage"],
        "rel_base": base["rel_rmse"], "rel_plus": plus["rel_rmse"],
    })
    delta.to_csv(os.path.join(_OUT, "compare_factors.csv"))
    pd.options.display.width = 200
    print("\n=== DELTA (negative rmse_delta => extras improve 2-stage) ===")
    print(delta.round(4).to_string())
    nb, npf = results["baseline"][1], results["plus_ppi_freight"][1]
    print(f"\nlive nowcast: baseline {nb['forecast']:.3f}  ->  plus {npf['forecast']:.3f}")
    print("written", os.path.join(_OUT, "compare_factors.csv"))


if __name__ == "__main__":
    main()
