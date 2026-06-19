"""
Live scorecard updater (frozen-model era). Tracks 5 forecasters + 1 reference:
  aa, current_production (old AA+BVAR+TVP+MIDAS), final_production (AA+0.25TVP+0.25LGBM),
  consensus, ucl  [+ experimental_overlay (lambda=1) as reference].

Usage:
  --seed                                  write the May-2026 genesis row (permanent record)
  --add --month 2026-06 --date 2026-06-30 --aa .. --current .. --final .. --consensus .. --ucl .. [--experimental ..]
  --actual --month 2026-06 --value 2.8    fill the realised print -> recompute errors
  (no args)                               just recompute errors from actuals

State: data/live_scorecard.csv. Report: run generate_live_report.py.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SC = os.path.join(_ROOT, "data", "live_scorecard.csv")
FORECASTERS = ["aa", "current_production", "final_production", "consensus", "ucl",
               "experimental_overlay"]
COLS = ["release_month", "forecast_date"] + FORECASTERS + ["actual"] + [f"{f}_error" for f in FORECASTERS]

# May-2026 genesis row (permanent; do NOT overwrite/reinterpret)
GENESIS = dict(release_month="2026-05", forecast_date="2026-05-31",
               aa=2.71, current_production=2.917, final_production=2.91,
               consensus=3.00, ucl=3.05, experimental_overlay=3.111, actual=2.80)


def load():
    if os.path.exists(SC):
        return pd.read_csv(SC)
    return pd.DataFrame(columns=COLS)


def recompute(df):
    for f in FORECASTERS:
        df[f"{f}_error"] = df[f] - df["actual"]
    return df


def save(df):
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLS].sort_values("release_month")
    df.to_csv(SC, index=False)


def seed():
    df = load()
    if (df["release_month"] == "2026-05").any():
        print("genesis row already present — not overwriting (permanent record)."); return
    df = pd.concat([df, pd.DataFrame([GENESIS])], ignore_index=True)
    save(recompute(df)); print("seeded May-2026 genesis row."); print(load().to_string(index=False))


def add(a):
    df = load()
    row = dict(release_month=a.month, forecast_date=a.date, aa=a.aa,
               current_production=a.current, final_production=a.final,
               consensus=a.consensus, ucl=a.ucl,
               experimental_overlay=a.experimental, actual=np.nan)
    df = pd.concat([df[df["release_month"] != a.month], pd.DataFrame([row])], ignore_index=True)
    save(recompute(df)); print(f"added {a.month}")


def set_actual(a):
    df = load(); df.loc[df["release_month"] == a.month, "actual"] = a.value
    save(recompute(df)); print(f"set actual {a.month}={a.value}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", action="store_true"); p.add_argument("--add", action="store_true")
    p.add_argument("--actual", action="store_true")
    p.add_argument("--month"); p.add_argument("--date"); p.add_argument("--value", type=float)
    for f in ["aa", "current", "final", "consensus", "ucl", "experimental"]:
        p.add_argument(f"--{f}", type=float)
    a = p.parse_args()
    if a.seed: seed()
    elif a.add: add(a)
    elif a.actual: set_actual(a)
    else:
        save(recompute(load())); print(load().to_string(index=False))


if __name__ == "__main__":
    main()
