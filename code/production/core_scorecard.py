"""Live scorecard updater for the CORE CPI nowcast (parallel to update_live_scorecard.py).

Tracks: core_aa (AutoARIMA anchor), core_final (AA + 0.5*(0.5*TVP+0.5*LGBM) on output-PPI + GBP-EER),
core_consensus (representative bank core forecast). State: data/live_scorecard_core.csv.

Usage:
  --seed                                    write the June-2026 genesis row
  --add --month YYYY-MM --date YYYY-MM-DD --core-aa .. --core-final .. --core-consensus ..
  --actual --month YYYY-MM --value 2.6      fill the realised core print -> recompute errors
  (no args)                                 recompute errors from actuals + print

Report: generate_live_report.py renders the core section from this file.
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SC = os.path.join(_ROOT, "data", "live_scorecard_core.csv")
FORECASTERS = ["core_aa", "core_final", "core_consensus"]
COLS = ["release_month", "forecast_date"] + FORECASTERS + ["actual"] + [f"{f}_error" for f in FORECASTERS]

# June-2026 genesis (first core observation). core_aa/core_final from production.model core block;
# core_consensus = representative of the quoted bank core forecasts (Goldman 2.5 / Barclays 2.45 /
# Nomura 2.3 — all undershot); actual core (ONS DKO8) = 2.6.
GENESIS = dict(release_month="2026-06", forecast_date="2026-06-30",
               core_aa=2.488, core_final=2.435, core_consensus=2.5, actual=2.6)


def load():
    return pd.read_csv(SC) if os.path.exists(SC) else pd.DataFrame(columns=COLS)


def recompute(df):
    for f in FORECASTERS:
        df[f"{f}_error"] = df[f] - df["actual"]
    return df


def save(df):
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    df[COLS].sort_values("release_month").to_csv(SC, index=False)


def seed():
    df = load()
    if len(df) and (df["release_month"] == "2026-06").any():
        print("June-2026 genesis already present — not overwriting."); return
    df = pd.concat([df, pd.DataFrame([GENESIS])], ignore_index=True)
    save(recompute(df)); print("seeded June-2026 core genesis row."); print(load().to_string(index=False))


def add(a):
    df = load()
    row = dict(release_month=a.month, forecast_date=a.date, core_aa=a.core_aa,
               core_final=a.core_final, core_consensus=a.core_consensus, actual=np.nan)
    df = pd.concat([df[df["release_month"] != a.month], pd.DataFrame([row])], ignore_index=True)
    save(recompute(df)); print(f"added {a.month}")


def set_actual(a):
    df = load(); df.loc[df["release_month"] == a.month, "actual"] = a.value
    save(recompute(df)); print(f"set core actual {a.month}={a.value}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", action="store_true"); p.add_argument("--add", action="store_true")
    p.add_argument("--actual", action="store_true")
    p.add_argument("--month"); p.add_argument("--date"); p.add_argument("--value", type=float)
    p.add_argument("--core-aa", dest="core_aa", type=float)
    p.add_argument("--core-final", dest="core_final", type=float)
    p.add_argument("--core-consensus", dest="core_consensus", type=float)
    a = p.parse_args()
    if a.seed: seed()
    elif a.add: add(a)
    elif a.actual: set_actual(a)
    else:
        save(recompute(load())); print(load().to_string(index=False))


if __name__ == "__main__":
    main()
