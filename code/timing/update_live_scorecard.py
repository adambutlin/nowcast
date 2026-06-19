"""
Live scorecard for the FINAL production model (AA + 0.5 TVP + 0.5 LGBM).
Prospective evaluation only — no retraining, no architecture change.

Usage:
  # regenerate report from current scorecard
  PYTHONPATH=code python code/timing/update_live_scorecard.py
  # append a new forecast (actual left blank until released)
  ... --add --month 2026-06 --date 2026-06-30 --model 2.93 --aa 2.73 --consensus 2.9 --ucl 2.95
  # fill the actual once released (errors + metrics update)
  ... --actual --month 2026-06 --value 2.8

Files: data/live_scorecard.csv (state), data/timing/live_report.md (report).
"""
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SC = os.path.join(_ROOT, "data", "live_scorecard.csv")
REP = os.path.join(_ROOT, "data", "timing", "live_report.md")
FCS = {"model": "model_forecast", "aa": "aa_forecast",
       "consensus": "consensus_forecast", "ucl": "ucl_forecast"}


def load():
    return pd.read_csv(SC) if os.path.exists(SC) else pd.DataFrame()


def recompute_errors(df):
    for k, col in FCS.items():
        df[f"{k}_error"] = df[col] - df["actual_cpi"]
    return df


def add_forecast(month, date, model, aa, consensus, ucl):
    df = load()
    row = dict(release_month=month, forecast_date=date, model_forecast=model,
               aa_forecast=aa, consensus_forecast=consensus, ucl_forecast=ucl,
               actual_cpi=np.nan, model_error=np.nan, aa_error=np.nan,
               consensus_error=np.nan, ucl_error=np.nan)
    df = pd.concat([df[df["release_month"] != month], pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values("release_month"); df.to_csv(SC, index=False)
    print(f"added forecast {month}"); print(df.to_string(index=False))


def set_actual(month, value):
    df = load(); df.loc[df["release_month"] == month, "actual_cpi"] = value
    df = recompute_errors(df); df.to_csv(SC, index=False)
    print(f"set actual {month}={value}")


def _metrics(err):
    err = pd.Series(err).dropna()
    if len(err) == 0:
        return dict(n=0, rmse=np.nan, mae=np.nan)
    return dict(n=len(err), rmse=float(np.sqrt((err**2).mean())), mae=float(err.abs().mean()))


def report():
    df = load()
    done = df[df["actual_cpi"].notna()].copy()
    lines = ["# Live scorecard — final production model (AA + 0.5 TVP + 0.5 LGBM)", "",
             f"Releases scored: **{len(done)}** (forecasts logged: {len(df)})", ""]
    if len(done) == 0:
        lines.append("_No realised actuals yet._")
    else:
        # per-forecaster cumulative
        lines += ["## Cumulative accuracy", "",
                  "| forecaster | n | RMSE | MAE | beats AA | beats consensus |",
                  "|---|---|---|---|---|---|"]
        for k, col in FCS.items():
            m = _metrics(done[f"{k}_error"])
            ba = (done[f"{k}_error"].abs() < done["aa_error"].abs()).mean() if k != "aa" else np.nan
            bc = (done[f"{k}_error"].abs() < done["consensus_error"].abs()).mean() if k != "consensus" else np.nan
            lines.append(f"| {k} | {m['n']} | {m['rmse']:.3f} | {m['mae']:.3f} | "
                         f"{'' if k=='aa' else f'{ba:.0%}'} | {'' if k=='consensus' else f'{bc:.0%}'} |")
        # hit rates
        hit_aa = (done["model_error"].abs() < done["aa_error"].abs()).mean()
        hit_c = (done["model_error"].abs() < done["consensus_error"].abs()).mean()
        hit_u = (done["model_error"].abs() < done["ucl_error"].abs()).mean()
        lines += ["", "## Model hit rate (final model beats benchmark)", "",
                  f"- vs AutoARIMA: **{hit_aa:.0%}** ({int((done['model_error'].abs()<done['aa_error'].abs()).sum())}/{len(done)})",
                  f"- vs Consensus: **{hit_c:.0%}**", f"- vs UCL: **{hit_u:.0%}**"]
        # rolling 6
        r6 = done.tail(6)
        lines += ["", "## Rolling 6-release", "",
                  f"- model RMSE: {_metrics(r6['model_error'])['rmse']:.3f} | "
                  f"AA: {_metrics(r6['aa_error'])['rmse']:.3f} | "
                  f"consensus: {_metrics(r6['consensus_error'])['rmse']:.3f} | "
                  f"UCL: {_metrics(r6['ucl_error'])['rmse']:.3f}"]
        # per-release table
        lines += ["", "## Per-release", "", "| month | model | AA | cons | UCL | actual | "
                  "|err_model| | |err_AA| | |err_cons| | |err_UCL| |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in done.iterrows():
            lines.append(f"| {r['release_month']} | {r['model_forecast']:.2f} | {r['aa_forecast']:.2f} | "
                         f"{r['consensus_forecast']:.2f} | {r['ucl_forecast']:.2f} | {r['actual_cpi']:.2f} | "
                         f"{abs(r['model_error']):.2f} | {abs(r['aa_error']):.2f} | "
                         f"{abs(r['consensus_error']):.2f} | {abs(r['ucl_error']):.2f} |")
    open(REP, "w").write("\n".join(lines) + "\n")
    print("wrote", REP); print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="store_true"); ap.add_argument("--actual", action="store_true")
    ap.add_argument("--month"); ap.add_argument("--date"); ap.add_argument("--value", type=float)
    ap.add_argument("--model", type=float); ap.add_argument("--aa", type=float)
    ap.add_argument("--consensus", type=float); ap.add_argument("--ucl", type=float)
    a = ap.parse_args()
    if a.add:
        add_forecast(a.month, a.date, a.model, a.aa, a.consensus, a.ucl)
    elif a.actual:
        set_actual(a.month, a.value)
    report()


if __name__ == "__main__":
    main()
