"""
Generate the live scorecard report from data/live_scorecard.csv.
Metrics per forecaster: RMSE, MAE, hit rate (beats AA / beats consensus), rolling-6, cumulative.
Out: docs/live_report.md (tracked).
Run: PYTHONPATH=code python code/production/generate_live_report.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SC = os.path.join(_ROOT, "data", "live_scorecard.csv")
REP = os.path.join(_ROOT, "docs", "live_report.md")
FCS = ["aa", "current_production", "final_production", "consensus", "ucl", "experimental_overlay"]


def m(err):
    e = pd.Series(err).dropna()
    return (len(e), float(np.sqrt((e**2).mean())) if len(e) else np.nan,
            float(e.abs().mean()) if len(e) else np.nan)


def core_section():
    """Render the Core CPI scorecard from data/live_scorecard_core.csv (parallel file)."""
    scc = os.path.join(_ROOT, "data", "live_scorecard_core.csv")
    if not os.path.exists(scc):
        return []
    cdone = pd.read_csv(scc)
    cdone = cdone[cdone["actual"].notna()]
    if len(cdone) == 0:
        return []
    L = ["", "---", "", "## Core CPI nowcast (ex energy, food, alcohol & tobacco)", "",
         "Core model: **AA + 0.25·TVP + 0.25·LGBM** on 5 factors (services-PMI, output-PPI, "
         "1y inflation-swap, corporate-profits, GBP-EER; λ=0.5); validation rel-RMSE **0.934** "
         "(DM p=0.006, n=120). See docs/core_factor_race.md.", "",
         f"Releases scored: **{len(cdone)}**", "",
         "| forecaster | n | RMSE | MAE |", "|---|---|---|---|"]
    for f in ["core_aa", "core_final", "core_consensus"]:
        n, r, ma = m(cdone[f"{f}_error"])
        L.append(f"| {f} | {n} | {r:.3f} | {ma:.3f} |")
    L += ["", "| month | core AA | **core final** | consensus | actual |", "|---|---|---|---|---|"]
    for _, r in cdone.iterrows():
        L.append(f"| {r['release_month']} | {r['core_aa']:.2f} | **{r['core_final']:.2f}** | "
                 f"{r['core_consensus']:.2f} | {r['actual']:.2f} |")
    return L


def main():
    df = pd.read_csv(SC)
    done = df[df["actual"].notna()].copy()
    L = ["# Live scorecard — UK CPI nowcast (frozen production model)", "",
         "Production model: **AA + 0.25·TVP + 0.25·LGBM** (λ=0.5). Genesis: May 2026.", "",
         f"Releases scored: **{len(done)}** | forecasts logged: {len(df)}", ""]
    if len(done) == 0:
        L.append("_No realised actuals yet._")
    else:
        L += ["## Cumulative accuracy", "",
              "| forecaster | n | RMSE | MAE | beats AA | beats consensus |",
              "|---|---|---|---|---|---|"]
        for f in FCS:
            n, r, ma = m(done[f"{f}_error"])
            ba = (done[f"{f}_error"].abs() < done["aa_error"].abs()).mean() if f != "aa" else np.nan
            bc = (done[f"{f}_error"].abs() < done["consensus_error"].abs()).mean() if f != "consensus" else np.nan
            L.append(f"| {f} | {n} | {r:.3f} | {ma:.3f} | "
                     f"{'' if f=='aa' else f'{ba:.0%}'} | {'' if f=='consensus' else f'{bc:.0%}'} |")
        # final-production hit rates
        fp = done["final_production_error"].abs()
        L += ["", "## Final production — hit rate", "",
              f"- beats AutoARIMA : {(fp < done['aa_error'].abs()).mean():.0%}",
              f"- beats consensus : {(fp < done['consensus_error'].abs()).mean():.0%}",
              f"- beats UCL       : {(fp < done['ucl_error'].abs()).mean():.0%}",
              f"- beats current-prod: {(fp < done['current_production_error'].abs()).mean():.0%}"]
        r6 = done.tail(6)
        L += ["", "## Rolling 6-release RMSE", "",
              "| forecaster | rolling-6 RMSE |", "|---|---|"]
        for f in FCS:
            L.append(f"| {f} | {m(r6[f'{f}_error'])[1]:.3f} |")
        L += ["", "## Per-release (signed error)", "",
              "| month | AA | curr-prod | **final** | consensus | UCL | exp(λ=1) | actual |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in done.iterrows():
            L.append(f"| {r['release_month']} | {r['aa']:.2f} | {r['current_production']:.2f} | "
                     f"**{r['final_production']:.2f}** | {r['consensus']:.2f} | {r['ucl']:.2f} | "
                     f"{r['experimental_overlay']:.2f} | {r['actual']:.2f} |")
        L += ["", "## May 2026 — GENESIS (permanent record, not reinterpreted)", "",
              "First true prospective observation. Final production 2.91 vs actual 2.80 "
              "(err +0.11). AutoARIMA 2.71 (err −0.09) was best; the λ=1 experimental overlay "
              "(3.11) was worst — a calm/base-effect month where the cost-push overlay overshot. "
              "λ=0.5 halved that error vs λ=1. One adverse point; the forward record decides."]
    L += core_section()
    open(REP, "w").write("\n".join(L) + "\n")
    print("wrote", REP); print("\n".join(L))


if __name__ == "__main__":
    main()
