"""Core-CPI factor race + SHAP screen (reuses factor_race methodology with target=uk_core_cpi).

Decision metric: univ_rel_rmse — walk-forward single-factor ridge on the CORE AA residual,
reconstructed RMSE relative to AA-only (<1 => the factor helps core when added alone).
Also mean|SHAP| from a GBT on the residual (attribution/screen).

Candidate universe = genuinely core-relevant drivers (wages, slack, expectations, FX, output PPI,
rents) plus, flagged separately, other CPI-family aggregates (services/trimmed/cpih) which are
near-components of core and must be judged for circularity, not pinned blindly.

Out: data/new_factors/core_factor_race.csv, docs/core_factor_race.md
Run: set -a; . ./.env; set +a; PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python -u code/new_factors/core_race.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import factors as F, uk_model_zoo as Z
import factor_race as FR                      # reuse wf_univariate_rel_rmse, START/END window
from targets import CORE

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_ROOT, "data", "new_factors"); os.makedirs(_OUT, exist_ok=True)
_DOCS = os.path.join(_ROOT, "docs")

EXOGENOUS = ["uk_ashe_pay", "uk_awg", "uk_paye", "uk_vacancies", "uk_monthly_gdp",
             "uk_pmi_services", "uk_pmi_composite", "uk_pmi_input_prices", "uk_infl_swap_1y",
             "gbp_eer", "uk_ppi_output", "uk_rents", "uk_retail_sales", "uk_quarterly_gdp"]
CPI_FAMILY = ["uk_services_cpi", "uk_trimmed_mean_cpi", "uk_cpih", "uk_hciall"]  # near-components: circularity risk
CANDIDATES = EXOGENOUS + CPI_FAMILY
TRAIN_FROM = 1997


def core_aa_residual():
    df, _ = F.build_matrix(names=[CORE.name])
    df = df[df.index.year >= TRAIN_FROM].resample("ME").last()
    aa = Z.AutoARIMA().backtest(df, [], CORE.name, start_year=CORE.aa_start, end_year=FR.END)
    aa = aa[(aa.index.year >= FR.START) & (aa.index.year <= FR.END)]
    resid = (aa["actual"] - aa["pred"]).rename("resid")
    return resid, aa


def load_candidates():
    df, status = F.build_matrix(names=CANDIDATES)
    live = [n for n in CANDIDATES if status.get(n) != "unavailable"]
    df = df[df.index.year >= TRAIN_FROM]
    df = F.apply_publication_lags(df, live)
    return df.resample("ME").last(), live, status


def main():
    print("Core AutoARIMA residual (target=uk_core_cpi) …")
    resid, aa = core_aa_residual()
    print(f"  resid n={len(resid.dropna())}  {resid.dropna().index.min().date()}→{resid.dropna().index.max().date()}")
    print("Loading core candidates …")
    fac, live, status = load_candidates()
    X = fac[live].reindex(resid.index)

    rows = []
    for f in live:
        rel, ra, n = FR.wf_univariate_rel_rmse(resid, X[f], aa)
        c = X[f].corr(resid)
        rows.append(dict(factor=f, group=("cpi_family" if f in CPI_FAMILY else "exogenous"),
                         n=n, abs_corr=abs(c) if np.isfinite(c) else np.nan, univ_rel_rmse=rel))
    race = pd.DataFrame(rows).set_index("factor")

    # SHAP screen (GBT on residual, exogenous only — avoid circular CPI-family dominating)
    exo = [f for f in live if f in EXOGENOUS]
    Xm = X[exo].ffill().fillna(X[exo].median(numeric_only=True))
    mask = resid.notna() & Xm.notna().all(axis=1)
    Xm2, ym = Xm[mask], resid[mask]
    shap_imp = {}
    if len(Xm2) > 40:
        from sklearn.ensemble import GradientBoostingRegressor
        gbt = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                        subsample=0.8, random_state=0).fit(Xm2, ym)
        try:
            import shap
            sv = shap.TreeExplainer(gbt).shap_values(Xm2)
            shap_imp = dict(zip(exo, np.abs(sv).mean(axis=0)))
        except Exception as e:
            print("  [shap unavailable, using GBT feature_importances_]", type(e).__name__)
            shap_imp = dict(zip(exo, gbt.feature_importances_))
    race["shap_mean_abs"] = race.index.map(lambda f: shap_imp.get(f, np.nan))
    race = race.sort_values("univ_rel_rmse")

    race.to_csv(os.path.join(_OUT, "core_factor_race.csv"))
    pd.options.display.width = 200
    print("\n=== CORE FACTOR RACE (univ_rel_rmse<1 => helps core) ===")
    print(race.round(4).to_string())

    beats = race[(race["univ_rel_rmse"] < 1.0)]
    print("\nfactors with rel_rmse<1:", list(beats.index))
    print("  exogenous among them  :", [f for f in beats.index if f in EXOGENOUS])

    # write docs
    with open(os.path.join(_DOCS, "core_factor_race.md"), "w") as fh:
        fh.write("# Core CPI factor race + SHAP screen\n\n")
        fh.write(f"Target `uk_core_cpi`; AA residual {resid.dropna().index.min().date()}→"
                 f"{resid.dropna().index.max().date()}; window {FR.START}-{FR.END}. "
                 "Decision metric `univ_rel_rmse` (<1 = helps core alone, purged walk-forward).\n\n")
        fh.write("```\n" + race.round(4).to_string() + "\n```\n\n")
        fh.write(f"**rel_rmse<1:** {list(beats.index)}\n\n")
        fh.write(f"**exogenous & rel_rmse<1 (pin candidates):** {[f for f in beats.index if f in EXOGENOUS]}\n")
    print("\nwritten: data/new_factors/core_factor_race.csv, docs/core_factor_race.md")


if __name__ == "__main__":
    main()
