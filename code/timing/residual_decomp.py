"""
Residual decomposition audit: explain resid_t = CPI_yoy - AutoARIMA via 3 economic blocks.

Blocks (pub-lagged, real-time factor matrix):
  CostPressure : uk_ppi_input, uk_ppi_output
  EnergyShock  : oil_brent, gas_eu, deep_sea_freight, imf_all_commodity
  RegulatoryShock : ofgem_cap_delta, budget_event, mpc_rate_change, mpc_vote_split

Each block -> a single composite (PC1 of its standardised factors, sign-aligned to +corr with
residual on full eval) so the variance decomposition is robust even on small windows. Then
OLS resid ~ Cost + Energy + Reg, and an LMG (Lindeman-Merenda-Gold) decomposition averages
each block's incremental R^2 over all orderings (collinearity-robust). Windows: full / 2022_23
/ ex_shock / pre_2020 (within eval 2015-2024). Explanatory (in-sample) — not a forecast.

Out: data/timing/decomp/{variance_decomposition,block_detail}.csv
Run: set -a; . ./.env; set +a; PYTHONPATH=code .venv/bin/python -u code/timing/residual_decomp.py
"""
import os, sys, warnings, itertools
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import factors as F, uk_model_zoo as Z, two_stage as TS

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing", "decomp")
os.makedirs(_OUT, exist_ok=True)
EVAL_START, END = 2015, 2024
BLOCKS = {
    "CostPressure":  ["uk_ppi_input", "uk_ppi_output"],
    "EnergyShock":   ["oil_brent", "gas_eu", "deep_sea_freight", "imf_all_commodity"],
    "Regulatory":    ["ofgem_cap_delta", "budget_event", "mpc_rate_change", "mpc_vote_split"],
}
REG = ["ofgem_cap_delta", "budget_event", "mpc_rate_change", "mpc_vote_split"]
ALL = sum(BLOCKS.values(), [])
WIN = {"full": lambda i: i.year >= EVAL_START, "2022_23": lambda i: i.year.isin([2022, 2023]),
       "ex_shock": lambda i: (i.year >= EVAL_START) & ~i.year.isin([2022, 2023]),
       "pre_2020": lambda i: (i.year >= EVAL_START) & (i.year <= 2019)}


def r2(y, X):
    if X.shape[1] == 0:
        return 0.0
    m = LinearRegression().fit(X, y)
    return float(1 - np.sum((y - m.predict(X))**2) / np.sum((y - y.mean())**2))


def lmg(y, cols):
    """LMG decomposition: each regressor's avg incremental R^2 over all orderings."""
    k = len(cols); idx = list(range(k))
    from math import factorial
    contrib = {c: 0.0 for c in cols}
    for perm in itertools.permutations(idx):
        prev = 0.0; used = []
        for j in perm:
            used2 = used + [j]
            cur = r2(y, X_[:, used2]) if used2 else 0.0
            contrib[cols[j]] += (cur - prev)
            prev = cur; used = used2
    n = factorial(k)
    return {c: contrib[c] / n for c in cols}


def main():
    # build residual (full vintage so AA is the same as production)
    df, live, status = TS.load_matrix()
    # add ppi_output (not in PINNED) for the decomposition
    extra, st2 = F.build_matrix(names=["uk_ppi_output"])
    extra = F.apply_publication_lags(extra, ["uk_ppi_output"]).resample("ME").last()
    df = df.join(extra[["uk_ppi_output"]]) if "uk_ppi_output" in extra else df
    aa = Z.AutoARIMA().backtest(df, [], TS.TARGET, start_year=TS.AA_START, end_year=END)
    resid = (aa["actual"] - aa["pred"]).rename("resid")

    use = [c for c in ALL if c in df.columns]
    for c in REG:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    X = df[use].reindex(resid.index)
    data = X.join(resid).dropna(subset=["resid"])
    data = data[(data.index.year >= EVAL_START) & (data.index.year <= END)]
    # standardise factors on full eval
    Z_ = (data[use] - data[use].mean()) / (data[use].std(ddof=0) + 1e-9)
    Z_ = Z_.fillna(0.0)

    # block composites = PC1, sign-aligned to +corr with residual (full eval)
    comp = pd.DataFrame(index=data.index)
    loadings = {}
    for b, fs in BLOCKS.items():
        fs = [f for f in fs if f in Z_.columns]
        if not fs:
            continue
        pc = PCA(1).fit(Z_[fs])
        c = pd.Series(pc.transform(Z_[fs])[:, 0], index=data.index)
        if c.corr(data["resid"]) < 0:
            c = -c
        comp[b] = c
        loadings[b] = dict(zip(fs, np.round(pc.components_[0], 3)))

    blocks = list(comp.columns)
    rows, detail_rows = [], []
    for w, fn in WIN.items():
        m = fn(data.index)
        y = data.loc[m, "resid"].values
        if len(y) < 6:
            continue
        global X_
        X_ = comp.loc[m, blocks].values
        total = r2(y, X_)
        lm = lmg(y, blocks)
        # univariate + LOO incremental per block
        uni = {b: r2(y, comp.loc[m, [b]].values) for b in blocks}
        loo = {}
        for b in blocks:
            others = [c for c in blocks if c != b]
            loo[b] = total - (r2(y, comp.loc[m, others].values) if others else 0.0)
        rec = dict(window=w, n=len(y), total_R2=total, resid_var=float(np.var(y)))
        for b in blocks:
            rec[f"LMG_{b}"] = lm[b]
            rec[f"LMGshare_{b}_%"] = 100 * lm[b] / total if total > 0 else np.nan
        rows.append(rec)
        for b in blocks:
            detail_rows.append(dict(window=w, block=b, LMG_R2=lm[b], univ_R2=uni[b],
                                    LOO_incr_R2=loo[b]))
    dec = pd.DataFrame(rows).set_index("window")
    det = pd.DataFrame(detail_rows).set_index(["window", "block"])
    dec.to_csv(os.path.join(_OUT, "variance_decomposition.csv"))
    det.to_csv(os.path.join(_OUT, "block_detail.csv"))
    pd.options.display.width = 200
    print("PC1 loadings per block:")
    for b, l in loadings.items():
        print(f"  {b:13} {l}")
    print("\n=== VARIANCE DECOMPOSITION (LMG share of explained R^2) ===")
    cols = ["n", "total_R2"] + [f"LMGshare_{b}_%" for b in blocks]
    print(dec[cols].round(3).to_string())
    print("\n=== block detail (LMG R^2 / univariate R^2 / LOO incremental R^2) ===")
    print(det.round(4).to_string())
    print("\nwritten variance_decomposition.csv, block_detail.csv")


if __name__ == "__main__":
    main()
