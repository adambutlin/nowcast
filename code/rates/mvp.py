"""
rates/mvp.py — Deliverable 4. Walk-forward front-end repricing model.

Causal expanding-window OLS: front-end move ~ my_surprise (+ regime, days_to_mpc).
One row per release; coefficients fit ONLY on prior releases. Produces an
out-of-sample predicted move per release + accuracy metrics. Gated on Gate 2
PASS by the orchestrator (run.py) — this module just builds/evaluates.
"""

import numpy as np
import pandas as pd

from . import config as C
from .gates import _design


def walk_forward_mvp(panel, target=None, min_train=24, exclude_ldi=True):
    target = target or C.PRIMARY_MOVE
    y, X = _design(panel, target, controls=True, exclude_ldi=exclude_ldi)
    n = len(y)
    if n < min_train + 5 or "my_surprise" not in X.columns:
        return pd.DataFrame(), {"n": int(n), "status": "insufficient"}
    import statsmodels.api as sm
    Xc = sm.add_constant(X, has_constant="add")
    rows = []
    for i in range(min_train, n):
        b = np.linalg.lstsq(Xc.iloc[:i].values, y.iloc[:i].values, rcond=None)[0]
        pred = float(Xc.iloc[i].values @ b)
        rows.append(dict(ref_month=y.index[i], pred_move=pred,
                         realized_move=float(y.iloc[i])))
    bt = pd.DataFrame(rows).set_index("ref_month")
    e = bt["realized_move"] - bt["pred_move"]
    ss_res = float((e ** 2).sum())
    ss_tot = float(((bt["realized_move"] - bt["realized_move"].mean()) ** 2).sum()) or np.nan
    metrics = dict(
        n=int(len(bt)), target=target,
        rmse_bp=float(np.sqrt((e ** 2).mean())),
        oos_r2=float(1 - ss_res / ss_tot) if ss_tot else np.nan,
        sign_hit=float(np.mean(np.sign(bt["pred_move"]) == np.sign(bt["realized_move"]))),
        corr=float(bt["pred_move"].corr(bt["realized_move"])),
    )
    bt.to_csv(C.MVP_CSV)
    return bt, metrics
