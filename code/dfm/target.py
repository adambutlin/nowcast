"""
dfm/target.py — the CPI target in MoM space, and the YoY reconstruction.

Modelling MoM instead of YoY is the single most important choice in this design
(spec §3.2). CPI YoY has lag-1 autocorrelation 0.983, so 11/12 of its variance is
mechanical carry-over the factors would be forced to re-learn; MoM has 0.048. The
repo's prior DFMs modelled YoY and scored ~1.97 RMSE against AutoARIMA's 0.467.

The carry term is fully known at T — every index level it uses is already
published — so all model risk concentrates on the single unknown increment:

    c_T   = 100 * (I_{T-1} / I_{T-12} - 1)
    YoY_T = 100 * ((1 + c_T/100) * exp(pi_T / 100) - 1)

The registry carries only YoY (D7G7), so the index D7BT is defined here, reusing
factors._dbnomics. That keeps the diff to factors.py at zero.
"""
import numpy as np
import pandas as pd

import factors as F
from dfm import config as C

_CACHE = {}


def cpi_index(force=False):
    """UK CPI all-items index D7BT (ONS MM23), month-end."""
    if not force and "idx" in _CACHE:
        return _CACHE["idx"]
    s = F._dbnomics("ONS", "MM23", "D7BT.M")
    s = s.resample("ME").last().dropna().rename("cpi_index")
    _CACHE["idx"] = s
    return s


def mom_from_index(index):
    """pi_t = 100 * dlog(I_t). First observation is NaN."""
    return (100.0 * np.log(index).diff()).rename("cpi_mom")


def carry(index, ref_month):
    """
    Base-effect carry for `ref_month`: 100*(I_{T-1}/I_{T-12} - 1).

    Uses only months strictly before `ref_month`, all of them published by T.
    """
    m = pd.Timestamp(ref_month) + pd.offsets.MonthEnd(0)
    prev = m - pd.offsets.MonthEnd(1)
    base = m - pd.offsets.MonthEnd(12)
    if prev not in index.index or base not in index.index:
        raise ValueError(f"need 12 months of index history before {m.date()}")
    i_prev, i_base = float(index.loc[prev]), float(index.loc[base])
    if not np.isfinite(i_prev) or not np.isfinite(i_base) or i_base == 0:
        raise ValueError(f"non-finite index values for the carry at {m.date()}")
    return 100.0 * (i_prev / i_base - 1.0)


def yoy_from_mom(mom, index, ref_month):
    """Rebuild CPI YoY for `ref_month` from the known carry and a modelled MoM."""
    c = carry(index, ref_month)
    return 100.0 * ((1.0 + c / 100.0) * np.exp(mom / 100.0) - 1.0)


def training_frame(asof_ts=None):
    """DataFrame [cpi_index, cpi_mom] through the last published month."""
    idx = cpi_index()
    if asof_ts is not None:
        idx = idx[idx.index <= pd.Timestamp(asof_ts)]
    return pd.DataFrame({"cpi_index": idx, "cpi_mom": mom_from_index(idx)})
