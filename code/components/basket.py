"""Phase 0 — reconstruct the CPI basket exactly, or refuse to proceed.

    PYTHONPATH=code python code/components/basket.py

Before any component can be modelled, the aggregation identity has to hold: the
component weights must sum to the headline basket and the weighted component
movements must reproduce published headline CPI. If that fails, every downstream
number is built on sand. This module is the gate.

Source: the full ONS MM23 dataset (4,054 series), not the per-series JSON API —
the weight series are only discoverable by title, and their titles are the only
place the ECOICOP hierarchy is documented.

WHAT THE RECONCILIATION FOUND (2026-08-21)

MM23 publishes 49 `CPI wts:` series. They are overlapping ANALYTICAL AGGREGATES,
not a hierarchical partition: the eight goods subgroups sum to 333-409 per 1000
against a `Non-energy industrial goods` weight of 276-347, and goods + the four
services groups leave 61-75 per 1000 unaccounted for. No COICOP division weights
(01..12) exist anywhere in MM23, although all twelve division INDICES do.

Exactly two partitions reconcile to 1000.0:

    2-way   efat + core
    4-way   household_energy + motor_fuel + food_alc_tob + core

both exact, because `core + efat = 1000.0` and `energy = household_energy +
motor_fuel` to 0.0. The 4-way is the deepest exhaustive split available from
published MM23 weights, and it is what this module verifies.

THE RECONSTRUCTION FLOOR

Even with an exact partition the identity does not close perfectly: reproducing
headline MoM from published components leaves ~0.057pp RMSE. That is not a
modelling error and cannot be removed by better components. MM23 publishes
indices to ONE decimal place, so an index near 143 carries +/-0.035% quantisation
and a MoM built from two rounded levels inherits ~0.029pp per index. Two
diagnostics confirm rounding rather than a structural mistake:

  - the error's lag-1 autocorrelation is -0.54, against a theoretical -0.5 for
    differenced white noise, which is what rounding a level and then differencing
    produces;
  - the error barely moves with the number of components (2-way 0.0552, 4-way
    0.0576), so it is dominated by rounding in the HEADLINE index, not by
    compounding across components.

Chain-linking was tested and is NOT the cause: the correct December-based
chained form (I_t = I_dec * sum_i w_i * I_i,t / I_i,dec) gives 0.0570 against
0.0576 for the naive weighted-MoM sum.

This floor must be pre-registered. A perfect component nowcast still carries
~0.057pp of reconstruction error, against production's 0.431pp headline RMSE —
about 13% of it in RMSE terms. Real, not disqualifying, and it is a property of
public data rather than of the method.
"""
import csv
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CACHE = os.path.join(_ROOT, "data", "components")
os.makedirs(_CACHE, exist_ok=True)
MM23_CSV = os.path.join(_CACHE, "mm23.csv")
MM23_URL = ("https://www.ons.gov.uk/file?uri=/economy/inflationandpriceindices"
            "/datasets/consumerpriceindices/current/mm23.csv")

# ── the reconciling partition ────────────────────────────────────────────────
# index CDID, weight CDID. Weights are parts per 1000, dated to the year they apply.
PARTITION = {
    "household_energy": dict(index="DK9U", weight="A9F4",
                             desc="Electricity, gas & misc. energy (G)"),
    "motor_fuel":       dict(index="DKC4", weight="A9FS",
                             desc="Liquid fuels, vehicle fuels & lubricants (G)"),
    "food_alc_tob":     dict(index="DK9O", weight="A9EW",
                             desc="Food, alcoholic beverages & tobacco (G)"),
    "core":             dict(index="DKC6", weight="A9FU",
                             desc="Excluding energy, food, alcohol & tobacco"),
}
HEADLINE_INDEX = "D7BT"

_MONTHS = {m.upper()[:3]: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
_MON_RE = re.compile(r"^(\d{4})\s+([A-Z]{3})$", re.I)
_YR_RE = re.compile(r"^\d{4}$")


def _download(force=False):
    if os.path.exists(MM23_CSV) and not force:
        return MM23_CSV
    r = requests.get(MM23_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
    r.raise_for_status()
    with open(MM23_CSV, "wb") as f:
        f.write(r.content)
    return MM23_CSV


def load_mm23(force=False):
    """Return (monthly DataFrame, annual DataFrame, {cdid: title}) for all series.

    MM23 interleaves annual, quarterly and monthly rows under one column set; the
    period label in column 0 is the only discriminator, so rows are routed by
    matching its shape rather than by position.
    """
    path = _download(force)
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.reader(f))
    titles, cdids = rows[0], rows[1]
    meta = dict(zip(cdids, titles))
    mrec, yrec = {}, {}
    for r in rows[2:]:
        if not r or not r[0]:
            continue
        lab = r[0].strip()
        m = _MON_RE.match(lab)
        if m:
            key = pd.Timestamp(int(m.group(1)), _MONTHS[m.group(2).upper()], 1) \
                + pd.offsets.MonthEnd(0)
            mrec[key] = r
        elif _YR_RE.match(lab):
            yrec[int(lab)] = r

    def frame(rec):
        out = {}
        for c, j in ((c, j) for j, c in enumerate(cdids) if c):
            col = {}
            for k, r in rec.items():
                v = r[j] if j < len(r) else ""
                if v:
                    try:
                        col[k] = float(v)
                    except ValueError:
                        pass
            if col:
                out[c] = pd.Series(col)
        return pd.DataFrame(out).sort_index()

    return frame(mrec), frame(yrec), meta


def basket(start_year=2005):
    """Component indices and their weights, aligned monthly.

    Each month takes the weight of its calendar year: ONS refreshes CPI weights
    every January and holds them for the year.
    """
    mon, ann, meta = load_mm23()
    idx = pd.DataFrame({k: mon[v["index"]] for k, v in PARTITION.items()})
    idx["headline"] = mon[HEADLINE_INDEX]
    idx = idx.dropna()
    idx = idx[idx.index.year >= start_year]
    wts = pd.DataFrame({
        k: pd.Series(idx.index.year, index=idx.index).map(ann[v["weight"]].dropna())
        for k, v in PARTITION.items()})
    return idx, wts, meta


def check(start_year=2005):
    """The Phase 0 gate. Returns a dict of diagnostics; prints a verdict."""
    idx, wts, meta = basket(start_year)
    comps = list(PARTITION)

    wsum = wts.sum(axis=1)
    mom = idx.pct_change() * 100.0
    recon = sum((wts[c] / 1000.0) * mom[c] for c in comps)
    d = pd.DataFrame({"actual": mom["headline"], "recon": recon}).dropna()
    err = d["recon"] - d["actual"]

    res = dict(
        n=len(d), first=str(d.index[0].date()), last=str(d.index[-1].date()),
        weight_sum_min=float(wsum.min()), weight_sum_max=float(wsum.max()),
        weight_max_dev=float((wsum - 1000.0).abs().max()),
        mom_mean_abs=float(err.abs().mean()), mom_max_abs=float(err.abs().max()),
        mom_rmse=float(np.sqrt((err ** 2).mean())),
        err_autocorr1=float(err.autocorr(1)),
        corr=float(d["actual"].corr(d["recon"])))

    print("=== PHASE 0: CPI basket reconciliation ===")
    print(f"  partition : {comps}")
    for k, v in PARTITION.items():
        print(f"    {k:<18} index {v['index']}  weight {v['weight']}  — {v['desc']}")
    print(f"  window    : {res['first']} … {res['last']}  (n={res['n']})\n")
    print(f"  weights sum to 1000: min {res['weight_sum_min']:.4f}  "
          f"max {res['weight_sum_max']:.4f}  max deviation {res['weight_max_dev']:.6f}")
    print(f"  MoM identity       : mean |err| {res['mom_mean_abs']:.5f} pp   "
          f"max {res['mom_max_abs']:.5f} pp   RMSE {res['mom_rmse']:.5f} pp")
    print(f"  error lag-1 autocorr: {res['err_autocorr1']:+.3f}   "
          f"(≈ -0.5 ⇒ differenced rounding noise, not structural)")

    # The gate: weights must partition exactly; the residual must look like rounding.
    ok_w = res["weight_max_dev"] < 0.01
    ok_r = res["err_autocorr1"] < -0.25 and res["mom_rmse"] < 0.10
    print()
    print(f"  weights exact          : {'PASS' if ok_w else 'FAIL'}")
    print(f"  residual is rounding   : {'PASS' if ok_r else 'FAIL'}")
    print(f"  RECONSTRUCTION FLOOR   : {res['mom_rmse']:.5f} pp RMSE — "
          f"pre-register this; no component model can beat it.")
    res["pass"] = bool(ok_w and ok_r)
    print(f"\n  PHASE 0 {'PASSES' if res['pass'] else 'FAILS'} for the 4-way partition.")
    return res


if __name__ == "__main__":
    r = check()
    sys.exit(0 if r["pass"] else 1)
