"""
intramonth/targets.py — configurable forecast targets (Part H).

resolve(target_key) -> (monthly Series, status).  status in {"live","csv","derived","unavailable"}.
Targets are defined in config.TARGETS. YoY targets load a registry factor directly;
MoM targets derive month-over-month % from a price index (FRED or registry).

All series are month-end indexed. No standardization here — that happens causally
downstream inside each model's walk-forward fit.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import factors as F
from intramonth import config as C

# index series for MoM derivation (FRED ids); None = no public index → unavailable
_INDEX_SOURCES = {
    "GBRCPIALLMINMEI": "GBRCPIALLMINMEI",   # UK headline CPI index (FRED/OECD)
    "uk_core_cpi_idx": None,                # no free core index → unavailable
    "uk_services_idx": None,                # no free services index → unavailable
}


def _yoy(source):
    """YoY target: load registry factor (already YoY %)."""
    s, st = F.load_factor(source)
    if st == "unavailable" or s is None or not len(s.dropna()):
        return None, "unavailable"
    return s.dropna().resample("ME").last(), ("live" if st == "live" else st)


def _mom(source):
    """MoM target: derive month-over-month % change from a price index."""
    fred_id = _INDEX_SOURCES.get(source, source)
    if fred_id is None:
        return None, "unavailable"
    try:
        idx = F._fred(fred_id)
        if idx is None or not len(idx.dropna()):
            return None, "unavailable"
        mom = idx.resample("ME").last().pct_change(1).mul(100).dropna()
        return mom, "derived"
    except Exception:
        return None, "unavailable"


def resolve(target_key):
    """Return (Series, status) for a configured target key."""
    if target_key not in C.TARGETS:
        raise KeyError(f"unknown target {target_key!r}; choices {list(C.TARGETS)}")
    spec = C.TARGETS[target_key]
    if spec["kind"] == "yoy":
        s, st = _yoy(spec["source"])
    elif spec["kind"] == "mom":
        s, st = _mom(spec["source"])
    else:
        raise ValueError(f"bad target kind {spec['kind']!r}")
    if s is not None:
        s.name = target_key
    return s, st


def available_targets():
    """List target keys whose data resolves (status != unavailable)."""
    out = {}
    for k in C.TARGETS:
        _, st = resolve(k)
        out[k] = st
    return out


if __name__ == "__main__":
    for k, st in available_targets().items():
        s, _ = resolve(k)
        last = f"{s.dropna().index[-1].date()}={s.dropna().iloc[-1]:.2f}" if st != "unavailable" else "—"
        print(f"  {k:20} {st:12} {last}")
