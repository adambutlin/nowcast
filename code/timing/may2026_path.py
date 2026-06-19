"""PART F — May-2026 live forecast path T-30..T-1 (per-origin nowcast). Actual = 2.8.
Requires cpi_yoy ending April so the nowcast target is May. Run after that is set."""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE)
import numpy as np, pandas as pd
from intramonth import config as C, panel as P
from intramonth.stack import ModelStack

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing")
TARGET = "cpi_headline_yoy"
ORIGINS = [30, 21, 14, 10, 7, 5, 2, 1]
W = {"factor": 0.375, "regime_tvp": 0.25, "intramonth": 0.375}
rows = []
for k in ORIGINS:
    pan, meta = P.build_panel(TARGET, k=k)
    nc = ModelStack(pan, meta, end_year=2024).nowcast()
    aa = nc["baseline"]
    def ov(layer):
        v = nc.get(layer + "_resid", np.nan); return v if np.isfinite(v) else 0.0
    full = aa + W["factor"]*ov("factor") + W["regime_tvp"]*ov("regime_tvp") + W["intramonth"]*ov("intramonth")
    rows.append(dict(origin=f"T-{k}", k=k, date=str(pd.Timestamp(nc["_nowcast_date"]).date()),
                     AA=round(float(aa),3), full=round(float(full),3),
                     bvar_resid=round(ov("factor"),3), tvp_resid=round(ov("regime_tvp"),3),
                     midas_resid=round(ov("intramonth"),3)))
df = pd.DataFrame(rows).set_index("origin")
df["actual"] = 2.8
df.to_csv(os.path.join(_OUT, "may2026_path.csv"))
pd.options.display.width = 200
print(df.to_string())
print("\nactual May-2026 = 2.8 | AA path flat (univariate) | full path driven by TVP overlay")
