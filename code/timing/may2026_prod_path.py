"""PART H — May 2026 path under the PRODUCTION model with as-of truncation.
Requires cpi_yoy ending April (so the nowcast target is May). Restores after."""
import os, sys, warnings
warnings.filterwarnings("ignore")
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "new_factors"))
import numpy as np, pandas as pd
import uk_model_zoo as Z, two_stage as TS
import production_asof as PA

_OUT = os.path.join(os.path.dirname(_CODE), "data", "timing", "prod_asof")
W = TS.WEIGHTS
daily = PA._daily()
df0, live, status = TS.load_matrix()
rows = []
for h in PA.HORIZONS:
    df = df0.copy()
    if h > 0:
        for f, src in [("oil_brent","brent"), ("gas_eu","gas")]:
            if f in df.columns and src in daily.columns:
                df[f] = PA.asof_logret(daily[src], h).reindex(df.index)
        Z._MIDAS_CACHE.clear(); Z._MIDAS_CACHE["mm"] = PA.build_mm(daily, h)
    else:
        Z._MIDAS_CACHE.clear()
    nc = TS.nowcast(df, live)
    Z._MIDAS_CACHE.clear()
    m = nc["members"]
    rows.append(dict(horizon=PA.LABEL[h], h=h, date=nc["nowcast_date"],
                     AA=round(nc["aa_pred"],3),
                     bvar=round(m.get("bvar",np.nan),3), tvp=round(m.get("tvp",np.nan),3),
                     midas=round(m.get("midas",np.nan),3),
                     full=round(nc["forecast"],3), actual=2.8))
out = pd.DataFrame(rows).set_index("horizon")
out.to_csv(os.path.join(_OUT, "may2026_forecast_path.csv"))
pd.options.display.width = 200
print(out.to_string())
