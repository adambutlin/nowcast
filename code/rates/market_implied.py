"""
rates/market_implied.py — Part B. Build the strongest available market-implied
UK inflation series and write it to the existing data/market_implied_cpi.csv
slot (so event_panel._anchor flips naive_rw -> market_implied).

Source: BoE implied-inflation spot curve (glcinflation*), the RPI-implied
breakeven curve. We take the 1Y point. BoE inflation curves are RPI-linked, so
we subtract a CAUSAL RPI->CPI wedge. NOTE: the downstream gap is standardized by
an EXPANDING mean (stage1/_design), which absorbs any constant/slow level error
in the wedge — so the wedge mainly affects the raw level, not the test.
"""

import os
import io
import zipfile
import datetime
import numpy as np
import pandas as pd

from . import config as C

BOE_BASE = "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves/"
WEDGE_CSV = os.path.join(C.DATA, "rpi_cpi_wedge.csv")      # optional override [date,value]
WEDGE_CONST = 0.8                                          # pp, RPI - CPI prior


# ─────────────────────────────────────────────────────────────────────────────
# generalized BoE spot-curve parser (factors._boe_spot_5y, any maturity)
# ─────────────────────────────────────────────────────────────────────────────

def _boe_spot(zip_url, maturity_years):
    import requests
    try:
        import certifi
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass
    r = requests.get(zip_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    frames = []
    for fname in sorted(z.namelist()):
        try:
            xl = pd.ExcelFile(z.open(fname))
        except Exception:
            continue
        if "4. spot curve" not in xl.sheet_names:
            continue
        df = xl.parse("4. spot curve", header=None)
        mats = df.iloc[3, 1:].values.astype(float)
        data = df.iloc[5:].copy()
        data.columns = ["date"] + list(mats)
        data = data[data["date"].apply(
            lambda x: isinstance(x, (pd.Timestamp, datetime.datetime)))]
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date").apply(pd.to_numeric, errors="coerce")
        col = min(mats, key=lambda x: abs(x - maturity_years))
        frames.append(data[[col]].rename(columns={col: "v"}))
    if not frames:
        return pd.Series(dtype=float)
    return pd.concat(frames).sort_index()["v"].resample("ME").last()


# BoE implied-inflation curve has NO 1Y/2Y point — its short end is 2.5Y, and
# anything the spline extrapolates below that blows up (seen: 17-23%). 2.5Y is
# the shortest stable maturity → used as the near-term market-implied proxy.
SHORT_MATURITY = 2.5
SANE_BAND = (-2.0, 12.0)   # % ; outside this = spline artifact -> NaN

def boe_implied_inflation(maturity=SHORT_MATURITY):
    """BoE implied-inflation (RPI) spot at `maturity` years, month-end %.
    Values outside SANE_BAND (short-end spline artifacts) -> NaN."""
    s = _boe_spot(BOE_BASE + "glcinflationmonthedata.zip", maturity)
    return s.where((s >= SANE_BAND[0]) & (s <= SANE_BAND[1]))


# ─────────────────────────────────────────────────────────────────────────────
# causal RPI -> CPI wedge
# ─────────────────────────────────────────────────────────────────────────────

def rpi_cpi_wedge(index):
    """Return a causal wedge Series aligned to `index` (month-end).
    Precedence: data/rpi_cpi_wedge.csv (trailing-mean'd) -> dbnomics RPI/CPI
    trailing 24m mean -> constant prior. Always backward-looking."""
    # 1) explicit override CSV
    if os.path.exists(WEDGE_CSV):
        w = (pd.read_csv(WEDGE_CSV, parse_dates=["date"]).set_index("date")
               .iloc[:, 0].sort_index().resample("ME").last())
        return w.reindex(index).ffill().fillna(WEDGE_CONST)
    # 2) best-effort RPI - CPI trailing mean (dbnomics ONS), sanity-guarded.
    #    RPI-CPI is historically ~0..+1.5pp; anything outside [-2,3] is a bad
    #    fetch -> fall back to the constant prior element-wise. The downstream
    #    gap is expanding-standardized, so a constant wedge is harmless anyway.
    WEDGE_BAND = (-2.0, 3.0)
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import factors as F
        rpi = F._dbnomics("ONS", "MM23", "CZBH.M").pct_change(12).mul(100)   # RPI YoY
        cpi = F._dbnomics("ONS", "MM23", "D7G7.M")                            # CPI YoY
        diff = (rpi - cpi).dropna()
        wedge = diff.rolling(24, min_periods=12).mean().shift(1)             # causal
        wedge = wedge.where((wedge >= WEDGE_BAND[0]) & (wedge <= WEDGE_BAND[1]))
        return wedge.reindex(index).ffill().fillna(WEDGE_CONST)
    except Exception:
        return pd.Series(WEDGE_CONST, index=index)


# ─────────────────────────────────────────────────────────────────────────────
# build + persist  (CPI-equivalent market-implied near-term inflation)
# ─────────────────────────────────────────────────────────────────────────────

def build_market_implied_cpi(maturity=SHORT_MATURITY, save=True):
    rpi_implied = boe_implied_inflation(maturity=maturity)
    if rpi_implied.empty:
        raise RuntimeError("BoE implied-inflation fetch returned empty — network/cert issue")
    wedge = rpi_cpi_wedge(rpi_implied.index)
    cpi_implied = (rpi_implied - wedge).rename("value")
    cpi_implied.index.name = "date"
    if save:
        cpi_implied.to_csv(C.MKT_IMPLIED_CSV)
    return cpi_implied


if __name__ == "__main__":
    s = build_market_implied_cpi()
    print(f"market-implied CPI (1Y, RPI-adj): {s.index.min().date()}..{s.index.max().date()}  "
          f"n={len(s)}  -> {C.MKT_IMPLIED_CSV}")
    print(s.tail(4).to_string())
