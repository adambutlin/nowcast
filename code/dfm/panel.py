"""
dfm/panel.py — the mixed-frequency observation panel.

32 series in 6 economic blocks (spec §1.6), plus CPI as MoM log-change and a
3-series observed exogenous reg block. Every series loads on the pervasive
`global` factor; `block` names the second factor it loads on under the
structured-loading arms (E5/E6).

`freq` decides how a month's value is formed at an as-of timestamp:
  daily      partial-month aggregation over days <= asof   (dfm.daily)
  monthly    the whole month, visible only once published  (release calendar)
  quarterly  Mariano-Murasawa 3-month aggregation; intra-quarter months missing

Only five series are genuinely daily at source — verified against FRED on
2026-08-19. uk_gilt_10y, copper_price and wheat_price are monthly at source
despite the registry describing them as market rates, and gas_eu blends daily
TTF (2017+) onto IMF monthly (1960+), so it stays monthly rather than silently
losing 1997-2016. See the comment in dfm/daily.py FETCHERS.

No LSEG series appear here: a multi-hour sweep must not depend on the Workspace
desktop app being open (spec §1.7).
"""

BLOCKS = ["activity", "nominal", "energy", "supply", "fx", "food"]

# name -> block, transform, frequency
PANEL_SERIES = {
    # ── activity (7) ─────────────────────────────────────────────────────────
    "uk_quarterly_gdp":  dict(block="activity", transform="yoy",    freq="quarterly"),
    "uk_retail_sales":   dict(block="activity", transform="yoy",    freq="monthly"),
    "uk_vacancies":      dict(block="activity", transform="yoy",    freq="monthly"),
    "uk_awg":            dict(block="activity", transform="yoy",    freq="monthly"),
    "uk_house_prices":   dict(block="activity", transform="yoy",    freq="monthly"),
    "vix":               dict(block="activity", transform="level",  freq="daily"),
    "move_index":        dict(block="activity", transform="level",  freq="monthly"),
    # ── nominal (8) ──────────────────────────────────────────────────────────
    "uk_ppi_output":     dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_ppi_input":      dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_core_cpi":       dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_services_cpi":   dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_rents":          dict(block="nominal",  transform="yoy",    freq="monthly"),
    "uk_be5":            dict(block="nominal",  transform="level",  freq="monthly"),
    "uk_gilt_10y":       dict(block="nominal",  transform="diff",   freq="monthly"),
    "slope_2s10s":       dict(block="nominal",  transform="level",  freq="monthly"),
    # ── energy (5) ───────────────────────────────────────────────────────────
    "oil_brent":         dict(block="energy",   transform="logret", freq="daily"),
    "gas_eu":            dict(block="energy",   transform="logret", freq="monthly"),
    "gas_hh":            dict(block="energy",   transform="logret", freq="daily"),
    "imf_all_commodity": dict(block="energy",   transform="logret", freq="monthly"),
    "oil_vol_6m":        dict(block="energy",   transform="level",  freq="monthly"),
    # ── supply (6) ───────────────────────────────────────────────────────────
    "deep_sea_freight":  dict(block="supply",   transform="logret", freq="monthly"),
    "semiconductors_ppi":dict(block="supply",   transform="logret", freq="monthly"),
    "chemicals_ppi":     dict(block="supply",   transform="logret", freq="monthly"),
    "metals_index":      dict(block="supply",   transform="logret", freq="monthly"),
    "copper_price":      dict(block="supply",   transform="logret", freq="monthly"),
    "timber_price":      dict(block="supply",   transform="logret", freq="monthly"),
    # ── fx (3) ───────────────────────────────────────────────────────────────
    "gbp_eer":           dict(block="fx",       transform="diff",   freq="monthly"),
    "gbpusd":            dict(block="fx",       transform="logret", freq="daily"),
    "gbp_eur":           dict(block="fx",       transform="logret", freq="daily"),
    # ── food (3) — the only genuinely new channel vs production ──────────────
    "food_price_index":  dict(block="food",     transform="logret", freq="monthly"),
    "wheat_price":       dict(block="food",     transform="logret", freq="monthly"),
    "vegetable_oil_price": dict(block="food",   transform="logret", freq="monthly"),
}

# Observed exogenous shifters. Sparse, mostly-zero administered events cannot
# identify a latent factor, so they enter the measurement equation directly.
REG_SERIES = ["mpc_rate_change", "ofgem_cap_delta", "budget_event"]

TARGET_SERIES = "cpi_index"


def block_members():
    """block -> list of series names loading on that block factor."""
    out = {b: [] for b in BLOCKS}
    for name, spec in PANEL_SERIES.items():
        out[spec["block"]].append(name)
    return out


def all_series():
    """Every series the release calendar must cover."""
    return [TARGET_SERIES] + list(PANEL_SERIES) + list(REG_SERIES)
