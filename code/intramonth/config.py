"""
intramonth/config.py — switchable configuration for the intramonth nowcasting system.

Single source of truth for: forecast origins, targets, model stack, regimes, paths.
Everything downstream reads from here so the whole pipeline changes behaviour by
editing this file (or the matching env vars) — no hard-coding in the modules.
"""
import os

_THIS = os.path.dirname(os.path.abspath(__file__))
ROOT  = os.path.dirname(os.path.dirname(_THIS))            # repo root
DATA  = os.path.join(ROOT, "data")
PLOTS = os.path.join(ROOT, "plots")

# output folders (Part I)
DIR_INTRA_DATA  = os.path.join(DATA, "intramonth")
DIR_SCEN_DATA   = os.path.join(DATA, "scenarios")
DIR_PROD_DATA   = os.path.join(DATA, "production")
DIR_INTRA_PLOTS = os.path.join(PLOTS, "intramonth")
DIR_SCEN_PLOTS  = os.path.join(PLOTS, "scenarios")
for _d in (DIR_INTRA_DATA, DIR_SCEN_DATA, DIR_PROD_DATA, DIR_INTRA_PLOTS, DIR_SCEN_PLOTS):
    os.makedirs(_d, exist_ok=True)

# ── Forecast origins (Part F) ────────────────────────────────────────────────
# Measured as CALENDAR DAYS BEFORE the reference month-end. At origin T-k the
# high-frequency data is truncated to (month_end - k days): smaller k = more of
# the month's HF data has accrued = later, sharper nowcast.
ORIGINS = [30, 21, 14, 10, 7, 5, 2, 1]    # "T-k" days before reference month-end

# ── Targets (Part H) ─────────────────────────────────────────────────────────
# Each target resolves to a monthly series via intramonth.targets.resolve().
# kind: "yoy" or "mom"; source: registry factor name or "index" (derive from index).
TARGETS = {
    "cpi_headline_yoy":  dict(kind="yoy", source="cpi_yoy",         label="UK CPI YoY"),
    "cpi_headline_mom":  dict(kind="mom", source="GBRCPIALLMINMEI", label="UK CPI MoM"),
    "cpi_core_yoy":      dict(kind="yoy", source="uk_core_cpi",     label="UK core CPI YoY"),
    "cpi_core_mom":      dict(kind="mom", source="uk_core_cpi_idx", label="UK core CPI MoM"),
    "cpi_services_yoy":  dict(kind="yoy", source="uk_services_cpi", label="UK services CPI YoY"),
    "cpi_services_mom":  dict(kind="mom", source="uk_services_idx", label="UK services CPI MoM"),
}
DEFAULT_TARGET = os.getenv("INTRA_TARGET", "cpi_headline_yoy")

# ── Model stack (Part C) — switchable via env or kwarg ───────────────────────
# Layer name -> uk_model_zoo class name. The stack instantiates by name so a
# model can be swapped without touching pipeline code.
STACK = {
    "baseline":   os.getenv("INTRA_BASELINE",  "AutoARIMA"),   # Layer 1 persistence
    "factor":     os.getenv("INTRA_FACTOR",    "BVAR"),        # Layer 2 factor residual
    "regime_tvp": os.getenv("INTRA_TVP",       "TVP"),         # Layer 3 regime-aware
    "intramonth": os.getenv("INTRA_MIDAS",     "MIDAS"),       # Layer 4 high-frequency
}
REGIME_DETECTOR = os.getenv("INTRA_REGIME", "HMM")             # Layer 5 gate
# Residual framework: factor/tvp/midas models predict (CPI - baseline) residual.
RESIDUAL_FRAMEWORK = os.getenv("INTRA_RESIDUAL", "1") == "1"

# ── High-frequency inputs (Part B) ───────────────────────────────────────────
# yfinance daily tickers -> short name. Partial-month aggregation truncated as-of.
HF_TICKERS = {
    "brent": "BZ=F",      # energy (Brent crude)
    "gas":   "TTF=F",     # energy (EU gas)
    "gbp":   "GBPUSD=X",  # FX
    "vix":   "^VIX",      # vol / risk
}
HF_START = "2010-01-01"   # daily HF history start (yfinance)

# Monthly macro factors pinned into the panel (causal, pub-lagged).
MONTHLY_FACTORS = ["oil_brent", "gas_eu", "uk_quarterly_gdp", "imf_all_commodity",
                   "mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]

# ── Regimes (Part D/E) ───────────────────────────────────────────────────────
# Core probabilistic regimes from the HMM (3 latent states relabelled by mean).
REGIMES = ["disinflation", "normal", "shock"]
N_REGIME_STATES = 3
# Interpretable driver overlays (deterministic tags from HF/monthly momentum).
DRIVER_TAGS = ["energy_led", "services_led", "policy_tightening"]

# ── Scenarios (Part E) ───────────────────────────────────────────────────────
# scenario -> which regime mass feeds it + directional skew used to split tails.
SCENARIOS = {
    "normalisation":      dict(regime="disinflation", skew=-1),
    "base":               dict(regime="normal",       skew=0),
    "energy_shock":       dict(regime="shock",        skew=+1, driver="energy_led"),
    "services_stickiness":dict(regime="shock",        skew=+1, driver="services_led"),
    "upside_surprise":    dict(regime="tail",         skew=+2),
    "downside_surprise":  dict(regime="tail",         skew=-2),
}

# ── Walk-forward / training ──────────────────────────────────────────────────
AA_START   = 2001    # AutoARIMA baseline walk-forward start
TRAIN_FROM = 2010    # HF daily data begins ~2010 → train intramonth models from here
EVAL_START = 2018    # intramonth evolution evaluation window start
WEIGHT_TEMP = 0.15   # softmax temperature for performance weights (smaller = sharper)
WEIGHT_HALFLIFE = 18 # months; rolling performance half-life for weight decay
# Regime posterior tempering for the LIVE nowcast posterior: p ∝ p_hmm**β.
# β<1 widens the (over-confident) HMM filtered posterior to reflect uncertainty in
# the regime classification itself, so the scenario tree is not degenerate. Causal.
REGIME_TEMPER = float(os.getenv("INTRA_REGIME_TEMPER", "0.45"))

# energy scenario perturbation size (σ of recent HF energy log-returns)
SCEN_ENERGY_SIGMA = 1.0
