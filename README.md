# nowcast

UK CPI YoY nowcasting (13 operational + 9 experimental model zoo, 38 live
factors, mixed-frequency pub-lag discipline, regime identification, ensemble
combination, regime-model-combine) **plus** a downstream rates-repricing research
pipeline (`code/rates/`) that tests whether the nowcast carries information beyond
consensus and can be turned into a UK 2Y gilt signal.

**Documentation:**
- [docs/STATE.md](docs/STATE.md) — current results, factor matrix, rates pipeline status, pending work
- [docs/SPEC.md](docs/SPEC.md) — system specification and design decisions
- [docs/PROCESS.md](docs/PROCESS.md) — chronological build log
- [docs/handoff/HANDOFF.md](docs/handoff/HANDOFF.md) — session handoff notes

**Branches / tags:** `main` carries the audited+remediated CPI nowcast (H6/C4/C1
fixes, commit `518f528`). `alpha-gen` (tag `rates-alpha`) carries the rates
pipeline. See the Rates Repricing Pipeline section below.

---

## UK CPI YoY Model Zoo

**Target:** UK CPI YoY (ONS D7G7.M via dbnomics, %)
**Factors:** 30 live (26 pub_lag=0, 4 pub_lag≥1)
**Mixed-frequency:** pub_lag applied per factor — financial data contemporaneous
(pub_lag=0), ONS/economic stats lagged 1–2 months (pub_lag=1,2)
**Backtest:** Expanding window 2015–2024 (blind test: 2025+ never evaluated)
**Training start:** 1992 (post-ERM crisis)

### Results (30-factor run, 2026-06-06)

| Model             | RMSE  | MAE   | Dir%  | beats AR(1) |
|-------------------|-------|-------|-------|-------------|
| Combined-Static   | **0.310** | 0.246 | 89.2% | ✓ |
| Combined-Dynamic  | 0.312 | 0.246 | 89.2% | ✓ |
| Combined-Absolute | 0.313 | 0.250 | 89.2% | ✓ |
| MedianElasticNet  | 0.345 | 0.240 | 91.9% | ✓ |
| ElasticNet        | 0.353 | 0.263 | 89.2% | ✓ |
| RegimeEns         | 0.466 | 0.349 | 86.5% | ✓ |
| **AR(1) baseline**| **0.495** | 0.322 | 93.3% | — |
| TVP               | 0.529 | 0.381 | 91.9% | — |
| UCM               | 0.605 | 0.468 | 83.8% | — |
| HuberNet          | 0.714 | 0.535 | 86.5% | — |
| SARIMAX           | 0.857 | 0.658 | 81.1% | — |
| PCR               | 0.865 | 0.654 | 89.2% | — |
| DFM               | 1.100 | 0.758 | 91.9% | — |
| RAMM-LGBM         | 2.170 | 1.351 | 91.9% | — |
| HMM               | 2.807 | 1.824 | 91.9% | — |

Combined-Static reduces RMSE by 37% vs AR(1) (0.310 vs 0.495).
Backtest n=37 (quarterly step 2015–2024); DM test under-powered at this n.

### Top Factor Importances (cross-model, 30-factor run)

| Factor           | pub_lag | Signal across models |
|------------------|---------|----------------------|
| uk_rents_lag1    | 0       | #1 by large margin: UCM=2.185, TVP=1.098, GBM=1.932 |
| metals_index     | 0       | UCM=0.264, TVP=0.195 — new factor, strong signal |
| copper_price     | 0       | DFM=0.732 factor loading (new) |
| gbp_eur          | 0       | TVP=0.154, ElasticNet=0.132 (new) |
| uk_ftse250       | 0       | DFM=0.616, TVP=0.147 (new) |
| chemicals_ppi    | 0       | ElasticNet=0.110 (new) |
| uk_awg           | 1       | RAMM-LGBM=0.338 (new: ONS AWE KAB9) |
| uk_monthly_gdp   | 1       | BVAR=0.088 (new: OECD industrial prod) |
| cpi_lag1 / cpi_3m_chg | — | AR features auto-added by tree models |

### Current Nowcast (May 2026)

Consensus: **~2.5–2.8% YoY** (April actual: 3.5%)

| Model             | May 2026 nowcast |
|-------------------|-----------------|
| Combined-Static   | ~2.78%          |
| MedianElasticNet  | 2.79%           |
| ElasticNet        | 2.80%           |
| TVP               | 2.72%           |
| UCM               | 2.61%           |

### Factor Set

**26 pub_lag=0 (financial/market, contemporaneous):**
oil_brent, gbpusd, uk_be5, vix, gas_eu, uk_gilt_10y, oil_vol_6m, gbpusd_vol_6m,
oil_brent_3m, gbpusd_3m, gbp_eur, gbp_eer, semiconductors_ppi, deep_sea_freight,
metals_index, copper_price, nickel_price, iron_ore_price, timber_price,
chemicals_ppi, uk_ftse250, uk_ftse100, food_price_index, wheat_price,
vegetable_oil_price, uk_rents_lag1

**4 pub_lag≥1 (ONS/economic stats, 1–2 month lag):**
uk_monthly_gdp (lag=1), uk_awg (lag=1), uk_vacancies (lag=1), uk_house_prices (lag=2)

### Regime-Model-Combine Framework

Run with `--rmc` flag. Identifies regime labels (HMM, LSTAR, DFM, VIX-threshold),
trains each model on regime-specific sub-samples, keeps only models that beat AR(1)
within that regime, builds a metamodel that selects models based on the current
regime signal.

---

## Rates Repricing Pipeline (`code/rates/`)

Tests one hypothesis: **does the CPI nowcast contain information about future UK
rates repricing not already in consensus / market pricing?** Built on `alpha-gen`
(tag `rates-alpha`).

**Flow:** model forecast → event panel → forecast gap → Stage 1 (gap predicts
realized surprise?) → Stage 2 (gap reprices rates?) → regime/confidence/risk
production pipeline → 2Y gilt position.

**Run:**
```bash
python -m rates.run_production              # config.MODEL (default HuberNet)
RATES_MODEL=TVP python -m rates.run_production --compare
python -c "from rates import event_panel,stage1; print(stage1.stage1_test(event_panel.build_event_panel()))"
```

**Empirical verdicts (see STATE.md for detail):**
- Stage 1 vs **market-implied** (BoE 2.5Y RPI curve) → `INVALID_MECHANICAL`
  (horizon/index mismatch; guard rejects it).
- Stage 1 vs **univariate consensus** (AutoARIMA) → PASS but economically tiny
  (b=0.09, HAC t=3.0, OOS R²≈0.05); survives ex-2022/23 & ex-COVID, fails pre-2020.
- Model sweep: TVP has the largest full-sample signal but it is a 2022-23
  regime artifact; **HuberNet** is the most robust.
- Production backtest is honestly negative (Sharpe ≈ −0.7 across all models); the
  risk layer suppresses trading and the latest live recommendation is FLAT.
- Posterior that the nowcast beats a real **survey** consensus: ~15%. Blocker =
  point-in-time survey consensus data (licensed). Drop `data/consensus_cpi.csv`.

Key modules: `event_panel.py`, `gates.py` (Gate1/Gate2), `stage1.py` (guarded),
`market_implied.py`, `consensus.py`, `model_sweep.py`, `regime.py`,
`prod_signal.py`, `risk.py`, `production.py`, `run_production.py`.

---

## Files

| File | Description |
|------|-------------|
| `code/factors.py` | 38-factor registry, `apply_publication_lags()`, `factor_health()`, fetchers |
| `code/uk_model_zoo.py` | 13 operational + 9 experimental models, `dm_test()`, `score_backtest()`, `nowcast()` |
| `code/main.py` | Main CPI runner: backtest, `combine_recursive`, `common_sample_metrics`, RMC, nowcast |
| `code/sweep_factors.py` | Forward factor-addition sweep |
| `code/plot_nowcast_history.py` | Regenerates history plot from CSVs |
| `code/rates/` | Rates-repricing pipeline (event panel → gates → production) |
| `code/tests/` | `test_main.py`, `test_remediation.py`, `test_rates_*.py` (66 tests) |
| `docs/STATE.md` | Current system state, last results, rates pipeline, pending work |
| `docs/SPEC.md` | System specification and design decisions |

---

## Adding New Factors

Drop a CSV in `data/<name>.csv` with columns `[date, value]`. Then register in `factors.py`:

```python
"my_factor": dict(
    fetch=None,               # None = CSV-only, or lambda: _fred("SERIES_ID")
    transform="level",        # "level" | "yoy" | "mom" | "logret" | "diff"
    pub_lag=0,                # 0=financial, 1=ONS monthly, 2=quarterly
    candidate=True,
    csv="my_factor.csv",
    note="Source description"),
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy yfinance lightgbm shap scikit-learn statsmodels \
            fredapi requests openpyxl dbnomics pytest scipy xgboost
```

**UK model zoo:**
```bash
export FRED_API_KEY=your_key_here
# Full backtest (blind test: --end 2024 enforced)
python -W ignore code/main.py --start 2015 --end 2024 --train-from 1992 --shap-screen
# With RMC (~5–10 min extra):
python -W ignore code/main.py --start 2015 --end 2024 --train-from 1992 --shap-screen --rmc
# Tests:
.venv/bin/python -m pytest code/tests/ -q
```

**Rates pipeline:**
```bash
python -m rates.run_production --compare
```
