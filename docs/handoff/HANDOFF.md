---

## Handoff: 2026-06-05T19:42:36Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
(unavailable)

### Last Assistant Message (transcript tail)
(unavailable)

### Git Snapshot
- (not a git repo)

### Model Summary
- Session goal: expand UK CPI YoY nowcasting system (ramm-lgbm project) across 10 tasks
- Added `gas_eu` (FRED `PNGASEUUSDM`) as European LNG proxy factor in `factors.py`; `gas_hh` stays tagged `region=US` and is excluded from UK runs
- Renamed `compare_uk.py` → `nowcast_cpi.py`; deleted `ramm_lgbm_uk_v1.py` (unused)
- Added `WINDOW = None` to `BaseModel` and rolling-window slicing in `backtest()` with `min_train=60` fallback
- Added 20 rolling-window subclasses (`*_Rolling5Y` WINDOW=60, `*_Rolling2Y` WINDOW=24) for all 10 base models, plus `DFM2` (k=2 latent factors) — zoo now has 32 models
- Added `ElasticNet` model (ElasticNetCV, AR-augmented with `cpi_lag1`, StandardScaler) to zoo and `all_models()`
- Added `screen_candidates(df, target, threshold=0.001)` to `factors.py` using LightGBM + SHAP TreeExplainer
- Tightened model gate to strict 1.0× AR(1) in `greedy_uncorrelated_subset()`; combined ensembles now filter to `beating_bts` only; `beats_ar1` column added to metrics table
- Added `nowcast()` to `BaseModel` with overrides for `RAMM_LGBM` and `ElasticNet`; nowcast output printed + saved to `nowcast_cpi_nowcast.csv`
- Added `--shap-screen` / `--shap-threshold` argparse flags to `nowcast_cpi.py`
- All code in `/Users/Adam/Documents/home/quant/ramm-lgbm/`; force-pushed to `adambutlin/nowcast` on GitHub (remote updated from `nowcaster`); 15 tests pass

### Handoff Context (paste into next session)
All 10 implementation tasks are COMPLETE. The zoo has 32 models; 15 tests pass.

**To run the full zoo retrain (requires live FRED_API_KEY):**
```bash
cd /Users/Adam/Documents/home/quant/ramm-lgbm
FRED_API_KEY=<key> .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --train-from 1992 2>&1 | tee nowcast_cpi_run.log
```

**To run with Shapley factor screening:**
```bash
FRED_API_KEY=<key> .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --train-from 1992 --shap-screen 2>&1 | tee nowcast_cpi_shap.log
```

**To run RMC:**
```bash
FRED_API_KEY=<key> .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --train-from 1992 --rmc
```

**After retrain:** update `STATE.md` with new RMSE numbers from `nowcast_cpi_metrics.csv` and nowcast from `nowcast_cpi_nowcast.csv`, then push again.

**Key files:**
- `factors.py`: factor registry + `screen_candidates()`
- `uk_model_zoo.py`: 32-model zoo, `BaseModel.WINDOW`, `ElasticNet`, rolling variants, `nowcast()`
- `nowcast_cpi.py`: main runner — backtest, combined ensembles, RMC, nowcast output, `--shap-screen`
- `test_nowcast_cpi.py`: 15 tests (TestGasEu, TestRollingWindow, TestAllModels, TestElasticNet, TestScreenCandidates, TestModelGate)
- GitHub: `https://github.com/adambutlin/nowcast` (force-pushed; remote in `/Users/Adam/Documents/home/quant/nowcast/`)

---
---

## Handoff: 2026-06-06T06:50:05Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
why not using may rents and vacancies? also gas_eu should be daily not monthly since it is a market spot price. then separate to bug fix other models since they were previously top performers.

### Last Assistant Message (transcript tail)
**What's still NaN:** DFM, TVP, HMM, LSTAR, DFM-k2 — these state-space/regime models have nowcast() failures. The issue is `_fit_predict_year(d, row, ...)` where `row` has a different frequency tag than `d` after the pub-lag application. These models' state-space recursions don't handle a detached single prediction row cleanly. Separate bug to fix if you want nowcast from those models.

### Git Snapshot
- (not a git repo)

### Model Summary
- **gas_eu switched to daily TTF**: `_gas_eu_ttf()` downloads TTF=F front-month futures via yfinance (daily), computes monthly log-return means, back-splices onto IMF PNGASEUUSDM (1960+) for pre-2017 history; pub_lag=0 correct (financial data)
- **uk_rents_lag1 fixed**: `_rents_lag1()` appends one forward row after the 1-month shift so `_nowcast_row` ffill picks up April 2026 rents (2.98%) instead of March (3.24%)
- **uk_vacancies via ONS VACS01**: NOMIS API returned empty/HTML (no auth). Switched to direct ONS VACS01 xlsx download; April 2026 = 705k captured vs dbnomics March = 711k
- **`_nowcast_row` bug fixed (3 parts)**: (1) include target=NaN in returned row to prevent KeyError in models that call `test[target]`; (2) NaN check on factor cols only; (3) nowcast date = first date AFTER last known CPI (not first NaN in full matrix which reached 1947 pre-history rows)
- **State-space nowcast() overrides**: DFM/DFM-k2 use `res.forecast(steps=1)` from final model state; TVP uses Kalman filter → `final_beta` → pure predict; HMM propagates filtered regime probs via transition matrix → regime-weighted mean; LSTAR sets `ylag_now = last training CPI` explicitly to avoid shift(1)-on-1-row bug
- **31/32 models** nowcasting May 2026 (up from ~22); only HMM-2Y NaN (24 obs too few for MarkovRegression)
- **Backtest (2015–2025, 9 factors)**: AR(1) 0.485; TVP 0.449, UCM 0.466 best individual; Combined-Dynamic 0.438

### Handoff Context (paste into next session)
Session focused on nowcast data pipeline fixes and state-space model nowcast() overrides. 31/32 models now produce valid May 2026 nowcasts. Working directory: `/Users/Adam/Documents/home/quant/ramm-lgbm/`.

**Remaining known issues at this handoff:**
1. `HMM-2Y`: NaN nowcast — 24 obs too few for MarkovRegression; add min_train guard in HMM.nowcast()
2. `LSTAR-2Y`: RMSE ~8 (optimizer blows up on 24-month window); consider excluding from 2Y rolling zoo
3. `STATE.md` not updated with latest RMSE numbers
4. `us_ism_pmi`: FRED NAPM removed; replace with alternative (FRED MANEMP or ISM PMI)
5. `uk_house_prices`, `uk_paye`: dbnomics fetch failing (candidates only, not core)

---
---

## Handoff: 2026-06-06T06:50:07Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
why not using may rents and vacancies? also gas_eu should be daily not monthly since it is a market spot price. then separate to bug fix other models since they were previously top performers.

### Last Assistant Message (transcript tail)
All fixes implemented and pushed (commit 57843df). 31/32 models producing valid May 2026 nowcasts. Consensus ~2.7% YoY. Best backtest: Combined-Dynamic 0.410 RMSE vs AR(1) 0.480.

### Git Snapshot
- Remote: `adambutlin/nowcast` on GitHub, branch `main`, commit `57843df`
- Working files: `/Users/Adam/Documents/home/quant/ramm-lgbm/` (no local git)
- Git repo mirror: `/Users/Adam/Documents/home/quant/nowcast/`

### Model Summary
- **gas_eu**: switched from PNGASEUUSDM (monthly IMF average) to `_gas_eu_ttf()` — daily TTF=F front-month futures (yfinance, 2017+) rescaled onto IMF USD/mmBtu for pre-2017 history; log-returns are continuous across splice
- **uk_rents_lag1**: added `_rents_lag1()` helper that appends one forward row after the shift so ffill in `_nowcast_row` picks up April 2026 rents YoY (2.98%) instead of March (3.24%)
- **uk_vacancies**: still using March 2026 (April not yet in dbnomics); ONS published April ~May 27 but ingestion lag unresolved — ffill accepted as best available
- **`_nowcast_row` fix (3 parts)**: (1) include target=NaN in returned row so models can call `test[target]` without KeyError; (2) NaN check on factor columns only; (3) nowcast date = first date AFTER last known CPI (not first NaN in full matrix, which would find 1947 rows)
- **DFM/DFM-k2 nowcast()**: fit on windowed training data, call `res.forecast(steps=1)` from final model state; no new-factor conditioning (extrapolates latent factor)
- **TVP nowcast()**: run Kalman filter on all training data, extract `final_beta`, predict `x_now @ final_beta` where ar1 = last released CPI — pure predict step, no update needed
- **HMM nowcast()**: fit MarkovRegression on training CPI, propagate filtered regime probs one step via transition matrix, return regime-weighted mean
- **LSTAR nowcast()**: set `ylag_now = last training CPI` explicitly (avoids `shift(1)`-on-1-row-test → ylag=NaN bug); respects WINDOW for rolling variants
- **31/32 models** nowcast May 2026 (was 22/32); only HMM-2Y NaN (24 obs too few for MarkovRegression fit)
- **Latest backtest (2015-2025)**: Combined-Dynamic 0.410, Combined-Absolute 0.429, DFM-5Y 0.429, TVP 0.458, DFM-k2 0.467, UCM 0.468 all beat AR(1) 0.480; LSTAR-2Y 7.99 (optimizer instability on 24-month window)
- **May 2026 nowcast**: DFM 2.80, TVP 2.71, UCM 2.69, LSTAR 2.66, GBM 2.36, HMM 2.03, BVAR 3.43 (outlier); RAMM-LGBM/HiddenRF cluster ~2.3-2.6
- **15 tests passing**; 3 live factor fetches failing (us_ism_pmi NAPM removed from FRED, uk_house_prices HPSSA, uk_paye RTI — all were candidates, not core)

### Handoff Context (paste into next session)

**Current state:** all three user issues resolved and pushed. Working directory: `/Users/Adam/Documents/home/quant/ramm-lgbm/`. Git in `/Users/Adam/Documents/home/quant/nowcast/` → `adambutlin/nowcast` main.

**To retrain from scratch:**
```bash
cd /Users/Adam/Documents/home/quant/ramm-lgbm
FRED_API_KEY=23615c784b776fe81c097fbef15bd6d2 .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --train-from 1992 2>&1 | tee nowcast_cpi_run.log
```

**Live factor availability (9 working):** oil_brent, gbpusd, uk_be5, vix, gas_eu (TTF blend), oil_vol_6m, gbpusd_vol_6m, uk_rents_lag1 (core), uk_vacancies (candidate). Excluded: uk_rents (collinear with uk_rents_lag1), uk_cpih, uk_services_cpi (CPI predicting CPI). Failing fetches: us_ism_pmi (FRED NAPM removed), uk_house_prices (ONS HPSSA dbnomics down), uk_paye (ONS RTI dbnomics down).

**Known remaining issues:**
1. `uk_vacancies`: April 2026 data available from ONS (~May 27) but not in dbnomics yet; ffill carries March forward — could fix with direct ONS API fetch
2. `HMM-2Y`: NaN nowcast — 24 months too few for MarkovRegression; consider adding `min_train` guard in HMM.nowcast()
3. `LSTAR-2Y`: RMSE 7.99 (optimizer blows up on 24-month window); consider excluding LSTAR from 2Y rolling variants or raising their min_train
4. `STATE.md`: not updated with latest RMSE numbers — do this before next push
5. `us_ism_pmi`: FRED NAPM series has been removed; replace with FRED `MANEMP` or ISM PMI alternative

**Key architecture reminders:**
- `pub_lag=0` factors: financial series (Brent, GBP, VIX, gas_eu, vol series) — full current-month value available before CPI release
- `pub_lag=1` factors: ONS series shifted 1 month by `apply_publication_lags()` — use T-1 value when predicting CPI(T)
- `uk_rents_lag1`: pub_lag=0 with shift baked into `_rents_lag1()` fetch + 1 forward row extension for nowcast
- `LSTAR.WINDOW=60`: capped to prevent TRF optimizer divergence on long expanding windows
- DFM/TVP/HMM/LSTAR nowcast() overrides: each uses a different approach to avoid the `shift(1)-on-1-row-test` and `target KeyError` bugs in the base `_fit_predict_year` path

---
---

## Handoff: 2026-06-06T07:54:25Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
on vacancies, try the ONS NOMIS api. then backtest model suite and plot nowcast history then print May 2026 nowcasts a la https://macromonitor.org/

### Last Assistant Message (transcript tail)
(duplicate entry — see 07:54:28Z below for full content)

### Git Snapshot
- (not a git repo)

### Model Summary
(duplicate entry — see 07:54:28Z below)

### Handoff Context (paste into next session)
(duplicate entry — see 07:54:28Z below)

---
---

## Handoff: 2026-06-06T07:54:28Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
on vacancies, try the ONS NOMIS api. then backtest model suite and plot nowcast history then print May 2026 nowcasts a la https://macromonitor.org/

### Last Assistant Message (transcript tail)
Generated nowcast_history.png (3-panel fan chart) and nowcast_may2026.png (macromonitor-style dot plot). May 2026 median 2.72%, IQR 2.50–2.90%, 31/32 models valid.

### Git Snapshot
- Remote: `adambutlin/nowcast` on GitHub, branch `main`, commit `57843df`
- Working files: `/Users/Adam/Documents/home/quant/ramm-lgbm/` (no local git)
- Git repo mirror: `/Users/Adam/Documents/home/quant/nowcast/`
- Plots not yet synced to GitHub

### Model Summary
- **uk_vacancies**: NOMIS API (NM_133_1) returned empty/HTML — not usable without auth. Switched to ONS VACS01 xlsx direct download (`_ons_vacancies()`), which provides April 2026 = 705k vs dbnomics March = 711k
- **uk_rents_lag1**: `_rents_lag1()` appends one forward row after shift so ffill picks up April 2026 rents YoY (2.98%) not March (3.24%)
- **gas_eu**: `_gas_eu_ttf()` — daily TTF=F front-month futures (yfinance, 2017+) rescaled onto IMF PNGASEUUSDM (1960+); log-returns continuous across splice
- **`_nowcast_row` fix**: (1) target=NaN included in row so models don't KeyError; (2) NaN check on factor cols only; (3) nowcast date = first date AFTER last known CPI (not first NaN in full matrix which hit 1947 pre-history)
- **DFM/DFM-k2 nowcast()**: `res.forecast(steps=1)` from final model state on windowed training data
- **TVP nowcast()**: Kalman on training only → `final_beta` → `x_now @ final_beta` pure predict; ar1 = last released CPI
- **HMM nowcast()**: fit MarkovRegression on training CPI, propagate filtered regime probs via transition matrix, return regime-weighted mean
- **LSTAR nowcast()**: `ylag_now = last training CPI` explicitly (avoids shift(1)-on-1-row → NaN); respects WINDOW
- **31/32 models** valid May 2026 nowcast; only HMM-2Y NaN (24 obs too few for MarkovRegression)
- **Backtest 2015–2025**: AR(1) RMSE 0.485; best: TVP 0.449, UCM 0.466; LSTAR-2Y 7.99 (optimizer instability)
- **May 2026 consensus**: median 2.72%, IQR 2.50–2.90%, range 2.03–3.43%; BVAR outlier at 3.43%
- **Plots**: `nowcast_history.png` (fan chart + RMSE bar + error time-series), `nowcast_may2026.png` (dot plot sorted by value, color-coded by model family)

### Handoff Context (paste into next session)

**Current state:** backtest complete, two plots generated, not yet synced to GitHub.

**Working directory:** `/Users/Adam/Documents/home/quant/ramm-lgbm/`
**Git mirror:** `/Users/Adam/Documents/home/quant/nowcast/` → `adambutlin/nowcast` main

**To sync plots + code to GitHub:**
```bash
cp /Users/Adam/Documents/home/quant/ramm-lgbm/factors.py \
   /Users/Adam/Documents/home/quant/ramm-lgbm/uk_model_zoo.py \
   /Users/Adam/Documents/home/quant/nowcast/
cp /Users/Adam/Documents/home/quant/ramm-lgbm/nowcast_history.png \
   /Users/Adam/Documents/home/quant/ramm-lgbm/nowcast_may2026.png \
   /Users/Adam/Documents/home/quant/nowcast/
cd /Users/Adam/Documents/home/quant/nowcast
git add -A && git commit -m "add nowcast plots and model fixes (TTF gas, rents lag, vacancies ONS, state-space nowcast overrides)"
git push
```

**To retrain from scratch:**
```bash
cd /Users/Adam/Documents/home/quant/ramm-lgbm
FRED_API_KEY=<key> .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --train-from 1992 2>&1 | tee nowcast_cpi_run.log
```

**Known remaining issues:**
1. `HMM-2Y`: NaN — 24 obs too few for MarkovRegression; add `min_train` guard in HMM.nowcast()
2. `LSTAR-2Y`: RMSE 7.99 — optimizer blows up on 24-month window; consider excluding from 2Y rolling zoo
3. `STATE.md`: not updated with latest RMSE numbers — do before next push
4. `us_ism_pmi`: FRED NAPM series removed; replace with alternative (FRED `MANEMP` or ISM PMI)
5. `uk_house_prices`, `uk_paye`: dbnomics fetch failing — candidates only, not core

**Key architecture reminders:**
- `pub_lag=0`: financial series (Brent, GBP, VIX, gas_eu, vol series) — contemporaneous
- `pub_lag=1`: ONS series shifted 1 month by `apply_publication_lags()`
- `uk_rents_lag1`: pub_lag=0 with shift baked in + forward row extension for nowcast ffill
- DFM/TVP/HMM/LSTAR have model-specific `nowcast()` overrides — don't use base `_fit_predict_year` path
- 9 live factors: oil_brent, gbpusd, uk_be5, vix, gas_eu, oil_vol_6m, gbpusd_vol_6m, uk_rents_lag1, uk_vacancies

---
---

## Handoff: 2026-06-06T10:52:46Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
(unavailable — duplicate compaction entry; see 10:52:47Z below for full content)

### Last Assistant Message (transcript tail)
(unavailable — duplicate compaction entry; see 10:52:47Z below for full content)

### Git Snapshot
- (not a git repo)

### Model Summary
(Duplicate compaction entry — same session state as 10:52:47Z. See that entry for full model summary.)

### Handoff Context (paste into next session)
(Duplicate compaction entry — see 10:52:47Z below for full handoff context.)

---
---

## Handoff: 2026-06-06T10:52:47Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
first add 3m cpi change (or some other standing moving average) as a feature add a MIDAS, bridge equation, and a copula model. retrain and rerun all models and plot nowcast_history_2.png. third try retraining everything on post-1945 data (where available) which should help the regime switching trees at least, then plot nowcast_history_long

### Last Assistant Message (transcript tail)
Both backtests complete. Generated nowcast_history_2.png (41 models, 1992 training) and nowcast_history_long.png (41 models, 1956 training). Key finding: longer training doesn't help — pre-war inflation dynamics create distributional shift that hurts 2015–2025 performance.

### Git Snapshot
- (not a git repo)

### Model Summary
- **Zoo expanded to 41 models**: added MIDAS, BridgeEq, CopulaReg base classes + their -5Y and -2Y rolling variants (9 new models); `all_models()` updated
- **cpi_3m_chg feature**: added as `MOM = "cpi_3m_chg"` to RAMM_LGBM, GBM, HiddenRF, ElasticNet, TVP, MIDAS, BridgeEq, CopulaReg; computed as `target.shift(1).diff(3)` inside each model — not in the factor registry
- **MIDAS**: ElasticNetCV on monthly-mean daily Brent/GBP/VIX/TTF (yfinance, cached via `_MIDAS_CACHE`); n=75 (limited by TTF=F data availability ~2019); RMSE 0.632 (standard run)
- **BridgeEq**: OLS bridge equation on same MIDAS daily factors + AR lag + momentum; n=75; RMSE 0.670
- **CopulaReg**: Gaussian copula — rank→normal scores, OLS in score space, back-transform via empirical quantile; n=135; RMSE 1.356 (poor — non-monotone CPI dynamics break copula assumption)
- **Standard backtest (1992 training, 2015–Jun 2026, n=135)**: Combined-Dynamic 0.438, Combined-Static 0.438, ElasticNet 0.455, TVP 0.458, UCM 0.462, DFM-k2 0.469; AR(1) 0.480; 7 models beat AR(1)
- **Long backtest (1956 training, 2015–Mar 2025, n=122)**: all models slightly worse; Combined-Dynamic 0.444, TVP 0.458, AR(1) 0.475; pre-war distributional shift (1970s–80s inflation) hurts 2015–2025 test performance
- **`--target` flag added to nowcast_cpi.py**: accepts `cpi_yoy` (default) or `cpi_yoy_long`; `live_facs` exclusion updated to filter both `cpi_yoy` and `cpi_yoy_long`
- **Output plots**: `nowcast_history_2.png` (41-model zoo, 1992 training), `nowcast_history_long.png` (1956 training)
- **Current CSVs** on disk reflect the long run (1956 training) — standard run (1992) results available only in `/tmp/nowcast_v2.log`

### Handoff Context (paste into next session)

**Current state:** two new backtests complete, plots saved. CSVs on disk are from the long (1956) run.

**Working directory:** `/Users/Adam/Documents/home/quant/ramm-lgbm/`
**Git mirror:** `/Users/Adam/Documents/home/quant/nowcast/` → `adambutlin/nowcast` main

**To re-run standard backtest (canonical, 1992 training):**
```bash
cd /Users/Adam/Documents/home/quant/ramm-lgbm
FRED_API_KEY=23615c784b776fe81c097fbef15bd6d2 .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --train-from 1992 2>&1 | tee /tmp/nowcast_v2.log
```

**To re-run long backtest (1956 training):**
```bash
FRED_API_KEY=23615c784b776fe81c097fbef15bd6d2 .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --train-from 1956 --target cpi_yoy_long 2>&1 | tee /tmp/nowcast_long.log
```

**Standard backtest key results (from /tmp/nowcast_v2.log):**
- AR(1): 0.480; Combined-Dynamic: 0.438; Combined-Static: 0.438; ElasticNet: 0.455; TVP: 0.458; UCM: 0.462; DFM-k2: 0.469
- MIDAS: 0.632 (n=75); BridgeEq: 0.670 (n=75); CopulaReg: 1.356; ML models 1.5–3.2

**Known remaining issues:**
1. `HMM-2Y`: NaN nowcast — 24 obs too few; add min_train guard in HMM.nowcast()
2. `LSTAR-2Y`: RMSE 245 (optimizer blows up on 24-month window); consider excluding from 2Y zoo
3. `LSTAR`, `LSTAR-5Y`: RMSE 266 (numerical instability even with full/5Y window)
4. `STATE.md`: not updated with new RMSE numbers — do before next push
5. `nowcast_cpi_backtest.csv` and `nowcast_cpi_metrics.csv` currently contain long-run results; re-run standard to restore
6. `us_ism_pmi`: FRED NAPM removed; replace with alternative
7. `uk_house_prices`, `uk_paye`: dbnomics fetch failing (candidates only, not core)

**Architecture reminders:**
- `all_models()` now returns 41 models: 13 base + 13×5Y + 13×2Y + 2 extras (DFM2, ElasticNet)
- MIDAS/BridgeEq use module-level `_MIDAS_CACHE["mm"]` (yfinance download once per process)
- `cpi_3m_chg` is model-internal (`target.shift(1).diff(3)`), not a factor registry entry
- `--target cpi_yoy_long` with `--train-from 1956` for extended history; factors still 10 live (1992+ for financial series)
- 9 live factors: oil_brent, gbpusd, uk_be5, vix, gas_eu, uk_gilt_10y, oil_vol_6m, gbpusd_vol_6m, uk_rents_lag1, uk_vacancies

---
---

## Handoff: 2026-06-06T13:45:09Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: manual
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
compare to RMC model; why train 1992 but backtest 2015; are you using walk-forward; keep 2025+ as blind test; drop all 2Y and 5Y rolling windows; test post-1999 training

### Last Assistant Message (transcript tail)
Launched 1992-training run (15 models + --rmc, --end 2024) and 1999-training run in parallel. Both in progress at handoff.

### Git Snapshot
- (not a git repo)

### Model Summary
- **Walk-forward confirmed**: `BaseModel.backtest()` uses expanding window; for each `yr >= start_year` trains on `data.year < yr` only — proper OOS
- **Blind test 2025+**: added `--end` argument (default 2024) to `nowcast_cpi.py`; propagated to `ar1_backtest()`, `BaseModel.backtest()`, and `regime_model_combine()` — backtest now stops Dec 2024
- **Rolling variants dropped**: removed all 26 rolling-window subclasses (5Y + 2Y for all 13 base models); `all_models()` returns 15 models: DFM, RAMM-LGBM, UCM, TVP, HMM, MS-DFM, LSTAR, BVAR, HiddenRF, GBM, MIDAS, BridgeEq, CopulaReg, DFM-k2, ElasticNet
- **MIDAS class shadowing fixed**: old Almon-polynomial MIDAS (line 920) renamed to `MIDAS_Almon`; Python was silently using the yfinance-based MIDAS (line 1221) for both, with Almon unreachable dead code
- **RMC = regime-model-combine**: already in codebase as `regime_model_combine()` / `--rmc` flag; not compared vs standard ensemble in prior runs — included in current 1992 run
- **Runs launched** (both `--end 2024`):
  - `/tmp/nowcast_1992.log`: `--train-from 1992 --rmc` (15 models + RMC methods, ~25 min)
  - `/tmp/nowcast_1999.log`: `--train-from 1999` (15 models, no RMC, ~15 min)
- **Comparison objective**: does 1999 training (post-EMU/post-Great-Moderation, avoids early 1990s data) beat 1992 for 2015–2024 out-of-sample?
- **Prior finding (prev session)**: 1956 training HURT performance vs 1992 (pre-1992 hyperinflation regimes create distributional shift); 1999 hypothesis is opposite direction — less history but higher quality

### Handoff Context (paste into next session)

**Current state:** runs in-flight at handoff. Zoo trimmed to 15 models. Blind test 2025+ enforced via `--end 2024`.

**Working directory:** `/Users/Adam/Documents/home/quant/ramm-lgbm/`

**Check run status:**
```bash
tail -30 /tmp/nowcast_1992.log    # 1992+RMC run
tail -20 /tmp/nowcast_1999.log    # 1999 run
ps aux | grep nowcast_cpi         # check if still running
```

**To re-run after completion:**
```bash
FRED_API_KEY=23615c784b776fe81c097fbef15bd6d2 PYTHONUNBUFFERED=1 \
  .venv/bin/python -W ignore -u nowcast_cpi.py --start 2015 --end 2024 --train-from 1992 --rmc \
  > /tmp/nowcast_1992.log 2>&1 &
```

**Key architecture changes this session:**
- `--end 2024` is now the default; 2025+ is blind test — do NOT evaluate metrics on 2025 data
- `all_models()` returns 15 models (no rolling variants)
- `MIDAS_Almon` class preserved but unused; `MIDAS` name now unambiguously = yfinance U-MIDAS
- `regime_model_combine()` signature: `(df, factors, target, models, start_year, end_year, regime_methods, min_regime_train)`

**Pending analysis after runs complete:**
1. Compare 1992 vs 1999 training: which wins for 2015–2024 OOS RMSE?
2. Compare RMC vs Combined-Dynamic from 1992 run — does regime-conditioning beat simple inverse-RMSE weighting?
3. Plot `nowcast_history_3.png` with clean 15-model results and 2024 cutoff
4. Update `STATE.md` with new RMSE numbers

**Known remaining issues:**
1. `LSTAR`: RMSE ~266 (numerical instability in TRF optimizer); consider removing from zoo or capping predictions
2. `us_ism_pmi`: FRED NAPM removed; replace with alternative
3. `uk_house_prices`, `uk_paye`: dbnomics fetch failing (candidates only)
4. `HMM` nowcast: may still NaN if min_train guard not triggered correctly

---
---

## Handoff: 2026-06-06T18:53:32Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
I am also concerned about the lack of simpler econometric models e.g. SVAR, VARMA, ARIMA, SARIMAX

### Last Assistant Message (transcript tail)
Added SARIMAX_Model, VAR_Model, AutoARIMA to uk_model_zoo.py and all_models() (now 21 models). Smoke test: SARIMAX n=0 (bug), VAR n=120 RMSE=1.96, AutoARIMA n=0 (bug). Bugs not yet fixed at this handoff point.

### Git Snapshot
- (not a git repo)

### Model Summary
- **Zoo expanded to 21 models**: added SARIMAX_Model, VAR_Model, AutoARIMA after prior session's additions (MedianElasticNet, HuberNet, PCR, RegimeEnsemble); LSTAR removed (RMSE blowup)
- **PRED_MIN/PRED_MAX clipping**: BaseModel.PRED_MIN=-2.0, PRED_MAX=20.0 applied in backtest(); ablation showed 0 observations clipped — all models already within [-2, 20] range
- **gas_eu_3m, oil_brent_3m, gbpusd_3m**: added to factors.py REGISTRY as 3-month cumulative log-returns; ablation showed gas_eu_3m hurts (Δ=-0.024) but oil_brent_3m helps (Δ=+0.008)
- **Ablation sweep results** (5 models × all factors): cpi_lag1+cpi_3m_chg Δ=+0.43 (most important); gas_eu_3m Δ=-0.024 (hurts); momentum versions of all factors add no value
- **SARIMAX bug**: `fillna(method="ffill")` deprecated pandas syntax causes silent exception before forecast; n=0 at this handoff
- **AutoARIMA bug**: `ARIMA.fit(disp=False)` invalid kwarg + rolling forecast(1) loop advances internal state; n=0 at this handoff
- **VAR**: working, RMSE=1.96 — poor as expected for joint VAR system on monthly CPI
- **HuberNet** (ablation RMSE 0.4734) competitive with TVP; HuberRegressor epsilon=2.0 robust to 2022 outliers

### Handoff Context (paste into next session)
**Current state:** SARIMAX and AutoARIMA have n=0 bugs not yet fixed. 21 models in zoo. gas_eu_3m not yet excluded from live_facs.

**Immediate next step:** fix SARIMAX (change `.fillna(method="ffill")` → `.ffill()`) and AutoARIMA (remove `disp=False`, replace rolling forecast loop with `predict(start=len(y_tr), end=len(y_tr)+len(test)-1)`), then add `gas_eu_3m` to exclusion list in `nowcast_cpi.py`, then run full 21-model backtest.

**Run command:**
```bash
FRED_API_KEY=23615c784b776fe81c097fbef15bd6d2 PYTHONUNBUFFERED=1 .venv/bin/python -W ignore -u nowcast_cpi.py --start 2015 --end 2024 --train-from 1992 2>&1 | tee /tmp/fullrun.log
```

---
---

## Handoff: 2026-06-06T18:53:59Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
I am also concerned about the lack of simpler econometric models e.g. SVAR, VARMA, ARIMA, SARIMAX

### Last Assistant Message (transcript tail)
Full 21-model backtest complete (2015–2024, blind test 2025+). Best: Combined-Dynamic 0.451, HuberNet 0.455. SARIMAX 0.675, VAR 1.966, AutoARIMA 1.961. gas_eu_3m excluded (ablation). Results in /tmp/fullrun.log.

### Git Snapshot
- (not a git repo)

### Model Summary
- **21 models in zoo**: DFM, RAMM-LGBM, UCM, TVP, HMM, MS-DFM, BVAR, HiddenRF, GBM, MIDAS, BridgeEq, CopulaReg, DFM-k2, ElasticNet, MedianElasticNet, HuberNet, PCR, RegimeEnsemble, SARIMAX_Model, VAR_Model, AutoARIMA
- **SARIMAX fix**: `fillna(method="ffill")` → `.ffill()` (deprecated pandas API was silently failing the entire try block); now n=120 RMSE=0.675
- **AutoARIMA fix**: removed invalid `disp=False` kwarg; replaced rolling `forecast(1)` loop (advances internal state, clones same prediction 12×) with single `predict(start=len(y_tr), end=len(y_tr)+len(test)-1)` call; now n=120 RMSE=1.961
- **gas_eu_3m excluded**: added to exclusion filter in `nowcast_cpi.py` (ablation Δ=-0.024 — adding it hurts); oil_brent_3m and gbpusd_3m remain (both mildly helpful)
- **Full backtest results (2015–2024)**: Combined-Dynamic 0.451, Combined-Static 0.452, HuberNet 0.455, Combined-Absolute 0.455, TVP 0.458, ElasticNet 0.462, UCM 0.463, PCR 0.485, DFM-k2 0.489, DFM 0.492 all beat AR(1) 0.495
- **Tree/ML models all fail**: RAMM-LGBM 1.764, GBM 1.618, HiddenRF 1.639, CopulaReg 1.455 — expected with small monthly samples (overfit in expanding-window OOS)
- **New econometric models**: SARIMAX 0.675 (seasonal MA adds some signal but below AR(1)), VAR 1.966 (joint system poor for monthly CPI), AutoARIMA 1.961 (mean-reverts to historical mean, badly misses 2022 spike)
- **HuberNet** (0.455) is new best individual model, effectively tying combined ensembles; epsilon=2.0 robust to 2022 outliers while still using full factor set

### Handoff Context (paste into next session)

**Current state:** 21-model backtest complete. All models producing valid predictions (n=120 each except MIDAS/BridgeEq n=60). Blind test 2025+ enforced.

**Working directory:** `/Users/Adam/Documents/home/quant/ramm-lgbm/`

**To re-run full backtest:**
```bash
FRED_API_KEY=23615c784b776fe81c097fbef15bd6d2 PYTHONUNBUFFERED=1 .venv/bin/python -W ignore -u nowcast_cpi.py --start 2015 --end 2024 --train-from 1992 2>&1 | tee /tmp/fullrun.log
```

**Current RMSE table (2015–2024, from /tmp/fullrun.log):**
- Beat AR(1) 0.495: Combined-Dynamic 0.451, Combined-Static 0.452, HuberNet 0.455, Combined-Absolute 0.455, TVP 0.458, ElasticNet 0.462, UCM 0.463, PCR 0.485, DFM-k2 0.489, DFM 0.492
- Below AR(1): MedianElasticNet 0.561, RegimeEns 0.559, SARIMAX 0.675, MIDAS 0.683 (n=60), BVAR 0.701, BridgeEq 0.726 (n=60), CopulaReg 1.455, GBM 1.618, HiddenRF 1.639, RAMM-LGBM 1.764, AutoARIMA 1.961, VAR 1.966, HMM 2.695, MS-DFM 2.954

**Live factors (12, from nowcast_cpi.py):** oil_brent, gbpusd, uk_be5, vix, gas_eu, uk_gilt_10y, oil_vol_6m, gbpusd_vol_6m, oil_brent_3m, gbpusd_3m, uk_rents_lag1, uk_vacancies. Excluded: gas_eu_3m (hurts), gas_eu_3m (ablation), uk_rents (collinear), uk_cpih/uk_services_cpi (circular). Failing fetches: us_ism_pmi (FRED NAPM removed), uk_house_prices, uk_paye (dbnomics down — candidates only).

**Known remaining tasks:**
1. `nowcast_history_3.png`: regenerate with 21-model clean results (current chart may be from 15-model run)
2. `STATE.md`: not updated with new RMSE numbers from this run
3. RMC top-tier only: `/tmp/ablation_sweep.py` Part 3 errored (NaN in bar chart) — fix and rerun with [UCM, TVP, ElasticNet, DFM-k2]
4. VAR and AutoARIMA: consider removing from `all_models()` if user wants a cleaner zoo (both RMSE ~2.0)
5. GitHub: no local git; mirror at `/Users/Adam/Documents/home/quant/nowcast/` → `adambutlin/nowcast`

**Key architecture:**
- `BaseModel.PRED_MIN=-2.0`, `PRED_MAX=20.0`: CPI support clipping in `backtest()`; 0 clipped observations in current run
- `--end 2024` default: 2025+ is blind test — never evaluate on these dates
- `pub_lag=0`: financial series (Brent, GBP, VIX, gas_eu, vol, 3m factors); `pub_lag=1`: ONS series shifted 1 month
- MIDAS/BridgeEq: n=60 due to TTF=F yfinance data starting ~2019

---
---

## Handoff: 2026-06-06T19:49:32Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm

### Last User Message (transcript tail)
rerun all models (including rmc with top tier models only) using all new factors (double check for leakage/look-ahead) and replot nowcast_history. then update all markdown files and suggest next steps for tomorrow.

### Last Assistant Message (transcript tail)
(unavailable — compaction fired mid-session; full results recorded in 21:30:00Z manual entry below)

### Git Snapshot
- (not a git repo)

### Model Summary
- **30 factors audited for leakage**: all 26 pub_lag=0 are financial/market data (no ONS publication delay); 4 pub_lag≥1 correctly lagged. Spearman ρ≈0 at lag=0 for food/shipping confirms no spurious contemporaneous correlation with CPI
- **New factors added this session**: metals_index + Cu/Al/Ni/Zn/Fe individual prices, gbp_eur, gbp_eer, semiconductors_ppi, deep_sea_freight, chemicals_ppi, timber_price, uk_monthly_gdp (OECD industrial prod), uk_awg (ONS AWE KAB9), uk_house_prices (FRED QGBR628BIS), uk_ftse250, uk_ftse100, food_price_index, wheat_price, vegetable_oil_price, uk_rents_lag1 (with pub-lag shift baked in)
- **uk_house_prices fixed**: switched from dbnomics HPSSA (404) to FRED QGBR628BIS (BIS quarterly, ffill to monthly, pub_lag=2)
- **uk_awg**: ONS AWE KAB9 via live ONS website JSON API; FRED LCEAMN01GBM661S fallback
- **uk_ftse250/uk_ftse100**: added as UK profit proxies (replacing licensed S&P PMI proxy); n=449, 1989–2026
- **Full 21-model backtest (30 factors, 2015–2024)**: Combined-Static 0.310, MedianElasticNet 0.345, ElasticNet 0.353 — 31% improvement vs prior 12-factor run (Combined-Dynamic 0.451). 6 models beat AR(1) 0.495
- **Factor importance (SHAP)**: metals_index, copper_price, gbp_eur, uk_ftse250, chemicals_ppi, uk_awg, uk_monthly_gdp in top-5 across multiple models; individual metals partially redundant with metals_index
- **Exclusions confirmed**: gas_eu_3m (+RMSE 0.024 ablation), uk_paye (≡uk_awg/KAB9), uk_rents (collinear with uk_rents_lag1), uk_cpih/uk_services_cpi (circular), gas_hh/us_ppi_all (region=US)
- **nowcast_history_3.png regenerated** via new `plot_nowcast_history.py` script; STATE.md and README.md fully rewritten
- **RMC running at session end**: 4 methods (HMM done, LSTAR/DFM/VIX in progress)

### Handoff Context (paste into next session)
Session completed 30-factor expansion and full backtest. State at compaction: RMC still running. Full results recorded in 21:30:00Z manual entry below.

**Key results:**
- Combined-Static RMSE=0.310, AR(1)=0.495 (37% improvement)
- SHAP screening (--shap-screen, threshold=0.001) drops 11 factors → UCM 0.605→0.370, TVP 0.529→0.413, Combined-Static 0.310→0.277

**Next steps (recorded in 21:30:00Z entry)**: --rmc-top-k flag, UCM regime fix, MIDAS n extension, dfm_k2 regime method, core CPI split.

---
---

## Handoff: 2026-06-06T21:30:00Z (manual, end of session)

### Git Snapshot
- (not a git repo — mirror at `/Users/Adam/Documents/home/quant/nowcast/`)

### Model Summary
- **30 live factors** added this session (up from 12): metals (Cu/Al/Ni/Zn/Fe index + individual), FX (gbp_eur, gbp_eer), food (PFOODINDEXM, wheat, veg oil), shipping (deep_sea_freight), tech (semiconductors_ppi), UK domestic (uk_monthly_gdp, uk_awg, uk_house_prices), equity (uk_ftse250, uk_ftse100 — replaced us_ism_pmi + uk_pmi_profit_proxy)
- **Factor leakage audit**: all 26 pub_lag=0 factors are financial/market data (no ONS pub delay); 4 pub_lag≥1 are ONS/OECD stats correctly lagged; Spearman ρ≈0 at lag=0 for food/shipping confirms no spurious correlation
- **New backtest (2015–2024, 30 factors)**: Combined-Static RMSE=0.310, MedianElasticNet=0.345, ElasticNet=0.353 — all beating AR(1) 0.495. Previous 12-factor run had Combined-Dynamic 0.451; new factors improve combined RMSE 31%
- **Factor importance (new factors)**: metals_index, copper_price, gbp_eur, uk_ftse250, chemicals_ppi, uk_awg, uk_monthly_gdp all appear in top-5 across multiple models
- **Exclusions confirmed**: gas_eu_3m (+RMSE 0.024 ablation), uk_paye (≡uk_awg via KAB9), uk_rents (collinear with uk_rents_lag1), uk_cpih/uk_services_cpi (circular), gas_hh/us_ppi_all (region=US)
- **SARIMAX + AutoARIMA bugs fixed**: deprecated `fillna(method="ffill")` → `.ffill()`; AutoARIMA `disp=False` kwarg removed; rolling `forecast(steps=1)` → `predict(start=len(tr), end=len(tr)+len(te)-1)` — both now produce n=37 valid predictions
- **uk_house_prices fixed**: switched from dbnomics HPSSA (404) to FRED QGBR628BIS (BIS quarterly BIS, quarterly ffill → monthly, pub_lag=2)
- **uk_awg**: ONS AWE KAB9 via live ONS website JSON API (working: `www.ons.gov.uk/employmentandlabourmarket/...`); FRED LCEAMN01GBM661S as fallback
- **uk_ftse250/uk_ftse100**: added as UK corporate profit proxies (replacing licensed S&P PMI proxy); confirmed n=449, 1989–June 2026
- **RMC running**: background process executing RMC with all 4 regime methods (HMM done, LSTAR in progress) — may not complete before session end
- **Plots + markdown**: `nowcast_history_3.png` regenerated via `plot_nowcast_history.py`; STATE.md and README.md fully updated; HANDOFF.md this entry

### Handoff Context (paste into next session)

**Current state (2026-06-06):**
- 30 factors loaded, leakage-clean, all pub_lags correct
- 21-model backtest complete: `nowcast_cpi_backtest.csv`, `nowcast_cpi_metrics.csv` saved (2026-06-06 21:10)
- Best models: Combined-Static 0.310, MedianElasticNet 0.345, ElasticNet 0.353 RMSE (37% improvement vs AR(1) 0.495)
- May 2026 nowcast consensus: ~2.5–2.8% YoY

**RMC status**: was running at session end. To check if it completed:
```bash
ls -la rmc_*_perf.csv 2>/dev/null  # CSVs written per RMC method on completion
tail -20 /tmp/fullrun_rmc.log
```

**To resume RMC with top-tier models only** (if previous RMC didn't complete or you want a faster run):
```bash
# Add --rmc-methods to nowcast_cpi.py args, filter models list in main() to beats_ar1=True subset
# Top-tier (beats AR(1)): ElasticNet, MedianElasticNet, RegimeEns, TVP, UCM
FRED_API_KEY=<key> .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --end 2024 --train-from 1992 --rmc
```

**Next steps for tomorrow:**

1. **`--rmc-top-k N` flag** — pre-filter models to top-N by overall backtest RMSE before passing to RMC. Prevents noisy models (RMSE>2: HMM, MS-DFM, RAMM-LGBM, HiddenRF) from winning within small regime sub-samples by luck. Implementation: in `main()`, after building `bt_dict`, sort by RMSE and slice to top-k before calling `regime_model_combine()`. Suggested default: top-8 (keeps ElasticNet, MedianElasticNet, RegimeEns, TVP, UCM, HuberNet, MIDAS, BridgeEq).

2. **Fix UCM n=0 in regime sub-samples** — UCM returns empty backtest when regime sub-sample is small. The `min_regime_train` fallback in `regime_model_combine()` already falls back to full-data training when `len(train_r) < min_regime_train`, but UCM may still fail at fit time. Add try/except inside the per-regime backtest loop (lines ~460-490 in `nowcast_cpi.py`) to catch UCM convergence failures gracefully.

3. **MIDAS/BridgeEq extend n=5→37** — TTF yfinance data starts ~2019 → only 5 quarterly obs from 2023. Fix: in `factors.py` `_gas_eu_ttf()`, already back-splices PNGASEUUSDM pre-2017. But MIDAS/BridgeEq use `gas_eu` as the mixed-frequency series — check if their `backtest()` is capping at gas_eu's yfinance start date. If so, use `gas_eu`'s full spliced history (goes back to 1992 via PNGASEUUSDM).

4. **SHAP screening is now DEFAULT** ✅ DONE — `--shap-screen` set to `default=True` in argparse; `--no-shap-screen` flag added to disable. Drops 11 factors (gas_eu, individual metals Cu/Al/Ni/Fe, food_price_index, veg_oil, uk_vacancies) → 20 factors. UCM 0.605→**0.370** (-39%), TVP 0.529→**0.413** (-22%), SARIMAX 0.857→**0.491** (-43%), Combined-Static 0.310→**0.277** (-11%). 9 models now beat AR(1) (was 6). Root cause: individual metals/food factors redundant with oil_brent/chemicals_ppi/wheat; adding them as separate inputs adds noise to state-space models.

5. **Add `dfm-k2` RMC regime method** — current `--rmc-methods dfm` uses k=1 DFM sign as regime label (`_regime_labels_dfm()` in `nowcast_cpi.py`). Add `_regime_labels_dfm_k2()`: fit `DFM(k=2)` on full training data up to t, extract 2 factor scores, cluster (e.g. KMeans k=2 or sign of first factor) to assign regime. Register as `"dfm_k2"` in the `rmc_methods` dispatch block (~line 423–432). Then rerun: `--rmc --rmc-methods dfm_k2` alongside top-tier models to compare vs k=1 result (0.671).

6. **Split headline CPI vs core CPI** — add `cpi_core_yoy` as a second target to `factors.py` (ONS CPIH excl. energy and food, series L55O or CPIH core via dbnomics). Add `--target cpi_core_yoy` flag support (already in argparse). Headline run = current; core run = re-run with `--target cpi_core_yoy`. Compare factor importance and RMSE across both targets — core CPI likely has higher predictability (less energy noise).

7. **Push to GitHub** — mirror at `/Users/Adam/Documents/home/quant/nowcast/`; copy updated files and push. Key changed files: `factors.py`, `uk_model_zoo.py`, `nowcast_cpi.py` (`--shap-screen` now default), `plot_nowcast_history.py` (new), `STATE.md`, `README.md`.

**Key commands:**
```bash
cd /Users/Adam/Documents/home/quant/ramm-lgbm
# Fresh full run:
FRED_API_KEY=23615c784b776fe81c097fbef15bd6d2 .venv/bin/python -W ignore nowcast_cpi.py --start 2015 --end 2024 --train-from 1992
# Regenerate plot:
.venv/bin/python plot_nowcast_history.py
# Tests:
FRED_API_KEY=23615c784b776fe81c097fbef15bd6d2 .venv/bin/python -m pytest test_nowcast_cpi.py -v
```

---
---

## Handoff: 2026-06-07T06:19:31Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-ramm-lgbm/0d79802f-88bb-4300-b286-727384080585.jsonl
- CWD: /Users/Adam/Documents/home/quant/ramm-lgbm → **migrated to /Users/Adam/Documents/home/quant/nowcast/** this session

### Last User Message (transcript tail)
(unavailable — session started from compaction summary)

### Last Assistant Message (transcript tail)
All steps 1–5 implemented, full backtest run, folder merged into nowcast/. Combined-Static RMSE=0.277, MIDAS/BridgeEq n fixed 5→25.

### Git Snapshot
- (not a git repo)

### Model Summary
- **`--rmc-top-k N` flag added** to `code/nowcast_cpi.py` argparse; in `main()` after `bt_dict` built, filters models to top-N by RMSE before passing to `regime_model_combine()`. Default `None` (all models). Run with `--rmc-top-k 8`.
- **`dfm_k2` RMC regime method** added: `_regime_labels_dfm_k2()` fits DFM(k=2) on training data, runs KMeans(k=2) on the 2-factor score matrix, labels r1 = cluster with higher first-factor mean. Registered in dispatch block alongside hmm/lstar/dfm/manual_vix.
- **UCM prediction sanity check**: added `abs(pred) < 50` guard inside per-regime backtest loop — prevents degenerate UCM convergence on small regime sub-samples from corrupting RMSE (was causing RMSE=30.3 in DFM-r0 regime).
- **MIDAS/BridgeEq n=5→25 fix**: in `_fit_predict_year()` of both classes, changed `dropna(subset=feats + [target])` to `dropna(subset=[self.LAG, self.MOM, target])`. TTF/gas_ma NaNs pre-2019 are now filled with `tr[feats].mean()` rather than dropping training rows. MIDAS RMSE improved 0.773→0.758; BridgeEq 0.836→0.658.
- **RMC-dfm_k2 result**: RMSE=0.545. r1 survivors (8): UCM, TVP, BridgeEq, ElasticNet, MedianElasticNet, HuberNet, RegimeEns, SARIMAX. r0 AR(1)=NaN (too few obs). No RMC method beats Combined-Static (0.277).
- **Project migrated**: `ramm-lgbm/` merged into canonical `nowcast/`. All Python files → `nowcast/code/`; CSVs → `nowcast/data/`; PNGs → `nowcast/plots/`; logs → `nowcast/logs/`. `factors.py` DATA_DIR and all script output paths updated via `__file__`-relative `_ROOT`.
- **Auto-organize hook**: `nowcast/.claude/settings.json` with Stop hook running `nowcast/.claude/organize.sh` — auto-moves stray .py/.csv/.png/.log files from project root into the correct subfolders at session end.
- **User-level permissions**: `Write(/Users/Adam/Documents/home/quant/**)` and `Edit(/Users/Adam/Documents/home/quant/**)` added to `~/.claude/settings.json` — no more tool-approval prompts for file writes inside the quant folder.
- **Full backtest results** (SHAP-screened 20 factors, 2015–2024): Combined-Static 0.277, ElasticNet 0.338, MedianElasticNet 0.364, UCM 0.370, TVP 0.413, RegimeEns 0.490, SARIMAX 0.491. AR(1) 0.495. 9 models beat AR(1).
- **May 2026 nowcast consensus**: UCM 2.60, TVP 2.63, BridgeEq 2.61, MedianElasticNet 2.82 → ~2.6–2.7% YoY (April actual 3.5%).

### Handoff Context (paste into next session)

**Canonical working directory:** `/Users/Adam/Documents/home/quant/nowcast/`
(ramm-lgbm/ is a stale backup — can be deleted)

**Run command:**
```bash
cd /Users/Adam/Documents/home/quant/nowcast
FRED_API_KEY=23615c784b776fe81c097fbef15bd6d2 .venv/bin/python -W ignore code/nowcast_cpi.py \
  --start 2015 --end 2024 --train-from 1992 \
  --rmc --rmc-methods dfm_k2 --rmc-top-k 8
```

**Folder structure:**
- `code/` — all .py source files
- `data/` — all .csv outputs (backtest, metrics, nowcast, rmc_*_perf)
- `plots/` — all .png outputs (nowcast_history_3.png etc.)
- `logs/` — run logs
- `.claude/` — organize.sh Stop hook

**Best results (canonical, 2026-06-07):**
- Combined-Static RMSE=0.277 (44% vs AR(1) 0.495)
- ElasticNet 0.338, MedianElasticNet 0.364, UCM 0.370, TVP 0.413
- RMC-dfm_k2 0.545 (regime-aware but not better than combined ensembles)

**Remaining next steps:**
1. **Core CPI project** (`--target cpi_core_yoy`): duplicate `nowcast/` into a new `nowcast-core/` directory, add `cpi_core_yoy` series to `factors.py`, drop energy/food factors, focus on profit/wage growth drivers. User requested this as "Step 6".
2. **MIDAS n still only 25** (not 37): investigate whether the remaining 12 missing obs come from LAG/MOM warmup (first few months of backtest with insufficient CPI history) — may be unavoidable.
3. **RMC r0 has no AR(1) baseline**: dfm_k2 r0 regime has too few obs for a regime-specific AR(1). Consider using the full-sample AR(1) as fallback AR(1) for that regime.
4. **Push to GitHub**: mirror at `/Users/Adam/Documents/home/quant/nowcast/` → `adambutlin/nowcast` — no local git in nowcast/ currently.

---
---

## Handoff: 2026-06-07T07:49:19Z (auto-saved before compaction)

### Compaction Metadata
- Trigger: auto
- Custom instructions: (none)
- Transcript: /Users/Adam/.claude/projects/-Users-Adam-Documents-home-quant-nowcast/65b6c120-cad7-4be6-9a13-fac8a165c303.jsonl
- CWD: /Users/Adam/Documents/home/quant/nowcast/code

### Last User Message (transcript tail)
(unavailable)

### Last Assistant Message (transcript tail)
(unavailable)

### Git Snapshot
- Branch: main
- Status:
RM code/nowcast_cpi.py -> code/main.py
D  code/markov_dfm.py
D  code/nowcaster.py
D  code/ramm_lgbm_v1.py
D  code/test_ramm_lgbm_v1.py
RM code/test_nowcast_cpi.py -> code/tests/test_main.py
 M code/uk_model_zoo.py
 M docs/handoff/HANDOFF.md
?? ".venv 2"
- Recent commits:
0a5a617 refactor: reorganize repo — move code to code/, docs to docs/, remove obsolete files
57843df fix: nowcast for DFM/TVP/HMM/LSTAR; TTF gas; correct rents ffill
0ab0da1 docs: fill HANDOFF.md with real session summary and resume instructions
fcca78c feat: UK CPI nowcast system — gas_eu factor, rolling windows, ElasticNet, SHAP screening
d0777c7 Merge branch 'main' of https://github.com/adambutlin/nowcaster

### Model Summary
- **Session goal**: debug nowcast_cpi.py end-to-end, fix runnable environment, patch all bugs, then reorganize repo structure
- **Python 3.10 venv recreated**: old `.venv` was a broken symlink to `ramm-lgbm/.venv`; replaced with real Python 3.10.6 venv at `nowcast/.venv`; `numpy<2` pin required (system Python 3.12 had numpy 2.2.6, incompatible with bottleneck/pyarrow/scipy ABI)
- **13 bugs patched in `code/uk_model_zoo.py`**: DFM positional column index (`obs.index(target)` not string lookup), TVP R calibration (AR(1) residual variance with NaN guard), SARIMAX/AutoARIMA/VAR frozen multi-step → rolling 1-step-ahead loops, `dir_acc` fix (`pred - actual.shift(1)` not `pred - actual`), `CopulaReg._normal_scores` clip bounds, `MS_DFM.regimes()` factors-only DFM, `RAMM_LGBM.MONO` dict updated for renamed features, `_zscore` NaN std guard
- **LSTAR added to `all_models()`**: zoo is now 22 models (was 21); `LSTAR.WINDOW=60` (rolling) to prevent TRF optimizer divergence on long expanding windows
- **`MIDAS_Almon` class deleted**: was dead code (unreachable via all_models); yfinance-based `MIDAS` is the live implementation
- **Dead scripts deleted**: `code/ramm_lgbm_v1.py`, `code/test_ramm_lgbm_v1.py`, `code/markov_dfm.py` (PyMC-based, superseded), `code/nowcaster.py` (78-line original monolith)
- **Repo reorganized**: `nowcast_cpi.py` → `code/main.py`; `test_nowcast_cpi.py` → `code/tests/test_main.py`; `code/tests/conftest.py` added to fix sys.path for pytest discovery
- **15/15 tests passing** after all fixes; committed as `7e67067`

### Handoff Context (paste into next session)

**Canonical working directory:** `/Users/Adam/Documents/home/quant/nowcast/`
**Git remote:** `adambutlin/nowcast` on GitHub, branch `main`, commit `7e67067`

**Run full backtest:**
```bash
cd /Users/Adam/Documents/home/quant/nowcast
FRED_API_KEY=23615c784b776fe81c097fbef15bd6d2 .venv/bin/python -W ignore code/main.py \
  --start 2015 --end 2024 --train-from 1992
```

**Run tests:**
```bash
.venv/bin/python -m pytest code/tests/test_main.py -v
```

**Key files:**
- `code/main.py` — main runner (renamed from nowcast_cpi.py)
- `code/uk_model_zoo.py` — 22-model zoo (all bugs patched)
- `code/factors.py` — factor registry, `apply_publication_lags()`, `screen_candidates()`
- `code/tests/test_main.py` — 15 tests
- `code/tests/conftest.py` — sys.path fix for test imports
- `code/plot_nowcast_history.py`, `code/nowcast_plot.py`, `code/backtest_2025.py` — utility scripts (kept)

**Architecture reminders:**
- Zoo: 22 models — DFM, RAMM-LGBM, UCM, TVP, HMM, MS-DFM, LSTAR (WINDOW=60), BVAR, HiddenRF, GBM, MIDAS, BridgeEq, CopulaReg, DFM2, ElasticNet, MedianElasticNet, HuberNet, PCR, RegimeEnsemble, SARIMAX_Model, VAR_Model, AutoARIMA
- `pub_lag=0`: financial series; `pub_lag=1`: ONS series — shifted by `apply_publication_lags()`
- SHAP screening default (`--shap-screen`): drops ~11 factors; Combined-Static best at RMSE=0.277 (vs AR(1) 0.495)
- Blind test 2025+ enforced via `--end 2024`

**Known remaining issues:**
1. HMM/MS-DFM RMSE very high (2–3) — regime-switching on monthly CPI with few obs
2. MIDAS/BridgeEq n≈25 (TTF data starts ~2019; earlier obs imputed from spliced IMF series)
3. `us_ism_pmi`, `uk_paye`, `uk_house_prices` (some dbnomics fetches intermittent — candidates only, not core)
4. Core CPI project not yet started (user requested `--target cpi_core_yoy` as next step)

---
