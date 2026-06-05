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
