# Core CPI Nowcast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Research tasks (T4–T7) have decision gates, not fixed asserts.

**Goal:** `code/production/model.py` prints two nowcasts each run — the frozen headline CPI (unchanged) and a new Core CPI YoY — via a target-parameterized pipeline, with the core overlay gated on headline-grade validation.

**Architecture:** Introduce a `Target` config carrying all target-specific params (series name, pinned factors, overlay weights, λ, backtest window). `two_stage.py` and `model.py` take a `Target`, defaulting to `HEADLINE` so the headline path stays byte-identical (regression-locked). Core research (data → screen → race → calibrate → validate) reuses existing `new_factors/` tools with `uk_core_cpi` as target. Core prints AA-only unless its overlay beats AA out-of-sample.

**Tech Stack:** Python 3.11 (`/opt/anaconda3/envs/quant/bin/python` — the ONLY interpreter; see `docs/CLAUDE.md`), statsmodels, lightgbm, pandas, matplotlib, pytest, dbnomics. Live fetches need network → run **foreground with the sandbox off**; never launch parallel Python here.

**Environment invariants (every run):**
- `cd /Users/Adam/Documents/home/quant/nowcast && set -a; . ./.env; set +a`
- `PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python …`
- Runs that fetch data: foreground, sandbox off. Runs that only read local CSV/compute: sandbox fine.

---

## File Structure

- **Create** `code/targets.py` — `Target` dataclass + `HEADLINE`, `CORE` instances (single source of target-specific config).
- **Modify** `code/new_factors/two_stage.py` — replace module-level `TARGET`/`PINNED`/`WEIGHTS`/`LAMBDA`/`AA_START`/`START`/`END` with `Target`-parameterized functions; keep thin module constants delegating to `HEADLINE` for back-compat.
- **Modify** `code/production/model.py` — `nowcast(target=HEADLINE)`; `main()` runs + prints both blocks.
- **Modify** `code/factors.py` — give `uk_core_cpi` a real ONS MM23 fetch (Phase 0).
- **Create** `data/uk_core_cpi.csv` — CSV fallback for core history (gitignored; drop-in).
- **Create** `code/production/core_scorecard.py` — parallel `data/live_scorecard_core.csv` updater + report block.
- **Create** `code/plot_core.py` — the four inspection figures → `plots/core/`.
- **Create** `code/tests/test_targets.py`, `code/tests/test_headline_regression.py` — pipeline + regression tests.
- **Modify** `docs/final_model.md`, `docs/MODEL_CARD.md` — core spec + validation results (T10).

---

## Task 1: `Target` config module

**Files:**
- Create: `code/targets.py`
- Test: `code/tests/test_targets.py`

- [ ] **Step 1: Write the failing test**

```python
# code/tests/test_targets.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new_factors"))
import targets as T

def test_headline_matches_frozen_params():
    h = T.HEADLINE
    assert h.name == "cpi_yoy"
    assert h.lam == 0.5
    assert h.overlay_weights == {"tvp": 0.5, "lgbm": 0.5}
    assert h.aa_start == 2001 and h.start == 2015
    assert set(h.pinned) == {
        "oil_brent", "gas_eu", "uk_quarterly_gdp", "imf_all_commodity",
        "global_supply_chain_pressure", "mpc_rate_change", "ofgem_cap_delta",
        "uk_ppi_input", "deep_sea_freight"}

def test_core_target_declared():
    c = T.CORE
    assert c.name == "uk_core_cpi"
    assert c.lam == 0.5  # prior; may change after calibration (T6)
    # pinned starts empty → AA-only until the race (T5) fills it
    assert isinstance(c.pinned, tuple)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/Adam/Documents/home/quant/nowcast && PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python -m pytest code/tests/test_targets.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'targets'`

- [ ] **Step 3: Write minimal implementation**

```python
# code/targets.py
"""Per-target model configuration. HEADLINE mirrors the frozen two_stage/model params
byte-for-byte; CORE is filled in by the core research phases (data → screen → race → calibrate)."""
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

@dataclass(frozen=True)
class Target:
    name: str                      # target column / factor-registry series name
    label: str                     # human label for printing
    pinned: Tuple[str, ...]        # pinned overlay factors ( () → AA-only )
    overlay_weights: Dict[str, float]
    lam: float                     # shrinkage haircut
    aa_start: int                  # AutoARIMA backtest first year
    start: int                     # walk-forward scoring first year
    end: Optional[int]             # None → to latest actual
    reg: Tuple[str, ...] = ("mpc_rate_change", "ofgem_cap_delta")  # fill-0 regulatory factors

HEADLINE = Target(
    name="cpi_yoy", label="Headline CPI YoY",
    pinned=("oil_brent", "gas_eu", "uk_quarterly_gdp", "imf_all_commodity",
            "global_supply_chain_pressure", "mpc_rate_change", "ofgem_cap_delta",
            "uk_ppi_input", "deep_sea_freight"),
    overlay_weights={"tvp": 0.5, "lgbm": 0.5}, lam=0.5,
    aa_start=2001, start=2015, end=None)

# CORE: pinned/weights/lam are PRIORS; the research phases (T5/T6) overwrite pinned and
# may adjust lam. aa_start/start set in T4 once core history depth is known.
CORE = Target(
    name="uk_core_cpi", label="Core CPI YoY (ex energy, food, alcohol & tobacco)",
    pinned=(), overlay_weights={"tvp": 0.5, "lgbm": 0.5}, lam=0.5,
    aa_start=2001, start=2015, end=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python -m pytest code/tests/test_targets.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add code/targets.py code/tests/test_targets.py
git commit -m "feat(core): Target config; HEADLINE mirrors frozen params"
```

---

## Task 2: Parameterize `two_stage.py` by Target (headline byte-identical)

**Files:**
- Modify: `code/new_factors/two_stage.py` (load_matrix, backtest, nowcast, module constants)
- Test: `code/tests/test_headline_regression.py`

**Approach:** Add optional `target: Target = HEADLINE` params to `load_matrix`, `backtest`, and the nowcast path. Keep module constants (`TARGET`, `PINNED`, `WEIGHTS`, `LAMBDA`, `AA_START`, `START`, `END`) as aliases of `HEADLINE.*` so any other caller is unaffected. `PURGE_HORIZON=12`, `EMBARGO=1`, and the LGBM hyperparams stay shared module constants.

- [ ] **Step 1: Write the failing regression test** (locks headline numbers before refactor)

```python
# code/tests/test_headline_regression.py
"""Headline output must not move when the pipeline is parameterized by target.
Golden values captured 2026-07-23 from the frozen model on the quant env."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new_factors"))
import pytest

pytestmark = pytest.mark.slow  # fetches + backtests; run explicitly

def test_headline_nowcast_unchanged():
    import production.model as M
    import targets as T
    nc = M.nowcast(target=T.HEADLINE)          # new signature
    assert nc["nowcast_date"] == "2026-06-30"
    assert round(nc["aa"], 3) == 2.726
    assert round(nc["forecast"], 3) == 2.683
```

- [ ] **Step 2: Run to verify it fails**

Run (foreground, sandbox off): `PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python -m pytest code/tests/test_headline_regression.py -q -m slow`
Expected: FAIL — `nowcast()` doesn't yet accept `target=`.

- [ ] **Step 3: Refactor `two_stage.py`** — parameterize the three entry points

Replace the hardcoded target/pinned usages. Key edits:

```python
# top of two_stage.py, after imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from targets import Target, HEADLINE

# module constants become HEADLINE aliases (back-compat)
TARGET = HEADLINE.name
PINNED = list(HEADLINE.pinned)
WEIGHTS = {"bvar": 0.375, "tvp": HEADLINE.overlay_weights["tvp"], "midas": 0.375}
AA_START, START, END, TRAIN_FROM = HEADLINE.aa_start, HEADLINE.start, HEADLINE.end, 1997
PURGE_HORIZON, EMBARGO = 12, 1

def load_matrix(pinned=None, target: Target = HEADLINE):
    pinned = list(target.pinned) if pinned is None else pinned
    raw, status = F.build_matrix(names=pinned + [target.name])
    live = [n for n in pinned if status.get(n) != "unavailable"]
    raw = raw[raw.index.year >= TRAIN_FROM]
    df = F.apply_publication_lags(raw, live)
    for rf in target.reg:
        if rf in df.columns:
            df[rf] = df[rf].fillna(0)
    return df.resample("ME").last(), live, status

def backtest(df, live, target: Target = HEADLINE):
    aa = Z.AutoARIMA().backtest(df, [], target.name, start_year=target.aa_start, end_year=target.end)
    aa = aa[(aa.index.year >= target.start) & ((target.end is None) | (aa.index.year <= (target.end or 9999)))]
    # …rest unchanged except TARGET→target.name, START/END→target.start/target.end…
```

(Apply the same `target.name` / `target.start` / `target.aa_start` / `target.end` substitutions through the remaining `backtest`/`nowcast` bodies. Do NOT change `PURGE_HORIZON`/`EMBARGO`/LGBM params.)

- [ ] **Step 4: Run the regression test**

Run (foreground, sandbox off): `PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python -m pytest code/tests/test_headline_regression.py -q -m slow`
Expected: PASS (requires Task 3's `model.nowcast(target=)` — do Step 3 of Task 3 first if needed, or mark xfail until then).

- [ ] **Step 5: Commit**

```bash
git add code/new_factors/two_stage.py code/tests/test_headline_regression.py
git commit -m "refactor(core): parameterize two_stage by Target; headline aliases preserved"
```

---

## Task 3: Parameterize `model.py`; `main()` prints both blocks

**Files:**
- Modify: `code/production/model.py`

- [ ] **Step 1: Add `target` param to `nowcast()`**

Change signature to `def nowcast(df=None, live=None, target=HEADLINE):` (import `from targets import HEADLINE, CORE`). Replace `TS.TARGET`→`target.name`, `TS.PINNED`→`target.pinned`, `LAMBDA`→`target.lam`, `OVERLAY_WEIGHTS`→`target.overlay_weights`, `TS.AA_START`→`target.aa_start`, `END`→`target.end`. `load_matrix` call → `TS.load_matrix(target=target)`. Include `target=target.name`/`label` in the returned dict.

- [ ] **Step 2: Rewrite `main()` to print both**

```python
def _print_block(nc):
    print(f"  {nc['label']}")
    print(f"    month     : {nc['nowcast_date']}")
    print(f"    AutoARIMA : {nc['aa']}")
    if nc["pinned_n"]:
        print(f"    + TVP     : {nc['tvp_contribution']:+.3f}  (resid {nc['tvp_resid']})")
        print(f"    + LGBM    : {nc['lgbm_contribution']:+.3f}  (resid {nc['lgbm_resid']})")
        print(f"    FORECAST  : {nc['forecast']}   [overlay: {nc['n_live']}/{nc['pinned_n']} factors]")
    else:
        print(f"    FORECAST  : {nc['forecast']}   [AA-only — no core overlay shipped]")

def main():
    print("=== PRODUCTION NOWCAST (headline + core) ===")
    for tgt in (HEADLINE, CORE):
        try:
            _print_block(nowcast(target=tgt))
        except Exception as e:
            print(f"  {tgt.label}: FAILED — {type(e).__name__}: {e}")
```

(Add `pinned_n=len(target.pinned)` to the `nowcast` return dict. When `target.pinned` is empty the overlay members are skipped and `forecast=aa_pred`.)

- [ ] **Step 3: Guard the empty-overlay path in `nowcast()`**

```python
    if not target.pinned:
        forecast = float(aa_pred)
        return dict(nowcast_date=str(pd.Timestamp(nd).date()), label=target.label,
                    target=target.name, aa=round(float(aa_pred),3), forecast=round(forecast,3),
                    pinned_n=0, n_live=0, tvp_resid=0.0, lgbm_resid=0.0, overlay=0.0,
                    tvp_contribution=0.0, lgbm_contribution=0.0)
```

- [ ] **Step 4: Run headline regression + smoke both blocks**

Run (foreground, sandbox off):
`PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python -m pytest code/tests/test_headline_regression.py -q -m slow`
then `PYTHONPATH=code /opt/anaconda3/envs/quant/bin/python -u code/production/model.py`
Expected: headline block prints 2.726/2.683; core block prints an AA-only nowcast (a number) with "AA-only — no core overlay shipped".

- [ ] **Step 5: Commit**

```bash
git add code/production/model.py
git commit -m "feat(core): model.py prints headline + core (core AA-only for now)"
```

---

## Task 4 (Phase 0): Core CPI data source — DISCOVERY + acceptance gate

**Files:** Modify `code/factors.py` (`uk_core_cpi` entry); create `data/uk_core_cpi.csv` (fallback).

- [ ] **Step 1: Identify the ONS MM23 core series code.** Fetch candidate MM23 codes via dbnomics and match the known core prints (**Jun-2026 = 2.6, May-2026 = 2.6**, and the recent 2.6–2.8 path). Script:

```python
# scratchpad: try candidate ONS MM23 core codes
from factors import _dbnomics
for code in ["D7BT","D7G8","DKC3","D7G7"]:  # D7BT = CPI ex energy/food/alc/tob annual rate (verify)
    try:
        s = _dbnomics("ONS","MM23",f"{code}.M").dropna()
        print(code, s.index.max().date(), round(float(s.iloc[-1]),2), "| 2026-06:",
              round(float(s.get('2026-06-30', float('nan'))),2))
    except Exception as e:
        print(code, "ERR", e)
```
Run foreground, sandbox off. **Acceptance:** exactly one code returns ≈2.6 for 2026-06 and ≈2.6 for 2026-05.

- [ ] **Step 2: Wire the fetch** in `code/factors.py` `uk_core_cpi` entry: set `fetch=lambda: _dbnomics("ONS","MM23","<CODE>.M")`, keep `transform="level"`, `csv="uk_core_cpi.csv"`. Write the fetched history to `data/uk_core_cpi.csv` as the drop-in fallback.

- [ ] **Step 3: Verify history depth + set core window.** Check `s.index.min()`. If core history starts >2001, set `CORE.aa_start`/`CORE.start` in `code/targets.py` accordingly (e.g. `aa_start=max(2001, first_full_year+2)`), and note the reduced sample. Acceptance: ≥10y of monthly core history for a walk-forward.

- [ ] **Step 4: Commit**

```bash
git add code/factors.py code/targets.py
git commit -m "feat(core): source uk_core_cpi from ONS MM23 (code <CODE>); set core backtest window"
```

---

## Task 5 (Phase 2): Core factor screen + race — RESEARCH + record

**Files:** run `code/new_factors/shap_pinned.py`, `code/new_factors/factor_race.py` with core target; record results in `docs/core_factor_race.md`.

- [ ] **Step 1: Adapt the screen/race entry points to accept `--target uk_core_cpi`** (or a `Target`), reusing the existing look-ahead-free pre-2015 SHAP screen and univariate rel-RMSE race over the core candidate universe: `uk_ashe_pay, uk_awg, uk_paye, uk_vacancies, uk_monthly_gdp, uk_pmi_services, uk_pmi_composite, uk_services_cpi, uk_rents, uk_infl_swap_1y, gbp_eer, uk_ppi_output, uk_trimmed_mean_cpi, uk_retail_sales`.
- [ ] **Step 2: Run screen + race** (foreground, sandbox off). Record per-factor SHAP importance and univariate rel-RMSE (vs AA core baseline) to `docs/core_factor_race.md`.
- [ ] **Step 3: Pin the core set** = factors with rel-RMSE < 1 that survive the SHAP screen and a leakage check (`leakage_audit.py`). Set `CORE.pinned` in `code/targets.py`.
- [ ] **Step 4: Commit** `git commit -am "feat(core): pin core factors from screen+race (see docs/core_factor_race.md)"`

**Decision gate:** if <2 factors beat AA, core stays AA-only — skip to Task 8 (integration) and note it.

---

## Task 6 (Phase 3): Core overlay calibration — RESEARCH + record

- [ ] **Step 1:** Fit TVP + LGBM on the core AA-residual with the pinned set. Run `weight_sweep.py`/`alloc_sweep.py` for the core target.
- [ ] **Step 2:** Default to equal TVP/LGBM split + λ=0.5 haircut; adopt different weights/λ only with clear OOS evidence (record rationale in `docs/core_factor_race.md`). Update `CORE.overlay_weights`/`CORE.lam` if changed.
- [ ] **Step 3: Commit** `git commit -am "feat(core): calibrate core overlay weights + lambda"`

---

## Task 7 (Phase 4): Validation gate — RESEARCH + SHIP/FALLBACK decision

- [ ] **Step 1:** Purged+embargoed walk-forward (`validation.purge_embargo`, `PURGE_HORIZON=12`, `EMBARGO=1`) of core overlay vs AA-only core baseline, `CORE.start`→embargo.
- [ ] **Step 2:** Diebold–Mariano overlay-vs-AA; record rel-RMSE + DM p-value in `docs/core_factor_race.md`.
- [ ] **Step 3 (GATE):** Ship the overlay (keep `CORE.pinned` populated) **iff** rel-RMSE < 1 OOS with the same discipline as headline; else set `CORE.pinned=()` (AA-only). Record the decision explicitly.
- [ ] **Step 4: Commit** `git commit -am "feat(core): validation gate — <ship|fallback> core overlay (rel-RMSE=…, DM p=…)"`

---

## Task 8 (Phase 5): Integration — core scorecard + seed June

**Files:** Create `code/production/core_scorecard.py` (mirror `update_live_scorecard.py`, parallel `data/live_scorecard_core.csv`); extend `generate_live_report.py` with a core section.

- [ ] **Step 1:** Implement `core_scorecard.py` with the same `--add/--actual` interface; forecasters `core_aa`, `core_final`, `core_consensus`.
- [ ] **Step 2:** Seed June-2026: `--add --month 2026-06 --date 2026-06-30 --core-aa <core AA> --core-final <core forecast> --core-consensus 2.6` then `--actual --value 2.6` (actual core = 2.6). Values from `model.py`'s core block.
- [ ] **Step 3:** Add a "Core CPI" section to `generate_live_report.py` output.
- [ ] **Step 4: Commit** `git commit -am "feat(core): parallel core scorecard + report section; seed June-2026"`

---

## Task 9 (Phase 6): Visualizations → `plots/core/` (all YoY %)

**Files:** Create `code/plot_core.py`. **Before writing chart code, load the `dataviz` skill; when saving figures, load `captioning-figures`** to bake a one-sentence caption into each PNG.

- [ ] **Step 1:** F1 `cpi_vs_core_history.png` — headline vs core YoY % line chart over full history, shaded headline−core wedge, BoE 2% line.
- [ ] **Step 2:** F2 `core_factor_screen_shap.png` — horizontal bar of core candidates by pre-2015 mean |SHAP|, passed/rejected coloured, threshold marked.
- [ ] **Step 3:** F3 `core_factor_race_relrmse.png` — horizontal bar of candidates by rel-RMSE vs AA (1.0 line), pinned highlighted.
- [ ] **Step 4:** F4 `core_overlay_validation.png` — AA vs TVP/LGBM/combo walk-forward RMSE (the T7 gate outcome).
- [ ] **Step 5:** Run headless; assert the four PNGs exist and are non-empty. **Commit** `git commit -am "feat(core): inspection figures (history, screen, race, validation)"`

---

## Task 10: Docs

- [ ] **Step 1:** Add a "Core CPI nowcast" section to `docs/final_model.md` + `docs/MODEL_CARD.md`: core spec, pinned set, calibration, validation result (ship/fallback), and the June-2026 core nowcast vs actual 2.6.
- [ ] **Step 2: Commit** `git commit -am "docs(core): document core nowcast spec + validation + June result"`

---

## Self-Review notes
- **Spec coverage:** T1–T3 = architecture/refactor; T4 = Phase 0 data; T5 = Phase 2 screen/race; T6 = Phase 3 calibration; T7 = Phase 4 gate; T8 = Phase 5 integration/scorecard; T9 = Phase 6 viz; T10 = docs. All spec phases mapped.
- **Regression:** headline byte-identical guarded by `test_headline_regression.py` (golden 2.726/2.683) before/after refactor.
- **Graceful degrade:** empty `CORE.pinned` ⇒ AA-only core prints and validates; the gate (T7) can set it back to `()`.
- **Env:** every run uses the quant interpreter; fetches foreground+sandbox-off; no parallel Python.
