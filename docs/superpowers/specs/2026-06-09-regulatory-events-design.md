# Regulatory Events Factors — Design Spec

**Date:** 2026-06-09
**Branch:** `feat/regulatory-events`
**Motivation:** Models underweight known structural discontinuities (Ofgem price cap resets, MPC policy shifts, fiscal events) that are mechanically causal rather than statistically learned. Root cause identified: April 2026 CPI drop driven by Ofgem cap cut was treated as a trending disinflation signal rather than a one-off regulatory event, causing models to forecast ~2.3% vs market 2.8–2.9%.

---

## Scope

Four new factors added to `factors.py` REGISTRY. No changes to models, `main.py`, `apply_publication_lags`, or SHAP screening — all existing plumbing picks them up automatically.

Out of scope: ONS basket reweighting events, NHS prescription charge changes, fuel duty changes (too granular and heterogeneous to encode consistently).

---

## New Factors

### 1. `mpc_rate_change`
- **Source:** FRED BOEBRATE (BoE Bank Rate, daily → month-end → diff × 100)
- **Values:** bps change. e.g. `+25`, `-50`, `0` in unchanged months
- **Transform:** `level` (diff computed in fetch, stored as bps)
- **pub_lag:** 0 (MPC decision announced same day, weeks before CPI release)
- **candidate:** False (always included)
- **CSV override:** `data/mpc_rate_change.csv` — [date, value]
- **History:** FRED BOEBRATE from 1975

### 2. `mpc_vote_split`
- **Source:** `data/mpc_vote_split.csv` — [date, hike_votes, hold_votes, cut_votes] curated from BoE MPC minutes
- **Encoding:** `hike_votes - cut_votes` = net hawkishness (-9 to +9)
  - e.g. 7-2 hold (2 cutters) = -2; 6-3 hold (3 hikers) = +3; 5-4 hike = +5
- **Forward-fill:** Between meetings, last vote split carries forward (committee stance persists)
- **Transform:** `level`
- **pub_lag:** 0 (vote split announced on decision day)
- **candidate:** False
- **Rationale:** Vote split captures forward policy direction without NLP on speeches; on par with LLM analysis of MPC member communications per domain knowledge

### 3. `ofgem_cap_delta`
- **Source:** `data/ofgem_cap.csv` — [date, value] cap level £/quarter for typical household dual-fuel
- **Encoding:** Monthly delta (diff of level). Non-change months = 0.
- **Transform:** `diff`
- **pub_lag:** 0 (Ofgem announces cap 4–6 weeks before effective date, before CPI release)
- **candidate:** False
- **History:** Oct 2018 (standard cap introduced). Pre-2018 = 0 (no cap regime).
- **Key events to populate:** Oct 2021 spike, Oct 2022 spike (£3,549), EPG 2022–23, quarterly resets from 2022

### 4. `budget_event`
- **Source:** `data/budget_event.csv` — [date, value] binary 0/1
- **Scope:** All fiscal events: Spring Budget, Autumn Statement, Spring Statement, and off-cycle events
  - Explicitly includes: Sep 2022 Kwarteng mini-budget, Oct 2022 Hunt reversal
- **Transform:** `level`
- **pub_lag:** 0 (announced on the day)
- **candidate:** False

---

## Code Changes

**File: `code/factors.py`** only.

### New fetch helper
```python
def _mpc_vote_split():
    """Load MPC vote CSV → net hawks (hike-cut), forward-filled monthly."""
    path = os.path.join(DATA_DIR, "mpc_vote_split.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df["value"] = df["hike_votes"] - df["cut_votes"]
    s = df.set_index("date")["value"].resample("ME").last()
    return s.ffill()
```

### REGISTRY additions (after existing entries)
```python
"mpc_rate_change": dict(
    fetch=lambda: _fred("BOEBRATE").diff() * 100,
    transform="level", pub_lag=0, candidate=False,
    csv="mpc_rate_change.csv",
    note="BoE Bank Rate monthly change (bps). FRED BOEBRATE diff×100. pub_lag=0."),

"mpc_vote_split": dict(
    fetch=_mpc_vote_split,
    transform="level", pub_lag=0, candidate=False,
    csv="mpc_vote_split.csv",
    note="MPC net hawks (hike_votes - cut_votes). Forward-filled between meetings. "
         "Source: BoE MPC minutes. pub_lag=0."),

"ofgem_cap_delta": dict(
    fetch=None, transform="diff", pub_lag=0, candidate=False,
    csv="ofgem_cap.csv",
    note="Ofgem quarterly price cap level £/quarter (typical dual-fuel household). "
         "diff transform → monthly delta, 0 in non-change months. Pre-Oct 2018 = 0. "
         "pub_lag=0: announced 4-6 weeks before effective date."),

"budget_event": dict(
    fetch=None, transform="level", pub_lag=0, candidate=False,
    csv="budget_event.csv",
    note="Fiscal event binary (1=event month). Covers Spring/Autumn budgets, "
         "Spring Statements, and off-cycle events (Sep 2022 mini-budget, Oct 2022 reversal). "
         "pub_lag=0."),
```

---

## Data Files to Create

### `data/mpc_vote_split.csv`
Format: `date,hike_votes,hold_votes,cut_votes`
Populate from BoE MPC minutes, 2000–present. One row per MPC decision date.

### `data/ofgem_cap.csv`
Format: `date,value` (cap level £/quarter, month-end dates)
- Pre-Oct 2018: omit or 0 (no standard cap)
- Oct 2018 onwards: biannual then quarterly resets
- Must include the full quarterly grid from 2022 (Jan/Apr/Jul/Oct)

### `data/budget_event.csv`
Format: `date,value` (1 on event month, 0 elsewhere or just rows with value=1)
- Cover 2000–present
- Include off-cycle events explicitly

### `data/mpc_rate_change.csv` (optional override)
Format: `date,value` (bps change). Only needed if FRED fetch fails.

---

## Integration Points

- `build_matrix()`: picks up new factors automatically via CSV drop-in + REGISTRY
- `apply_publication_lags()`: all four factors have pub_lag=0, no shift applied
- `screen_candidates()`: skipped (candidate=False for all four)
- `main.py` exclusion list: no additions needed (these are not CPI measures or collinear duplicates)

---

## Success Criteria

1. `build_matrix()` returns all four new factors with no errors
2. `mpc_vote_split` is non-zero and forward-filled in non-meeting months
3. `ofgem_cap_delta` shows large spikes in Oct 2021, Oct 2022, Apr 2023 and 0 in non-change months
4. `budget_event` = 1 in Sep 2022 and Oct 2022 (mini-budget + reversal)
5. Backtest RMSE does not degrade vs baseline (regression check)
6. April 2026 Ofgem cap cut visible as negative `ofgem_cap_delta` spike
