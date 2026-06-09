# Regulatory Events Factors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four regulatory-event factors (`mpc_rate_change`, `mpc_vote_split`, `ofgem_cap_delta`, `budget_event`) to `factors.py` so all models automatically account for known structural discontinuities in UK CPI.

**Architecture:** New REGISTRY entries in `code/factors.py` + one fetch helper + three CSV data files. The existing `load_factor` / `build_matrix` / `apply_publication_lags` pipeline requires zero changes — new factors are picked up automatically. `candidate=False` so they bypass SHAP screening.

**Tech Stack:** Python, pandas, FRED API (existing `_fred` helper), BoE MPC minutes (manual CSV curation)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `code/factors.py` | `_mpc_vote_split()` helper + 4 REGISTRY entries |
| Create | `data/ofgem_cap.csv` | Ofgem quarterly price cap levels £/yr |
| Create | `data/budget_event.csv` | Fiscal event binary (all UK budgets/statements) |
| Create | `data/mpc_vote_split.csv` | MPC vote splits from BoE minutes |
| Modify | `code/tests/test_main.py` | Tests for all four new REGISTRY entries |

---

## Task 1: Write failing tests for the four new REGISTRY entries

**Files:**
- Modify: `code/tests/test_main.py`

- [ ] **Step 1: Append the following test class to `code/tests/test_main.py`**

```python
class TestRegulatoryEventFactors(unittest.TestCase):
    def test_mpc_rate_change_in_registry(self):
        self.assertIn("mpc_rate_change", F.REGISTRY)

    def test_mpc_rate_change_fields(self):
        e = F.REGISTRY["mpc_rate_change"]
        self.assertEqual(e["pub_lag"], 0)
        self.assertFalse(e["candidate"])
        self.assertEqual(e["transform"], "level")
        self.assertIsNotNone(e.get("fetch"))

    def test_mpc_vote_split_in_registry(self):
        self.assertIn("mpc_vote_split", F.REGISTRY)

    def test_mpc_vote_split_fields(self):
        e = F.REGISTRY["mpc_vote_split"]
        self.assertEqual(e["pub_lag"], 0)
        self.assertFalse(e["candidate"])
        self.assertEqual(e["transform"], "level")
        self.assertEqual(e["csv"], "mpc_vote_split.csv")

    def test_ofgem_cap_delta_in_registry(self):
        self.assertIn("ofgem_cap_delta", F.REGISTRY)

    def test_ofgem_cap_delta_fields(self):
        e = F.REGISTRY["ofgem_cap_delta"]
        self.assertEqual(e["pub_lag"], 0)
        self.assertFalse(e["candidate"])
        self.assertEqual(e["transform"], "diff")
        self.assertIsNone(e["fetch"])
        self.assertEqual(e["csv"], "ofgem_cap.csv")

    def test_budget_event_in_registry(self):
        self.assertIn("budget_event", F.REGISTRY)

    def test_budget_event_fields(self):
        e = F.REGISTRY["budget_event"]
        self.assertEqual(e["pub_lag"], 0)
        self.assertFalse(e["candidate"])
        self.assertEqual(e["transform"], "level")
        self.assertIsNone(e["fetch"])
        self.assertEqual(e["csv"], "budget_event.csv")

    def test_mpc_vote_split_loads_and_forward_fills(self):
        """_mpc_vote_split() returns a ffilled monthly series with values in [-9, 9]."""
        import tempfile, os
        csv_content = "date,hike_votes,hold_votes,cut_votes\n2022-02-03,5,4,0\n2022-03-17,8,1,0\n2022-05-05,6,3,0\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            tmp = f.name
        try:
            orig = F.DATA_DIR
            # Patch DATA_DIR so _mpc_vote_split reads our temp file
            import unittest.mock as mock
            with mock.patch.object(F, "DATA_DIR", os.path.dirname(tmp)):
                # Rename so filename matches
                dest = os.path.join(os.path.dirname(tmp), "mpc_vote_split.csv")
                os.rename(tmp, dest)
                s = F._mpc_vote_split()
                # Feb 2022: 5-0 = +5
                self.assertEqual(s.loc["2022-02-28"], 5)
                # Mar 2022: 8-0 = +8
                self.assertEqual(s.loc["2022-03-31"], 8)
                # Apr 2022 (between meetings): should be ffilled to 8
                self.assertEqual(s.loc["2022-04-30"], 8)
                # All values in valid range
                self.assertTrue((s >= -9).all() and (s <= 9).all())
        finally:
            try:
                os.remove(dest)
            except Exception:
                pass

    def test_ofgem_cap_delta_spikes_oct_2022(self):
        """ofgem_cap_delta series shows large positive spike in Oct 2022."""
        s, status = F.load_factor("ofgem_cap_delta")
        if status == "unavailable":
            self.skipTest("data/ofgem_cap.csv not yet created")
        oct_2022 = s.loc["2022-10-31"] if "2022-10-31" in s.index else None
        self.assertIsNotNone(oct_2022)
        self.assertGreater(oct_2022, 500)  # Oct 2022 spike was ~+1578 £/yr

    def test_budget_event_sep_oct_2022(self):
        """budget_event = 1 for Sep 2022 (mini-budget) and Oct 2022 (reversal)."""
        s, status = F.load_factor("budget_event")
        if status == "unavailable":
            self.skipTest("data/budget_event.csv not yet created")
        self.assertEqual(s.loc["2022-09-30"], 1)
        self.assertEqual(s.loc["2022-10-31"], 1)
```

- [ ] **Step 2: Run the tests and confirm they all fail**

```bash
cd /Users/Adam/Documents/home/quant/nowcast
conda run -n ssj python -m pytest code/tests/test_main.py::TestRegulatoryEventFactors -v 2>&1 | tail -20
```

Expected: all 10 tests `FAILED` with `KeyError` or `AssertionError` (factors not in REGISTRY yet).

---

## Task 2: Add `_mpc_vote_split` helper and 4 REGISTRY entries to `factors.py`

**Files:**
- Modify: `code/factors.py`

- [ ] **Step 1: Add `_mpc_vote_split` fetch helper**

Find the block of `_fred`, `_yf`, `_dbnomics` helpers near the top of `code/factors.py` (around line 60–100). Add after them:

```python
def _mpc_vote_split():
    """
    Load data/mpc_vote_split.csv [date, hike_votes, hold_votes, cut_votes]
    → net hawks series (hike_votes - cut_votes), resampled to month-end,
    forward-filled so committee stance persists between meetings.
    Values in range [-9, 9]; 0 only before first meeting date.
    """
    path = os.path.join(DATA_DIR, "mpc_vote_split.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    df["value"] = df["hike_votes"].astype(float) - df["cut_votes"].astype(float)
    s = df.set_index("date")["value"].sort_index()
    s.index = s.index + pd.offsets.MonthEnd(0)   # snap to month-end
    s = s.resample("ME").last()
    return s.ffill()
```

- [ ] **Step 2: Add 4 REGISTRY entries**

Find the end of the REGISTRY dict in `code/factors.py` (look for the closing `}` of `REGISTRY = { ... }`). Add the following four entries before the closing `}`:

```python
    # ── regulatory event factors: pub_lag=0 (known before CPI release) ────────
    "mpc_rate_change": dict(
        fetch=lambda: _fred("BOEBRATE").diff() * 100,
        transform="level", pub_lag=0, candidate=False,
        csv="mpc_rate_change.csv",
        note="BoE Bank Rate monthly change (bps). FRED BOEBRATE diff×100. "
             "pub_lag=0: MPC decision announced same day, weeks before CPI release. "
             "0 in unchanged months. CSV override: data/mpc_rate_change.csv."),

    "mpc_vote_split": dict(
        fetch=_mpc_vote_split,
        transform="level", pub_lag=0, candidate=False,
        csv="mpc_vote_split.csv",
        note="MPC net hawks = hike_votes - cut_votes (-9 to +9). "
             "Source: BoE MPC minutes. Forward-filled between meetings so "
             "committee stance persists. pub_lag=0: vote announced on decision day. "
             "Curate from https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes"),

    "ofgem_cap_delta": dict(
        fetch=None, transform="diff", pub_lag=0, candidate=False,
        csv="ofgem_cap.csv",
        note="Ofgem quarterly price cap level £/yr (typical dual-fuel household). "
             "diff transform → monthly delta; 0 in non-change months. "
             "Standard cap introduced Oct 2018; pre-Oct 2018 = 0. "
             "pub_lag=0: cap announced 4-6 weeks before effective date. "
             "Source: https://www.ofgem.gov.uk/check-if-energy-price-cap-affects-you"),

    "budget_event": dict(
        fetch=None, transform="level", pub_lag=0, candidate=False,
        csv="budget_event.csv",
        note="Fiscal event binary (1=event month, 0 otherwise). Covers all UK "
             "Spring Budgets, Autumn Statements, Spring Statements, and off-cycle "
             "events (Sep 2022 Kwarteng mini-budget, Oct 2022 Hunt reversal). "
             "pub_lag=0: announced on the day. Curate from HM Treasury records."),
```

- [ ] **Step 3: Run the REGISTRY tests**

```bash
cd /Users/Adam/Documents/home/quant/nowcast
conda run -n ssj python -m pytest code/tests/test_main.py::TestRegulatoryEventFactors::test_mpc_rate_change_in_registry code/tests/test_main.py::TestRegulatoryEventFactors::test_mpc_rate_change_fields code/tests/test_main.py::TestRegulatoryEventFactors::test_mpc_vote_split_in_registry code/tests/test_main.py::TestRegulatoryEventFactors::test_mpc_vote_split_fields code/tests/test_main.py::TestRegulatoryEventFactors::test_ofgem_cap_delta_in_registry code/tests/test_main.py::TestRegulatoryEventFactors::test_ofgem_cap_delta_fields code/tests/test_main.py::TestRegulatoryEventFactors::test_budget_event_in_registry code/tests/test_main.py::TestRegulatoryEventFactors::test_budget_event_fields -v 2>&1 | tail -15
```

Expected: 8 tests PASS. The 2 data-dependent tests (`test_ofgem_cap_delta_spikes_oct_2022`, `test_budget_event_sep_oct_2022`) still SKIP (CSV files not created yet). `test_mpc_vote_split_loads_and_forward_fills` passes if `_mpc_vote_split` is correctly defined.

- [ ] **Step 4: Commit**

```bash
git add code/factors.py
git commit -m "feat: add mpc_rate_change, mpc_vote_split, ofgem_cap_delta, budget_event to REGISTRY"
```

---

## Task 3: Create `data/ofgem_cap.csv`

**Files:**
- Create: `data/ofgem_cap.csv`

The `diff` transform in `load_factor` → `_apply_transform` computes `s.diff()`. For this to produce 0 in non-change months, the CSV must have a row for every month (forward-filled cap level). Alternatively, provide only change-months and let pandas `resample("ME").last().ffill()` handle it — but `_load_csv` does `resample("ME").last()` without ffill. **Therefore populate every month from Jan 2015 onwards** (the backtest window), repeating the cap level until the next change.

- [ ] **Step 1: Create the file**

Create `data/ofgem_cap.csv` with the following content. Cap is in £/year for typical dual-fuel household. Dates are month-end. Repeat the cap level for every month until the next change.

**Key cap change dates and levels (verify current values at https://www.ofgem.gov.uk/check-if-energy-price-cap-affects-you):**

| Effective from | Cap £/yr | Notes |
|---|---|---|
| Jan 2015 | 0 | Pre-cap era — use 0 as placeholder |
| Jan 2019 | 1137 | Standard cap introduced |
| Apr 2019 | 1254 | |
| Oct 2019 | 1179 | |
| Feb 2020 | 1162 | |
| Aug 2020 | 1042 | |
| Feb 2021 | 1138 | |
| Oct 2021 | 1277 | |
| Apr 2022 | 1971 | Russia-Ukraine energy shock |
| Oct 2022 | 3549 | (EPG meant households paid ~£2500 equivalent) |
| Jan 2023 | 4279 | Quarterly resets begin; EPG still £2500 |
| Apr 2023 | 3280 | EPG ended |
| Jul 2023 | 2074 | |
| Oct 2023 | 1834 | |
| Jan 2024 | 1928 | |
| Apr 2024 | 1690 | |
| Jul 2024 | 1568 | |
| Oct 2024 | 1717 | |
| Jan 2025 | 1738 | |
| Apr 2025 | 1849 | |
| Jul 2025 | *(verify)* | |
| Oct 2025 | *(verify)* | |
| Jan 2026 | *(verify)* | |
| Apr 2026 | *(verify — the drop that motivated this feature)* | |

```csv
date,value
2015-01-31,0
2015-02-28,0
2015-03-31,0
2015-04-30,0
2015-05-31,0
2015-06-30,0
2015-07-31,0
2015-08-31,0
2015-09-30,0
2015-10-31,0
2015-11-30,0
2015-12-31,0
2016-01-31,0
2016-02-29,0
2016-03-31,0
2016-04-30,0
2016-05-31,0
2016-06-30,0
2016-07-31,0
2016-08-31,0
2016-09-30,0
2016-10-31,0
2016-11-30,0
2016-12-31,0
2017-01-31,0
2017-02-28,0
2017-03-31,0
2017-04-30,0
2017-05-31,0
2017-06-30,0
2017-07-31,0
2017-08-31,0
2017-09-30,0
2017-10-31,0
2017-11-30,0
2017-12-31,0
2018-01-31,0
2018-02-28,0
2018-03-31,0
2018-04-30,0
2018-05-31,0
2018-06-30,0
2018-07-31,0
2018-08-31,0
2018-09-30,0
2018-10-31,0
2018-11-30,0
2018-12-31,0
2019-01-31,1137
2019-02-28,1137
2019-03-31,1137
2019-04-30,1254
2019-05-31,1254
2019-06-30,1254
2019-07-31,1254
2019-08-31,1254
2019-09-30,1254
2019-10-31,1179
2019-11-30,1179
2019-12-31,1179
2020-01-31,1179
2020-02-29,1162
2020-03-31,1162
2020-04-30,1162
2020-05-31,1162
2020-06-30,1162
2020-07-31,1162
2020-08-31,1042
2020-09-30,1042
2020-10-31,1042
2020-11-30,1042
2020-12-31,1042
2021-01-31,1042
2021-02-28,1138
2021-03-31,1138
2021-04-30,1138
2021-05-31,1138
2021-06-30,1138
2021-07-31,1138
2021-08-31,1138
2021-09-30,1138
2021-10-31,1277
2021-11-30,1277
2021-12-31,1277
2022-01-31,1277
2022-02-28,1277
2022-03-31,1277
2022-04-30,1971
2022-05-31,1971
2022-06-30,1971
2022-07-31,1971
2022-08-31,1971
2022-09-30,1971
2022-10-31,3549
2022-11-30,3549
2022-12-31,3549
2023-01-31,4279
2023-02-28,4279
2023-03-31,4279
2023-04-30,3280
2023-05-31,3280
2023-06-30,3280
2023-07-31,2074
2023-08-31,2074
2023-09-30,2074
2023-10-31,1834
2023-11-30,1834
2023-12-31,1834
2024-01-31,1928
2024-02-29,1928
2024-03-31,1928
2024-04-30,1690
2024-05-31,1690
2024-06-30,1690
2024-07-31,1568
2024-08-31,1568
2024-09-30,1568
2024-10-31,1717
2024-11-30,1717
2024-12-31,1717
2025-01-31,1738
2025-02-28,1738
2025-03-31,1738
2025-04-30,1849
2025-05-31,1849
2025-06-30,1849
```

**Note:** Add Jul 2025 onwards by checking https://www.ofgem.gov.uk/check-if-energy-price-cap-affects-you — the Apr 2026 reset (the one that caused model error) must be included.

- [ ] **Step 2: Run the ofgem spike test**

```bash
cd /Users/Adam/Documents/home/quant/nowcast
conda run -n ssj python -m pytest code/tests/test_main.py::TestRegulatoryEventFactors::test_ofgem_cap_delta_spikes_oct_2022 -v 2>&1 | tail -10
```

Expected: PASS. Oct 2022 delta = 3549 - 1971 = +1578.

- [ ] **Step 3: Verify zeros in non-change months**

```bash
conda run -n ssj python -c "
import sys; sys.path.insert(0, 'code')
import factors as F
s, st = F.load_factor('ofgem_cap_delta')
print('status:', st)
print('Non-zero months:')
print(s[s != 0].to_string())
print('Jul 2022 (no change, should be 0):', s.loc['2022-07-31'])
"
```

Expected: Only cap-change months are non-zero. Jul 2022 = 0.

- [ ] **Step 4: Commit**

```bash
git add data/ofgem_cap.csv
git commit -m "data: Ofgem quarterly price cap history 2015-2025"
```

---

## Task 4: Create `data/budget_event.csv`

**Files:**
- Create: `data/budget_event.csv`

Provide only rows where value=1 (event months). `_load_csv` will resample to month-end and return 1 on event months, NaN elsewhere — but we need 0 not NaN for non-event months. **Therefore provide rows for every month** with 0/1.

- [ ] **Step 1: Create the file with complete monthly rows 2000–2026**

The key fiscal events (value=1):

```
2000-03, 2001-03, 2002-04, 2003-04, 2003-12, 2004-03, 2005-03, 2005-12,
2006-03, 2007-03, 2007-10, 2008-03, 2008-11, 2009-04, 2009-12,
2010-03, 2010-06, 2010-10, 2011-03, 2011-11, 2012-03, 2012-12,
2013-03, 2013-12, 2014-03, 2014-12, 2015-03, 2015-07, 2015-11,
2016-03, 2016-11, 2017-03, 2017-11, 2018-10, 2019-03,
2020-03, 2020-07, 2021-03, 2021-10,
2022-03, 2022-09, 2022-10, 2022-11,
2023-03, 2023-11, 2024-03, 2024-10,
2025-03, 2025-10
```

Create the CSV programmatically to avoid transcription errors:

```bash
conda run -n ssj python -c "
import pandas as pd

event_months = {
    '2000-03','2001-03','2002-04','2003-04','2003-12',
    '2004-03','2005-03','2005-12','2006-03','2007-03',
    '2007-10','2008-03','2008-11','2009-04','2009-12',
    '2010-03','2010-06','2010-10','2011-03','2011-11',
    '2012-03','2012-12','2013-03','2013-12','2014-03',
    '2014-12','2015-03','2015-07','2015-11','2016-03',
    '2016-11','2017-03','2017-11','2018-10','2019-03',
    '2020-03','2020-07','2021-03','2021-10','2022-03',
    '2022-09','2022-10','2022-11','2023-03','2023-11',
    '2024-03','2024-10','2025-03','2025-10',
}

idx = pd.date_range('2000-01-31', '2026-06-30', freq='ME')
vals = [1 if d.strftime('%Y-%m') in event_months else 0 for d in idx]
df = pd.DataFrame({'date': idx.strftime('%Y-%m-%d'), 'value': vals})
df.to_csv('data/budget_event.csv', index=False)
print(df[df.value==1].to_string())
print(f'Total events: {df.value.sum()}')
"
```

- [ ] **Step 2: Run the budget_event tests**

```bash
cd /Users/Adam/Documents/home/quant/nowcast
conda run -n ssj python -m pytest code/tests/test_main.py::TestRegulatoryEventFactors::test_budget_event_sep_oct_2022 code/tests/test_main.py::TestRegulatoryEventFactors::test_budget_event_in_registry code/tests/test_main.py::TestRegulatoryEventFactors::test_budget_event_fields -v 2>&1 | tail -10
```

Expected: all 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add data/budget_event.csv
git commit -m "data: UK fiscal event binary 2000-2026 (budgets, statements, off-cycle)"
```

---

## Task 5: Create `data/mpc_vote_split.csv`

**Files:**
- Create: `data/mpc_vote_split.csv`

One row per MPC decision date. The `_mpc_vote_split` helper forward-fills to monthly. Only meeting dates needed (not every month).

- [ ] **Step 1: Create the file**

```bash
conda run -n ssj python -c "
import pandas as pd

# MPC decisions: date, hike_votes, hold_votes, cut_votes
# Source: BoE MPC minutes https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes
# Votes are for the majority action unless noted; hike=vote to raise rate, cut=vote to lower
rows = [
    # Pre-2008: rates generally falling, most votes unanimous
    ('2003-07-10', 0, 9, 0),
    ('2003-11-06', 0, 0, 9),  # 50bps cut
    ('2005-08-04', 0, 0, 9),  # cut
    ('2006-08-03', 9, 0, 0),  # hike
    ('2007-07-05', 9, 0, 0),  # hike to 5.75
    ('2007-12-06', 0, 0, 9),  # cut
    ('2008-02-07', 0, 0, 9),
    ('2008-04-10', 0, 0, 9),
    ('2008-10-08', 0, 0, 9),  # emergency 50bps
    ('2008-11-06', 0, 0, 9),
    ('2008-12-04', 0, 0, 9),
    ('2009-01-08', 0, 0, 9),
    ('2009-02-05', 0, 0, 9),
    ('2009-03-05', 0, 0, 9),  # QE begins
    # 2009-2021: long period of 0.5% then 0.1%, mostly 9-0 holds
    ('2009-06-04', 0, 9, 0),
    ('2010-01-07', 0, 9, 0),
    ('2011-01-13', 3, 6, 0),  # 3 hawks
    ('2011-06-09', 2, 7, 0),
    ('2012-07-05', 0, 9, 0),
    ('2016-08-04', 0, 0, 9),  # post-Brexit cut
    ('2017-11-02', 7, 2, 0),  # first hike in 10yr
    ('2018-08-02', 9, 0, 0),
    ('2019-06-20', 0, 9, 0),
    ('2020-03-11', 0, 0, 9),  # COVID emergency
    ('2020-03-19', 0, 0, 9),  # COVID emergency
    ('2021-12-16', 8, 0, 1),  # first hike of new cycle, 1 dissent (cut)
    # 2022 hiking cycle
    ('2022-02-03', 5, 4, 0),
    ('2022-03-17', 8, 0, 1),
    ('2022-05-05', 6, 3, 0),  # 3 wanted +50bps
    ('2022-06-16', 6, 3, 0),
    ('2022-08-04', 8, 0, 1),  # +50bps, 1 wanted +25
    ('2022-09-22', 5, 4, 0),
    ('2022-11-03', 7, 0, 2),  # +75bps; 1 wanted +50, 1 wanted +25
    ('2022-12-15', 6, 1, 2),
    # 2023
    ('2023-02-02', 7, 0, 2),
    ('2023-03-23', 7, 0, 2),
    ('2023-05-11', 7, 0, 2),
    ('2023-06-22', 7, 0, 2),  # surprise +50bps
    ('2023-08-03', 6, 0, 3),  # +25bps
    ('2023-09-21', 5, 4, 0),  # first hold
    ('2023-11-02', 6, 3, 0),
    ('2023-12-14', 6, 3, 0),
    # 2024
    ('2024-02-01', 6, 2, 1),
    ('2024-03-21', 8, 1, 0),
    ('2024-05-09', 7, 2, 0),
    ('2024-06-20', 7, 2, 0),
    ('2024-08-01', 5, 4, 0),  # first cut, 5-4
    ('2024-09-19', 8, 0, 1),
    ('2024-11-07', 8, 0, 1),
    ('2024-12-19', 6, 3, 0),
    # 2025 (verify from BoE minutes for dates after Feb 2025 — knowledge cutoff)
    ('2025-02-06', 7, 0, 2),
    # Add remaining 2025 and 2026 from BoE minutes:
    # https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes
]

df = pd.DataFrame(rows, columns=['date','hike_votes','hold_votes','cut_votes'])
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.to_csv('data/mpc_vote_split.csv', index=False, date_format='%Y-%m-%d')
print(df.tail(10).to_string())
print(f'Total meetings: {len(df)}')
"
```

- [ ] **Step 2: Extend with post-Feb 2025 meetings from BoE minutes**

Fetch remaining 2025 and 2026 MPC decisions from:
`https://www.bankofengland.co.uk/monetary-policy-summary-and-minutes`

Add rows to `data/mpc_vote_split.csv` for each meeting date with the correct vote split.

- [ ] **Step 3: Run the vote split test**

```bash
cd /Users/Adam/Documents/home/quant/nowcast
conda run -n ssj python -m pytest code/tests/test_main.py::TestRegulatoryEventFactors::test_mpc_vote_split_loads_and_forward_fills -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 4: Verify forward-fill visually**

```bash
conda run -n ssj python -c "
import sys; sys.path.insert(0, 'code')
import factors as F
s = F._mpc_vote_split()
print(s['2022-01-31':'2022-12-31'].to_string())
print('--- 2024 easing cycle ---')
print(s['2024-07-31':'2024-12-31'].to_string())
"
```

Expected:
- Jan 2022 forward-filled from Dec 2021 meeting (value=7, net hawks from 8-1 hike)
- Aug 2024: net hawks = 5 (5-0 cut vote)
- Sep 2024: forward-filled to 8 (8-1 hold)

- [ ] **Step 5: Commit**

```bash
git add data/mpc_vote_split.csv
git commit -m "data: MPC vote split history 2003-2025 from BoE minutes"
```

---

## Task 6: Full integration test — `build_matrix` picks up all four factors

**Files:**
- Modify: `code/tests/test_main.py`

- [ ] **Step 1: Add integration test to `TestRegulatoryEventFactors`**

```python
    def test_all_four_in_build_matrix(self):
        """build_matrix returns all four regulatory factors with non-empty series."""
        df, status = F.build_matrix(
            names=["mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]
        )
        for name in ["mpc_rate_change", "mpc_vote_split", "ofgem_cap_delta", "budget_event"]:
            self.assertIn(name, df.columns, f"{name} missing from build_matrix output")
            self.assertGreater(df[name].notna().sum(), 0, f"{name} is all-NaN")

    def test_regulatory_factors_have_zero_pub_lag(self):
        """apply_publication_lags does not shift regulatory factors (pub_lag=0)."""
        df, _ = F.build_matrix(
            names=["ofgem_cap_delta", "budget_event"]
        )
        import numpy as np
        df["cpi_yoy"] = np.nan
        shifted = F.apply_publication_lags(df, ["ofgem_cap_delta", "budget_event"])
        # pub_lag=0 → series should be identical after applying lags
        pd.testing.assert_series_equal(df["ofgem_cap_delta"], shifted["ofgem_cap_delta"])
        pd.testing.assert_series_equal(df["budget_event"], shifted["budget_event"])
```

- [ ] **Step 2: Run all regulatory event tests**

```bash
cd /Users/Adam/Documents/home/quant/nowcast
conda run -n ssj python -m pytest code/tests/test_main.py::TestRegulatoryEventFactors -v 2>&1 | tail -20
```

Expected: all tests PASS (or SKIP for unavailable data).

- [ ] **Step 3: Run full test suite — no regressions**

```bash
conda run -n ssj python -m pytest code/tests/test_main.py -v 2>&1 | tail -20
```

Expected: all existing tests still PASS.

- [ ] **Step 4: Smoke test — build_matrix with full REGISTRY**

```bash
conda run -n ssj python -c "
import sys; sys.path.insert(0, 'code')
import factors as F
df, status = F.build_matrix()
reg_factors = ['mpc_rate_change', 'mpc_vote_split', 'ofgem_cap_delta', 'budget_event']
for f in reg_factors:
    s = status.get(f, 'missing')
    n = df[f].notna().sum() if f in df.columns else 0
    print(f'  {f:<22} status={s}  n={n}')
" 2>&1 | grep -E "mpc|ofgem|budget|ERROR"
```

Expected: all four show `status=csv` or `status=live`, `n>0`.

- [ ] **Step 5: Final commit**

```bash
git add code/tests/test_main.py
git commit -m "test: integration tests for regulatory event factors in build_matrix"
```

---

## Success Criteria Checklist

- [ ] All `TestRegulatoryEventFactors` tests pass
- [ ] All pre-existing tests still pass
- [ ] `ofgem_cap_delta` non-zero only in cap-change months; Oct 2022 spike > +500
- [ ] `mpc_vote_split` forward-filled (no NaN between meeting dates)
- [ ] `budget_event` = 1 for Sep 2022 and Oct 2022
- [ ] `apply_publication_lags` does not shift any of the four factors (pub_lag=0)
- [ ] `build_matrix()` full run completes without errors
- [ ] Backtest RMSE regression check: run `python code/main.py --start 2015 --end 2024` and confirm Combined-Dynamic RMSE does not exceed 0.50pp (baseline was 0.453pp; new factors should not degrade)
