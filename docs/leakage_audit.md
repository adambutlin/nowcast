# Forensic leakage audit — frozen two-stage model (2026-06-17)

Architecture unchanged. `code/new_factors/leakage_audit.py`; artifacts in
`data/new_factors/audit/`.

## Information boundary (Part A/B) — proven from code, not asserted
| channel | rule | latest obs for forecasting CPI(T) | post-month-end? |
|---|---|---|---|
| brent/gas/imf/mpc/ofgem/deep_sea_freight (pub_lag 0) | ME.last, no shift | month-T end | NO |
| uk_ppi_input, cpi target (pub_lag 1) | ME.last, shift 1 | month-(T-1) end | NO |
| uk_quarterly_gdp (pub_lag 2) | ME.last, shift 2 | month-(T-2) end | NO |
| MIDAS daily {Brent,GBP,VIX,TTF} | resample('ME').mean, NO shift | mean of days 1..end of month T | NO |
0/9 factors use post-month-end data. Weights fixed (no selection leakage); each member
trains on years < test year.

## Part E — blinded backtest
| window | rmse_AA | 2stage baseline | month-end-blind | strict-premonth(T-30) | edge survived |
|---|---|---|---|---|---|
| full | 0.4687 | 0.4355 | **0.4355 (no-op)** | 0.4546 | **42%** |
| 2022_23 | 0.7715 | 0.6876 | 0.6876 | 0.7156 | 67% |
| ex_shock | 0.3546 | 0.3447 | 0.3447 | 0.3610 | **negative (edge vanishes)** |
| pre_2020 | 0.1908 | 0.1950 | 0.1950 | 0.1942 | n/a (already <AA) |

**Month-end-blind == baseline: structural NO-OP → no post-month-end leakage.**
Strict-premonth (blind all within-reference-month data; true ~T-30 standpoint): Stage-2
edge ~halves full-sample, vanishes/negative in calm. Within-month info ≈ 4.4% RMSE.

## Part F — model leakage ranking (within-month dependence)
| model | rmse base | rmse strict | %worse |
|---|---|---|---|
| **TVP** | 0.488 | 0.576 | **+17.9%** |
| MIDAS | 0.554 | 0.569 | +2.6% |
| AutoARIMA | 0.469 | 0.469 | 0% (CPI history only) |
| BVAR | 0.521 | 0.511 | −1.8% (doesn't need within-month) |

Live June nowcast: **2.933 → 2.794** under reference-month blinding; gap is entirely TVP
(overlay +0.641 → +0.035).

## Verdict
1. **Leakage present? NO** — no factor uses post-month-end (post-reference-month) data;
   month-end blinding is a no-op. No post-RELEASE leakage (release is ~17 days after the
   information boundary).
2. **Largest "forward-looking" source:** within-reference-month financials, via **TVP**
   (and MIDAS daily mean) — data dated days 13–31 of month T, which postdate the ~12th-of-
   month ONS price collection but precede month-end and release. Conventional nowcasting,
   NOT leakage by the audited definition.
3. **Most affected model: TVP** (+17.9%).
4. **ΔRMSE month-end blinding: 0.000 (no-op).** ΔRMSE reference-month blinding (T-30 proxy):
   +0.019 full (+4.4%).
5. **Δ live forecast (reference-month blinding): 2.933 → 2.794** (−0.14, ~all TVP).
6. **Architecture valid?** Yes — as a MONTH-END nowcast with fixed weights. No code change.
7. **Backtest valid?** Valid for a **month-end standpoint** (uses full month-T data). It is
   **OPTIMISTIC / mislabeled if presented as T-30**: a true T-30 origin has only partial
   month-T data, and MIDAS uses the FULL month-T mean at every backtest origin. The honest
   T-30 backtest is the strict-premonth one (full rel_rmse ~0.97 vs the headline 0.93).
8. **Inflation nowcast or release-day forecast?** Reference-month **inflation nowcast** (uses
   ≤ month-T-end info; does NOT use post-May/June-release data for the May print). It is NOT
   a release-day forecast. Mild hybrid only in that it leans on within-reference-month data
   (mostly TVP) that postdates the collection date.

## Does the Stage-2 edge survive removing all post-month-end information?
**YES — 100%** (no post-month-end information is used; month-end blinding changes nothing).
If instead ALL within-reference-month information is removed (true earliest / T-30):
**only ~42% of the edge survives full-sample, and it disappears (goes negative) in calm
months** — concentrated in TVP. So the headline backtest edge is a *month-end* edge; a
genuine T-30 product would retain under half of it.
