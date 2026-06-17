# Stage-2 BVAR/MIDAS allocation (TVP locked 0.25) — final ensemble weights

`code/new_factors/alloc_sweep.py`. TVP fixed 0.25; sweep w_bvar+w_midas=0.75.
One walk-forward backtest, cheap weight sweep. Deliverables in data/new_factors/.

## Full-sample RMSE (Part B) — flat, shallow min ~0.45/0.30 ≈ equal
| w_bvar / w_midas | rmse_full | rel | shock 22/23 | ex_shock | pre_2020 |
|---|---|---|---|---|---|
| 0.75 / 0.00 | 0.4392 | 0.937 | 0.7058 | **0.3414** | 0.1949 |
| 0.55 / 0.20 | 0.4359 | 0.930 | 0.6941 | 0.3421 | **0.1947** |
| 0.45 / 0.30 | **0.4354** | 0.929 | 0.6900 | 0.3434 | 0.1948 |
| 0.375/0.375 | 0.4355 | 0.929 | 0.6876 | 0.3447 | 0.1950 |
| 0.30 / 0.45 | 0.4360 | 0.930 | 0.6859 | 0.3464 | 0.1953 |
| 0.10 / 0.65 | 0.4398 | 0.938 | **0.6845** | 0.3530 | 0.1980 |
| 0.00 / 0.75 | 0.4436 | 0.946 | 0.6856 | 0.3584 | 0.2032 |

Full range ~2%. BVAR wins calm (ex_shock/pre_2020), MIDAS wins shock (22/23), full optimum middle.

## Live May-2026 (Part D) — insensitive to split
2.914 → 2.920 across the entire sweep (6bp). The overshoot vs the 2.8 print is driven by
TVP (locked 0.25, overlay +0.64), NOT the BVAR/MIDAS allocation.

## Error correlation (Part E)
|       | bvar | tvp | midas |
|---|---|---|---|
| bvar | 1.00 | 0.69 | **0.93** |
| tvp  | 0.69 | 1.00 | 0.70 |
| midas| 0.93 | 0.70 | 1.00 |
BVAR & MIDAS errors are 0.93 correlated → near-redundant; the split cannot matter much.
TVP (0.69) is the real diversifier. This explains the flat RMSE surface.

## Factor exposure (Part F)
More BVAR ⇒ more cost-push (PPI/freight/Ofgem) exposure (0.263→0.293); energy_fx low for
all members (MIDAS daily-channel correlation modest, ~0.087). TVP most cost-push (0.48).

## Hostile review (Part G)
DM vs equal-split (0.375/0.375): every allocation p>0.32 — **no allocation is statistically
distinguishable from equal**. Optimum unstable across windows (calm→BVAR, shock→MIDAS,
full→middle). Differences are noise; a PM would not care. → recommend the simplest robust
allocation: **equal split**.

## Final answer — production weights
- TVP = 0.25 (locked), **BVAR = 0.375, MIDAS = 0.375** (equal split of the 0.75).
- Best full-sample (argmin) 0.45/0.30, but within DM noise of equal; equal is robust + simplest.
- Full RMSE 0.4355 (rel 0.929) | shock 0.6876 | calm(ex_shock) 0.3447 | live May 2.917.
- Given TVP fixed 0.25, the most robust BVAR/MIDAS allocation is EQUAL (0.375/0.375):
  the two are 0.93-correlated, the RMSE surface is flat (~2%), and no split beats equal
  significantly. Architecture frozen. WEIGHTS = {bvar 0.375, tvp 0.25, midas 0.375}.
