# Nowcast-window value audit (branch `timing`)

Question: standing AFTER month-end T but BEFORE release R (T+15..T+21), does the model
improve as new info arrives? Is it materially better at R-1 than at T?
`code/timing/nowcast_window.py`; artifacts in `data/timing/`.

## PART A — release calendar (real ONS dates, n=136, 2015-2026)
R−T: min 15, max 21, median 18 days. Release is NEVER < T+15 → the nowcast window is
T+1..R-1 (T-1 never exists), exactly as posed.

## Information-set logic
For reference month T, a factor's month-T value enters only once published (offset days
after T): financials/MIDAS-daily/scheduled = 0 (closed at T); pub_lag≥1 (uk_ppi_input,
gdp) use pre-T vintages (always in); IMF commodity ≈ +7; US deep-sea-freight PPI ≈ +13.
AutoARIMA uses month-(T-1) CPI vintage → CONSTANT across the window (no new CPI until
release). MIDAS reference-month daily data is complete at T. So only imf(+7)/freight(+13)
can move the model after month-end.

## PART B — forecast movement T → R-1
mean |fc(R-1) − fc(T)| = **0.035pp**, max 0.31pp. fc(T) == fc(T+5) exactly; fc(T+15) ==
fc(R-1) exactly. The forecast is flat T→T+5, a tiny step at T+10 (IMF) and T+15 (freight),
then flat to release. The model is **essentially frozen at month-end**.

## PART C — model vs AA by horizon (full sample, n=120)
| horizon | rmse_AA | rmse_model | rel_rmse | hit |
|---|---|---|---|---|
| T | 0.4687 | 0.4377 | 0.934 | 0.45 |
| T+5 | 0.4687 | 0.4377 | 0.934 | 0.45 |
| T+10 | 0.4687 | 0.4395 | 0.938 (worse) | 0.47 |
| T+15 | 0.4687 | 0.4355 | 0.929 | 0.45 |
| R-1 | 0.4687 | 0.4355 | 0.929 | 0.45 |

## PART D — when does the edge arrive?
Δrmse(R-1 vs T) = **−0.0022 (0.5%)**, DM p=0.77 — insignificant. IMF at T+10 slightly
HURTS; freight at T+15 slightly helps; net noise. **MIDAS and TVP add ZERO post-month-end**
(their inputs are frozen at T). The only in-window movers are two monthly cost indices,
and their net effect is statistically zero.

## PART E — vs consensus at R-1 (n=120 overlap)
RMSE: model 0.4355 / AA 0.4687 / consensus 0.4591. MAE: model 0.3080 / consensus 0.3109
(≈tied). Model beats consensus in **48%** of months (coin flip). DM consensus-vs-model
p=0.34 — not significant. CAVEAT: the consensus series is a model-derived proxy (123/136
unique, repeated values), NOT a clean point-in-time survey → this comparison is weak.

## PART F — hostile
1. Model already optimal at T? **Yes** — it barely changes (0.035pp) through the window.
2. TVP add value after month-end? **No** — inputs frozen at T.
3. MIDAS add value after month-end? **No** — reference-month daily complete at T.
4. Edge concentrated? The 0.5% T→R-1 gain is a single freight step, statistically zero.
5. Survive ex-2022/23? The whole post-month-end gain is noise in every window.

## Final deliverable
1. RMSE at T: **0.4377**.
2. RMSE at R-1: **0.4355**.
3. Incremental gain from post-month-end info: **0.0022 RMSE (0.5%), p=0.77 — negligible**.
4. Incremental gain from TVP (post-month-end): **0** (frozen at T).
5. Incremental gain from MIDAS (post-month-end): **0** (frozen at T).
6. vs consensus at R-1: RMSE 5% better but MAE tied, beats in 48% of months, DM p=0.34 —
   not significant (and consensus is a proxy).
7. Genuinely useful nowcast (post-month-end updating)? **NO.**

### Answer: if I stand the day before release, is the model materially better than at month-end?
**NO.** Forecast moves 0.035pp on average T→R-1; RMSE improves 0.5% (p=0.77). The model's
information set CLOSES at month-end T — the reference month is over, so no new
reference-month data arrives in the T+1..R-1 window (only two minor cost indices republish,
to no significant effect). This is a **month-end nowcast that sits static until release**,
not an updating intramonth nowcast. Any genuine intramonth value accrues DURING the
reference month (T-30→T, as partial-month financials fill in) — explicitly outside this
window.
