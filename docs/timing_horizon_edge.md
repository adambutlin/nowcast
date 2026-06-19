# When does the edge arrive? Within-month horizon audit (branch `timing`)

Causal as-of reconstruction via the intramonth panel: build_panel(target, k) gives, for
origin T-k (k days before reference month-end), a panel where every month's HF features
are aggregated ONLY over days <= T-k (partial month). ModelStack runs AutoARIMA(baseline)
+ BVAR(factor) + TVP(regime_tvp) + MIDAS(intramonth = U-MIDAS/ElasticNet on HF as-of)
walk-forward at each origin. Reconstructed with frozen weights 0.375/0.25/0.375. n=156.

Caveat: this stack is the as-of-capable intramonth variant (MIDAS = HF-ElasticNet, intra
panel feature set), NOT byte-identical to the production two_stage (which uses Z.MIDAS,
PINNED monthly factors, full-month data, n=120, edge +0.033 / rel 0.929 BUT insignificant
p=0.17). AutoARIMA baseline is ~origin-invariant here, so the measured EDGE is cleanly the
residual-models' contribution at each origin. Direction of result is robust.

## PART B/C — horizon accuracy + edge (full sample); AA RMSE = 0.4152 (flat)
| origin | rmse_AA+BVAR | rmse_AA+BVAR+MIDAS | rmse_full | edge_full |
|---|---|---|---|---|
| T-30 | 0.4210 | 0.4151 | 0.4154 | **−0.0001** |
| T-21 | 0.4752 | 0.4396 | 0.4279 | −0.0127 |
| T-14 | 0.4551 | 0.4317 | 0.4264 | −0.0112 |
| T-10 | 0.4540 | 0.4270 | 0.4304 | −0.0152 |
| T-7 | 0.4548 | 0.4273 | 0.4290 | −0.0138 |
| T-5 | 0.4553 | 0.4299 | 0.4309 | **−0.0157** |
| T-2 | 0.4638 | 0.4333 | 0.4240 | −0.0087 |
| T-1 | 0.4632 | 0.4345 | 0.4254 | −0.0102 |

**The edge is NEGATIVE at every horizon.** On a strictly causal as-of panel the Stage-2
overlay is WORSE than AutoARIMA from T-30 through T-1. There is no horizon at which the
model beats the univariate benchmark. It is worst mid-month (T-5/T-10) and least-bad at
T-30 (overlays ~net to zero with no data) and T-2.

## PART D — contribution by horizon (mean |overlay|)
TVP overlay is largest at every horizon (0.70 at T-30 with ~no data!, ~0.46 later); BVAR
~0.28; MIDAS ~0.20. TVP MOVES the most and the edge is negative -> TVP never earns its
keep; it throws large (wrong) overlays, biggest when least informed (T-30).

## PART E — by window (edge vs AA)
- ex_shock: NEGATIVE at every horizon (−0.013 to −0.023). Model hurts in calm throughout.
- 2022_23: small/inconsistent (T-7 +0.011, T-10 +0.005, but T-1 −0.011, T-21 −0.014).
- pre_2020: no data in this start window.
Even in the shock the intramonth edge is tiny and flips sign; in calm it is uniformly negative.

## PART F — May 2026 path (actual = 2.8)
| origin | AA | full | TVP overlay |
|---|---|---|---|
| T-30 | 2.819 | 3.242 | +0.639 |
| T-21 | 2.819 | 2.977 | +0.303 |
| T-14 | 2.819 | 2.917 | +0.152 |
| T-10 | 2.819 | 2.840 | −0.066 |
| T-2 | 2.819 | 2.831 | −0.096 |
| T-1 | 2.819 | 2.839 | −0.086 |
**AutoARIMA = 2.819 (≈ actual) at EVERY origin from T-30.** The full model starts at 3.24
(TVP +0.64 with no May data), then decays back toward AA as data arrives — never beating
it. The "evolution" is the overlay un-learning its own early error. Charts:
plots/timing/{may2026_path,edge_by_horizon}.png.

## PART G — hostile review
1. Edge at T-30? Effectively zero/negative (overlays net out with no data).
2. Edge only late? No — never positive; least-bad at T-30, WORSE mid-month.
3. TVP reacting to final days? TVP's biggest move is at T-30 (least data) — it extrapolates
   wildly early then decays; its overlay is noise, not late-month learning.
4. MIDAS contributing? Small overlay (~0.20), no positive edge.
5. Would a PM care? No — at no horizon does the model beat AutoARIMA on this causal panel.

## Final deliverable
1. RMSE by horizon: AA flat 0.4152; full 0.4154→0.4309 (always ≥ AA).
2. Edge by horizon: NEGATIVE everywhere (−0.0001 to −0.0157).
3. Earliest horizon with meaningful improvement: NONE.
4. TVP contribution: largest overlay at all horizons, biggest at T-30; edge still negative.
5. MIDAS contribution: ~0.20 overlay, no positive edge.
6. BVAR contribution: ~0.28 overlay; AA+BVAR is the WORST combo (rmse up to 0.475).
7. May 2026: AA correct (2.82) at T-30; overlay only adds error, decaying 3.24→2.84.
8. Genuinely useful nowcast? On the causal as-of panel: NO.

### Answer — how much of the final month-end edge is earned by T-30/T-14/T-7/T?
There is **no positive edge to earn**: on a strictly causal intramonth panel the model is
worse than AutoARIMA at every horizon, and AutoARIMA already has the answer at T-30
(May: 2.82 vs actual 2.8). The model does not "learn the truth" during the month — it
starts with a wrong overlay (largest at T-30) and spends the month decaying back toward the
univariate baseline it should not have left. The production month-end edge (+0.033, rel
0.929) is wiring/sample-specific and statistically insignificant (p=0.17); it does not
reproduce on a causal intramonth panel.
