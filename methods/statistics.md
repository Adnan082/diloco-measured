# Statistics: Median vs. Mean, Repeats, Uncertainty

**Status:** `[PROPOSED]`. Specifies the aggregation behavior of `analysis/aggregate.py` and
`analysis/predictor.py`.

---

## 1. Why median + IQR, never mean-only

Wall-clock timing on shared cloud infrastructure is right-skewed by nature (a straggler node,
a noisy-neighbor blip, an ENA burst-credit stall). A mean lets one bad run silently dominate a
figure; median + interquartile range makes the spread visible instead of hiding it. This is a
`[CONFIRMED]` requirement (§27.1), not a style preference.

## 2. Discrepancy factor `F` (the headline number, G2)

`F` = measured required bandwidth ÷ analytically predicted required bandwidth, at a fixed CU
target (e.g. 90%). Reported at 50/75/90/95% CU, each with a confidence interval derived from the
repeat spread at neighboring grid points — `[UNKNOWN]` exact CI method (bootstrap over repeats
vs. parametric) pending Phase 3 data volume.

## 3. Confidence intervals on the predictor (FR-07)

`PredictorModel.holdout_validation` reports `{predicted_H, measured_H, regret_pct}` on a
held-out configuration never used for fitting (G4). `[UNKNOWN]` exact regret metric definition
(wall-clock time lost by using the predicted `H` instead of the true optimum) — to be finalized
alongside `predictor.py` in Phase 5.

## 4. Sample size honesty

Every figure states, in caption or metadata, the number of runs contributing (FR-13). A figure
built from fewer than the planned repeat count must say so, not present as if fully powered.

## 5. What must never be computed as if it were data

- `tttl == null` is never converted to a large finite number for averaging (`ConvergenceCurve`
  invariant, §15.2, §25.3, §30.2).
- Excluded runs (crashed/diverged/loader-bound/version-mismatched) never enter an aggregate —
  see `methods/measurement_windows.md` §3.
