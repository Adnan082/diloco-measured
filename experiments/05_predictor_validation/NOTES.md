# NOTES — 05_predictor_validation

**Mandatory (CLAUDE.md §14): what broke, what changed, what you'd redo.**

**First real fit + holdout validation run 2026-08-15** (CLAUDE.md ADR-038, G4). Not run via
`configs/grids/phase_d.yaml` (still a scaffold, and there is no genuinely-different-config
held-out set yet — see below) — `fit_and_validate_predictor.py` (this directory), pure
analysis-layer work, no cluster needed.

**Split:** fit on the shaped CU grid's repeats 0+1 (32 runs), validate on repeat 2 (16 runs,
never touched during fitting). This is a held-out *repeat*, not a held-out *configuration* —
the warning in `configs/grids/phase_d.yaml` about genuine held-out-ness still applies to a
proper G4 validation; what ran here specifically tests whether the recommendation is stable
against repeat-to-repeat noise, which is a real and useful check, just a narrower one than
"generalizes to an unseen bandwidth/H combination."

**Result: `predicted_H == measured_H` at all 4 tested bandwidth levels, 0% regret
throughout.** Consistent with the tight repeat variance already observed in the CU grid itself
(<1.5% spread on typical points, `experiments/01_cu_grid/NOTES.md`) — the decision boundaries
`recommend()`'s "smallest H clearing a CU target" rule depends on turned out to be stable, not
noise-sensitive, within this narrow domain.

**A real bug found and fixed before this ever ran for real:** the first draft of
`validate_holdout()` compared the fitted model's prediction against "whichever H had the
highest holdout `tokens_per_s`" — a different objective than `recommend()`'s own CU-threshold
rule. Since throughput is maximized by the largest tested `H` almost by construction (CU is
monotone increasing in `H` in every series measured so far), this produced a structurally
inflated ~21% "regret" at 5 Gbit/s that had nothing to do with prediction error — purely an
objective mismatch between the two sides of the comparison. Caught by running the function
against the real corpus before committing it, not by inspection. Fixed by applying
`recommend()`'s own decision rule directly to the holdout ground truth.

**Sample real recommendations** (target_cu=0.5, fit on repeats 0+1):

| Bandwidth | Recommended H | Expected CU | In calibration domain? |
| --- | --- | --- | --- |
| 50 Mbit/s | 128 (best available) | 0.112 | No — target unreachable at any tested H |
| 200 Mbit/s | 128 (best available) | 0.354 | No — target unreachable at any tested H |
| 1 Gbit/s | 128 | 0.712 | Yes |
| 5 Gbit/s | 32 | 0.677 | Yes |

Outputs: `fitted_predictor_model.json`, `holdout_validation_result.json` (both committed,
derived artifacts — not `RunResult`s, don't belong in `results/raw/`, same reasoning as
`experiments/01_cu_grid/required_bandwidth_table.json`).

**Not yet done:** a genuinely held-out configuration (a bandwidth level or `H` never run at
all, not just an unused repeat). Calibration beyond one model size / one algorithm (DiLoCo
only — `model_size_range` in the fitted domain is a single point, not a range). The actual
`diloco-measured plan --probe` CLI surface FR-07 specifies as its trigger — this validates the
underlying model, not the end-user tool. Fig 6 (predicted-vs-measured-H-with-regret, as a
figure rather than a JSON file).
