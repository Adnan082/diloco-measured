# NOTES — 01_cu_grid

**Rule (CLAUDE.md §14.1):** this file records what actually happened, including mistakes — a
sanitized narrative does not belong here.

**First real slice run 2026-08-14** (see CLAUDE.md ADR-034): `algorithm=diloco`,
`model_config=30m-realvocab` (30,846,720 real params, real gpt2 vocab), **unshaped** baseline
(no `tc` shaping — this is not the full multi-bandwidth `phase_a.yaml` grid yet, just the first
H-sweep at one bandwidth level), `H ∈ {1, 8, 32, 128}`, 150–200 steps each, 20 steps warmup
discarded, 1 repeat per point (`r0` — no repeat-variance data yet, CLAUDE.md §40 Q6 still
open). Real 4x g6e.2xlarge cluster, `us-east-1b`, placement group `pg-04ac04963de1615d8`.

Driven by `train_driver.py` (this directory) via a direct `torchrun` invocation across all 4
nodes, **not** through `measurement/train.py::run()`'s full FR-03 orchestration — see
`train_driver.py`'s own docstring and ADR-034 for exactly what that gap is (no automated
precondition gate, no automated shaping gate — moot here since unshaped, no in-process
fingerprinting). Raw per-step telemetry: `raw_step_telemetry/result_h{1,8,32,128}.json`.
Aggregated into schema-valid `RunResult` records by `aggregate_results.py` →
`results/raw/cu_grid-diloco-30m-h{1,8,32,128}-bwunshaped-r0.json`.

**Headline result** (`cu_measured` vs. both analytic variants, FR-04):

| H | cu_measured | cu_analytic_link | cu_analytic_achieved | discrepancy_link |
| --- | --- | --- | --- | --- |
| 1 | 0.1670 | 0.1989 | 0.2797 | 1.19x |
| 8 | 0.6023 | 0.6560 | 0.7490 | 1.09x |
| 32 | 0.7101 | 0.8827 | 0.9217 | 1.24x |
| 128 | 0.8393 | 0.9682 | 0.9794 | 1.15x |

Measured CU is below both analytic predictions at every H tested — the pre-registered
hypothesis (CLAUDE.md §2.7) holds on this first real slice. Discrepancy factor is roughly
1.1–1.25x across this range, not monotone in H on this single-repeat, single-bandwidth-level
sample — a real trend (if any) needs the shaped multi-bandwidth grid and repeats, not this
slice alone. Figure: `results/figures/fig4_cu_vs_h_diloco_bwunshaped.png`.
All 4 records pass schema validation and step-time reconciliation (residuals 0.01–0.04%, well
under the 5% tolerance).

**Not yet done:** the shaped, multi-bandwidth grid (`configs/grids/phase_a.yaml`, Phase 2/3,
M3/M4) — this slice is unshaped only, so it says nothing yet about bandwidth as a variable,
only about H at one (very high, effectively uncapped) bandwidth. Repeats (§40 Q6). The
torchft-vs-reference cross-implementation equivalence check (US-06) — this slice used the
reference `DiLoCoTrainer` only. Per-step Parquet (§16.1/ADR-023) — raw telemetry is committed
as JSON in `raw_step_telemetry/`, not yet converted to the schema's Parquet format.
