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

**Not yet done (as of the unshaped slice above):** the shaped, multi-bandwidth grid
(`configs/grids/phase_a.yaml`, Phase 2/3, M3/M4) — this slice is unshaped only, so it says
nothing yet about bandwidth as a variable, only about H at one (very high, effectively
uncapped) bandwidth. Repeats (§40 Q6). The torchft-vs-reference cross-implementation
equivalence check (US-06) — this slice used the reference `DiLoCoTrainer` only. Per-step
Parquet (§16.1/ADR-023) — raw telemetry is committed as JSON in `raw_step_telemetry/`, not yet
converted to the schema's Parquet format.

---

## Shaped bandwidth × H grid — 2026-08-14/15 (CLAUDE.md ADR-035)

The real headline comparison: same `algorithm=diloco`, same `model_config=30m-realvocab`, but
now with real `tc` shaping and a real FR-02 verification gate at 4 bandwidth levels
(`5g`/`1g`/`200m`/`50m`) × the same 4 `H` values = **16 points**, all completed, zero
shaping-gate failures, zero crashes. Cluster was relaunched for this (torn down at the end of
the unshaped-slice session) — new placement group `pg-0ee059f5ef7da671b`, `us-east-1b`, same
AMIs/pins. Driven by `run_shaped_grid.py` (this directory) against `netshape.py`'s real
`apply`/`verify`/`restore`, not through `measurement/train.py::run()`'s FR-03 orchestration
(same gap as the unshaped slice — see ADR-035 for exactly what that does and doesn't mean).
Aggregated by `aggregate_shaped_grid.py` → `results/raw/cu_grid-diloco-30m-h*-bw*-r0.json`
(16 files). Raw per-step telemetry: `raw_step_telemetry_shaped/*.json`.

**Real bugs hit and fixed getting the cluster back up** (see ADR-035 for full detail): a stale
security-group SSH rule (operator IP had changed), a Git-Bash/MSYS path-mangling bug in the
`aws` CLI invocation (`/dev/sda1` → a Windows path), and the control node's root volume being
too small (8GB default, no `--block-device-mappings` in `launch_control_node()` — fixed live
via `ec2:ModifyVolume` + `growpart`/`resize2fs`, and fixed in `infra/launch_cluster.sh` for
future launches).

**Headline result** (`cu_measured`, DiLoCo, 30.8M params, 4 replicas):

| H | 50 Mbit/s | 200 Mbit/s | 1 Gbit/s | 5 Gbit/s |
| --- | --- | --- | --- | --- |
| 1 | 0.07% | 0.32% | 1.74% | 7.76% |
| 8 | 0.55% | 2.48% | 12.35% | 40.24% |
| 32 | 2.41% | 10.01% | 36.01% | 68.10% |
| 128 | 11.55% | 36.09% | 71.52% | 87.00% |

`discrepancy_link` (measured vs. naive analytic) ranges 1.08×–1.92× across all 16 points,
worst at the low-bandwidth/low-H corner. Figure: `results/figures/fig1_cu_surface_diloco.png`
— the first time this figure has ever rendered with real data (it needs ≥2 bandwidth levels,
which no earlier campaign had). All 16 records pass schema validation and step-time
reconciliation.

**Still not done:** DDP/FSDP2/LocalSGD (no training driver exists for them yet — this grid is
DiLoCo only, not the full `phase_a.yaml` 4-algorithm comparison). The 1B model
`phase_a.yaml` specifies (still the 30.8M model, for continuity with the unshaped slice).
Repeats (§40 Q6) — 1 per point, 16 points, no variance estimate. US-06. The orchestration
script's cluster config now reads from `DILOCO_NODES`/`DILOCO_SSH_KEY` env vars rather than a
hardcoded snapshot (fixed before commit, per CLAUDE.md §23's private-IP discipline) — set
those before re-running against a new cluster.
