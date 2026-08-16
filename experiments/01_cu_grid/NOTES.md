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

**Still not done (as of the first shaped pass above):** DDP/FSDP2/LocalSGD (no training driver
exists for them yet — this grid is DiLoCo only, not the full `phase_a.yaml` 4-algorithm
comparison). The 1B model `phase_a.yaml` specifies (still the 30.8M model, for continuity with
the unshaped slice). Repeats (§40 Q6) — 1 per point, 16 points, no variance estimate. US-06.
The orchestration script's cluster config now reads from `DILOCO_NODES`/`DILOCO_SSH_KEY` env
vars rather than a hardcoded snapshot (fixed before commit, per CLAUDE.md §23's private-IP
discipline) — set those before re-running against a new cluster.

---

## Repeats 1 and 2 — 2026-08-15 (CLAUDE.md ADR-037, G1/G2)

Same 16-point grid, same model, run twice more (cluster relaunched again — new placement
group, new node IPs — `DILOCO_REPEAT_INDEX=1` then `=2`, `run_shaped_grid.py`'s new env-var
support). All 32 additional points completed, zero shaping-gate failures, zero crashes. Total
across all 3 repeats: **48 real runs**, satisfying G1's "3 repeats each" for the first time.

Repeat variance is tight — e.g. `H=32, 5g` `cu_measured`: 0.6810 / 0.6728 / 0.6761 across
r0/r1/r2 (< 1.5% spread). This is itself informative: the single-repeat point estimates in the
section above were not noise-dominated.

`experiments/01_cu_grid/compute_required_bandwidth_table.py` (new) answers G2 directly:
per `H`, the bandwidth needed to reach 50/75/90/95% CU (log-linear interpolation between
measured bandwidth levels, never extrapolated past 5 Gbit/s — most `H×target` cells are
honestly `null` since only `H∈{32,128}` reach ≥50% CU within the tested range at all).
Concrete numbers: `H=32` needs ~2.06 Gbit/s measured (vs. ~1.26 Gbit/s analytic, F≈1.63×) to
hit 50% CU; `H=128` needs ~394 Mbit/s (vs. ~317 Mbit/s, F≈1.25×) for 50% CU and ~1.55 Gbit/s
(vs. ~937 Mbit/s, F≈1.66×) for 75% CU. Output: `required_bandwidth_table.json` (regenerable,
not a primary record — see the script's own docstring for why it doesn't live in
`results/raw/`).

Figures regenerated from the full 48-run corpus (`fig1_cu_surface_diloco.png` now says "48
contributing runs" in its caption, median-of-3 rather than a single point estimate).

**Still not done:** everything the first shaped pass's list already said (DDP/FSDP2/LocalSGD,
1B model, US-06) — repeats close the variance gap, not the breadth gap.

---

## DDP + LocalSGD grids — 2026-08-15/16 (CLAUDE.md ADR-039)

Cluster relaunched again (new placement group `pg-0f8e8774a16b3b7e7`, same `us-east-1b`,
same instance types). Two new training drivers (`train_driver_ddp.py`, `train_driver_
localsgd.py`) plus a new reference `LocalSGDTrainer` (`measurement/localsgd.py`) — the CU
grid becomes cross-algorithm for the first time. 5 DDP points (bandwidth only, H=1 by
definition) + 16 LocalSGD points (same H×bandwidth grid as DiLoCo's), all 21 completed,
zero shaping-gate failures, zero crashes.

**Two real bugs in the DDP calibration probe, both found on live hardware and fixed before
any grid data was trusted** (see ADR-039 for the full mechanism): Triton JIT compilation
contaminating first the synced-path warmup, then — after that fix shipped and the *entire*
5-point grid completed — the `no_sync()` path's *separate* JIT compilation, discovered because
every single point still showed the same outlier signature. The whole DDP grid was thrown
away and re-run from scratch under the fixed driver; nothing collected under the buggy
calibration ever reached `results/raw/`.

**One real orchestration bug**, also found live: the grid script's sequential per-node
`proc.wait(timeout=...)` had no exception handling, so when the 50 Mbit/s DDP point's rank-0
SSH session didn't return in time, the uncaught `TimeoutExpired` crashed the whole campaign —
at that moment all 16 LocalSGD points hadn't started yet. Investigation found the training
had actually finished cleanly (792.7s, all 15 steps, valid output already on disk) — the hang
was in post-training teardown, not training itself. Recovered that point's result by hand,
then hardened the orchestrator (shared-deadline polling, force-kill stragglers, always attempt
result fetch regardless of clean ssh exit, resume-awareness, defensive process cleanup between
every point). The fixed orchestrator then ran the rest of the campaign with zero further
timeouts.

**Headline finding:** DDP (no H-amortization at all) collapses to near-total idleness under
scarcity — `cu_measured` 16.68% unshaped → 0.04% at 50 Mbit/s — and lands dramatically below
*both* analytic predictions at every shaped level (~2 orders of magnitude below the naive
model at 50 Mbit/s). LocalSGD tracks DiLoCo's existing H×bandwidth numbers closely (within
~0.15pp at low H, a few points lower at high H) — the first real, direct comparison of the
"no outer optimizer" ablation against DiLoCo's pseudo-gradient+Nesterov design.

Figure generation needed **zero code changes** — `analysis/report.py` already discovers
algorithms dynamically from the corpus (ADR-029's design held up under real multi-algorithm
data for the first time). `fig5_bytes_on_wire_*` remains empty for every algorithm (already
flagged at ADR-038 — `wire` has never been populated by any driver, pre-existing debt, not
new).

**Still not done:** FSDP2 (not started this session). 1B model. Repeats (1 per point, both
algorithms — no variance estimate yet). US-06. `wire` accounting in any driver.
