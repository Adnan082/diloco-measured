# Measurement Windows: Warmup, Repeats, Outliers

**Status:** `[PROPOSED]` values below, `[CONFIRMED]` requirement that they be explicit and
recorded. Specifies the windowing behavior of `measurement/train.py` and `measurement/telemetry.py`.

---

## 1. Warmup discard

**Requirement (NFR-09, `[CONFIRMED]`):** warmup steps are discarded from every measurement
window, and the discard count is recorded on the `RunResult`.

**Value:** `[PROPOSED]` — to be set from the Day 1 warmup-sufficiency check (§30.5): compare
10/20/30-step discard; if the CU or throughput conclusion changes between them, warmup is too
short and the value must increase. Do not finalize this number before that check runs.

## 2. Repeat policy

**Value:** `[PROPOSED — CLAUDE.md §40 Q6, PENDING]` — plan is 3 repeats for throughput / 1 seed
for convergence, with an adaptive rule: if a configuration's throughput IQR exceeds a threshold
`[UNKNOWN]`, add repeats for that configuration only rather than uniformly. Resolve after Day 1
variance is observed.

## 3. Outlier / exclusion policy

A run is excluded from analysis aggregation, but never deleted, when:

- `status != completed` (crashed, oom, diverged, aborted_shaping).
- `loader_bound_warning == true` (dataloader stall exceeded `[PROPOSED]` 5% of step time — FR-03
  alt-flow 5a).
- Step-time component reconciliation residual exceeds `[PROPOSED]` 5% (`methods/cu_model.md` §5).
- `harness_version` does not match the version under analysis (unless an explicit
  `--allow-version-mix` override is used, and it is recorded in figure metadata).

Every exclusion is counted and reported, never silently dropped (§25.3, §30.2 `filter` test).

## 4. Aggregation statistic

**Requirement (`[CONFIRMED]`, §30.5 R10-adjacent):** median + IQR across repeats, **never
mean-only** — a single straggler run should not silently pull a mean.

## 5. Instrumentation overhead

Measured once, explicitly: an instrumented vs. uninstrumented run at one fixed configuration
(`[PROPOSED]` target: < 1% of step time, R8). `torch.cuda.Event` sync points can be surprisingly
expensive; this must be measured, not assumed.
