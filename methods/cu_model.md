# Compute Utilization Model

**Status:** `[CONFIRMED — form only]`. §40 Q3 was resolved by the project owner on 2026-08-09
(decision: Option 1, see `CLAUDE.md` ADR-015): the primary analytic form below is now the
implemented specification for `analysis/cu.py::analytic()`. This remains the single most
reviewer-sensitive document in the repository (FR-04's design note: "you compared apples to
oranges" is the first objection a serious reviewer raises) — the sensitivity analysis in §6
is still owed before publication, and every number this form produces is `[PROPOSED]` output
of a `[CONFIRMED]` formula until that analysis exists.

This document is the specification for `analysis/cu.py::analytic()`. The code implements this,
not the reverse (`CLAUDE.md` §45.2).

---

## 1. Definitions

- **`CU_measured`** = `Σ compute_time / Σ total_step_time`, over the post-warmup measurement
  window (FR-04 step 1). Measured directly from `torch.cuda.Event` decomposition (or the
  `perf_counter` fallback off-GPU — see `measurement/telemetry.py`). Implemented:
  `analysis/cu.py::measured()`.
- **`CU_analytic`** = `f(H, t_compute, bytes_synced, bandwidth)` — the literature's model, fed
  either the nominal link bandwidth (`cu_analytic_link`) or the measured achieved NCCL bandwidth
  at the relevant message size (`cu_analytic_achieved`). Implemented: `analysis/cu.py::analytic()`.

## 2. Functional form (§40 Q3 — RESOLVED, see ADR-015)

**Chosen form (Option 1):**

```
CU = H · t_compute / (H · t_compute + bytes_synced / B)
```

i.e. compute time accumulated over `H` inner steps, divided by that plus the wall time of one
synchronization at bandwidth `B`. `bytes_synced` is in bytes, `B` (`bandwidth_bps`) is in bits
per second, so the sync-time term is `bytes_synced · 8 / B` seconds.

**Rejected-for-now alternatives** (kept here, not deleted, because §6's sensitivity analysis
must still run them against real data before publication):

- **Option 2 — partial overlap.** A variant accounting for overlap between computation and
  communication. **No functional form has been specified anywhere in this repository** — this
  is not merely unimplemented, it is *undefined*. Before this option can be implemented, its
  exact formula must be derived or sourced and written here. Implementing anything under this
  name without that step would be inventing a model form, which `CLAUDE.md` §33.2.6 forbids.
- **Option 3 — per-paper reproduction.** Reproduce each source paper's literal formula
  individually (DeepMind DiLoCo scaling laws, Eager Updates, Decoupled DiLoCo) rather than one
  canonical "the literature" form. Most defensible, most work — each paper's exact equation
  needs to be transcribed and cited here before it's implementable.

**Decision:** `[CONFIRMED]` Option 1, by the project owner, 2026-08-09 (`CLAUDE.md` ADR-015).
Options 2 and 3 are explicitly out of scope for the initial headline comparison and remain for
the sensitivity pass in §6.

## 3. Every assumption

Per Architecture Principle #7 (honest labelling) and FR-04, everything Option 1 assumes:

- **`t_compute` is per-run, not necessarily per-rank-homogeneous.** The formula takes a single
  `t_compute_s` value; if ranks have heterogeneous step times (stragglers, R10), the caller
  must decide what to pass in (e.g. the max over ranks, since a blocking barrier waits for the
  slowest) — `analytic()` does not make that choice implicitly. **This is the most likely
  source of a "the model's input is wrong, not the model" finding (ADR-008 pattern) and must
  be stated explicitly wherever a run's `t_compute_s` is chosen.**
- **Synchronization is a non-overlapped, instantaneous-once-initiated blocking barrier.** No
  compute/communication overlap is modeled; the sync cost is applied in full, once per `H`
  steps, as dead time. This is the literature's simplifying assumption, not a claim about how
  the actual measured rig behaves — the entire point of `cu_measured` existing is to check it.
- **`bytes_synced` is the FULL synchronized-tensor size for the round** (see
  `methods/wire_model.md` §2-3 — same `N` used in the ring all-reduce byte prediction), not a
  per-rank ring-topology-adjusted figure. The model treats the sync cost as `bytes/B`, a
  single logical transfer, deliberately simpler than the ring all-reduce's actual
  `2·N·(P-1)/P` byte count — this is *why* `cu_analytic_achieved` (fed the measured NCCL
  bandwidth, which already reflects ring overhead in its rate) and `cu_analytic_link` (fed raw
  link bandwidth, which does not) can diverge even when both use the same `bytes_synced`.
- **Which bandwidth is `B`:** `cu_analytic_link` is fed the nominal/requested link bandwidth
  (the papers' implicit assumption: achieved == link). `cu_analytic_achieved` is fed the
  measured NCCL achieved-bandwidth at the run's message size, or `null` if no `NetworkProfile`
  NCCL curve covers that size (FR-04 alt-flow 3a/failure condition) — never guessed.
- **Attribution:** this functional form is the project's own simplification for a shared
  "the literature" baseline, not a literal transcription of any single paper's equation —
  see Option 3 above for why a literal per-paper reproduction is separate, harder, future work.

## 4. Interpolation policy

Per FR-04 alternative flow 3a: if the run's message size falls outside the measured NCCL
bandwidth curve for its `NetworkProfile`, `cu_analytic_achieved` is computed by interpolation
and the record sets `nccl_bw_interpolated: true`. If no NCCL curve exists at all for the relevant
bandwidth level, `cu_analytic_achieved` is `null` — never guessed (FR-04 failure condition).

## 5. Reconciliation invariant

`compute_s + sync_blocked_s + optimizer_s + loader_stall_s` must sum to `total_s` within a
recorded residual. A residual above `[PROPOSED]` 5% invalidates the `CUObservation` (see
`CLAUDE.md` §15.2 `CUObservation` entity). Implemented: `measurement/telemetry.py`'s
`StepTiming.reconciliation_residual_pct`; enforced at the aggregation layer by
`analysis/filter.py`.

## 6. Sensitivity analysis (required before publication) — STILL OWED

`[PROPOSED]` Now that Option 1 is implemented, this is unblocked but not done: re-run the
discrepancy-factor computation under Option 2 (once its form is specified — see §2) and, for
at least the primary source papers, Option 3, and report whether the headline conclusion
(§2.7 hypothesis) survives the choice of model form. This requires real Phase 3 grid data and
cannot be done from synthetic fixtures alone — tracked as follow-up work, not resolved by
ADR-015.
