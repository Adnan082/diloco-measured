# Compute Utilization Model

**Status:** `[PROPOSED — blocks M3, CLAUDE.md §40 Q3]`. This is the single most reviewer-sensitive
document in the repository (FR-04's design note: "you compared apples to oranges" is the first
objection a serious reviewer raises). Nothing here is final until Q3 is resolved.

This document is the specification for `analysis/cu.py::analytic()`. The code implements this,
not the reverse (`CLAUDE.md` §45.2).

---

## 1. Definitions

- **`CU_measured`** = `Σ compute_time / Σ total_step_time`, over the post-warmup measurement
  window (FR-04 step 1). Measured directly from `torch.cuda.Event` decomposition.
- **`CU_analytic`** = `f(H, t_compute, bytes_synced, bandwidth)` — the literature's model, fed
  either the nominal link bandwidth (`cu_analytic_link`) or the measured achieved NCCL bandwidth
  at the relevant message size (`cu_analytic_achieved`).

## 2. Candidate functional forms (§40 Q3 — PENDING)

**Option 1 (primary candidate):**

```
CU = H · t_compute / (H · t_compute + bytes_synced / B)
```

i.e. compute time accumulated over `H` inner steps, divided by that plus the wall time of one
synchronization at bandwidth `B`. This assumes synchronization is a non-overlapped, instantaneous
blocking barrier once initiated.

**Option 2:** a variant accounting for partial overlap between computation and communication.

**Option 3:** reproduce the exact form used in each source paper individually and report all of
them, rather than picking one as "the" literature form.

**Decision:** PENDING. Recommendation is Option 1 as primary + a sensitivity analysis against
Option 2/3 (`CLAUDE.md` §40 Q3 recommendation). **Do not implement `analytic()` against this
document until Q3 is resolved and this line is updated to `[CONFIRMED]`.**

## 3. Every assumption (fill in as each is confirmed)

`[UNKNOWN]` — to be listed exhaustively once Option 1 (or its replacement) is confirmed. At
minimum this section must state, per Architecture Principle #7 (honest labelling) and FR-04:

- Whether `t_compute` is measured per-rank or assumed homogeneous across ranks.
- Whether the model assumes synchronous barrier semantics or allows overlap.
- Which bandwidth value is substituted for `B` in each variant (`link` vs. `achieved`), and why
  `cu_analytic_achieved` is `null` when no NCCL curve covers the message size (FR-04 alt-flow 3a).
- The source paper(s) each assumption is attributed to.

## 4. Interpolation policy

Per FR-04 alternative flow 3a: if the run's message size falls outside the measured NCCL
bandwidth curve for its `NetworkProfile`, `cu_analytic_achieved` is computed by interpolation
and the record sets `nccl_bw_interpolated: true`. If no NCCL curve exists at all for the relevant
bandwidth level, `cu_analytic_achieved` is `null` — never guessed (FR-04 failure condition).

## 5. Reconciliation invariant

`compute_s + sync_blocked_s + optimizer_s + loader_stall_s` must sum to `total_s` within a
recorded residual. A residual above `[PROPOSED]` 5% invalidates the `CUObservation` (see
`CLAUDE.md` §15.2 `CUObservation` entity).

## 6. Sensitivity analysis (required before publication)

`[PROPOSED]` Once Option 1 is implemented, re-run the discrepancy-factor computation under
Option 2 and (for at least the primary source papers) Option 3, and report whether the headline
conclusion (§2.7 hypothesis) survives the choice of model form. This is what makes the
"apples to oranges" objection answerable rather than merely asserted away.
