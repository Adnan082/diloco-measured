# Limitations

**Status:** `[CONFIRMED — set of known confounds]`, `[content pending measurement]`.

Per `CLAUDE.md` Architecture Principle #10 ("The limitation section is part of the deliverable"),
every known confound is stated here *before* a reviewer finds it. This list is seeded from the
non-goals (§4.5), technical debt (§39), and risks (§38) already accepted in `CLAUDE.md` — it must
be kept in sync with that document (§45.1 sync checklist) and rewritten in reader-facing language
before publication.

---

## Scope limitations

- **No real WAN emulation.** `tc`/`tbf` shapes bandwidth only — no added latency, jitter, or
  packet loss (TD-3). The measured discrepancy this project reports is therefore a **lower
  bound** on the discrepancy a real, lossy, high-latency WAN would show.
- **Four replicas only**, fixed by the 32-vCPU AWS quota (TD-4). Conclusions may not generalize
  to 8, 16, or more replicas.
- **Single-GPU replicas.** No multi-GPU-per-replica (FSDP-inside-a-DiLoCo-worker) hierarchy is
  tested (TD-5) — untestable on this hardware.
- **Models ≤ ~1B parameters**, short token budgets (~400M tokens) (TD-6). Results may not hold
  at frontier scale.
- **One GPU generation** (NVIDIA L40S, Ada sm89), one cloud (AWS), one region/AZ.
- **One seed per convergence configuration** (TD-7) — convergence conclusions have not been
  checked for seed-sensitivity.

## Methodological limitations

- The analytic CU model attributed to "the literature" is one specific functional form
  (`CLAUDE.md` §40 Q3, pending); a sensitivity analysis against alternative forms is planned but
  not yet done.
- `torchft`'s LocalSGD/DiLoCo paths are explicitly experimental upstream (R2); mitigated by a
  cross-validated in-repo reference implementation (ADR-003), not eliminated.
- Straggler heterogeneity across nominally-identical EC2 instances is measured but not fully
  separated from bandwidth effects in every figure (R10).

## What is explicitly NOT claimed

- This project does not claim DiLoCo is good or bad as an algorithm.
- This project does not claim to have invented DiLoCo, LocalSGD, or the analytic scaling-law
  model — see `PRIOR_ART.md`.
- This project does not claim results generalize beyond the scope stated above (R11 — accepted,
  not mitigated).

---

`[UNKNOWN]` The remaining content of this file is written incrementally as findings land, and
finalized during Phase 6 close-out (`CLAUDE.md` §35, §46 "Honesty" checklist).
