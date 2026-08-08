"""Compute utilization: measured AND analytic, from one shared input schema (FR-04).

This module IS the project's headline contribution (FR-04 status note). See
methods/cu_model.md — the analytic model form is [PROPOSED] and PENDING (§40 Q3); `analytic()`
below must not be implemented against a guessed form. Do not fill this in until Q3 is resolved
and methods/cu_model.md §2 says [CONFIRMED].

STATUS: [PROPOSED] scaffold — deliberately unimplemented pending Q3.
"""

from __future__ import annotations


def measured(steps: "StepRecords", warmup: int) -> float:  # noqa: F821 — StepRecords TBD (pandas/polars frame)
    """CU_measured = sum(compute_time) / sum(total_step_time), over steps after `warmup`."""
    raise NotImplementedError("Phase 0 — see methods/cu_model.md §1")


def analytic(
    spec: dict,
    t_compute_s: float,
    bytes_synced: int,
    bandwidth_bps: int,
) -> float:
    """The literature's analytic CU model (methods/cu_model.md §2, form PENDING §40 Q3).

    CONTRACT (CLAUDE.md §17.2): `bandwidth_bps` is an EXPLICIT, required parameter. There is
    no default. The caller must decide, visibly, whether it is passing link bandwidth or
    achieved bandwidth — this is intentional friction, not an oversight.
    """
    raise NotImplementedError("BLOCKED on CLAUDE.md §40 Q3 — do not guess the model form")
