"""Compute utilization: measured AND analytic, from one shared input schema (FR-04).

This module IS the project's headline contribution (FR-04 status note). See
methods/cu_model.md — the analytic model form is [PROPOSED] and PENDING (§40 Q3); `analytic()`
below must not be implemented against a guessed form. Do not fill this in until Q3 is resolved
and methods/cu_model.md §2 says [CONFIRMED].

STATUS: [PROPOSED] scaffold — deliberately unimplemented pending Q3.
"""

from __future__ import annotations

from typing import Any


def _field(record: Any, name: str) -> Any:
    """Read `name` off a StepRecord-like `record`, whether it's a dict (JSON-shaped, matching
    schemas/step_record.v1.json) or an attribute-style row (e.g. a pandas itertuples()/polars
    iter_rows(named=True) result). Isolates the one place that would need to change if the
    concrete container type (CLAUDE.md §13.3, [PROPOSED]) is later pinned to something else.
    """
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def measured(steps: Any, warmup: int) -> float:
    """CU_measured = sum(compute_time_ms) / sum(wall_time_ms), over steps after `warmup` are
    discarded (NFR-09 — warmup steps are always discarded, and the discard count is recorded
    by the caller alongside this value).

    `steps` type is intentionally `Any`: the concrete container (pandas vs. polars DataFrame
    over StepRecord rows, CLAUDE.md §13.3, still [PROPOSED]) isn't decided — this function
    only requires iteration in step order plus per-record field access (see `_field()`), so
    it works unchanged regardless of which container wins that decision. Unlike
    `analytic()` below, this has no dependency on §40 Q3 — it is pure arithmetic over
    already-measured per-step telemetry, not a model.
    """
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup!r}")

    records = list(steps)[warmup:]
    if not records:
        raise ValueError(f"no StepRecords remain after discarding warmup={warmup}")

    total_compute_ms = sum(_field(r, "compute_time_ms") for r in records)
    total_wall_ms = sum(_field(r, "wall_time_ms") for r in records)
    if total_wall_ms <= 0:
        raise ValueError(f"sum of wall_time_ms is {total_wall_ms!r}, must be > 0")

    return total_compute_ms / total_wall_ms


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
