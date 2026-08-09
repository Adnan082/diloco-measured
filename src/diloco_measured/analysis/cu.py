"""Compute utilization: measured AND analytic, from one shared input schema (FR-04).

This module IS the project's headline contribution (FR-04 status note). §40 Q3 (which
analytic CU model form to use) was RESOLVED by the project owner on 2026-08-09 — see
CLAUDE.md ADR-015 and methods/cu_model.md §2, both now [CONFIRMED] on the form itself. The
sensitivity analysis against the rejected alternatives (methods/cu_model.md §6) is still owed
before publication — that is real, separate, data-dependent work, not resolved by this file.
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
    """The literature's analytic CU model — methods/cu_model.md §2, Option 1, `[CONFIRMED]`
    per ADR-015:

        CU = H · t_compute / (H · t_compute + bytes_synced · 8 / B)

    `H` comes from `spec["H"]`. `bytes_synced` is in bytes; `bandwidth_bps` is bits/second
    (matching the `_bps` naming convention, CLAUDE.md §14.3), hence the `· 8`.

    Every assumption behind this form (non-overlapped blocking sync, what `t_compute_s`
    means when ranks are heterogeneous, why `bytes_synced` is the full round-trip tensor
    size) is listed in methods/cu_model.md §3 — read that before changing this function's
    behavior, per §45.2 ("the specification; the code implements it, not the reverse").

    CONTRACT (CLAUDE.md §17.2): `bandwidth_bps` is an EXPLICIT, required parameter. There is
    no default. The caller must decide, visibly, whether it is passing link bandwidth
    (`cu_analytic_link`) or achieved bandwidth (`cu_analytic_achieved`) — this is intentional
    friction, not an oversight.
    """
    H = spec["H"]
    if H < 1:
        raise ValueError(f"H must be >= 1, got {H!r}")
    if t_compute_s <= 0:
        raise ValueError(f"t_compute_s must be > 0, got {t_compute_s!r}")
    if bytes_synced < 0:
        raise ValueError(f"bytes_synced must be >= 0, got {bytes_synced!r}")
    if bandwidth_bps <= 0:
        raise ValueError(f"bandwidth_bps must be > 0, got {bandwidth_bps!r}")

    compute_budget_s = H * t_compute_s
    sync_time_s = (bytes_synced * 8) / bandwidth_bps
    return compute_budget_s / (compute_budget_s + sync_time_s)
