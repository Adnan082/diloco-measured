"""Exclusion rules: non-completed, loader-bound, reconciliation-failed, and
harness_version-mismatched records.

See methods/measurement_windows.md §3 for the full policy this module implements. Every
exclusion is counted and reported (CLAUDE.md §25.3) — never a silent drop. `apply()` never
mutates or discards a record; it only partitions the corpus and reports the partition.
"""

from __future__ import annotations

from dataclasses import dataclass

# [PROPOSED] — CLAUDE.md §15.2 CUObservation invariant: "A residual above 5% invalidates the
# observation." Matches methods/cu_model.md §5.
DEFAULT_RECONCILIATION_TOLERANCE_PCT = 5.0


@dataclass(frozen=True)
class FilterReport:
    """Counts of what was excluded and why — must be surfaced in every figure/table that uses
    the filtered corpus (FR-13). A record is counted in exactly one bucket: the first
    applicable rule wins, in the order checked by `apply()`.
    """

    total: int
    kept: int
    excluded_crashed: int
    excluded_diverged: int
    excluded_other_status: int
    excluded_version_mismatch: int
    excluded_loader_bound: int
    excluded_reconciliation_failed: int

    @property
    def excluded_total(self) -> int:
        return self.total - self.kept


def _reconciliation_residual_pct(cu: dict) -> float | None:
    """|total_s - (compute_s + sync_blocked_s + optimizer_s + loader_stall_s)| / total_s * 100.

    Returns None if `cu` doesn't carry enough fields to check (e.g. total_s missing/zero) —
    callers treat None as "cannot evaluate", not as "passed".
    """
    total_s = cu.get("total_s")
    if not total_s or total_s <= 0:
        return None
    component_sum = sum(
        cu.get(field, 0.0)
        for field in ("compute_s", "sync_blocked_s", "optimizer_s", "loader_stall_s")
    )
    return abs(total_s - component_sum) / total_s * 100.0


def apply(
    records: list[dict],
    harness_version: str | None = None,
    allow_version_mix: bool = False,
    reconciliation_tolerance_pct: float = DEFAULT_RECONCILIATION_TOLERANCE_PCT,
) -> tuple[list[dict], FilterReport]:
    """Partition `records` into (kept, report).

    Exclusion order (first match wins, so the report's buckets are mutually exclusive):
      1. status != "completed"                              (crashed / diverged / other)
      2. harness_version mismatch, unless allow_version_mix   (CLAUDE.md §16.3, ADR-006)
      3. loader_bound_warning == true                          (FR-03 alt-flow 5a)
      4. step-time reconciliation residual exceeds tolerance   (methods/cu_model.md §5)

    `harness_version=None` disables the version-mismatch check entirely (useful when the
    caller has already partitioned by version, e.g. one figure per harness_version).
    """
    kept: list[dict] = []
    excluded_crashed = 0
    excluded_diverged = 0
    excluded_other_status = 0
    excluded_version_mismatch = 0
    excluded_loader_bound = 0
    excluded_reconciliation_failed = 0

    for record in records:
        status = record.get("status")
        if status != "completed":
            if status == "crashed":
                excluded_crashed += 1
            elif status == "diverged":
                excluded_diverged += 1
            else:
                excluded_other_status += 1
            continue

        if (
            harness_version is not None
            and not allow_version_mix
            and record.get("harness_version") != harness_version
        ):
            excluded_version_mismatch += 1
            continue

        if record.get("loader_bound_warning"):
            excluded_loader_bound += 1
            continue

        cu = record.get("cu")
        if cu:
            residual_pct = _reconciliation_residual_pct(cu)
            if residual_pct is not None and residual_pct > reconciliation_tolerance_pct:
                excluded_reconciliation_failed += 1
                continue

        kept.append(record)

    report = FilterReport(
        total=len(records),
        kept=len(kept),
        excluded_crashed=excluded_crashed,
        excluded_diverged=excluded_diverged,
        excluded_other_status=excluded_other_status,
        excluded_version_mismatch=excluded_version_mismatch,
        excluded_loader_bound=excluded_loader_bound,
        excluded_reconciliation_failed=excluded_reconciliation_failed,
    )
    return kept, report
