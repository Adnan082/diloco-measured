"""Exclusion rules: crashed, loader-bound, harness_version-mismatched records.

See methods/measurement_windows.md §3 for the full policy this module implements. Every
exclusion must be counted and reported (CLAUDE.md §25.3) — never a silent drop.

STATUS: [PROPOSED] scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterReport:
    """Counts of what was excluded and why — must be surfaced in every figure/table that uses it."""

    total: int
    kept: int
    excluded_crashed: int
    excluded_diverged: int
    excluded_loader_bound: int
    excluded_version_mismatch: int
    excluded_reconciliation_failed: int


def apply(
    records: list[dict],
    harness_version: str | None = None,
    allow_version_mix: bool = False,
) -> tuple[list[dict], FilterReport]:
    """Exclude non-completed, loader-bound, reconciliation-failed, and (unless overridden)
    version-mismatched records. Returns (kept_records, FilterReport) — the report is not
    optional; callers must propagate it into figure metadata (FR-13).
    """
    raise NotImplementedError("Phase 0 — see methods/measurement_windows.md §3")
