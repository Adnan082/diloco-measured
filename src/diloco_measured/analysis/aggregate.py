"""Aggregate repeats into median + IQR. Never mean-only (CLAUDE.md §27.1, methods/statistics.md §1).

STATUS: [PROPOSED] scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AggregatedMetric:
    median: float
    q1: float
    q3: float
    n: int


def aggregate_repeats(values: list[float]) -> AggregatedMetric:
    """Median + IQR across repeats. `n` must be reported alongside every figure value (FR-13)."""
    raise NotImplementedError("Phase 0 — see methods/statistics.md §1")


def discrepancy_factor(
    measured_required_bandwidth_bps: float,
    analytic_required_bandwidth_bps: float,
) -> float:
    """F = measured / analytic, at a fixed CU target. The headline number (G2).

    See methods/statistics.md §2 — confidence interval method is [UNKNOWN] pending Phase 3
    data volume; this function returns the point estimate only.
    """
    raise NotImplementedError("Phase 3")
