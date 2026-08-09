"""Aggregate repeats into median + IQR. Never mean-only (CLAUDE.md §27.1, methods/statistics.md §1).

Wall-clock timing on shared cloud infrastructure is right-skewed by nature (a straggler node,
a noisy-neighbor blip, an ENA burst-credit stall) — a mean lets one bad run silently dominate a
figure. Uses the stdlib `statistics` module deliberately: this is small-N (repeats, not a big
data problem) and pandas/numpy would be complexity without a second user (Architecture
Principle #8).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class AggregatedMetric:
    median: float
    q1: float
    q3: float
    n: int

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


def aggregate_repeats(values: list[float]) -> AggregatedMetric:
    """Median + IQR across repeats. `n` must be reported alongside every figure value (FR-13)
    — a value backed by fewer than the planned repeat count must say so, not present as if
    fully powered (methods/statistics.md §4).
    """
    if not values:
        raise ValueError("aggregate_repeats requires at least one value")

    n = len(values)
    median = statistics.median(values)

    if n == 1:
        q1 = q3 = values[0]
    else:
        try:
            q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        except statistics.StatisticsError:
            # Fewer than 2 distinct points for a stable quartile split — degrade to the
            # median rather than raising, since this is still a valid (if unpowered) point.
            q1 = q3 = median

    return AggregatedMetric(median=median, q1=q1, q3=q3, n=n)


def discrepancy_factor(
    measured_required_bandwidth_bps: float,
    analytic_required_bandwidth_bps: float,
) -> float:
    """F = measured / analytic, at a fixed CU target. The headline number (G2).

    CONTRACT: never returns a value for an undefined ratio — a zero or non-positive analytic
    bandwidth is a data problem to surface, not to silently divide by (CLAUDE.md §33.2.6).
    See methods/statistics.md §2 — confidence-interval method around this point estimate is
    [UNKNOWN] pending Phase 3 data volume; this function returns the point estimate only.
    """
    if analytic_required_bandwidth_bps <= 0:
        raise ValueError(
            f"analytic_required_bandwidth_bps must be > 0, got {analytic_required_bandwidth_bps!r}"
        )
    return measured_required_bandwidth_bps / analytic_required_bandwidth_bps
