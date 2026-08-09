"""Unit tests for analysis/aggregate.py — median + IQR, never mean-only."""

from __future__ import annotations

import pytest

from diloco_measured.analysis.aggregate import aggregate_repeats, discrepancy_factor


@pytest.mark.unit
def test_aggregate_repeats_median_and_iqr_known_values():
    # [1, 2, 3, 4, 5, 6, 7, 8, 9] -> median 5, Q1 3, Q3 7 (inclusive method)
    result = aggregate_repeats([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert result.median == 5
    assert result.q1 == 3
    assert result.q3 == 7
    assert result.iqr == 4
    assert result.n == 9


@pytest.mark.unit
def test_aggregate_repeats_single_value():
    result = aggregate_repeats([42.0])
    assert result.median == 42.0
    assert result.q1 == result.q3 == 42.0
    assert result.n == 1


@pytest.mark.unit
def test_aggregate_repeats_rejects_empty():
    with pytest.raises(ValueError):
        aggregate_repeats([])


@pytest.mark.unit
def test_aggregate_repeats_is_not_pulled_by_a_single_straggler():
    """A single very slow run should shift the mean a lot but the median very little — this
    is the entire reason CLAUDE.md §27.1 mandates median+IQR over mean-only.
    """
    normal_runs = [10.0, 10.1, 9.9, 10.0, 10.2]
    straggler_run = [10.0, 10.1, 9.9, 10.0, 500.0]

    normal_result = aggregate_repeats(normal_runs)
    straggler_result = aggregate_repeats(straggler_run)

    assert straggler_result.median == pytest.approx(normal_result.median, abs=0.2)
    mean_normal = sum(normal_runs) / len(normal_runs)
    mean_straggler = sum(straggler_run) / len(straggler_run)
    assert mean_straggler - mean_normal > 50  # the mean WOULD have been badly distorted


@pytest.mark.unit
def test_discrepancy_factor_point_estimate():
    assert discrepancy_factor(200.0, 100.0) == pytest.approx(2.0)
    assert discrepancy_factor(50.0, 100.0) == pytest.approx(0.5)


@pytest.mark.unit
def test_discrepancy_factor_rejects_non_positive_analytic():
    with pytest.raises(ValueError):
        discrepancy_factor(100.0, 0.0)
    with pytest.raises(ValueError):
        discrepancy_factor(100.0, -10.0)
