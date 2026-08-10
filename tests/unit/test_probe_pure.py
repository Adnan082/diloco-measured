"""Unit tests for measurement/probe.py::log_spaced_message_sizes — the one piece of FR-01
step 3 that doesn't need a distributed process group or a real node to test.
"""

from __future__ import annotations

import pytest

from diloco_measured.measurement.probe import log_spaced_message_sizes


@pytest.mark.unit
def test_endpoints_are_exact():
    sizes = log_spaced_message_sizes(1024, 1_048_576, 5)
    assert sizes[0] == 1024
    assert sizes[-1] == 1_048_576


@pytest.mark.unit
def test_is_monotonically_increasing():
    sizes = log_spaced_message_sizes(1024, 4 * 1024**3, 10)
    assert sizes == sorted(sizes)
    assert len(sizes) == len(set(sizes))  # strictly increasing, no duplicates post-dedup


@pytest.mark.unit
def test_fr01_default_range_1mib_to_4gib():
    """CLAUDE.md FR-01 step 3: "log-spaced from 1 MiB to 4 GiB"."""
    sizes = log_spaced_message_sizes(1024**2, 4 * 1024**3, 8)
    assert sizes[0] == 1024**2
    assert sizes[-1] == 4 * 1024**3
    assert len(sizes) <= 8  # dedup may reduce count, never increase it


@pytest.mark.unit
def test_single_point_returns_min():
    assert log_spaced_message_sizes(100, 1000, 1) == [100]


@pytest.mark.unit
def test_min_equals_max():
    assert log_spaced_message_sizes(500, 500, 4) == [500]


@pytest.mark.unit
def test_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="must be > 0"):
        log_spaced_message_sizes(0, 1000, 5)
    with pytest.raises(ValueError, match="must be <="):
        log_spaced_message_sizes(1000, 100, 5)
    with pytest.raises(ValueError, match="n_points"):
        log_spaced_message_sizes(100, 1000, 0)


@pytest.mark.unit
def test_dedup_when_range_too_narrow_for_point_count():
    # A tiny range with many requested points will produce duplicate rounded values —
    # confirm they're collapsed rather than returned as literal repeats.
    sizes = log_spaced_message_sizes(100, 105, 20)
    assert len(sizes) == len(set(sizes))
