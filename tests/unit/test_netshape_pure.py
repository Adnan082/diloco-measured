"""Unit tests for the pure pieces of measurement/netshape.py: tolerance arithmetic and
tc argv construction. Real subprocess/SSH execution (apply/verify/restore) is untestable
without a real Linux node and is covered only by the E2E smoke test (CLAUDE.md §30.6).
"""

from __future__ import annotations

import pytest

from diloco_measured.measurement.netshape import (
    build_tbf_add_args,
    build_tbf_del_args,
    compute_error_pct,
    passes_tolerance,
)


@pytest.mark.unit
def test_compute_error_pct_exact_match_is_zero():
    assert compute_error_pct(1_000_000_000, 1_000_000_000) == 0.0


@pytest.mark.unit
def test_compute_error_pct_matches_worked_example():
    # CLAUDE.md US-01: 1 Gbit/s requested, 0.62 Gbit/s measured => ~38% error.
    error = compute_error_pct(1_000_000_000, 0.62e9)
    assert error == pytest.approx(38.0, abs=0.01)


@pytest.mark.unit
def test_compute_error_pct_rejects_non_positive_requested():
    with pytest.raises(ValueError):
        compute_error_pct(0, 100)


@pytest.mark.unit
def test_passes_tolerance_worked_examples():
    # US-01: 38% error against a 10% tolerance => fails.
    assert not passes_tolerance(1_000_000_000, 0.62e9, tolerance_pct=10)
    # 0.96 Gbit/s measured for a 1 Gbit/s request => 4% error, passes a 10% tolerance.
    assert passes_tolerance(1_000_000_000, 0.96e9, tolerance_pct=10)


@pytest.mark.unit
def test_passes_tolerance_boundary_is_inclusive():
    # Exactly at tolerance should pass (<=, not <).
    assert passes_tolerance(1_000_000_000, 1_100_000_000, tolerance_pct=10)


@pytest.mark.unit
def test_build_tbf_add_args_is_a_flat_list_with_no_shell_metacharacters():
    args = build_tbf_add_args("ens5", rate_bps=1_000_000_000, burst_bytes=32768, latency_ms=50)
    assert args[:6] == ["tc", "qdisc", "add", "dev", "ens5", "root"]
    assert "tbf" in args
    assert "rate" in args
    assert "1000000000bit" in args
    assert "latency" in args
    assert "50ms" in args
    # No element should contain shell metacharacters — this list goes straight to
    # subprocess.run(argv, shell=False), never through a shell (CLAUDE.md §23).
    for token in args:
        assert not any(c in token for c in ";|&$`\n")


@pytest.mark.unit
def test_build_tbf_add_args_rejects_non_positive_rate():
    with pytest.raises(ValueError):
        build_tbf_add_args("ens5", rate_bps=0, burst_bytes=1024, latency_ms=50)


@pytest.mark.unit
def test_build_tbf_del_args():
    assert build_tbf_del_args("ens5") == ["tc", "qdisc", "del", "dev", "ens5", "root"]
