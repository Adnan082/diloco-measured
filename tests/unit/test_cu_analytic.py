"""Unit tests for analysis/cu.py::analytic() — methods/cu_model.md §2 Option 1, [CONFIRMED]
per ADR-015 (§40 Q3 resolved 2026-08-09).

Test targets from CLAUDE.md §30.2: "Known-input/known-output cases hand-computed from
methods/cu_model.md; H=1 reduces to the DDP case; monotonicity in bandwidth and in H."
"""

from __future__ import annotations

import pytest

from diloco_measured.analysis.cu import analytic


def _spec(H: int) -> dict:
    return {"H": H}


@pytest.mark.unit
def test_known_value_h1():
    # sync_time = 125_000 bytes * 8 / 1_000_000 bps = 1.0s; compute_budget = 1 * 0.1 = 0.1s
    # CU = 0.1 / (0.1 + 1.0) = 0.1 / 1.1
    result = analytic(_spec(H=1), t_compute_s=0.1, bytes_synced=125_000, bandwidth_bps=1_000_000)
    assert result == pytest.approx(0.1 / 1.1)


@pytest.mark.unit
def test_known_value_h32():
    # Same sync cost as above, but H=32 amortizes it: compute_budget = 32 * 0.1 = 3.2s
    # CU = 3.2 / (3.2 + 1.0) = 3.2 / 4.2
    result = analytic(_spec(H=32), t_compute_s=0.1, bytes_synced=125_000, bandwidth_bps=1_000_000)
    assert result == pytest.approx(3.2 / 4.2)


@pytest.mark.unit
def test_h_equals_1_is_the_ddp_case():
    """H=1 is not special-cased anywhere in analytic() — the general formula must reduce to
    the same thing as a hand-derived DDP (sync-every-step) computation, since there is no
    separate DDP-only code path (methods/cu_model.md §2 makes no such distinction).
    """
    t_compute_s, bytes_synced, bandwidth_bps = 0.05, 500_000, 2_000_000
    expected = (1 * t_compute_s) / (1 * t_compute_s + (bytes_synced * 8) / bandwidth_bps)
    assert analytic(_spec(H=1), t_compute_s, bytes_synced, bandwidth_bps) == pytest.approx(expected)


@pytest.mark.unit
def test_zero_bytes_synced_gives_perfect_utilization():
    result = analytic(_spec(H=1), t_compute_s=1.0, bytes_synced=0, bandwidth_bps=1_000)
    assert result == pytest.approx(1.0)


@pytest.mark.unit
def test_monotonic_increasing_in_h():
    """More inner steps between syncs amortizes the fixed sync cost -> higher CU."""
    kwargs = {"t_compute_s": 0.1, "bytes_synced": 1_000_000, "bandwidth_bps": 1_000_000}
    cu_h1 = analytic(_spec(H=1), **kwargs)
    cu_h8 = analytic(_spec(H=8), **kwargs)
    cu_h128 = analytic(_spec(H=128), **kwargs)
    assert cu_h1 < cu_h8 < cu_h128
    assert cu_h128 < 1.0  # approaches but never reaches 1.0 for finite H


@pytest.mark.unit
def test_monotonic_increasing_in_bandwidth():
    """More bandwidth -> shorter sync time -> higher CU, holding everything else fixed."""
    kwargs = {"spec": _spec(H=8), "t_compute_s": 0.1, "bytes_synced": 1_000_000}
    cu_slow = analytic(bandwidth_bps=1_000_000, **kwargs)
    cu_medium = analytic(bandwidth_bps=10_000_000, **kwargs)
    cu_fast = analytic(bandwidth_bps=1_000_000_000, **kwargs)
    assert cu_slow < cu_medium < cu_fast
    assert cu_fast < 1.0


@pytest.mark.unit
def test_cu_is_bounded_in_zero_one_open_interval_for_positive_bytes():
    result = analytic(_spec(H=4), t_compute_s=0.01, bytes_synced=10_000_000, bandwidth_bps=1_000)
    assert 0.0 < result < 1.0


@pytest.mark.unit
@pytest.mark.parametrize("H", [0, -1])
def test_rejects_non_positive_h(H):
    with pytest.raises(ValueError, match="H must be >= 1"):
        analytic(_spec(H=H), t_compute_s=0.1, bytes_synced=1000, bandwidth_bps=1000)


@pytest.mark.unit
@pytest.mark.parametrize("t_compute_s", [0, -0.1])
def test_rejects_non_positive_t_compute(t_compute_s):
    with pytest.raises(ValueError, match="t_compute_s must be > 0"):
        analytic(_spec(H=1), t_compute_s=t_compute_s, bytes_synced=1000, bandwidth_bps=1000)


@pytest.mark.unit
def test_rejects_negative_bytes_synced():
    with pytest.raises(ValueError, match="bytes_synced must be >= 0"):
        analytic(_spec(H=1), t_compute_s=0.1, bytes_synced=-1, bandwidth_bps=1000)


@pytest.mark.unit
@pytest.mark.parametrize("bandwidth_bps", [0, -1000])
def test_rejects_non_positive_bandwidth(bandwidth_bps):
    with pytest.raises(ValueError, match="bandwidth_bps must be > 0"):
        analytic(_spec(H=1), t_compute_s=0.1, bytes_synced=1000, bandwidth_bps=bandwidth_bps)


@pytest.mark.unit
def test_bandwidth_is_required_with_no_default_link_vs_achieved_friction():
    """CLAUDE.md §17.2: bandwidth_bps has no default — the call site must always name it."""
    import inspect

    sig = inspect.signature(analytic)
    assert sig.parameters["bandwidth_bps"].default is inspect.Parameter.empty
