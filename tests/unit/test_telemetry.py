"""Unit tests for measurement/telemetry.py::StepTimer and measure_instrumentation_overhead.

Exercises the perf_counter backend only — there's no GPU in this environment, so the CUDA-
event backend is untestable here by construction (and is documented as UNVERIFIED ON REAL
HARDWARE in telemetry.py until Phase 1's `make smoke` exercises it for real).
"""

from __future__ import annotations

import time

import pytest

from diloco_measured.measurement.telemetry import StepTimer, measure_instrumentation_overhead


@pytest.mark.unit
def test_result_before_entering_context_raises():
    with pytest.raises(RuntimeError):
        StepTimer().result()


@pytest.mark.unit
def test_fully_marked_step_has_near_zero_residual():
    with StepTimer() as t:
        time.sleep(0.001)
        t.mark_loader_done()
        time.sleep(0.001)
        t.mark_compute_done()
        time.sleep(0.001)
        t.mark_sync_done()
        time.sleep(0.001)
        t.mark_optimizer_done()
    timing = t.result()

    assert timing.wall_time_ms > 0
    assert timing.loader_stall_ms > 0
    assert timing.compute_time_ms > 0
    assert timing.sync_blocked_ms > 0
    assert timing.optimizer_time_ms > 0
    # Should reconcile essentially exactly: perf_counter markers ARE the ground truth here
    # (no async execution to misrepresent), so the only residual is inter-call overhead.
    assert timing.reconciliation_residual_pct < 5.0


@pytest.mark.unit
def test_unmarked_phase_contributes_zero_duration_not_an_error():
    """A DDP step with no separate loader phase legitimately skips mark_loader_done()."""
    with StepTimer() as t:
        time.sleep(0.001)
        t.mark_compute_done()
        time.sleep(0.001)
        t.mark_sync_done()
        time.sleep(0.001)
        t.mark_optimizer_done()
    timing = t.result()

    assert timing.loader_stall_ms == 0.0
    assert timing.compute_time_ms > 0  # start..compute_done, absorbing the "loader" gap


@pytest.mark.unit
def test_skipped_final_marker_shows_up_as_reconciliation_residual():
    """Forgetting mark_optimizer_done() (but still doing optimizer work before __exit__)
    must show up as unaccounted time, not silently vanish into another bucket — this is the
    entire point of the reconciliation check (methods/cu_model.md §5).
    """
    with StepTimer() as t:
        t.mark_loader_done()
        t.mark_compute_done()
        t.mark_sync_done()
        time.sleep(0.02)  # "optimizer work" that is never marked
    timing = t.result()

    assert timing.optimizer_time_ms == 0.0
    # 20ms of real work happened but was attributed to no bucket -> large residual.
    assert timing.reconciliation_residual_pct > 50.0


@pytest.mark.unit
def test_reconciliation_residual_pct_handles_zero_wall_time():
    with StepTimer() as t:
        pass  # no sleep at all; wall time could legitimately be 0 on a fast clock
    timing = t.result()
    # Must not raise ZeroDivisionError regardless of how fast the clock resolution is.
    assert timing.reconciliation_residual_pct >= 0.0


@pytest.mark.unit
def test_measure_instrumentation_overhead_returns_a_small_finite_number():
    def step_fn(timer):
        if timer is not None:
            timer.mark_loader_done()
        # Synthetic "step" work, sized to comfortably clear the 1ms measurability floor
        # regardless of host speed.
        time.sleep(0.003)
        if timer is not None:
            timer.mark_compute_done()
            timer.mark_sync_done()
            timer.mark_optimizer_done()

    overhead = measure_instrumentation_overhead(step_fn, n_steps=20)
    assert isinstance(overhead, float)
    # Sanity bound only — this is a noisy wall-clock measurement, not a strict assertion that
    # overhead is under the CLAUDE.md §27 1% target (that check belongs on real hardware).
    assert -1.0 < overhead < 5.0


@pytest.mark.unit
def test_measure_instrumentation_overhead_rejects_a_noop_step_fn():
    with pytest.raises(ValueError):
        measure_instrumentation_overhead(lambda timer: None, n_steps=5)
