"""Unit tests for measurement/train.py::run()'s SEQUENCING logic — validation order,
precondition gating, the exactly-one-retry shaping rule, and unconditional restore-on-exit.

Uses fake apply_fn/verify_fn/restore_fn (plain Python functions returning canned results),
injected via run()'s own parameters. This is NOT the same as mocking netshape.py (forbidden,
CLAUDE.md §30.6, and netshape.py's own tests never do this) — these tests check that
train.py calls its dependencies in the right order and reacts to their results correctly;
they say nothing about whether a real iperf3 measurement is trustworthy, which is exactly
the thing that must never be faked.
"""

from __future__ import annotations

import pytest

from diloco_measured.measurement.netshape import ShapingHandle, ShapingVerification
from diloco_measured.measurement.spec import SpecValidationError
from diloco_measured.measurement.train import RunAbort, check_preconditions, run


def _spec(**overrides) -> dict:
    base = {
        "spec_id": "x", "phase": "cu_grid", "algorithm": "diloco", "implementation": "reference",
        "H": 32, "model_config": "m", "world_size": 4, "micro_batch_size": 1, "seq_len": 8,
        "grad_accum": 1, "budget_type": "steps", "budget_value": 1, "warmup_steps": 0,
        "compression": None, "seed": 0, "repeat_index": 0, "fault_schedule": None,
        "bandwidth_requested_bps": 1_000_000_000,
    }
    base.update(overrides)
    return base


def _fake_apply(rate_bps, nodes):
    return ShapingHandle(nodes=tuple(nodes), requested_bps=rate_bps)


def _fake_verify_always_passes(handle, tolerance_pct):
    return ShapingVerification(
        requested_bps=handle.requested_bps, measured_bps=handle.requested_bps,
        error_pct=0.0, tolerance_pct=tolerance_pct, passed=True, attempts=1,
        iperf_raw="{}", qdisc_dump="",
    )


def _fake_verify_always_fails(handle, tolerance_pct):
    return ShapingVerification(
        requested_bps=handle.requested_bps, measured_bps=handle.requested_bps * 0.3,
        error_pct=70.0, tolerance_pct=tolerance_pct, passed=False, attempts=1,
        iperf_raw="{}", qdisc_dump="",
    )


def _run_kwargs(**overrides):
    base = dict(
        spec=_spec(),
        nodes=["node0", "node1"],
        network_profile_exists=True,
        dataset_checksum_ok=True,
        gpu_clocks_locked=True,
        node_dirty=False,
        dataset_shard_checksum="deadbeef",
        apply_fn=_fake_apply,
        verify_fn=_fake_verify_always_passes,
        restore_fn=lambda handle: None,
    )
    base.update(overrides)
    return base


# ---- check_preconditions() -------------------------------------------------------------


@pytest.mark.unit
def test_check_preconditions_all_pass():
    checks = check_preconditions(_spec(), True, True, True, False)
    assert all(c.passed for c in checks)


@pytest.mark.unit
def test_check_preconditions_skips_network_profile_check_when_unshaped():
    checks = check_preconditions(_spec(bandwidth_requested_bps=None), False, True, True, False)
    names = [c.name for c in checks]
    assert "network_profile_exists" not in names


@pytest.mark.unit
def test_check_preconditions_requires_network_profile_when_shaped():
    checks = check_preconditions(_spec(), False, True, True, False)
    failed = {c.name for c in checks if not c.passed}
    assert "network_profile_exists" in failed


# ---- run() sequencing -------------------------------------------------------------------


@pytest.mark.unit
def test_invalid_spec_raises_before_touching_shaping():
    bad_spec = _spec()
    del bad_spec["world_size"]  # required field -> schema violation
    apply_calls = []

    with pytest.raises(SpecValidationError):
        run(**_run_kwargs(spec=bad_spec, apply_fn=lambda *a: apply_calls.append(a)))

    assert apply_calls == [], "shaping must never be touched when the spec itself is invalid"


@pytest.mark.unit
def test_failed_precondition_raises_run_abort_with_precondition_failed_class():
    with pytest.raises(RunAbort) as exc_info:
        run(**_run_kwargs(gpu_clocks_locked=False))
    assert exc_info.value.error_class == "precondition_failed"


@pytest.mark.unit
def test_precondition_failure_never_calls_apply():
    apply_calls = []
    with pytest.raises(RunAbort):
        run(**_run_kwargs(gpu_clocks_locked=False, apply_fn=lambda *a: apply_calls.append(a)))
    assert apply_calls == []


@pytest.mark.unit
def test_shaping_success_reaches_the_not_implemented_boundary():
    """Everything before step 6 (torchrun launch) is real; run() must get all the way to
    that documented boundary — not fail earlier — when every precondition and the shaping
    gate pass.
    """
    with pytest.raises(NotImplementedError, match="torchrun launch"):
        run(**_run_kwargs())


@pytest.mark.unit
def test_shaping_failure_retries_exactly_once_then_aborts():
    verify_calls = []

    def counting_verify_always_fails(handle, tolerance_pct):
        verify_calls.append(1)
        return _fake_verify_always_fails(handle, tolerance_pct)

    with pytest.raises(RunAbort) as exc_info:
        run(**_run_kwargs(verify_fn=counting_verify_always_fails))

    assert exc_info.value.error_class == "shaping_verification_failed"
    assert len(verify_calls) == 2, "FR-02: exactly one retry, i.e. exactly two total attempts"


@pytest.mark.unit
def test_shaping_succeeds_on_the_retry():
    """The retry isn't just attempted — a pass on attempt 2 must let the run proceed."""
    attempts = {"n": 0}

    def verify_fails_once_then_passes(handle, tolerance_pct):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _fake_verify_always_fails(handle, tolerance_pct)
        return _fake_verify_always_passes(handle, tolerance_pct)

    with pytest.raises(NotImplementedError, match="torchrun launch"):
        run(**_run_kwargs(verify_fn=verify_fails_once_then_passes))
    assert attempts["n"] == 2


@pytest.mark.unit
def test_restore_called_on_shaping_failure():
    restore_calls = []
    with pytest.raises(RunAbort):
        run(**_run_kwargs(
            verify_fn=_fake_verify_always_fails,
            restore_fn=lambda handle: restore_calls.append(handle),
        ))
    assert len(restore_calls) == 1


@pytest.mark.unit
def test_restore_called_even_on_the_not_implemented_boundary():
    """§25.3: restore is unconditional on every exit path — including the documented
    NotImplementedError this version of run() always currently ends in on the happy path.
    """
    restore_calls = []
    with pytest.raises(NotImplementedError):
        run(**_run_kwargs(restore_fn=lambda handle: restore_calls.append(handle)))
    assert len(restore_calls) == 1


@pytest.mark.unit
def test_unshaped_run_never_calls_shaping_functions_at_all():
    apply_calls = []
    verify_calls = []
    restore_calls = []
    with pytest.raises(NotImplementedError):
        run(**_run_kwargs(
            spec=_spec(bandwidth_requested_bps=None),
            network_profile_exists=False,  # not required when unshaped
            apply_fn=lambda *a: apply_calls.append(a),
            verify_fn=lambda *a: verify_calls.append(a),
            restore_fn=lambda *a: restore_calls.append(a),
        ))
    assert apply_calls == verify_calls == restore_calls == []
