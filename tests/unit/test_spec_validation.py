"""Unit tests for measurement/spec.py — closes the gap flagged in test_schemas.py:
`test_bare_schema_alone_cannot_express_the_h_algorithm_invariant`.
"""

from __future__ import annotations

import json

import pytest

from diloco_measured.measurement.spec import SpecValidationError, validate_experiment_spec


def _base_spec(**overrides) -> dict:
    base = {
        "spec_id": "x", "phase": "cu_grid", "algorithm": "diloco", "implementation": "reference",
        "H": 32, "model_config": "m", "world_size": 4, "micro_batch_size": 1, "seq_len": 8,
        "grad_accum": 1, "budget_type": "steps", "budget_value": 1, "warmup_steps": 0,
        "compression": None, "seed": 0, "repeat_index": 0, "fault_schedule": None,
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_fixture_experiment_spec_passes_full_validation(fixtures_dir):
    with open(fixtures_dir / "experiment_spec_example.json") as f:
        spec = json.load(f)
    validate_experiment_spec(spec)  # must not raise


@pytest.mark.unit
def test_ddp_with_h_not_equal_one_is_rejected():
    spec = _base_spec(algorithm="ddp", H=32)
    with pytest.raises(SpecValidationError, match="H == 1"):
        validate_experiment_spec(spec)


@pytest.mark.unit
def test_ddp_with_h_equal_one_is_accepted():
    spec = _base_spec(algorithm="ddp", H=1)
    validate_experiment_spec(spec)  # must not raise


@pytest.mark.unit
def test_h_equal_one_reserved_for_ddp_rejects_other_algorithms():
    spec = _base_spec(algorithm="diloco", H=1)
    with pytest.raises(SpecValidationError, match="reserved for algorithm 'ddp'"):
        validate_experiment_spec(spec)


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ["localsgd", "diloco"])
def test_compression_is_accepted_for_localsgd_and_diloco(algorithm):
    spec = _base_spec(algorithm=algorithm, H=8, compression="fp16")
    validate_experiment_spec(spec)  # must not raise


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", ["ddp", "fsdp2"])
def test_compression_is_rejected_for_ddp_and_fsdp2(algorithm):
    H = 1 if algorithm == "ddp" else 8
    spec = _base_spec(algorithm=algorithm, H=H, compression="fp16")
    with pytest.raises(SpecValidationError, match="compression"):
        validate_experiment_spec(spec)


@pytest.mark.unit
def test_convergence_phase_requires_tokens_budget():
    spec = _base_spec(phase="convergence", budget_type="steps")
    with pytest.raises(SpecValidationError, match="budget_type == 'tokens'"):
        validate_experiment_spec(spec)


@pytest.mark.unit
def test_convergence_phase_with_tokens_budget_is_accepted():
    spec = _base_spec(phase="convergence", budget_type="tokens")
    validate_experiment_spec(spec)  # must not raise


@pytest.mark.unit
def test_schema_violation_is_also_caught():
    spec = _base_spec()
    del spec["world_size"]  # required field
    with pytest.raises(SpecValidationError):
        validate_experiment_spec(spec)


@pytest.mark.unit
def test_multiple_violations_are_all_reported_in_one_error():
    # H==1 with a non-ddp algorithm AND compression on an algorithm that doesn't allow it.
    spec = _base_spec(algorithm="fsdp2", H=1, compression="fp16")
    with pytest.raises(SpecValidationError) as exc_info:
        validate_experiment_spec(spec)
    message = str(exc_info.value)
    assert "reserved for algorithm 'ddp'" in message
    assert "compression" in message
