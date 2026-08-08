"""Schema sanity checks — pure, fast, every commit (CLAUDE.md §30.2 'Schema validation').

These are deliberately minimal: they confirm each committed schema file is itself a
well-formed JSON Schema, and that a representative ExperimentSpec instance validates against
its schema. Full cross-schema $ref resolution (run_result.v1.json -> experiment_spec.v1.json)
and negative/invalid-case coverage belongs to `analysis/load.py`'s implementation in Phase 0
— this file only guards against the schema files themselves silently rotting.
"""

from __future__ import annotations

import json

import jsonschema
import pytest

SCHEMA_FILES = [
    "experiment_spec.v1.json",
    "run_result.v1.json",
    "network_profile.v1.json",
    "step_record.v1.json",
]


@pytest.mark.unit
@pytest.mark.parametrize("schema_file", SCHEMA_FILES)
def test_schema_is_well_formed(schemas_dir, schema_file):
    with open(schemas_dir / schema_file) as f:
        schema = json.load(f)
    jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.unit
def test_example_experiment_spec_validates(fixtures_dir, schemas_dir):
    with open(schemas_dir / "experiment_spec.v1.json") as f:
        schema = json.load(f)
    with open(fixtures_dir / "experiment_spec_example.json") as f:
        instance = json.load(f)
    jsonschema.validate(instance=instance, schema=schema)


@pytest.mark.unit
def test_ddp_with_h_greater_than_one_is_a_documented_invariant_not_yet_enforced(schemas_dir):
    """Documents a known gap: plain JSON Schema can't express 'H == 1 iff algorithm == ddp'
    (CLAUDE.md §15.2 ExperimentSpec invariant). That cross-field rule must be enforced in
    `analysis/load.py` / the spec-validation code path, not assumed to fall out of the schema.
    This test exists so the gap is visible rather than silently relied upon.
    """
    with open(schemas_dir / "experiment_spec.v1.json") as f:
        schema = json.load(f)
    invalid_but_schema_valid = {
        "spec_id": "x", "phase": "cu_grid", "algorithm": "ddp", "implementation": "reference",
        "H": 32,  # invalid per the cross-field rule, but the bare schema doesn't know that
        "model_config": "m", "world_size": 4, "micro_batch_size": 1, "seq_len": 8,
        "grad_accum": 1, "budget_type": "steps", "budget_value": 1, "warmup_steps": 0,
        "seed": 0, "repeat_index": 0,
    }
    # This currently PASSES schema validation despite violating the domain invariant —
    # that gap is the point of this test.
    jsonschema.validate(instance=invalid_but_schema_valid, schema=schema)
