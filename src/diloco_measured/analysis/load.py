"""Schema-validated loading of results/. Refuses invalid records (FR-12).

Pure, GPU-free, network-free, credential-free (CLAUDE.md §11.2) — this module never opens a
socket and never imports anything from `diloco_measured.measurement`.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"

# Versions this build of the analysis layer accepts. Unknown versions are rejected loudly
# (CLAUDE.md §16.2) rather than silently skipped or coerced.
ACCEPTED_RUN_RESULT_SCHEMA_VERSIONS = ("v1",)
ACCEPTED_NETWORK_PROFILE_SCHEMA_VERSIONS = ("v1",)


class SchemaValidationError(ValueError):
    """Raised when a results/ record fails schema validation.

    FR-12: "analysis shall refuse to load records that do not validate" — this exception IS
    the refusal. Callers must not catch-and-skip it silently (CLAUDE.md §25.3).
    """


def _load_schema_registry() -> Registry:
    """Build a referencing.Registry over every schema file in schemas/, keyed by $id, so
    $ref: "experiment_spec.v1.json" inside run_result.v1.json resolves correctly.
    """
    resources = []
    for schema_path in SCHEMA_DIR.glob("*.json"):
        with open(schema_path) as f:
            contents = json.load(f)
        resources.append((contents["$id"], Resource.from_contents(contents)))
    return Registry().with_resources(resources)


def _validator_for(schema_filename: str, registry: Registry) -> jsonschema.protocols.Validator:
    with open(SCHEMA_DIR / schema_filename) as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    return validator_cls(schema, registry=registry)


def _load_and_validate(path: Path, validator: jsonschema.protocols.Validator) -> dict:
    with open(path) as f:
        record = json.load(f)
    errors = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
    if errors:
        messages = "; ".join(
            f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
        )
        raise SchemaValidationError(f"{path}: {messages}")
    return record


def load_run_results(results_dir: Path | str = "results/raw") -> list[dict]:
    """Load and schema-validate every `RunResult` JSON under `results_dir`.

    CONTRACT: a record that fails schema validation raises `SchemaValidationError` — it is
    never silently skipped (FR-12). This function does not filter by status or
    `harness_version`; see `analysis/filter.py` for exclusion rules — loading and filtering
    are kept separate so a caller can always see the full unfiltered corpus if they choose to.
    """
    results_dir = Path(results_dir)
    registry = _load_schema_registry()
    validator = _validator_for("run_result.v1.json", registry)
    return [
        _load_and_validate(path, validator)
        for path in sorted(results_dir.glob("*.json"))
    ]


def load_network_profiles(results_dir: Path | str = "results/network") -> list[dict]:
    """Load and schema-validate every `NetworkProfile` JSON directly under `results_dir`.

    Raw `iperf3`/NCCL probe output lives in per-profile subdirectories (§16.1) and is not
    parsed by this function — it is the audit trail, read by a human, not by the loader.
    """
    results_dir = Path(results_dir)
    registry = _load_schema_registry()
    validator = _validator_for("network_profile.v1.json", registry)
    return [
        _load_and_validate(path, validator)
        for path in sorted(results_dir.glob("*.json"))
    ]
