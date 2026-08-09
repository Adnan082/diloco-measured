"""Schema-validated loading of results/. Refuses invalid records (FR-12).

Pure, GPU-free, network-free, credential-free (CLAUDE.md §11.2) — this module never opens a
socket and never imports anything from `diloco_measured.measurement`.
"""

from __future__ import annotations

import json
from pathlib import Path

from diloco_measured.schemas.registry import format_errors, load_registry, validator_for

# Versions this build of the analysis layer accepts. Unknown versions are rejected loudly
# (CLAUDE.md §16.2) rather than silently skipped or coerced.
ACCEPTED_RUN_RESULT_SCHEMA_VERSIONS = ("v1",)
ACCEPTED_NETWORK_PROFILE_SCHEMA_VERSIONS = ("v1",)


class SchemaValidationError(ValueError):
    """Raised when a results/ record fails schema validation.

    FR-12: "analysis shall refuse to load records that do not validate" — this exception IS
    the refusal. Callers must not catch-and-skip it silently (CLAUDE.md §25.3).
    """


def _load_and_validate(path: Path, validator) -> dict:
    with open(path) as f:
        record = json.load(f)
    errors = list(validator.iter_errors(record))
    if errors:
        raise SchemaValidationError(f"{path}: {format_errors(errors)}")
    return record


def load_run_results(results_dir: Path | str = "results/raw") -> list[dict]:
    """Load and schema-validate every `RunResult` JSON under `results_dir`.

    CONTRACT: a record that fails schema validation raises `SchemaValidationError` — it is
    never silently skipped (FR-12). This function does not filter by status or
    `harness_version`; see `analysis/filter.py` for exclusion rules — loading and filtering
    are kept separate so a caller can always see the full unfiltered corpus if they choose to.
    """
    results_dir = Path(results_dir)
    registry = load_registry()
    validator = validator_for("run_result.v1.json", registry)
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
    registry = load_registry()
    validator = validator_for("network_profile.v1.json", registry)
    return [
        _load_and_validate(path, validator)
        for path in sorted(results_dir.glob("*.json"))
    ]
