"""Schema-validated loading of results/. Refuses invalid records (FR-12).

STATUS: [PROPOSED] scaffold.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"

# Versions this build of the analysis layer accepts. Unknown versions are rejected loudly
# (CLAUDE.md §16.2) rather than silently skipped.
ACCEPTED_RUN_RESULT_SCHEMA_VERSIONS = ("v1",)


def load_run_results(results_dir: Path | str = "results/raw") -> list[dict]:
    """Load and schema-validate every RunResult JSON under `results_dir`.

    CONTRACT: a record that fails schema validation is not silently skipped — it is an error
    the caller must see (FR-12). This function does not filter by status; see analysis/filter.py
    for exclusion rules.
    """
    raise NotImplementedError("Phase 0")


def load_network_profiles(results_dir: Path | str = "results/network") -> list[dict]:
    """Load and schema-validate every NetworkProfile JSON under `results_dir`."""
    raise NotImplementedError("Phase 0")


def _validate(record: dict, schema_filename: str) -> None:
    schema_path = SCHEMA_DIR / schema_filename
    with open(schema_path) as f:
        schema = json.load(f)
    jsonschema.validate(instance=record, schema=schema)
