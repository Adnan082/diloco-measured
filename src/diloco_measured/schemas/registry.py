"""Shared `referencing.Registry` construction over schemas/*.json.

Lives in `schemas/`, not in `measurement/` or `analysis/`, deliberately: both packages need
to validate against these contracts (measurement validates an incoming `ExperimentSpec`
before touching a GPU; analysis validates `RunResult`/`NetworkProfile` records it loads), and
the forbidden edges in CLAUDE.md §11.2/§14.2 are `analysis -> measurement` and
`measurement -> analysis` — not `-> schemas`. Two concrete callers exist as of this module's
introduction (`analysis/load.py`, `measurement/spec.py`), so this is not a
single-implementation abstraction (CLAUDE.md §33.2.9).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
from referencing import Registry, Resource

SCHEMA_DIR = Path(__file__).resolve().parent


def load_registry() -> Registry:
    """Build a `referencing.Registry` over every schema file in schemas/, keyed by `$id`, so
    e.g. `$ref: "experiment_spec.v1.json"` inside `run_result.v1.json` resolves correctly.
    """
    resources = []
    for schema_path in SCHEMA_DIR.glob("*.json"):
        with open(schema_path) as f:
            contents = json.load(f)
        resources.append((contents["$id"], Resource.from_contents(contents)))
    return Registry().with_resources(resources)


def validator_for(schema_filename: str, registry: Registry) -> jsonschema.protocols.Validator:
    """A validator instance for `schema_filename`, wired to `registry` so its `$ref`s resolve."""
    with open(SCHEMA_DIR / schema_filename) as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    return validator_cls(schema, registry=registry)


def format_errors(errors: list[jsonschema.exceptions.ValidationError]) -> str:
    """Render a sorted list of jsonschema errors as one semicolon-joined string, each prefixed
    by its JSON path (or `<root>`). Shared so `analysis/load.py` and `measurement/spec.py`
    report violations in the same format.
    """
    ordered = sorted(errors, key=lambda e: list(e.path))
    return "; ".join(
        f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in ordered
    )
