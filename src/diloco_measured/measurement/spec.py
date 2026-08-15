"""ExperimentSpec validation: schema plus the cross-field invariants plain JSON Schema can't
express.

This IS run lifecycle step 1 (CLAUDE.md §10.1: "Schema validation ─► invalid → ABORT, no side
effects") and Architecture Principle #5 ("fail loud, fail early, fail cheap ... a misconfigured
run should die in seconds, not after twenty minutes") — validation happens before any node,
GPU, or network resource is touched, which is exactly why it's pure and safe to unit test
without a cluster.

Imports `diloco_measured.schemas.registry`, not `diloco_measured.analysis` — schemas/ is a
shared, neutral dependency for both packages (see that module's docstring); this file must
never import anything under `diloco_measured.analysis` (CLAUDE.md §11.2 forbidden edges).
"""

from __future__ import annotations

from diloco_measured.schemas.registry import format_errors, load_registry, validator_for


class SpecValidationError(ValueError):
    """Raised when an ExperimentSpec fails schema validation OR a cross-field invariant.

    CONTRACT: raised, never returned as a bool/warning — a spec that fails validation must
    abort the run with no side effects (CLAUDE.md §10.1 step 1, §25.1 `spec_invalid`).
    """


def validate_experiment_spec(spec: dict) -> None:
    """Validate `spec` against `experiment_spec.v1.json` AND the documented cross-field
    invariants (CLAUDE.md §15.2 `ExperimentSpec` entity — status `[PROPOSED]`):

      1. `algorithm == "ddp"` requires `H == 1` (one-directional, not a biconditional — see
         CLAUDE.md ADR-036 for why the reverse direction was dropped: real committed data
         (ADR-034/035, `results/raw/cu_grid-diloco-*-h1-*`) already uses `algorithm="diloco"`
         with `H=1` as a deliberate degenerate-case measurement — "what does DiLoCo do if you
         force it to sync every inner step" — which is a legitimate, already-published
         configuration, not an error. The original biconditional was written speculatively in
         Phase 0, before any real run existed to check it against; §44's change-management
         process is exactly for updating a `[PROPOSED]` rule once real evidence contradicts
         it, which is what happened here).
      2. `compression` is only valid with `algorithm in {"localsgd", "diloco"}`.
      3. `budget_type == "tokens"` is required when `phase == "convergence"`.

    Raises `SpecValidationError` listing every violation found (schema AND cross-field),
    rather than stopping at the first one — a config author fixing specs by hand benefits
    from seeing all the problems in one pass.
    """
    messages: list[str] = []

    registry = load_registry()
    validator = validator_for("experiment_spec.v1.json", registry)
    schema_errors = list(validator.iter_errors(spec))
    if schema_errors:
        messages.append(format_errors(schema_errors))

    messages.extend(_check_cross_field_invariants(spec))

    if messages:
        raise SpecValidationError("; ".join(messages))


def _check_cross_field_invariants(spec: dict) -> list[str]:
    errors: list[str] = []

    algorithm = spec.get("algorithm")
    H = spec.get("H")
    if algorithm == "ddp" and H != 1:
        errors.append(f"algorithm 'ddp' requires H == 1 (got H={H!r})")
    # NOTE: the reverse (H == 1 implies algorithm == "ddp") is deliberately NOT enforced —
    # see this function's docstring / CLAUDE.md ADR-036. H=1 with diloco/localsgd is a valid
    # degenerate-case configuration (sync every inner step) and is already real, committed
    # data in this project.

    compression = spec.get("compression")
    if compression is not None and algorithm not in ("localsgd", "diloco"):
        errors.append(
            f"compression={compression!r} is only valid with algorithm in "
            f"{{'localsgd', 'diloco'}} (got algorithm={algorithm!r})"
        )

    if spec.get("phase") == "convergence" and spec.get("budget_type") != "tokens":
        errors.append(
            "phase == 'convergence' requires budget_type == 'tokens' "
            f"(got budget_type={spec.get('budget_type')!r})"
        )

    return errors
