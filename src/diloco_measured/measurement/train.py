"""Run lifecycle orchestration (CLAUDE.md §10.1): schema validation -> preconditions ->
shaping -> verification gate -> fingerprint -> [torchrun launch -> measurement window] ->
restore.

Implements FR-03 (instrumented run) and FR-06 (convergence run) up to, and not past, the
point where continuing would mean guessing. Steps 1, 2, 3-4, 5, and 11 below are real,
implemented against the actual modules that back them (`spec.py`, `netshape.py`,
`fingerprint.py`), and their SEQUENCING (retry-once on a failed shaping check, unconditional
restore on every exit path including an exception) is unit-tested using injectable fake
shaping functions — see tests/unit/test_train_orchestration.py. That is different from
mocking `netshape.py` itself (forbidden, CLAUDE.md §30.6): these tests check that `run()`
calls its dependencies in the right order and handles their results correctly, not that a
fake `iperf3` produces a trustworthy number.

Step 6 (torchrun launch across real ranks running torchtitan/torchft) is NOT implemented.
Building it now would mean guessing at API surfaces this project hasn't validated: `torchft`
is pinned to a research candidate, not a confirmed SHA (CLAUDE.md §40 Q2), and ADR-009
(torchtitan as substrate) is explicitly still `[PROPOSED]`, "must be validated on Day 0."
Writing training-loop wiring against either before that validation would be exactly the kind
of invented-and-presented-as-real code CLAUDE.md §33.2.6 forbids. `run()` raises
`NotImplementedError` at precisely that point — after every precondition that CAN be checked
for real has been checked for real, so a caller gets every other correctness guarantee (spec
validity, shaping verified, environment fingerprinted, network restored) before hitting the
one part that's still genuinely unbuilt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from diloco_measured.measurement import fingerprint as fingerprint_module
from diloco_measured.measurement import netshape
from diloco_measured.measurement.spec import validate_experiment_spec

# Injectable so tests can exercise run()'s SEQUENCING without touching a real node — see the
# module docstring for why this is not the same thing as mocking netshape.py's own tests.
ApplyFn = Callable[[int | None, list], netshape.ShapingHandle]
VerifyFn = Callable[[netshape.ShapingHandle, float], netshape.ShapingVerification]
RestoreFn = Callable[[netshape.ShapingHandle], None]

DEFAULT_SHAPING_TOLERANCE_PCT = 10.0  # [PROPOSED] — CLAUDE.md FR-02 step 3


class RunAbort(Exception):
    """Raised when a precondition or the shaping verification gate fails.

    Per CLAUDE.md §10.1/§25.3: the run lifecycle aborts with NO analysis-eligible RunResult
    written — this exception IS that abort. `error_class` matches CLAUDE.md §25.1's error
    taxonomy (`spec_invalid`, `precondition_failed`, `shaping_verification_failed`, ...) so a
    caller can write the correctly-classified failure record without re-deriving it.
    """

    def __init__(self, error_class: str, message: str):
        self.error_class = error_class
        super().__init__(message)


@dataclass(frozen=True)
class Precondition:
    name: str
    passed: bool
    detail: str = ""


def check_preconditions(
    spec: dict,
    network_profile_exists: bool,
    dataset_checksum_ok: bool,
    gpu_clocks_locked: bool,
    node_dirty: bool,
) -> list[Precondition]:
    """CLAUDE.md §10.1 step 2. Pure — takes already-gathered facts as parameters rather than
    gathering them itself (gathering a real GPU-clock-lock state or a real dataset checksum
    needs SSH/filesystem access; this function's job is just the PASS/FAIL logic over facts
    someone else gathered, which is what's actually testable offline).
    """
    checks = [
        Precondition(
            "dataset_checksums_ok", dataset_checksum_ok, "tokenized shard checksum mismatch"
        ),
        Precondition("gpu_clocks_locked", gpu_clocks_locked, "GPU clocks not locked (NFR-08)"),
        Precondition("qdisc_clean", not node_dirty, "a previous run left a dirty qdisc (§19.4)"),
    ]
    if spec.get("bandwidth_requested_bps") is not None:
        checks.append(
            Precondition(
                "network_profile_exists", network_profile_exists,
                "a NetworkProfile is required before any shaped run (FR-02 precondition)",
            )
        )
    return checks


def run(
    spec: dict,
    nodes: list,
    *,
    network_profile_exists: bool,
    dataset_checksum_ok: bool,
    gpu_clocks_locked: bool,
    node_dirty: bool,
    dataset_shard_checksum: str,
    shaping_tolerance_pct: float = DEFAULT_SHAPING_TOLERANCE_PCT,
    apply_fn: ApplyFn = netshape.apply,
    verify_fn: VerifyFn = netshape.verify,
    restore_fn: RestoreFn = netshape.restore,
) -> dict:
    """Execute the run lifecycle for one `ExperimentSpec` up through fingerprinting.

    Steps performed for real: (1) schema + cross-field validation, (2) preconditions,
    (3-4) shaping + the verification gate with exactly one retry (FR-02), (5) environment
    fingerprint. Step 11 (restore) runs unconditionally in a `finally`, on every exit path,
    including the `NotImplementedError` this function currently always ends in past step 5 —
    see the module docstring for why steps 6+ are not built yet.

    Raises `RunAbort` (never returns a value) for `precondition_failed` or
    `shaping_verification_failed` — matching CLAUDE.md §25.1's rule that these abort before
    an analysis-eligible `RunResult` can exist. Schema violations raise
    `SpecValidationError` directly (from `spec.py`), which is also never caught here — an
    invalid spec aborts "with no side effects" (§10.1 step 1), before shaping is even touched.
    """
    validate_experiment_spec(spec)

    checks = check_preconditions(
        spec, network_profile_exists, dataset_checksum_ok, gpu_clocks_locked, node_dirty
    )
    failed = [c.name for c in checks if not c.passed]
    if failed:
        raise RunAbort("precondition_failed", f"preconditions failed: {failed}")

    handle: netshape.ShapingHandle | None = None
    bandwidth_requested_bps = spec.get("bandwidth_requested_bps")

    try:
        if bandwidth_requested_bps is not None:
            handle = apply_fn(bandwidth_requested_bps, nodes)
            verification = verify_fn(handle, shaping_tolerance_pct)

            if not verification.passed:
                # Exactly one retry, per FR-02 — never more, never zero.
                handle = apply_fn(bandwidth_requested_bps, nodes)
                verification = verify_fn(handle, shaping_tolerance_pct)

                if not verification.passed:
                    raise RunAbort(
                        "shaping_verification_failed",
                        f"measured {verification.measured_bps:.0f}bps vs requested "
                        f"{bandwidth_requested_bps}bps, error {verification.error_pct:.1f}% "
                        f"> tolerance {shaping_tolerance_pct}% (after 1 retry)",
                    )

        fp = fingerprint_module.capture(
            seed=spec["seed"],
            dataset_shard_checksum=dataset_shard_checksum,
            gpu_clocks_locked=gpu_clocks_locked,
        )

        raise NotImplementedError(
            "torchrun launch + measurement window: blocked on §40 Q2 (torchft/torchtitan "
            "pin) being finalized and validated on real GPU hardware (ADR-009 is still "
            "[PROPOSED]) — see this module's docstring. Spec validation, preconditions, "
            f"shaping+verification, and fingerprinting all completed for real (fingerprint "
            f"harness_git_sha={fp['harness_git_sha']!r})."
        )
    finally:
        # §25.3: "Restore is unconditional. Network state is restored on every exit path."
        if handle is not None:
            restore_fn(handle)


def run_convergence(spec: dict, reference_loss: float, **kwargs) -> dict:
    """FR-06: train to a fixed token budget, computing TTTL against `reference_loss`.

    CONTRACT: if the target is never reached, `tttl_s` is null and `final_loss` is still
    recorded — this must never be silently converted to a large finite number (methods/
    statistics.md §5). Not implemented for the same reason `run()` stops where it does — see
    the module docstring.
    """
    raise NotImplementedError(
        "Blocked on the same thing run() is: a validated torchtitan/torchft pin and real GPU "
        "hardware (§40 Q2, ADR-009). See train.py's module docstring."
    )
