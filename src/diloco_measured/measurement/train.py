"""torchtitan-based training loop, fixed-token-budget mode.

Implements FR-03 (instrumented run) and FR-06 (convergence run). Executes the run lifecycle
in CLAUDE.md §10.1: schema validation -> preconditions -> shaping -> verification gate ->
fingerprint -> torchrun launch -> warmup discard -> measurement window -> aggregation -> write
-> restore.

STATUS: [PROPOSED] scaffold. ADR-009 (torchtitan as substrate) is still Proposed, not Accepted
— must be validated on Day 0 before this module is built against it.
"""

from __future__ import annotations


def run(spec_path: str, dry_run: bool = False) -> dict:
    """Execute one instrumented run per an ExperimentSpec. Returns the written RunResult dict.

    CONTRACT: never writes a partially valid record (§25.3). On any failure, writes a failure
    record with status != completed and restores network state unconditionally.
    """
    raise NotImplementedError("Phase 0/1 — see CLAUDE.md §10.1")


def run_convergence(spec_path: str, reference_loss: float) -> dict:
    """FR-06: train to a fixed token budget, computing TTTL against `reference_loss`.

    CONTRACT: if the target is never reached, tttl_s is null and final_loss is still recorded
    — this must never be silently converted to a large finite number (see methods/statistics.md §5).
    """
    raise NotImplementedError("Phase 4 — see CLAUDE.md FR-06")
