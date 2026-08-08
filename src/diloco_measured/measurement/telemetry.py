"""Per-step CUDA-event decomposition: wall/compute/sync-blocked/optimizer/loader-stall time.

Feeds both the StepRecord Parquet output (schemas/step_record.v1.json) and the CU reconciliation
invariant in methods/cu_model.md §5. Overhead of this instrumentation must itself be measured
once and reported (CLAUDE.md §27, R8) — torch.cuda.Event sync points can be surprisingly
expensive.

STATUS: [PROPOSED] scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepTiming:
    wall_time_ms: float
    compute_time_ms: float
    sync_blocked_ms: float
    optimizer_time_ms: float
    loader_stall_ms: float


class StepTimer:
    """Wraps one training step with CUDA-event markers to decompose wall time.

    CONTRACT: the four component times must sum to wall_time_ms within a recorded residual
    (methods/cu_model.md §5) — a residual above the documented threshold invalidates the
    observation rather than being silently absorbed.
    """

    def __enter__(self):
        raise NotImplementedError("Phase 0/1")

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError("Phase 0/1")

    def result(self) -> StepTiming:
        raise NotImplementedError("Phase 0/1")


def measure_instrumentation_overhead(step_fn, n_steps: int = 50) -> float:
    """Run `step_fn` instrumented vs. uninstrumented and return the overhead as a fraction of
    step time. Target: < 1% (CLAUDE.md §27, R8) — must be measured, not assumed.
    """
    raise NotImplementedError("Phase 1 — run once, report the number")
