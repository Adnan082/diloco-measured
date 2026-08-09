"""Per-step timing decomposition: wall / compute / sync-blocked / optimizer / loader-stall.

Feeds both the `StepRecord` Parquet output (schemas/step_record.v1.json) and the CU
reconciliation invariant (methods/cu_model.md §5). Instrumentation overhead must itself be
measured once and reported (CLAUDE.md §27, R8) — `measure_instrumentation_overhead()` below
does that.

Two backends, chosen automatically by `StepTimer`:

  - **CUDA-event backend**, used when `torch.cuda.is_available()`: the spec's original design
    (CLAUDE.md §13.2 — "torch.cuda.Event: low overhead, GPU-timeline accurate"). GPU kernel
    launches are asynchronous, so a CPU-side timestamp at a phase boundary does not mean the
    GPU has actually reached that point in its work; `torch.cuda.Event`s are recorded on the
    GPU's own stream and diffed via `elapsed_time()`, which is accurate regardless of
    launch/queueing overhead. **[PROPOSED — UNVERIFIED ON REAL HARDWARE]**: this backend has
    not been exercised on an actual GPU (none is available in this dev environment); `make
    smoke` (Phase 1, CLAUDE.md §30.4) is the first real test of it — do not trust its numbers
    before that gate passes.
  - **`perf_counter` backend**, used otherwise (CPU nodes, unit tests, CI): plain wall-clock
    timestamps, which are exact here because there is no asynchronous execution to misrepresent.

Both backends implement the same phase-marker API, so calling code never branches on which
one is active.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

_PHASES = ("start", "loader_done", "compute_done", "sync_done", "optimizer_done", "end")


@dataclass(frozen=True)
class StepTiming:
    wall_time_ms: float
    compute_time_ms: float
    sync_blocked_ms: float
    optimizer_time_ms: float
    loader_stall_ms: float

    @property
    def reconciliation_residual_pct(self) -> float:
        """|wall - sum(components)| / wall * 100. See methods/cu_model.md §5.

        Components are NOT forced to sum to `wall_time_ms` — an unmarked phase boundary (a
        caller that forgets `mark_optimizer_done()`, say) shows up here as genuine
        unaccounted time rather than being silently absorbed into another bucket. That is the
        entire point of the reconciliation check: it catches instrumentation gaps.
        """
        if self.wall_time_ms <= 0:
            return 0.0
        component_sum = (
            self.compute_time_ms
            + self.sync_blocked_ms
            + self.optimizer_time_ms
            + self.loader_stall_ms
        )
        return abs(self.wall_time_ms - component_sum) / self.wall_time_ms * 100.0


class _Backend(Protocol):
    def mark(self, name: str) -> None: ...
    def result(self) -> StepTiming: ...


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


class _PerfCounterBackend:
    def __init__(self) -> None:
        self._timestamps: dict[str, float] = {}

    def mark(self, name: str) -> None:
        self._timestamps[name] = time.perf_counter()

    def result(self) -> StepTiming:
        if "start" not in self._timestamps or "end" not in self._timestamps:
            raise RuntimeError(
                "StepTimer.result() called without both start and end marks — use it as a "
                "context manager: `with StepTimer() as t: ...`"
            )
        # Any phase not explicitly marked collapses to the previous phase's timestamp, so it
        # contributes exactly zero duration rather than raising — a DDP step with no separate
        # loader phase, for instance, legitimately skips mark_loader_done().
        resolved: dict[str, float] = {}
        prev = self._timestamps["start"]
        for phase in _PHASES:
            prev = self._timestamps.get(phase, prev)
            resolved[phase] = prev

        return StepTiming(
            wall_time_ms=(resolved["end"] - resolved["start"]) * 1000,
            loader_stall_ms=(resolved["loader_done"] - resolved["start"]) * 1000,
            compute_time_ms=(resolved["compute_done"] - resolved["loader_done"]) * 1000,
            sync_blocked_ms=(resolved["sync_done"] - resolved["compute_done"]) * 1000,
            optimizer_time_ms=(resolved["optimizer_done"] - resolved["sync_done"]) * 1000,
        )


class _CudaEventBackend:
    """See module docstring — [PROPOSED, UNVERIFIED ON REAL HARDWARE]."""

    def __init__(self) -> None:
        import torch

        self._torch = torch
        self._events: dict[str, torch.cuda.Event] = {}

    def mark(self, name: str) -> None:
        event = self._torch.cuda.Event(enable_timing=True)
        event.record()
        self._events[name] = event

    def result(self) -> StepTiming:
        if "start" not in self._events or "end" not in self._events:
            raise RuntimeError(
                "StepTimer.result() called without both start and end marks — use it as a "
                "context manager: `with StepTimer() as t: ...`"
            )
        self._events["end"].synchronize()  # block until the GPU has actually reached "end"

        resolved = {}
        prev = self._events["start"]
        for phase in _PHASES:
            prev = self._events.get(phase, prev)
            resolved[phase] = prev

        return StepTiming(
            wall_time_ms=resolved["start"].elapsed_time(resolved["end"]),
            loader_stall_ms=resolved["start"].elapsed_time(resolved["loader_done"]),
            compute_time_ms=resolved["loader_done"].elapsed_time(resolved["compute_done"]),
            sync_blocked_ms=resolved["compute_done"].elapsed_time(resolved["sync_done"]),
            optimizer_time_ms=resolved["sync_done"].elapsed_time(resolved["optimizer_done"]),
        )


class StepTimer:
    """Wraps one training step with phase markers to decompose wall time.

    Usage::

        with StepTimer() as timer:
            ... dataloader fetch ...
            timer.mark_loader_done()
            ... forward + backward ...
            timer.mark_compute_done()
            ... outer sync / all-reduce wait, only on sync steps ...
            timer.mark_sync_done()
            ... optimizer.step() ...
            timer.mark_optimizer_done()
        timing = timer.result()

    Markers are optional and cumulative — see `_PerfCounterBackend`/`_CudaEventBackend` for
    exactly how an unmarked phase is handled (zero duration, not an error).
    """

    def __init__(self) -> None:
        self._backend: _Backend = (
            _CudaEventBackend() if _cuda_available() else _PerfCounterBackend()
        )

    def __enter__(self) -> StepTimer:
        self._backend.mark("start")
        return self

    def mark_loader_done(self) -> None:
        self._backend.mark("loader_done")

    def mark_compute_done(self) -> None:
        self._backend.mark("compute_done")

    def mark_sync_done(self) -> None:
        self._backend.mark("sync_done")

    def mark_optimizer_done(self) -> None:
        self._backend.mark("optimizer_done")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._backend.mark("end")

    def result(self) -> StepTiming:
        return self._backend.result()


def measure_instrumentation_overhead(
    step_fn: Callable[[StepTimer | None], None], n_steps: int = 50
) -> float:
    """Run `step_fn` `n_steps` times WITHOUT a timer and `n_steps` times WITH one, and return
    the relative overhead: `(mean_instrumented_s - mean_uninstrumented_s) / mean_uninstrumented_s`.

    `step_fn(timer)` performs one step's work, calling `timer.mark_*()` at phase boundaries
    when `timer is not None` and skipping marking entirely when `timer is None`. Target:
    `< 1%` (CLAUDE.md §27, R8) — this function measures the number, it does not assert
    against the target; that assertion belongs in a Phase 1 hardware check (`make smoke`),
    not here.
    """
    uninstrumented_times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        step_fn(None)
        uninstrumented_times.append(time.perf_counter() - t0)

    instrumented_times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        with StepTimer() as timer:
            step_fn(timer)
        instrumented_times.append(time.perf_counter() - t0)

    mean_uninstrumented = sum(uninstrumented_times) / n_steps
    mean_instrumented = sum(instrumented_times) / n_steps

    # Below ~1ms, plain Python call/loop overhead dominates and the relative-overhead ratio
    # becomes numerically meaningless (or divides by near-zero) — not a real measurement of
    # StepTimer's cost. Use a step_fn with representative work instead of a near no-op.
    min_measurable_s = 1e-3
    if mean_uninstrumented < min_measurable_s:
        raise ValueError(
            f"uninstrumented step time measured as {mean_uninstrumented * 1000:.4f}ms, "
            f"below the {min_measurable_s * 1000:.0f}ms floor this function trusts; "
            "step_fn is too fast/no-op to measure overhead reliably"
        )
    return (mean_instrumented - mean_uninstrumented) / mean_uninstrumented
