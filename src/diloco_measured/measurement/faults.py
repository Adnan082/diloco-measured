"""Scheduled fault injection and recovery timing (FR-09, secondary goal G7).

Kills a designated worker at a scheduled time and measures lighthouse reconfiguration:
detect->resume latency, inner steps lost since the last sync, post-recovery loss trajectory.

For DDP, a hang instead of a recovery is the EXPECTED outcome and is the point of the
comparison (FR-09 failure condition) — record `recovery: hung` with a timeout, not an error.

STATUS: [PROPOSED — secondary goal].
"""

from __future__ import annotations


def schedule_kill(rank: int, t_s: float) -> None:
    """Schedule a SIGKILL of `rank` at wall-clock offset `t_s` into the run."""
    raise NotImplementedError("Phase 5")


def observe_recovery(rank: int, timeout_s: float) -> dict:
    """Observe lighthouse reconfiguration after a scheduled kill.

    Returns a FaultEvent-shaped dict: injected_at_s, rank, detected_at_s, resumed_at_s,
    steps_lost, outcome ("recovered" | "hung" | "job_died").
    """
    raise NotImplementedError("Phase 5")
