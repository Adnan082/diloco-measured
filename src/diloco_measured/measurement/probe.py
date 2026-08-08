"""NCCL all-reduce achieved-bandwidth-vs-message-size characterization.

Implements FR-01 step 3. Independently publishable as a standalone artifact (G8) regardless
of the rest of the project. See methods/network_protocol.md §1.

STATUS: [PROPOSED] scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NcclBandwidthPoint:
    msg_bytes: int
    achieved_bps: float


def sweep_all_reduce_bandwidth(
    world_size: int,
    msg_sizes_bytes: list[int],
) -> list[NcclBandwidthPoint]:
    """Run an NCCL all-reduce probe across `world_size` ranks at each message size.

    Message sizes are log-spaced 1 MiB to 4 GiB per FR-01 step 3. Must be run under whatever
    shaping level is currently applied, so the curve reflects achieved (not link) bandwidth.
    """
    raise NotImplementedError("Phase 0/1")


def burst_decay_probe(duration_s: int = 600) -> list[tuple[float, float]]:
    """10-minute sustained transfer at the unshaped rate to detect ENA burst-credit decay.

    Returns (t_s, bps) points. FR-01 step 5: if sustained throughput decays > 20% over the
    window, the caller records burst_decay_detected=True — this is a finding, not an error.
    """
    raise NotImplementedError("Phase 0/1")
