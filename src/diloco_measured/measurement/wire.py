"""/proc/net/dev bytes-on-wire accounting + analytic prediction (FR-05).

See methods/wire_model.md for the full derivation (ring all-reduce byte counts) this module
implements. The measurement side (snapshot/account) and the prediction side (predict) are kept
separate on purpose so agreement between them is a real check, not a tautology.

STATUS: [PROPOSED] scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WireSnapshot:
    per_node_bytes: dict[str, int]
    taken_at_s: float


def snapshot(nodes: list) -> WireSnapshot:
    """Read /proc/net/dev on every node. Kernel-level counter, independent of framework accounting."""
    raise NotImplementedError("Phase 0/1 — see methods/wire_model.md §1")


def predict(spec: dict, model_params: int, dtype_bytes: int = 4) -> float:
    """Analytic prediction of bytes-on-wire PER RANK PER STEP for the given ExperimentSpec.

    Implements the [CONFIRMED] ring all-reduce derivation (methods/wire_model.md §2):

        bytes_per_rank_per_sync = 2 * N * (P - 1) / P
        bytes_per_rank_per_step = bytes_per_rank_per_sync / H

    where N = model_params * dtype_bytes is the size of the synchronized tensor (the full
    gradient for ddp/localsgd, the pseudo-gradient for diloco — same size, per methods/
    wire_model.md §3). This is pure arithmetic: no GPU, no network, safe to unit test
    directly (CLAUDE.md §30.2 "wire.predict — Ring all-reduce byte counts for known (N,P,H);
    DDP vs DiLoCo ratio equals H").

    FSDP2 is intentionally NOT handled here yet: its per-step communication volume depends on
    sharding configuration and is [UNKNOWN] pending empirical derivation (methods/wire_model.md
    §3, §6) — guessing it would violate CLAUDE.md §33.2.6 ("never invent a number").
    """
    algorithm = spec["algorithm"]
    if algorithm == "fsdp2":
        raise NotImplementedError(
            "FSDP2 per-step wire volume is [UNKNOWN] — see methods/wire_model.md §3/§6; "
            "must be derived empirically on Day 0/1, not guessed."
        )

    P = spec["world_size"]
    H = spec["H"]
    if P < 2:
        raise ValueError("ring all-reduce byte formula requires world_size >= 2")
    if H < 1:
        raise ValueError("H must be >= 1")

    n_bytes = model_params * dtype_bytes
    bytes_per_rank_per_sync = 2 * n_bytes * (P - 1) / P
    return bytes_per_rank_per_sync / H


def account(before: WireSnapshot, after: WireSnapshot, predicted: int) -> dict:
    """Difference before/after snapshots and compare against the analytic prediction.

    Returns a WireAccount-shaped dict: predicted_bytes, measured_bytes, overhead_ratio,
    bytes_per_training_token_{predicted,measured}, idle_baseline_bytes.
    """
    raise NotImplementedError("Phase 0/1 — see methods/wire_model.md §4")
