"""/proc/net/dev bytes-on-wire accounting + analytic prediction (FR-05).

See methods/wire_model.md for the full derivation this module implements. The measurement
side (snapshot/account) and the prediction side (predict) are kept separate on purpose so
agreement between them is a real check, not a tautology (FR-05 design note).

We snapshot TX (egress) byte counters only, per node, on the shaped interface. This avoids
double-counting a single transfer (sender's TX and receiver's RX both incrementing for the
same bytes) and matches what `tc`/`tbf` actually shapes (methods/network_protocol.md — egress
only).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WireSnapshot:
    """Per-node cumulative TX byte counters at one point in time."""

    per_node_bytes: dict[str, int]
    taken_at_s: float


def parse_proc_net_dev(text: str, iface: str) -> tuple[int, int]:
    """Parse `/proc/net/dev` content, returning (rx_bytes, tx_bytes) for `iface`.

    Pure text parsing — no filesystem or network access — so it is unit-testable without a
    real node (CLAUDE.md §30.2). The column layout after `<iface>:` is standardized by the
    kernel:

        rx: bytes packets errs drop fifo frame compressed multicast
        tx: bytes packets errs drop fifo colls carrier compressed

    i.e. rx_bytes is field 0, tx_bytes is field 8 of the 16 whitespace-separated fields
    following the colon.
    """
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        name = name.strip()
        if name != iface:
            continue
        fields = rest.split()
        if len(fields) < 9:
            raise ValueError(f"unexpected /proc/net/dev line for {iface!r}: {line!r}")
        return int(fields[0]), int(fields[8])
    raise KeyError(f"interface {iface!r} not found in /proc/net/dev output")


def snapshot(nodes: list) -> WireSnapshot:
    """Read `/proc/net/dev` on every node (via SSH) and return their TX byte counters.

    STATUS: [PROPOSED] scaffold — the parsing primitive (`parse_proc_net_dev`) is implemented
    and unit-tested; the SSH/remote-exec plumbing to actually reach the 4 GPU nodes is cluster
    infrastructure not buildable or testable offline, and is Phase 1 work (CLAUDE.md §35).
    """
    raise NotImplementedError(
        "Remote /proc/net/dev collection needs SSH plumbing to real nodes — Phase 1. "
        "The parser it will call, parse_proc_net_dev(), is implemented and tested."
    )


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


def account(
    before: WireSnapshot,
    after: WireSnapshot,
    predicted_bytes_per_rank: float,
    tokens_processed: int,
    idle_baseline_bytes: int = 0,
) -> dict:
    """Difference before/after snapshots, compare against the analytic prediction, and
    normalize by tokens processed during the window.

    Returns a WireAccount-shaped dict (schemas/run_result.v1.json#/$defs/WireAccount):
    predicted_bytes, measured_bytes, overhead_ratio, bytes_per_training_token_{predicted,
    measured}, idle_baseline_bytes.

    `predicted_bytes_per_rank` is the PER-RANK prediction from `predict()`; `measured_bytes`
    here is summed across all nodes present in the snapshots, so `predicted_bytes` in the
    returned dict is scaled by the number of nodes for a fair comparison.

    CONTRACT: raises rather than silently producing a nonsensical ratio if
    `predicted_bytes_per_rank` or `tokens_processed` is non-positive (CLAUDE.md §33.2.6).
    """
    if predicted_bytes_per_rank <= 0:
        raise ValueError(f"predicted_bytes_per_rank must be > 0, got {predicted_bytes_per_rank!r}")
    if tokens_processed <= 0:
        raise ValueError(f"tokens_processed must be > 0, got {tokens_processed!r}")

    nodes = after.per_node_bytes.keys()
    raw_measured = sum(
        after.per_node_bytes[node] - before.per_node_bytes.get(node, 0) for node in nodes
    )
    measured_bytes = raw_measured - idle_baseline_bytes

    predicted_bytes = predicted_bytes_per_rank * len(nodes)

    return {
        "predicted_bytes": predicted_bytes,
        "measured_bytes": measured_bytes,
        "overhead_ratio": measured_bytes / predicted_bytes,
        "bytes_per_training_token_predicted": predicted_bytes / tokens_processed,
        "bytes_per_training_token_measured": measured_bytes / tokens_processed,
        "idle_baseline_bytes": idle_baseline_bytes,
    }
