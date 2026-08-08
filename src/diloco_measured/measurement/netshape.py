"""Egress bandwidth shaping with a hard verification gate.

Implements FR-02. This is the project's central integrity mechanism (CLAUDE.md §33.1.5):
`verify()` must never return a passing result it did not measure, and no caller may proceed
past a failing result. See methods/network_protocol.md for the full protocol.

STATUS: [PROPOSED] scaffold — not yet implemented. No mocks belong in this module's tests
(CLAUDE.md §30.6): a mocked iperf3 would defeat the purpose of the gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    """A cluster node addressable for SSH + tc control. STATUS: [PROPOSED] field set."""

    host: str
    iface: str = "ens5"


@dataclass(frozen=True)
class ShapingHandle:
    """Opaque handle to an applied shaping configuration, needed to verify/restore it."""

    nodes: tuple[Node, ...]
    requested_bps: int | None


@dataclass(frozen=True)
class ShapingVerification:
    """See schemas/run_result.v1.json#/$defs/ShapingVerification."""

    requested_bps: int
    measured_bps: float
    error_pct: float
    tolerance_pct: float
    passed: bool
    attempts: int
    iperf_raw: str
    qdisc_dump: str


def apply(rate_bps: int | None, nodes: list[Node]) -> ShapingHandle:
    """Apply `tc qdisc ... tbf rate <R> burst <B> latency <L>` on every node's egress.

    rate_bps=None means unshaped (no qdisc applied). CONTRACT: this function only applies
    shaping — it does not verify it. Callers MUST call verify() before treating the network
    as at the requested rate.
    """
    raise NotImplementedError("Phase 0/1 — see methods/network_protocol.md §2")


def verify(handle: ShapingHandle, tolerance_pct: float) -> ShapingVerification:
    """Run iperf3 (>=15s) between two nodes and assert |measured - requested| / requested <= tolerance.

    CONTRACT: NEVER returns passed=True for a rate that was not actually measured this call.
    Retries are the CALLER's responsibility (exactly one retry, per FR-02) — this function does
    not retry itself, so its result is always a single honest measurement.
    """
    raise NotImplementedError("Phase 0/1 — see methods/network_protocol.md §2")


def restore(handle: ShapingHandle) -> None:
    """Restore the original qdisc on every node.

    CONTRACT: idempotent — calling this twice must be safe. Must be called on every exit path,
    including SIGINT (CLAUDE.md §25.3). On failure, the caller must mark the node dirty.
    """
    raise NotImplementedError("Phase 0/1 — see methods/network_protocol.md §2")
