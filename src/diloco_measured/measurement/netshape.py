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


def compute_error_pct(requested_bps: int, measured_bps: float) -> float:
    """|measured - requested| / requested * 100. Pure arithmetic — the piece of the
    verification gate (FR-02 step 3) that doesn't need a real node to test.
    """
    if requested_bps <= 0:
        raise ValueError(f"requested_bps must be > 0, got {requested_bps!r}")
    return abs(measured_bps - requested_bps) / requested_bps * 100.0


def passes_tolerance(requested_bps: int, measured_bps: float, tolerance_pct: float) -> bool:
    """FR-02 step 3: |measured - requested| / requested <= tolerance."""
    return compute_error_pct(requested_bps, measured_bps) <= tolerance_pct


def build_tbf_add_args(
    iface: str, rate_bps: int, burst_bytes: int, latency_ms: int
) -> list[str]:
    """Build the argv (never a shell string) for applying a `tbf` qdisc.

    CLAUDE.md §23: shaping commands are "a fixed, parameterized allowlist — no shell
    interpolation of user input into `tc` invocations." Returning a list here, for the
    caller to pass straight to `subprocess.run(argv, shell=False)`, is what makes that literal
    — there is no string concatenation step where an injection could hide.
    """
    if rate_bps <= 0:
        raise ValueError(f"rate_bps must be > 0, got {rate_bps!r}")
    return [
        "tc", "qdisc", "add", "dev", iface, "root", "tbf",
        "rate", f"{rate_bps}bit",
        "burst", str(burst_bytes),
        "latency", f"{latency_ms}ms",
    ]


def build_tbf_del_args(iface: str) -> list[str]:
    """Build the argv for removing the root qdisc (restore to default), i.e. unshaped."""
    return ["tc", "qdisc", "del", "dev", iface, "root"]


def apply(rate_bps: int | None, nodes: list[Node]) -> ShapingHandle:
    """Apply `tc qdisc ... tbf rate <R> burst <B> latency <L>` on every node's egress.

    rate_bps=None means unshaped (no qdisc applied). CONTRACT: this function only applies
    shaping — it does not verify it. Callers MUST call verify() before treating the network
    as at the requested rate.

    STATUS: [PROPOSED] scaffold — the argv-building primitives above are implemented and
    tested; the actual SSH + subprocess execution against real nodes is Phase 1 work.
    """
    raise NotImplementedError("Phase 0/1 — see methods/network_protocol.md §2")


def verify(handle: ShapingHandle, tolerance_pct: float) -> ShapingVerification:
    """Run iperf3 (>=15s) between two nodes and assert the measured rate is within tolerance.

    CONTRACT: NEVER returns passed=True for a rate that was not actually measured this call.
    Retries are the CALLER's responsibility (exactly one retry, per FR-02) — this function does
    not retry itself, so its result is always a single honest measurement.

    STATUS: [PROPOSED] scaffold — the tolerance arithmetic (`compute_error_pct`,
    `passes_tolerance`) is implemented and tested; running real iperf3 against real nodes is
    Phase 1 work. No mocks belong in that half (CLAUDE.md §30.6).
    """
    raise NotImplementedError("Phase 0/1 — see methods/network_protocol.md §2")


def restore(handle: ShapingHandle) -> None:
    """Restore the original qdisc on every node.

    CONTRACT: idempotent — calling this twice must be safe. Must be called on every exit path,
    including SIGINT (CLAUDE.md §25.3). On failure, the caller must mark the node dirty.
    """
    raise NotImplementedError("Phase 0/1 — see methods/network_protocol.md §2")
