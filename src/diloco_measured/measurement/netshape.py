"""Egress bandwidth shaping with a hard verification gate.

Implements FR-02. This is the project's central integrity mechanism (CLAUDE.md §33.1.5):
`verify()` must never return a passing result it did not measure, and no caller may proceed
past a failing result. See methods/network_protocol.md for the full protocol.

`apply`/`verify`/`restore` execute real commands on real nodes over SSH — they have never
run against a live node (no cluster is up as of this writing) and CANNOT be meaningfully
unit-tested (CLAUDE.md §30.6: "No mocks in the measurement path... a mocked iperf3 would
defeat the purpose of the gate"). The pure arithmetic/argv-building functions below them
(`compute_error_pct`, `passes_tolerance`, `build_tbf_add_args`, `build_tbf_del_args`) ARE
tested — see tests/unit/test_netshape_pure.py — because they don't need a real node to check.
Treat the SSH-executing functions as a first draft to debug live against `make smoke`
(Phase 1, CLAUDE.md §30.4), the same posture as `setup_node.sh`.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass

SSH_CONNECT_TIMEOUT_S = 10
IPERF3_PORT = 5201
IPERF3_SERVER_STARTUP_GRACE_S = 1.0


@dataclass(frozen=True)
class Node:
    """A cluster node addressable for SSH + tc control.

    `host` is what the OPERATOR uses to SSH in (the public IP in the current setup —
    CLAUDE.md §5.2, there is no bastion/VPN). `private_ip` is what OTHER cluster nodes use to
    reach this one for the traffic actually being measured/shaped (NCCL, iperf3) — using the
    VPC-internal address, not the public one, matches how real training traffic flows and is
    what `verify()` measures between.

    `iface` defaults to `ens5` per CLAUDE.md §11.1's architecture diagram — `[PROPOSED]`,
    confirm the real primary interface name (`ip link show`) against the actual pinned AMI on
    Day 1; it has not been checked against a live node.
    """

    host: str
    private_ip: str
    iface: str = "ens5"
    ssh_user: str = "ubuntu"  # matches the Ubuntu AMI launch_cluster.sh pins, ADR-027
    ssh_key_file: str = "~/.ssh/diloco-measured-key.pem"  # matches launch_cluster.sh's KEY_FILE


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


def _ssh_run(
    node: Node,
    remote_argv: list[str],
    use_sudo: bool = False,
    timeout_s: float = 30.0,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run `remote_argv` on `node` over SSH, returning the completed process.

    CONTRACT: `remote_argv` is a real argv list — this is the ONE place that turns it into
    the single string SSH's remote shell expects (`shlex.join`), so callers never build a
    shell string by hand (CLAUDE.md §23) — every value reaching this function comes from the
    fixed argv builders above or from admin-controlled `Node` config, never end-user input.
    `check=True` raises `subprocess.CalledProcessError` on a nonzero remote exit; `check=False`
    (default) lets the caller inspect `.returncode` itself — used for idempotent
    cleanup-if-present calls where "there was nothing to clean up" is an expected outcome,
    not a failure.
    """
    key_file = os.path.expanduser(node.ssh_key_file)
    remote_cmd = shlex.join((["sudo"] if use_sudo else []) + remote_argv)
    ssh_argv = [
        "ssh",
        "-i", key_file,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
        "-o", "BatchMode=yes",  # never hang on an interactive prompt (e.g. host key) — fail instead
        f"{node.ssh_user}@{node.host}",
        remote_cmd,
    ]
    return subprocess.run(ssh_argv, capture_output=True, text=True, timeout=timeout_s, check=check)


def apply(
    rate_bps: int | None,
    nodes: list[Node],
    burst_bytes: int = 32_768,
    latency_ms: int = 50,
) -> ShapingHandle:
    """Apply `tc qdisc ... tbf rate <R> burst <B> latency <L>` on every node's egress.

    `rate_bps=None` means unshaped: any existing qdisc is removed and none is applied.
    CONTRACT: this function only applies shaping — it does not verify it. Callers MUST call
    `verify()` before treating the network as at the requested rate (FR-02).

    Every node gets a "remove any existing qdisc first" pass regardless of `rate_bps`, so a
    qdisc left over from a previous run's failed `restore()` (a node marked dirty, CLAUDE.md
    §19.4) doesn't make this `apply()` fail with "File exists" — failures from that cleanup
    pass are deliberately ignored (`check=False`), since "there was nothing to remove" is the
    common case, not an error.
    """
    for node in nodes:
        _ssh_run(node, build_tbf_del_args(node.iface), use_sudo=True, timeout_s=15)

        if rate_bps is not None:
            result = _ssh_run(
                node,
                build_tbf_add_args(node.iface, rate_bps, burst_bytes, latency_ms),
                use_sudo=True,
                timeout_s=15,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"tc qdisc add failed on {node.host} ({node.iface}): "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )

    return ShapingHandle(nodes=tuple(nodes), requested_bps=rate_bps)


def verify(
    handle: ShapingHandle, tolerance_pct: float, duration_s: int = 15
) -> ShapingVerification:
    """Run iperf3 (>=15s) between the first two nodes in `handle` and assert the measured rate
    is within tolerance.

    Deliberately checks only ONE pair, not every pair — that is FR-01's job (network
    *characterization*, all ordered pairs, both directions, run once per session); this is
    FR-02's quick per-run gate, run before every shaped experiment.

    CONTRACT: NEVER returns `passed=True` for a rate that was not actually measured this call
    — the `iperf3 -J` JSON is parsed directly into `measured_bps`, nothing here falls back to
    the requested rate on any kind of failure; a failure raises instead. Retries are the
    CALLER's responsibility (exactly one retry, per FR-02) — this function does not retry
    itself, so its result is always a single honest measurement (`attempts` is always 1 here).
    """
    if handle.requested_bps is None:
        raise ValueError(
            "verify() requires a shaped handle (requested_bps is None — unshaped runs skip "
            "verification entirely per FR-02, they don't call verify() at all)"
        )
    if len(handle.nodes) < 2:
        raise ValueError("verify() needs at least 2 nodes to measure bandwidth between")

    server_node, client_node = handle.nodes[0], handle.nodes[1]

    # Best-effort: kill any stale iperf3 server from a previous failed run on this port.
    _ssh_run(server_node, ["pkill", "-f", f"iperf3 -s -p {IPERF3_PORT}"], timeout_s=10)

    _ssh_run(
        server_node,
        ["bash", "-c", f"nohup iperf3 -s -p {IPERF3_PORT} > /tmp/iperf3_server.log 2>&1 & disown"],
        timeout_s=10,
    )
    time.sleep(IPERF3_SERVER_STARTUP_GRACE_S)

    try:
        client_result = _ssh_run(
            client_node,
            [
                "iperf3", "-c", server_node.private_ip, "-p", str(IPERF3_PORT),
                "-t", str(duration_s), "-J",
            ],
            timeout_s=duration_s + 15,
        )
    finally:
        # Unconditional: the server must not be left running regardless of client outcome.
        _ssh_run(server_node, ["pkill", "-f", f"iperf3 -s -p {IPERF3_PORT}"], timeout_s=10)

    if client_result.returncode != 0:
        raise RuntimeError(
            f"iperf3 client ({client_node.host} -> {server_node.private_ip}) failed: "
            f"{client_result.stderr.strip() or client_result.stdout.strip()}"
        )

    try:
        parsed = json.loads(client_result.stdout)
        measured_bps = float(parsed["end"]["sum_received"]["bits_per_second"])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"could not parse iperf3 JSON output: {e}\nraw output: {client_result.stdout[:2000]}"
        ) from e

    error_pct = compute_error_pct(handle.requested_bps, measured_bps)
    passed = error_pct <= tolerance_pct

    qdisc_dump = _ssh_run(
        server_node, ["tc", "qdisc", "show", "dev", server_node.iface], timeout_s=10
    ).stdout

    return ShapingVerification(
        requested_bps=handle.requested_bps,
        measured_bps=measured_bps,
        error_pct=error_pct,
        tolerance_pct=tolerance_pct,
        passed=passed,
        attempts=1,
        iperf_raw=client_result.stdout,
        qdisc_dump=qdisc_dump,
    )


def restore(handle: ShapingHandle) -> None:
    """Restore the original qdisc (remove the `tbf` qdisc) on every node in `handle`.

    CONTRACT: idempotent — calling this twice must be safe, including on a node that was
    never shaped in the first place. `check=False` throughout: "there was nothing to remove"
    is success here, not failure. Must be called on every exit path, including SIGINT
    (CLAUDE.md §25.3) — that signal-handling wiring belongs to the caller (`train.py`'s run
    lifecycle), not here; this function's only job is "make the removal attempt, on every
    node, and don't raise over one that had nothing to remove."

    Returns nothing — a caller that needs to know whether restore actually succeeded on every
    node (to decide whether to mark a node dirty, §19.4) should check each node itself via
    `tc qdisc show`; this function intentionally does not swallow that decision.
    """
    for node in handle.nodes:
        _ssh_run(node, build_tbf_del_args(node.iface), use_sudo=True, timeout_s=15)
