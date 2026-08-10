"""NCCL all-reduce achieved-bandwidth-vs-message-size characterization.

Implements FR-01 step 3 (`sweep_all_reduce_bandwidth`) and step 5 (`burst_decay_probe`).
Independently publishable as a standalone artifact (G8) regardless of the rest of the
project. See methods/network_protocol.md §1.

`sweep_all_reduce_bandwidth()` is backend-agnostic — it calls whatever `torch.distributed`
backend the caller already initialized (`gloo` on CPU, `nccl` on GPU), so its correctness is
testable on CPU right now (tests/integration_cpu/test_probe.py, 2-process gloo) even though
"NCCL" is in the module's name; only the GPU/NCCL numbers themselves need real hardware to be
meaningful. `burst_decay_probe()` is SSH+iperf3-based like netshape.py, not NCCL-based (it's
characterizing the raw ENA link, not a collective) — it inherits the same "cannot be
meaningfully tested without a live node" status as netshape.py's SSH-executing functions.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from diloco_measured.measurement.netshape import IPERF3_PORT, Node, ssh_run

WARMUP_ITERS = 5
TIMED_ITERS = 10
DEFAULT_DTYPE_BYTES = 4  # fp32


@dataclass(frozen=True)
class NcclBandwidthPoint:
    msg_bytes: int
    achieved_bps: float


def log_spaced_message_sizes(min_bytes: int, max_bytes: int, n_points: int) -> list[int]:
    """`n_points` message sizes, log-spaced from `min_bytes` to `max_bytes` inclusive,
    rounded to whole bytes and deduplicated. Pure — no distributed/GPU dependency — so this
    piece of FR-01 step 3 ("log-spaced from 1 MiB to 4 GiB") is unit-tested directly.
    """
    if min_bytes <= 0 or max_bytes <= 0:
        raise ValueError("min_bytes and max_bytes must be > 0")
    if min_bytes > max_bytes:
        raise ValueError(f"min_bytes ({min_bytes}) must be <= max_bytes ({max_bytes})")
    if n_points < 1:
        raise ValueError("n_points must be >= 1")
    if n_points == 1:
        return [min_bytes]

    import math

    log_min, log_max = math.log(min_bytes), math.log(max_bytes)
    step = (log_max - log_min) / (n_points - 1)
    sizes = [round(math.exp(log_min + i * step)) for i in range(n_points)]
    # Rounding can collapse adjacent points at the low end — dedupe while preserving order.
    seen: set[int] = set()
    deduped = []
    for s in sizes:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


def sweep_all_reduce_bandwidth(
    world_size: int,
    msg_sizes_bytes: list[int],
    warmup_iters: int = WARMUP_ITERS,
    timed_iters: int = TIMED_ITERS,
) -> list[NcclBandwidthPoint]:
    """Run an all-reduce probe across `world_size` ranks at each message size, on whichever
    `torch.distributed` backend and device the CALLING process already set up (this function
    does not call `init_process_group` itself — one call per rank, same pattern as
    `measurement/diloco.py`'s `DiLoCoTrainer`).

    Message sizes are log-spaced 1 MiB to 4 GiB per FR-01 step 3 (use
    `log_spaced_message_sizes()` to build `msg_sizes_bytes`) — must be run under whatever
    shaping level is currently applied (after `netshape.apply()` + `verify()`), so the curve
    reflects ACHIEVED bandwidth, not the nominal link rate (methods/cu_model.md §3's
    `cu_analytic_achieved` depends on this being real).

    `achieved_bps` follows the ring all-reduce convention in methods/wire_model.md §2:
    `2*N*(P-1)/P` bytes moved per rank per collective, divided by the measured per-call wall
    time — the same byte-accounting formula `wire.py::predict()` uses, so the two are directly
    comparable (predicted vs. measured, same units).
    """
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "sweep_all_reduce_bandwidth() requires an already-initialized torch.distributed "
            "process group (e.g. launched via torchrun) — it does not call "
            "init_process_group() itself."
        )
    actual_world_size = dist.get_world_size()
    if actual_world_size != world_size:
        raise ValueError(
            f"world_size={world_size} was requested but the initialized process group has "
            f"{actual_world_size} ranks"
        )
    if actual_world_size < 2:
        raise ValueError("all-reduce bandwidth is undefined for world_size < 2")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    results: list[NcclBandwidthPoint] = []
    for msg_bytes in msg_sizes_bytes:
        n_elements = max(1, msg_bytes // DEFAULT_DTYPE_BYTES)
        actual_bytes = n_elements * DEFAULT_DTYPE_BYTES
        tensor = torch.ones(n_elements, dtype=torch.float32, device=device)

        for _ in range(warmup_iters):
            dist.all_reduce(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(timed_iters):
            dist.all_reduce(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = (time.perf_counter() - start) / timed_iters

        bytes_per_rank_per_call = 2 * actual_bytes * (actual_world_size - 1) / actual_world_size
        achieved_bps = (bytes_per_rank_per_call * 8) / elapsed_s

        results.append(NcclBandwidthPoint(msg_bytes=actual_bytes, achieved_bps=achieved_bps))

    return results


def burst_decay_probe(
    server_node: Node,
    client_node: Node,
    duration_s: int = 600,
    interval_s: int = 30,
) -> list[tuple[float, float]]:
    """10-minute sustained transfer at the unshaped rate to detect ENA burst-credit decay
    (FR-01 step 5). Returns `(t_s, bps)` points sampled every `interval_s` seconds, from
    `iperf3`'s own periodic interval reports (`-i`) rather than a separate polling loop.

    FR-01 alt-flow 5a: if sustained throughput decays > 20% over the window, the CALLER
    records `burst_decay_detected: true` — computing that decision is the caller's job (it
    needs to become a field in a committed `NetworkProfile`, per schemas/network_profile.v1.json,
    not something this probe decides unilaterally); this function only returns the raw curve.

    STATUS: same as netshape.py's SSH-executing functions — real, but has never run against a
    live node (no cluster is up as of this writing) and cannot be meaningfully unit-tested
    (mocking iperf3 here would be exactly the kind of mock CLAUDE.md §30.6 forbids).
    """
    ssh_run(server_node, ["pkill", "-f", f"iperf3 -s -p {IPERF3_PORT}"], timeout_s=10)
    ssh_run(
        server_node,
        ["bash", "-c", f"nohup iperf3 -s -p {IPERF3_PORT} > /tmp/iperf3_server.log 2>&1 & disown"],
        timeout_s=10,
    )
    time.sleep(1.0)

    try:
        client_result = ssh_run(
            client_node,
            [
                "iperf3", "-c", server_node.private_ip, "-p", str(IPERF3_PORT),
                "-t", str(duration_s), "-i", str(interval_s), "-J",
            ],
            timeout_s=duration_s + 30,
        )
    finally:
        ssh_run(server_node, ["pkill", "-f", f"iperf3 -s -p {IPERF3_PORT}"], timeout_s=10)

    if client_result.returncode != 0:
        raise RuntimeError(
            f"iperf3 sustained-transfer probe ({client_node.host} -> {server_node.private_ip}) "
            f"failed: {client_result.stderr.strip() or client_result.stdout.strip()}"
        )

    try:
        parsed = json.loads(client_result.stdout)
        intervals = parsed["intervals"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(
            f"could not parse iperf3 JSON output: {e}\nraw output: {client_result.stdout[:2000]}"
        ) from e

    points: list[tuple[float, float]] = []
    for interval in intervals:
        summary = interval["sum"]
        t_s = float(summary["end"])
        bps = float(summary["bits_per_second"])
        points.append((t_s, bps))

    return points
