"""CPU integration test for measurement/probe.py::sweep_all_reduce_bandwidth.

Uses gloo, not NCCL — the function is backend-agnostic by design (it never calls
init_process_group itself, just uses whatever the caller already set up), so this validates
the wiring and the byte-accounting formula for real, even though the *numbers* produced on
CPU/gloo/loopback are meaningless as actual bandwidth figures. Real NCCL/GPU numbers are
Phase 1 work (no GPU exists in this dev environment) — see the module's own docstring.
"""

from __future__ import annotations

import socket

import pytest
import torch.multiprocessing as mp

pytestmark = pytest.mark.integration_cpu


def _free_tcp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker_sweep(rank: int, world_size: int, port: int) -> None:
    import torch.distributed as dist

    from diloco_measured.measurement.probe import sweep_all_reduce_bandwidth

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        results = sweep_all_reduce_bandwidth(
            world_size=world_size,
            msg_sizes_bytes=[4096, 65536],
            warmup_iters=1,
            timed_iters=2,
        )
        assert len(results) == 2
        for point in results:
            if point.msg_bytes <= 0 or point.achieved_bps <= 0:
                raise AssertionError(f"non-positive result: {point}")
        # 4096 bytes / 4 bytes-per-fp32 = 1024 elements exactly -> no rounding needed.
        if results[0].msg_bytes != 4096:
            raise AssertionError(f"expected msg_bytes=4096, got {results[0].msg_bytes}")
    finally:
        dist.destroy_process_group()


def test_sweep_runs_across_two_gloo_ranks_and_returns_positive_bandwidth():
    world_size = 2
    port = _free_tcp_port()
    mp.spawn(_worker_sweep, args=(world_size, port), nprocs=world_size, join=True)


def test_raises_without_an_initialized_process_group():
    import torch.distributed as dist

    from diloco_measured.measurement.probe import sweep_all_reduce_bandwidth

    assert not dist.is_initialized()
    with pytest.raises(RuntimeError, match="already-initialized"):
        sweep_all_reduce_bandwidth(world_size=2, msg_sizes_bytes=[1024])
