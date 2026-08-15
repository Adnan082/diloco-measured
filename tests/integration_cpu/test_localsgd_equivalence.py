"""LocalSGD multi-rank invariant (gloo, CPU) -- mirrors
tests/integration_cpu/test_diloco_equivalence.py's structure: after `outer_step()`, every
replica must hold bit-identical parameters, given different local data per rank (so agreement
is a real assertion about the all-reduce, not a coincidence of identical inputs).
"""

from __future__ import annotations

import socket

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.integration_cpu


def _free_tcp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker_bit_identical_params(rank: int, world_size: int, port: int, H: int) -> None:
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch import nn

    from diloco_measured.measurement.localsgd import LocalSGDTrainer

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(0)  # identical init across ranks
        model = nn.Linear(4, 2)
        trainer = LocalSGDTrainer(model, optimizer_cfg={"name": "adamw", "lr": 0.05}, H=H)

        torch.manual_seed(100 + rank)  # DIFFERENT local data per rank
        x = torch.randn(8, 4)
        y = torch.randn(8, 2)

        for _ in range(H):
            trainer.inner_step(lambda: F.mse_loss(model(x), y))
        assert trainer.ready_for_outer_step()
        trainer.outer_step()

        params = [p.detach().clone() for p in model.parameters()]

        gathered: list | None = [None] * world_size if rank == 0 else None
        dist.gather_object(params, gathered, dst=0)

        if rank == 0:
            assert gathered is not None
            first = gathered[0]
            for other_rank, other in enumerate(gathered[1:], start=1):
                for param_idx, (t1, t2) in enumerate(zip(first, other, strict=True)):
                    if not torch.equal(t1, t2):
                        raise AssertionError(
                            f"param[{param_idx}] differs between rank 0 and rank {other_rank} "
                            f"after outer_step()"
                        )
    finally:
        dist.destroy_process_group()


def test_replicas_hold_bit_identical_params_after_outer_step():
    world_size = 2
    port = _free_tcp_port()
    mp.spawn(
        _worker_bit_identical_params,
        args=(world_size, port, 4),
        nprocs=world_size,
        join=True,
    )
