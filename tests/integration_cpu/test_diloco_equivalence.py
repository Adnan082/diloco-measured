"""DiLoCo cross-implementation equivalence and multi-rank invariants (ADR-003, US-06).

Two things live in this file, deliberately separated:

1. Multi-rank invariants of OUR OWN reference implementation (gloo, CPU, 2 processes) — these
   ARE implemented and run below, because measurement/diloco.py has no torchft dependency.
2. Reference-vs-torchft equivalence — still SKIPPED. That needs a pinned torchft (CLAUDE.md
   §40 Q2, PENDING); running against an unpinned `main` would violate the supply-chain rule
   (§33.2.8, no unpinned installs) and would make a passing/failing result meaningless.
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


def _worker_bit_identical_theta_outer(rank: int, world_size: int, port: int, H: int) -> None:
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch import nn

    from diloco_measured.measurement.diloco import DiLoCoTrainer

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # Same seed on every rank => identical model init (methods/diloco.md §1: "theta_outer
        # <- initial weights (broadcast to all replicas)" — seeding identically is the CPU-test
        # stand-in for an actual broadcast).
        torch.manual_seed(0)
        model = nn.Linear(4, 2)
        trainer = DiLoCoTrainer(
            model,
            inner_optimizer_cfg={"name": "adamw", "lr": 0.05},
            outer_optimizer_cfg={"name": "nesterov_sgd", "lr": 0.5, "momentum": 0.9},
            H=H,
        )

        # DIFFERENT local data per rank — this is what makes bit-identical theta_outer after
        # the all-reduce a real assertion about the outer step, not a coincidence of identical
        # inputs everywhere.
        torch.manual_seed(100 + rank)
        x = torch.randn(8, 4)
        y = torch.randn(8, 2)

        for _ in range(H):
            trainer.inner_step(lambda: F.mse_loss(model(x), y))
        assert trainer.ready_for_outer_step()
        trainer.outer_step()

        theta = trainer.theta_outer()

        gathered: list | None = [None] * world_size if rank == 0 else None
        dist.gather_object(theta, gathered, dst=0)

        if rank == 0:
            assert gathered is not None
            first = gathered[0]
            for other_rank, other in enumerate(gathered[1:], start=1):
                for param_idx, (t1, t2) in enumerate(zip(first, other, strict=True)):
                    if not torch.equal(t1, t2):
                        raise AssertionError(
                            f"theta_outer param[{param_idx}] differs between rank 0 and "
                            f"rank {other_rank} after outer_step() "
                            f"(methods/diloco.md §3 invariant 2)"
                        )
    finally:
        dist.destroy_process_group()


def test_replicas_hold_bit_identical_theta_outer_after_outer_step():
    """methods/diloco.md §3 invariant 2, gloo, 2 CPU processes, H=4."""
    world_size = 2
    port = _free_tcp_port()
    mp.spawn(
        _worker_bit_identical_theta_outer,
        args=(world_size, port, 4),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.skip(
    reason="Blocked on CLAUDE.md §40 Q2 (torchft SHA pin, PENDING) — running against an "
    "unpinned torchft main would violate §33.2.8 and make pass/fail meaningless."
)
def test_reference_and_torchft_agree_within_tolerance():
    """US-06: given the same seed, model, data order, and H, the reference diloco.py and the
    torchft path must produce loss curves agreeing within a documented tolerance over 200
    steps. Tolerance value itself is [PROPOSED], to be set empirically on Day 0 (methods/
    diloco.md §5) once a torchft SHA is pinned.
    """
    raise NotImplementedError
