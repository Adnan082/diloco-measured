"""Reference LocalSGD implementation — parameter averaging every H steps, no outer optimizer
(CLAUDE.md §43 Glossary: "Semi-sync training by parameter averaging every H steps; no outer
optimizer. Serves here as the no-outer-optimizer ablation against DiLoCo").

Written as an in-repo reference for the same reason `diloco.py` is (ADR-003/D3): `torchft`'s
semi-sync paths are marked experimental (R2), and `configs/algorithms/localsgd.yaml` originally
assumed torchft was the only available implementation ("no in-repo reference exists for it") —
that gap is closed here, consistent with how this project has actually run every real DiLoCo
measurement so far (the reference path, never torchft, per every real ADR from 034 onward).

STATUS: [CONFIRMED] algorithm form (LocalSGD's parameter-averaging definition is standard,
long predates this project — see PRIOR_ART.md). [PROPOSED] this specific implementation,
mirroring `DiLoCoTrainer`'s structure deliberately so the two are comparable apples-to-apples
(same inner loop code shape, same StepTimer usage pattern) with the outer step being the ONLY
structural difference between the two algorithms.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class LocalSGDStepResult:
    inner_loss: float
    did_outer_step: bool


class LocalSGDTrainer:
    """Reference LocalSGD training loop:

        for round r = 0, 1, 2, ...:
            for h in 1..H:                    # purely local, zero communication
                theta <- AdamW_step(theta, local_batch)
            theta <- all_reduce_mean(theta)    # OUTER: average raw parameters directly,
                                                # no pseudo-gradient, no outer optimizer

    The only structural difference from `DiLoCoTrainer` (methods/diloco.md §1): there is no
    separate theta_outer, no pseudo-gradient, and no outer optimizer state at all — the outer
    step directly all-reduces (mean) the model's live parameters and every replica adopts the
    average in place. `inner_step()`'s AdamW state is untouched by this (same invariant 1 as
    DiLoCo, methods/diloco.md §3), since only the PARAMETER values are touched, not the
    optimizer's internal exp_avg/exp_avg_sq buffers.

    Distributed behaviour matches `DiLoCoTrainer`: if `torch.distributed` is initialized, the
    outer step all-reduces; if not, it's skipped and this behaves as a single replica (usable
    in a plain single-process unit test).
    """

    def __init__(self, model: nn.Module, optimizer_cfg: dict, H: int) -> None:
        if H < 1:
            raise ValueError("H must be >= 1 (CLAUDE.md ExperimentSpec invariant)")

        self.model = model
        self.H = H
        self._h_count = 0

        opt_cfg = dict(optimizer_cfg)
        opt_name = opt_cfg.pop("name", "adamw")
        if opt_name != "adamw":
            raise NotImplementedError(
                f"Only the adamw optimizer is implemented for LocalSGD; got {opt_name!r}"
            )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **opt_cfg)

    @property
    def h_count(self) -> int:
        return self._h_count

    def ready_for_outer_step(self) -> bool:
        return self._h_count >= self.H

    def inner_step(self, loss_fn: Callable[[], torch.Tensor]) -> float:
        """One local AdamW step. Zero cross-replica communication."""
        self.optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        self.optimizer.step()
        self._h_count += 1
        return float(loss.detach())

    def outer_step(self) -> None:
        """All-reduce (mean) the live model parameters directly -- no pseudo-gradient, no
        outer optimizer. This is the ONLY cross-replica traffic in the loop, same communication
        volume per round as DiLoCo (one N-sized tensor all-reduced), differing only in what the
        all-reduced quantity IS (raw parameters here, a pseudo-gradient in DiLoCo) and what
        happens to it afterward (direct replacement here, an optimizer step in DiLoCo).
        """
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            with torch.no_grad():
                for p in self.model.parameters():
                    dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
                    p.data.div_(world_size)
        self._h_count = 0

    def step(self, loss_fn: Callable[[], torch.Tensor]) -> LocalSGDStepResult:
        """Convenience: one inner step, followed by an outer step once H has been reached."""
        loss = self.inner_step(loss_fn)
        did_outer = self.ready_for_outer_step()
        if did_outer:
            self.outer_step()
        return LocalSGDStepResult(inner_loss=loss, did_outer_step=did_outer)

    def inner_optimizer_state_fingerprint(self) -> list[int]:
        """Number of AdamW state tensors tracked per parameter — used by tests to confirm
        optimizer state is NOT reset by outer_step() (same invariant as DiLoCo's).
        """
        return [len(self.optimizer.state.get(p, {})) for p in self.model.parameters()]
