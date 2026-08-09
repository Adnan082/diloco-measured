"""Reference DiLoCo implementation — inner AdamW / outer Nesterov SGD.

This is OUR implementation (ADR-003), cross-validated against the `torchft` path because
torchft's semi-sync APIs are marked experimental (R2). See methods/diloco.md for the full
derivation and the invariants this module MUST satisfy.

STATUS: [CONFIRMED] algorithm form (methods/diloco.md §1 — this is DeepMind's published
algorithm, not ours; see PRIOR_ART.md). [PROPOSED] hyperparameters — callers must supply
inner/outer optimizer configs explicitly; this module does not choose lr/momentum defaults
(CLAUDE.md §33.2.6 — never invent a number).

This trainer is deliberately model-agnostic (plain `torch.optim` over any `nn.Module`) and
has no dependency on torchtitan or torchft — that independence is the entire point (D3).

INVARIANTS (methods/diloco.md §3 — each has a corresponding test in tests/integration_cpu/):
  1. Inner optimizer state persists across outer rounds.
  2. All replicas hold bit-identical θ_outer after every outer step.
  3. Communication volume is O(N) per round, O(N/H) per step (tested in wire.py, not here).
  4. With compression enabled, the error-feedback residual persists across rounds and is
     included in checkpoints (see measurement/compress.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class DiLoCoStepResult:
    """Returned by `DiLoCoTrainer.step()`."""

    inner_loss: float
    did_outer_step: bool


class DiLoCoTrainer:
    """Reference inner/outer training loop. See methods/diloco.md §1 for the pseudocode:

        theta_outer <- initial weights (broadcast to all replicas)
        for round r = 0, 1, 2, ...:
            theta_inner <- theta_outer                 # INNER starts from the global model
            for h in 1..H:                              # purely local, zero communication
                theta_inner <- AdamW_step(theta_inner, local_batch)
            delta <- theta_outer - theta_inner           # pseudo-gradient
            delta_bar <- all_reduce_mean(delta)          # OUTER: the ONLY cross-replica traffic
            theta_outer <- NesterovSGD_step(theta_outer, delta_bar)
            broadcast theta_outer to all replicas

    Distributed behaviour: if `torch.distributed` is initialized, `outer_step()` all-reduces
    the pseudo-gradient (mean) across the process group. If it is not initialized, the
    all-reduce is skipped and the trainer behaves as a single replica — this is what makes
    it usable in a plain single-process unit test as well as a multi-rank gloo/NCCL job.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_optimizer_cfg: dict,
        outer_optimizer_cfg: dict,
        H: int,
    ) -> None:
        if H < 1:
            raise ValueError("H must be >= 1 (CLAUDE.md ExperimentSpec invariant)")

        self.model = model
        self.H = H
        self._h_count = 0

        inner_cfg = dict(inner_optimizer_cfg)
        inner_name = inner_cfg.pop("name", "adamw")
        if inner_name != "adamw":
            raise NotImplementedError(
                f"Only the adamw inner optimizer is implemented (methods/diloco.md §1); "
                f"got {inner_name!r}"
            )
        self.inner_optimizer = torch.optim.AdamW(self.model.parameters(), **inner_cfg)

        outer_cfg = dict(outer_optimizer_cfg)
        outer_name = outer_cfg.pop("name", "nesterov_sgd")
        if outer_name != "nesterov_sgd":
            raise NotImplementedError(
                f"Only the nesterov_sgd outer optimizer is implemented (methods/diloco.md §1); "
                f"got {outer_name!r}"
            )
        outer_cfg.setdefault("nesterov", True)

        # theta_outer is a SEPARATE tensor set from the model's live parameters. The inner
        # optimizer only ever touches self.model's parameters; outer_step() is the only
        # thing that reads/writes self._outer_params. Keeping them distinct is what makes
        # "theta_inner <- theta_outer" at the top of each round an explicit, auditable copy
        # rather than an aliasing accident.
        self._outer_params: list[nn.Parameter] = [
            nn.Parameter(p.detach().clone(), requires_grad=True) for p in self.model.parameters()
        ]
        self.outer_optimizer = torch.optim.SGD(self._outer_params, **outer_cfg)

        self._sync_outer_to_model()

    # -- round bookkeeping ------------------------------------------------------------

    @property
    def h_count(self) -> int:
        """Inner steps taken since the last outer step."""
        return self._h_count

    def ready_for_outer_step(self) -> bool:
        return self._h_count >= self.H

    # -- inner loop ---------------------------------------------------------------------

    def inner_step(self, loss_fn: Callable[[], torch.Tensor]) -> float:
        """One local AdamW step. Zero cross-replica communication (methods/diloco.md §1).

        `loss_fn` takes no arguments, runs the forward pass against `self.model`, and
        returns a scalar loss tensor wired to `self.model`'s parameters via autograd.
        """
        self.inner_optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        self.inner_optimizer.step()
        self._h_count += 1
        return float(loss.detach())

    # -- outer loop ---------------------------------------------------------------------

    def outer_step(self) -> None:
        """Compute the pseudo-gradient, all-reduce it, and apply the outer Nesterov SGD step.

        This is the ONLY cross-replica traffic in the loop (methods/diloco.md §1). Per
        invariant 1 (methods/diloco.md §3), `self.inner_optimizer`'s internal state (AdamW's
        exp_avg / exp_avg_sq) is never touched here — only theta_outer and theta_inner
        change across a round.
        """
        with torch.no_grad():
            pseudo_grads = [
                outer_p.detach() - inner_p.detach()
                for outer_p, inner_p in zip(
                    self._outer_params, self.model.parameters(), strict=True
                )
            ]
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                for g in pseudo_grads:
                    dist.all_reduce(g, op=dist.ReduceOp.SUM)
                    g.div_(world_size)
            for outer_p, g in zip(self._outer_params, pseudo_grads, strict=True):
                outer_p.grad = g.clone()

        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()
        self._sync_outer_to_model()
        self._h_count = 0

    def step(self, loss_fn: Callable[[], torch.Tensor]) -> DiLoCoStepResult:
        """Convenience: one inner step, followed by an outer step once H has been reached."""
        loss = self.inner_step(loss_fn)
        did_outer = self.ready_for_outer_step()
        if did_outer:
            self.outer_step()
        return DiLoCoStepResult(inner_loss=loss, did_outer_step=did_outer)

    # -- state -----------------------------------------------------------------------------

    def _sync_outer_to_model(self) -> None:
        with torch.no_grad():
            for model_p, outer_p in zip(self.model.parameters(), self._outer_params, strict=True):
                model_p.copy_(outer_p)

    def theta_outer(self) -> list[torch.Tensor]:
        """Snapshot of theta_outer, detached — used to check bit-identical theta_outer across
        replicas after each outer step (methods/diloco.md §3 invariant 2, US-06).
        """
        return [p.detach().clone() for p in self._outer_params]

    def inner_optimizer_state_fingerprint(self) -> list[int]:
        """Number of AdamW state tensors tracked per parameter (nonzero once warmed up).
        Used by tests to confirm inner optimizer state is NOT reset by outer_step()
        (methods/diloco.md §3 invariant 1) — a reset would show up as this collapsing to
        all-empty right after an outer step.
        """
        return [len(self.inner_optimizer.state.get(p, {})) for p in self.model.parameters()]
