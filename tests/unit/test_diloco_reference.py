"""Single-process unit tests for measurement/diloco.py::DiLoCoTrainer.

These exercise the reference implementation without torch.distributed — outer_step() must
degrade gracefully to a single-replica no-op reduction when no process group is initialized
(see DiLoCoTrainer's docstring). Multi-rank behaviour (bit-identical theta_outer across
replicas) is covered separately in tests/integration_cpu/test_diloco_equivalence.py, since
that genuinely needs multiple processes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from diloco_measured.measurement.diloco import DiLoCoTrainer

INNER_CFG = {"name": "adamw", "lr": 0.05}
OUTER_CFG = {"name": "nesterov_sgd", "lr": 0.1, "momentum": 0.9}


def _tiny_trainer(H: int = 3) -> tuple[DiLoCoTrainer, nn.Module]:
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    trainer = DiLoCoTrainer(model, dict(INNER_CFG), dict(OUTER_CFG), H=H)
    return trainer, model


@pytest.mark.unit
def test_h_must_be_positive():
    with pytest.raises(ValueError):
        DiLoCoTrainer(nn.Linear(2, 2), dict(INNER_CFG), dict(OUTER_CFG), H=0)


@pytest.mark.unit
def test_rejects_unknown_inner_optimizer():
    with pytest.raises(NotImplementedError):
        DiLoCoTrainer(nn.Linear(2, 2), {"name": "sgd", "lr": 0.1}, dict(OUTER_CFG), H=1)


@pytest.mark.unit
def test_rejects_unknown_outer_optimizer():
    with pytest.raises(NotImplementedError):
        DiLoCoTrainer(nn.Linear(2, 2), dict(INNER_CFG), {"name": "adam", "lr": 0.1}, H=1)


@pytest.mark.unit
def test_h_count_advances_and_ready_flag_flips_at_h():
    trainer, model = _tiny_trainer(H=3)
    x, y = torch.randn(4, 4), torch.randn(4, 2)

    for _ in range(2):
        trainer.inner_step(lambda: F.mse_loss(model(x), y))
        assert not trainer.ready_for_outer_step()

    trainer.inner_step(lambda: F.mse_loss(model(x), y))
    assert trainer.ready_for_outer_step()
    assert trainer.h_count == 3


@pytest.mark.unit
def test_outer_step_resets_h_count_but_preserves_inner_optimizer_state():
    """methods/diloco.md §3 invariant 1: inner optimizer state persists across rounds."""
    trainer, model = _tiny_trainer(H=2)
    x, y = torch.randn(4, 4), torch.randn(4, 2)

    # Before any step: AdamW has no state yet.
    assert trainer.inner_optimizer_state_fingerprint() == [0, 0]  # weight, bias

    trainer.inner_step(lambda: F.mse_loss(model(x), y))
    fingerprint_after_one_step = trainer.inner_optimizer_state_fingerprint()
    assert all(n > 0 for n in fingerprint_after_one_step), "AdamW should have allocated state"

    trainer.inner_step(lambda: F.mse_loss(model(x), y))
    assert trainer.ready_for_outer_step()
    trainer.outer_step()

    assert trainer.h_count == 0, "outer_step must reset the inner-step counter"
    fingerprint_after_outer_step = trainer.inner_optimizer_state_fingerprint()
    assert fingerprint_after_outer_step == fingerprint_after_one_step, (
        "outer_step must NOT reset AdamW's internal state — that would silently turn this "
        "into naive FedOpt (methods/diloco.md §3 invariant 1)"
    )


@pytest.mark.unit
def test_outer_step_moves_theta_outer_toward_theta_inner():
    trainer, model = _tiny_trainer(H=1)
    theta_before = [t.clone() for t in trainer.theta_outer()]

    x, y = torch.randn(8, 4), torch.randn(8, 2)
    trainer.inner_step(lambda: F.mse_loss(model(x), y))
    assert trainer.ready_for_outer_step()
    trainer.outer_step()

    theta_after = trainer.theta_outer()
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(theta_before, theta_after, strict=True)
    ), "theta_outer should move once a nonzero pseudo-gradient is applied"


@pytest.mark.unit
def test_outer_step_works_without_torch_distributed_initialized():
    """Single-replica fallback: no process group => outer_step() skips the all-reduce branch
    entirely rather than raising.
    """
    import torch.distributed as dist

    assert not dist.is_initialized()
    trainer, model = _tiny_trainer(H=1)
    x, y = torch.randn(4, 4), torch.randn(4, 2)
    result = trainer.step(lambda: F.mse_loss(model(x), y))
    assert result.did_outer_step is True
    assert isinstance(result.inner_loss, float)


@pytest.mark.unit
def test_loss_trends_down_over_several_rounds():
    """Not a tight convergence guarantee — just a sanity check that the loop is wired
    correctly on a convex problem (linear model, MSE loss). Compares the mean of the first
    few losses against the mean of the last few, rather than single endpoints, since a
    momentum-based outer optimizer can legitimately overshoot on any individual step.
    """
    trainer, model = _tiny_trainer(H=2)
    torch.manual_seed(1)
    x, y = torch.randn(16, 4), torch.randn(16, 2)

    losses = []
    for _ in range(40):
        result = trainer.step(lambda: F.mse_loss(model(x), y))
        losses.append(result.inner_loss)

    early_mean = sum(losses[:5]) / 5
    late_mean = sum(losses[-5:]) / 5
    assert late_mean < early_mean
