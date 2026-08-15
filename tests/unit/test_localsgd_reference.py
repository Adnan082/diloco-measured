"""Single-process unit tests for measurement/localsgd.py::LocalSGDTrainer.

Mirrors tests/unit/test_diloco_reference.py's structure deliberately -- the two algorithms
share almost everything except the outer step, so the tests should too, making any future
divergence between them easy to spot.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from diloco_measured.measurement.localsgd import LocalSGDTrainer

OPT_CFG = {"name": "adamw", "lr": 0.05}


def _tiny_trainer(H: int = 3) -> tuple[LocalSGDTrainer, nn.Module]:
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    trainer = LocalSGDTrainer(model, dict(OPT_CFG), H=H)
    return trainer, model


@pytest.mark.unit
def test_h_must_be_positive():
    with pytest.raises(ValueError):
        LocalSGDTrainer(nn.Linear(2, 2), dict(OPT_CFG), H=0)


@pytest.mark.unit
def test_rejects_unknown_optimizer():
    with pytest.raises(NotImplementedError):
        LocalSGDTrainer(nn.Linear(2, 2), {"name": "sgd", "lr": 0.1}, H=1)


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
def test_outer_step_resets_h_count_but_preserves_optimizer_state():
    """Same invariant as DiLoCo's: averaging PARAMETERS must not disturb AdamW's internal
    exp_avg/exp_avg_sq state -- that state lives per-parameter-tensor-identity, and outer_step()
    only ever calls `.data.div_()`/`all_reduce` in-place, never replacing the parameter object.
    """
    trainer, model = _tiny_trainer(H=2)
    x, y = torch.randn(4, 4), torch.randn(4, 2)

    assert trainer.inner_optimizer_state_fingerprint() == [0, 0]  # weight, bias

    trainer.inner_step(lambda: F.mse_loss(model(x), y))
    fingerprint_after_one_step = trainer.inner_optimizer_state_fingerprint()
    assert all(n > 0 for n in fingerprint_after_one_step)

    trainer.inner_step(lambda: F.mse_loss(model(x), y))
    assert trainer.ready_for_outer_step()
    trainer.outer_step()

    assert trainer.h_count == 0
    fingerprint_after_outer_step = trainer.inner_optimizer_state_fingerprint()
    assert fingerprint_after_outer_step == fingerprint_after_one_step, (
        "outer_step must NOT reset AdamW's internal state"
    )


@pytest.mark.unit
def test_outer_step_works_without_torch_distributed_initialized():
    import torch.distributed as dist

    assert not dist.is_initialized()
    trainer, model = _tiny_trainer(H=1)
    x, y = torch.randn(4, 4), torch.randn(4, 2)
    result = trainer.step(lambda: F.mse_loss(model(x), y))
    assert result.did_outer_step is True
    assert isinstance(result.inner_loss, float)


@pytest.mark.unit
def test_single_replica_outer_step_is_a_no_op_on_parameters():
    """With no process group, outer_step() skips the all-reduce branch entirely (single
    replica IS the average of one thing) -- parameters must be numerically unchanged by the
    outer step itself (only the inner AdamW steps change them).
    """
    trainer, model = _tiny_trainer(H=1)
    x, y = torch.randn(4, 4), torch.randn(4, 2)
    trainer.inner_step(lambda: F.mse_loss(model(x), y))
    params_before_outer = [p.detach().clone() for p in model.parameters()]
    trainer.outer_step()
    params_after_outer = [p.detach().clone() for p in model.parameters()]
    for before, after in zip(params_before_outer, params_after_outer, strict=True):
        assert torch.equal(before, after)


@pytest.mark.unit
def test_loss_trends_down_over_several_rounds():
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
