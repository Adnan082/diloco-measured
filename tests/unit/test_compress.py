"""Unit tests for measurement/compress.py codecs.

The single most important assertion in this file is
`test_int8_error_feedback_residual_is_not_dropped_across_rounds` — CLAUDE.md §30.2 flags
exactly this as "the invariant most likely to be silently broken."
"""

from __future__ import annotations

import pytest
import torch

from diloco_measured.measurement.compress import (
    Fp16Codec,
    Int8ErrorFeedbackCodec,
    TopKCodec,
)


@pytest.mark.unit
def test_fp16_round_trip_is_bounded():
    torch.manual_seed(0)
    x = torch.randn(100) * 10
    codec = Fp16Codec()
    decoded = codec.decode(codec.encode(x))
    assert decoded.shape == x.shape
    # fp16 relative precision is ~2^-11; generous bound to avoid flakiness.
    assert torch.allclose(decoded, x, rtol=1e-2, atol=1e-2)


@pytest.mark.unit
def test_int8_single_shot_round_trip_is_bounded():
    torch.manual_seed(0)
    x = torch.randn(1000)
    codec = Int8ErrorFeedbackCodec()
    decoded = codec.decode(codec.encode(x))
    # 8-bit symmetric quantization step is max_abs/127; error per element <= step/2.
    max_abs = x.abs().max().item()
    step = max_abs / 127.0
    assert torch.all((decoded - x).abs() <= step / 2 + 1e-6)


@pytest.mark.unit
def test_int8_error_feedback_residual_is_not_dropped_across_rounds():
    """The residual from round 1's quantization error must show up compensated in round 2 —
    i.e. the codec is NOT equivalent to independently quantizing each round in isolation.
    """
    torch.manual_seed(0)
    # A tensor with values well below one quantization step of a much larger co-occurring
    # value would be crushed to zero WITHOUT error feedback; with EF, the residual accumulates
    # across rounds until it's large enough to be representable.
    codec = Int8ErrorFeedbackCodec()
    small_constant = torch.tensor([0.01, 0.01, 0.01, 0.01])
    large_once = torch.tensor([100.0, -100.0, 50.0, -50.0])

    # Round 1: dominated by the large tensor, so the small residual is invisible this round.
    codec.encode(large_once)
    residual_after_round_1 = codec.state_dict()["residual"].clone()
    assert residual_after_round_1 is not None
    assert torch.any(residual_after_round_1.abs() > 0), (
        "quantizing a wide-dynamic-range tensor should leave a nonzero residual to carry "
        "forward — if this is all-zero, the codec is silently dropping error"
    )

    # Round 2: same small constant contribution repeatedly, WITH the carried residual.
    total_residual_growth = torch.zeros_like(small_constant)
    for _ in range(50):
        codec.encode(small_constant)
        total_residual_growth = codec.state_dict()["residual"]

    # Without error feedback, encoding a tiny constant tensor round after round would never
    # move the residual (each round's residual would just reset to ~small_constant every
    # time). With EF, the residual should have grown to reflect accumulated bias.
    assert not torch.equal(total_residual_growth, residual_after_round_1)


@pytest.mark.unit
def test_int8_state_dict_round_trips():
    torch.manual_seed(0)
    codec = Int8ErrorFeedbackCodec()
    codec.encode(torch.randn(10))
    state = codec.state_dict()

    restored = Int8ErrorFeedbackCodec()
    restored.load_state_dict(state)
    assert torch.equal(restored.state_dict()["residual"], state["residual"])


@pytest.mark.unit
def test_int8_rejects_shape_change():
    codec = Int8ErrorFeedbackCodec()
    codec.encode(torch.randn(10))
    with pytest.raises(ValueError):
        codec.encode(torch.randn(20))


@pytest.mark.unit
def test_topk_keeps_only_k_fraction_nonzero():
    torch.manual_seed(0)
    x = torch.randn(100)
    codec = TopKCodec(k_fraction=0.1)
    decoded = codec.decode(codec.encode(x))
    nonzero = (decoded != 0).sum().item()
    assert nonzero <= 10  # 10% of 100, allowing for ties truncated by round()


@pytest.mark.unit
def test_topk_keeps_the_largest_magnitude_elements():
    x = torch.tensor([0.1, -5.0, 0.2, 3.0, -0.05, 0.01])
    codec = TopKCodec(k_fraction=1 / 3)  # k = 2
    decoded = codec.decode(codec.encode(x))
    kept_indices = {i for i, v in enumerate(decoded.tolist()) if v != 0}
    assert kept_indices == {1, 3}  # -5.0 and 3.0 are the two largest-magnitude elements


@pytest.mark.unit
def test_topk_error_feedback_residual_persists():
    torch.manual_seed(0)
    codec = TopKCodec(k_fraction=0.5, error_feedback=True)
    codec.encode(torch.randn(10))
    residual_1 = codec.state_dict()["residual"].clone()
    codec.encode(torch.randn(10))
    residual_2 = codec.state_dict()["residual"].clone()
    assert not torch.equal(residual_1, residual_2), "residual should evolve, not reset, each round"


@pytest.mark.unit
def test_topk_rejects_invalid_k_fraction():
    with pytest.raises(ValueError):
        TopKCodec(k_fraction=0)
    with pytest.raises(ValueError):
        TopKCodec(k_fraction=1.5)
