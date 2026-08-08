"""Unit tests for measurement/wire.py::predict — pure arithmetic, no GPU/network required.

Targets from CLAUDE.md §30.2: "Ring all-reduce byte counts for known (N, P, H); DDP vs DiLoCo
ratio equals H." See methods/wire_model.md §2-3 for the derivation being tested.
"""

from __future__ import annotations

import pytest

from diloco_measured.measurement.wire import predict


@pytest.mark.unit
def test_ring_all_reduce_matches_known_formula():
    # P=4, N=1000 bytes (model_params=250, dtype_bytes=4), H=1 (DDP-equivalent sync frequency)
    spec = {"algorithm": "diloco", "world_size": 4, "H": 1}
    bytes_per_step = predict(spec, model_params=250, dtype_bytes=4)
    expected = 2 * 1000 * (4 - 1) / 4  # = 1500.0
    assert bytes_per_step == pytest.approx(expected)


@pytest.mark.unit
def test_ddp_vs_diloco_ratio_equals_h():
    """DDP synchronizes every step (H=1); DiLoCo at H synchronizes 1/H as often. Per-step
    bytes must differ by exactly a factor of H, holding everything else constant — this is
    the specific invariant methods/wire_model.md §2 calls out as a test target.
    """
    ddp_spec = {"algorithm": "ddp", "world_size": 4, "H": 1}
    H = 32
    diloco_spec = {"algorithm": "diloco", "world_size": 4, "H": H}

    ddp_bytes = predict(ddp_spec, model_params=1_000_000, dtype_bytes=4)
    diloco_bytes = predict(diloco_spec, model_params=1_000_000, dtype_bytes=4)

    assert ddp_bytes / diloco_bytes == pytest.approx(H)


@pytest.mark.unit
def test_communication_volume_independent_of_h_per_round():
    """Per-ROUND (not per-step) bytes must be O(N), independent of H — methods/diloco.md §3
    invariant 3. per_round_bytes = per_step_bytes * H should be constant across H.
    """
    spec_h8 = {"algorithm": "diloco", "world_size": 4, "H": 8}
    spec_h128 = {"algorithm": "diloco", "world_size": 4, "H": 128}

    per_round_h8 = predict(spec_h8, model_params=500_000, dtype_bytes=4) * 8
    per_round_h128 = predict(spec_h128, model_params=500_000, dtype_bytes=4) * 128

    assert per_round_h8 == pytest.approx(per_round_h128)


@pytest.mark.unit
def test_fsdp2_is_explicitly_unimplemented_not_guessed():
    """FSDP2's per-step wire volume is [UNKNOWN] (methods/wire_model.md §3/§6). predict()
    must refuse rather than silently returning a guessed number (CLAUDE.md §33.2.6).
    """
    spec = {"algorithm": "fsdp2", "world_size": 4, "H": 1}
    with pytest.raises(NotImplementedError):
        predict(spec, model_params=1_000_000)


@pytest.mark.unit
@pytest.mark.parametrize("world_size", [0, 1])
def test_rejects_degenerate_world_size(world_size):
    spec = {"algorithm": "diloco", "world_size": world_size, "H": 1}
    with pytest.raises(ValueError):
        predict(spec, model_params=1000)


@pytest.mark.unit
def test_rejects_h_less_than_one():
    spec = {"algorithm": "diloco", "world_size": 4, "H": 0}
    with pytest.raises(ValueError):
        predict(spec, model_params=1000)
