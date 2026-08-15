"""Unit tests for analysis/predictor.py (FR-07, G4) -- fit/recommend/validate_holdout against
synthetic fixtures shaped like the real corpus (CLAUDE.md ADR-035/037), not real files, so
these stay fast and don't depend on results/raw/'s exact contents changing over time.
"""

from __future__ import annotations

import pytest

from diloco_measured.analysis import predictor


def _record(run_id: str, H: int, bw: int, cu_measured: float, tokens_per_s: float) -> dict:
    return {
        "run_id": run_id,
        "spec": {
            "phase": "cu_grid", "algorithm": "diloco", "H": H,
            "bandwidth_requested_bps": bw, "world_size": 4,
        },
        "cu": {"cu_measured": cu_measured},
        "throughput": {"tokens_per_s": tokens_per_s},
    }


# A small synthetic grid: 2 H values x 2 bandwidth levels, CU monotone increasing in both H
# and bandwidth, tokens_per_s likewise -- mirrors the real corpus's qualitative shape without
# depending on its exact numbers.
_GRID = [
    _record("r-h1-bw1g", 1, 1_000_000_000, 0.10, 10_000.0),
    _record("r-h1-bw5g", 1, 5_000_000_000, 0.30, 30_000.0),
    _record("r-h32-bw1g", 32, 1_000_000_000, 0.60, 60_000.0),
    _record("r-h32-bw5g", 32, 5_000_000_000, 0.85, 85_000.0),
]


@pytest.mark.unit
def test_fit_raises_on_empty_input():
    with pytest.raises(ValueError, match="no phase='cu_grid'"):
        predictor.fit([])


@pytest.mark.unit
def test_fit_ignores_unshaped_and_non_diloco_records():
    unshaped = _record("r-unshaped", 1, None, 0.5, 5000.0)  # type: ignore[arg-type]
    other_algo = dict(_GRID[0])
    other_algo["spec"] = {**_GRID[0]["spec"], "algorithm": "ddp"}
    model = predictor.fit([*_GRID, unshaped, other_algo])
    assert len(model.training_run_ids) == len(_GRID)


@pytest.mark.unit
def test_fit_calibration_domain_matches_tested_range():
    model = predictor.fit(_GRID)
    assert model.calibration_domain.bandwidth_bps_range == (1_000_000_000.0, 5_000_000_000.0)
    assert model.calibration_domain.h_range == (1, 32)


@pytest.mark.unit
def test_recommend_picks_smallest_h_clearing_target():
    model = predictor.fit(_GRID)
    # At 5g, H=1 gives cu=0.30 (below 0.5 target), H=32 gives cu=0.85 (clears it) -> H=32.
    rec = predictor.recommend(model, bandwidth_bps=5_000_000_000, model_config="30m-realvocab")
    assert rec.recommended_h == 32
    assert rec.expected_cu == pytest.approx(0.85)
    assert not rec.extrapolation_warning


@pytest.mark.unit
def test_recommend_flags_extrapolation_below_domain():
    model = predictor.fit(_GRID)
    rec = predictor.recommend(model, bandwidth_bps=1_000_000, model_config="30m-realvocab")
    assert rec.extrapolation_warning


@pytest.mark.unit
def test_recommend_flags_extrapolation_above_domain():
    model = predictor.fit(_GRID)
    rec = predictor.recommend(model, bandwidth_bps=50_000_000_000, model_config="30m-realvocab")
    assert rec.extrapolation_warning


@pytest.mark.unit
def test_recommend_flags_extrapolation_for_unrecognized_model_config():
    model = predictor.fit(_GRID)
    rec = predictor.recommend(
        model, bandwidth_bps=5_000_000_000, model_config="some-other-1b-model"
    )
    assert rec.extrapolation_warning


@pytest.mark.unit
def test_recommend_flags_extrapolation_when_target_unreachable():
    model = predictor.fit(_GRID)
    # No H reaches 0.99 CU anywhere in this fixture.
    rec = predictor.recommend(
        model, bandwidth_bps=5_000_000_000, model_config="30m-realvocab", target_cu=0.99
    )
    assert rec.extrapolation_warning
    assert rec.recommended_h == 32  # falls back to the best available, not silent


@pytest.mark.unit
def test_recommend_never_recommends_an_untested_h():
    model = predictor.fit(_GRID)
    for bw in (1_500_000_000, 2_000_000_000, 4_000_000_000):
        rec = predictor.recommend(model, bandwidth_bps=bw, model_config="30m-realvocab")
        assert rec.recommended_h in (1, 32)


@pytest.mark.unit
def test_expected_bytes_per_hour_is_positive_and_decreases_with_h():
    """More frequent sync (smaller H) means more bytes-on-wire per hour, all else equal --
    checked via two recommend() calls forced to different H by different target_cu.
    """
    model = predictor.fit(_GRID)
    rec_h32 = predictor.recommend(
        model, bandwidth_bps=5_000_000_000, model_config="30m-realvocab", target_cu=0.5
    )
    rec_h1 = predictor.recommend(
        model, bandwidth_bps=5_000_000_000, model_config="30m-realvocab", target_cu=0.0
    )
    assert rec_h32.recommended_h == 32
    assert rec_h1.recommended_h == 1
    assert rec_h1.expected_bytes_per_hour > rec_h32.expected_bytes_per_hour > 0


@pytest.mark.unit
def test_validate_holdout_zero_regret_when_holdout_matches_fit():
    model = predictor.fit(_GRID)
    result = predictor.validate_holdout(model, _GRID)
    assert result["mean_regret_pct"] == pytest.approx(0.0)
    for entry in result["per_bandwidth"].values():
        assert entry["predicted_H"] == entry["measured_H"]
        assert entry["regret_pct"] == pytest.approx(0.0)


@pytest.mark.unit
def test_validate_holdout_detects_a_flipped_recommendation():
    """Holdout data where H=1 unexpectedly clears the target at 5g (repeat noise) --
    predicted_H (from the fitted model) should differ from measured_H (from holdout ground
    truth). `regret_pct` here is reported in THROUGHPUT terms while the decision rule is
    CU-based (smallest H clearing target_cu, not "maximize tokens/s") -- so it is NOT
    guaranteed to be positive: if the noise-flipped ground-truth H=1 happens to have lower
    throughput than the model's (throughput-better) H=32 recommendation, following the
    prediction is actually a throughput WIN despite disagreeing with the CU-rule's pick, and
    `regret_pct` correctly comes out negative. What this test actually checks is that a
    genuine disagreement (predicted_H != measured_H) is detected and quantified at all.
    """
    model = predictor.fit(_GRID)
    noisy_holdout = [
        _record("noisy-h1-bw5g", 1, 5_000_000_000, 0.55, 40_000.0),  # now clears 0.5!
        _record("noisy-h32-bw5g", 32, 5_000_000_000, 0.85, 85_000.0),
    ]
    result = predictor.validate_holdout(model, noisy_holdout)
    entry = result["per_bandwidth"]["5000000000"]
    assert entry["measured_H"] == 1  # smallest H clearing target_cu in the noisy holdout
    assert entry["predicted_H"] == 32  # the fitted model still recommends what it saw in _GRID
    assert entry["regret_pct"] != 0
    assert entry["regret_pct"] == pytest.approx((40_000 - 85_000) / 40_000 * 100)


@pytest.mark.unit
def test_recommendation_calibration_domain_matches_model():
    model = predictor.fit(_GRID)
    rec = predictor.recommend(model, bandwidth_bps=5_000_000_000, model_config="30m-realvocab")
    assert rec.calibration_domain == model.calibration_domain
