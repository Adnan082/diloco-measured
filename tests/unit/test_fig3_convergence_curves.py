"""Unit tests for analysis/figures/fig3_convergence_curves.py — training loss vs. tokens,
single-GPU reference vs. DiLoCo across H, at one fixed bandwidth level.
"""

from __future__ import annotations

import pytest

from diloco_measured.analysis.figures import fig3_convergence_curves


def _convergence_record(
    run_id: str, H: int, world_size: int, bandwidth_requested_bps: int | None,
    losses: list[float], target_loss: float,
) -> dict:
    return {
        "run_id": run_id,
        "spec": {
            "phase": "convergence", "algorithm": "diloco", "H": H,
            "world_size": world_size, "bandwidth_requested_bps": bandwidth_requested_bps,
        },
        "convergence": {
            "points": [
                {"tokens": (i + 1) * 1000, "wall_s": float(i), "train_loss": loss, "val_loss": None}
                for i, loss in enumerate(losses)
            ],
            "target_loss": target_loss,
            "tttl_s": None,
            "tttl_smoothed_s": None,
            "final_loss": losses[-1],
            "reached_target": False,
        },
    }


@pytest.fixture
def campaign_records() -> list[dict]:
    ref = _convergence_record(
        "ref", H=999999, world_size=1, bandwidth_requested_bps=None,
        losses=[11.0, 9.0, 7.35], target_loss=7.35,
    )
    h1 = _convergence_record(
        "h1", H=1, world_size=4, bandwidth_requested_bps=None,
        losses=[11.0, 9.5, 8.65], target_loss=7.35,
    )
    h128 = _convergence_record(
        "h128", H=128, world_size=4, bandwidth_requested_bps=None,
        losses=[11.0, 9.2, 8.18], target_loss=7.35,
    )
    # Same H=1, different bandwidth -- must NOT appear when filtering bandwidth=None.
    h1_shaped = _convergence_record(
        "h1_shaped", H=1, world_size=4, bandwidth_requested_bps=200_000_000,
        losses=[11.0, 9.5, 8.65], target_loss=7.35,
    )
    return [ref, h1, h128, h1_shaped]


@pytest.mark.unit
def test_build_returns_a_figure_with_one_axes(campaign_records):
    fig = fig3_convergence_curves.build(
        campaign_records, algorithm="diloco", bandwidth_requested_bps=None
    )
    assert len(fig.axes) == 1


@pytest.mark.unit
def test_reference_and_both_h_curves_plus_target_line_all_plotted(campaign_records):
    fig = fig3_convergence_curves.build(
        campaign_records, algorithm="diloco", bandwidth_requested_bps=None
    )
    lines = fig.axes[0].get_lines()
    # reference + H=1 + H=128 + L* horizontal line = 4
    assert len(lines) == 4


@pytest.mark.unit
def test_reference_line_is_dashed_black(campaign_records):
    fig = fig3_convergence_curves.build(
        campaign_records, algorithm="diloco", bandwidth_requested_bps=None
    )
    ref_line = next(line for line in fig.axes[0].get_lines() if "reference" in line.get_label())
    assert ref_line.get_linestyle() == "--"


@pytest.mark.unit
def test_shaped_h1_record_excluded_from_unshaped_plot(campaign_records):
    fig = fig3_convergence_curves.build(
        campaign_records, algorithm="diloco", bandwidth_requested_bps=None
    )
    h1_lines = [line for line in fig.axes[0].get_lines() if line.get_label() == "H=1"]
    assert len(h1_lines) == 1
    # The unshaped H=1 curve has 3 points; if the shaped duplicate leaked in, this would
    # still be 3 points too (same shape), so check the actual y-data matches the unshaped one.
    assert list(h1_lines[0].get_ydata()) == [11.0, 9.5, 8.65]


@pytest.mark.unit
def test_raises_when_no_matching_records_and_no_reference_exist(campaign_records):
    # No reference (world_size==1 excluded) AND no DiLoCo record at this bandwidth level.
    diloco_only = [r for r in campaign_records if r["spec"]["world_size"] != 1]
    with pytest.raises(ValueError, match="nothing to plot"):
        fig3_convergence_curves.build(
            diloco_only, algorithm="diloco", bandwidth_requested_bps=999_999_999,
        )


@pytest.mark.unit
def test_reference_alone_plots_even_with_unmatched_bandwidth_filter(campaign_records):
    """The reference is bandwidth-independent (it never shapes anything) so it's included
    regardless of the DiLoCo bandwidth filter -- this is deliberate, not a leak.
    """
    fig = fig3_convergence_curves.build(
        campaign_records, algorithm="diloco", bandwidth_requested_bps=999_999_999,
    )
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "single-GPU reference" in labels
    assert "H=1" not in labels


@pytest.mark.unit
def test_title_states_bandwidth_as_unshaped(campaign_records):
    fig = fig3_convergence_curves.build(
        campaign_records, algorithm="diloco", bandwidth_requested_bps=None
    )
    assert "bandwidth=unshaped" in fig.axes[0].get_title()
