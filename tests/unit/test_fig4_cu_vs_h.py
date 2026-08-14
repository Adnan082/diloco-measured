"""Unit tests for analysis/figures/fig4_cu_vs_h.py — CU vs. H at one fixed bandwidth level
(the complementary grouping to fig1_cu_surface's CU vs. bandwidth at one fixed H; this is
what an unshaped-baseline H-sweep produces, e.g. the real 2026-08-14 training runs).
"""

from __future__ import annotations

import pytest

from diloco_measured.analysis.figures import fig4_cu_vs_h


def _record(run_id: str, H: int, cu_measured: float, cu_link: float, cu_achieved: float) -> dict:
    return {
        "run_id": run_id,
        "spec": {
            "phase": "cu_grid", "algorithm": "diloco", "H": H,
            "bandwidth_requested_bps": None,
        },
        "cu": {
            "cu_measured": cu_measured,
            "cu_analytic_link": cu_link,
            "cu_analytic_achieved": cu_achieved,
        },
    }


@pytest.fixture
def h_sweep_records() -> list[dict]:
    return [
        _record("r-h1", 1, 0.167, 0.199, 0.280),
        _record("r-h8", 8, 0.602, 0.656, 0.749),
        _record("r-h32", 32, 0.710, 0.883, 0.922),
        _record("r-h128", 128, 0.839, 0.968, 0.979),
    ]


@pytest.mark.unit
def test_build_returns_a_figure_with_one_axes(h_sweep_records):
    fig = fig4_cu_vs_h.build(
        h_sweep_records, algorithm="diloco", bandwidth_requested_bps=None
    )
    assert len(fig.axes) == 1


@pytest.mark.unit
def test_three_series_all_plotted(h_sweep_records):
    fig = fig4_cu_vs_h.build(
        h_sweep_records, algorithm="diloco", bandwidth_requested_bps=None
    )
    lines = fig.axes[0].get_lines()
    assert len(lines) == 3  # measured, analytic_link, analytic_achieved


@pytest.mark.unit
def test_measured_series_is_solid_analytic_series_are_not(h_sweep_records):
    """CLAUDE.md §18: the one convention that must never silently break."""
    fig = fig4_cu_vs_h.build(
        h_sweep_records, algorithm="diloco", bandwidth_requested_bps=None
    )
    for line in fig.axes[0].get_lines():
        label = line.get_label()
        if "measured" in label:
            assert line.get_linestyle() == "-", f"{label} must be solid"
        else:
            assert line.get_linestyle() in ("--", ":"), f"{label} must not be solid"


@pytest.mark.unit
def test_measured_cu_increases_with_h(h_sweep_records):
    """The headline DiLoCo trend: larger H amortizes the sync cost over more compute."""
    fig = fig4_cu_vs_h.build(
        h_sweep_records, algorithm="diloco", bandwidth_requested_bps=None
    )
    measured_line = next(
        line for line in fig.axes[0].get_lines() if "measured" in line.get_label()
    )
    ys = measured_line.get_ydata()
    assert list(ys) == sorted(ys), "cu_measured should increase monotonically with H here"


@pytest.mark.unit
def test_raises_when_no_matching_records_exist(h_sweep_records):
    with pytest.raises(ValueError, match="nothing to plot"):
        fig4_cu_vs_h.build(
            h_sweep_records, algorithm="ddp", bandwidth_requested_bps=None
        )


@pytest.mark.unit
def test_bandwidth_mismatch_excludes_records(h_sweep_records):
    with pytest.raises(ValueError, match="nothing to plot"):
        fig4_cu_vs_h.build(
            h_sweep_records, algorithm="diloco", bandwidth_requested_bps=1_000_000_000
        )


@pytest.mark.unit
def test_title_states_bandwidth_as_unshaped(h_sweep_records):
    fig = fig4_cu_vs_h.build(
        h_sweep_records, algorithm="diloco", bandwidth_requested_bps=None
    )
    assert "bandwidth=unshaped" in fig.axes[0].get_title()
