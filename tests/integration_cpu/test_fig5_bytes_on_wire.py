"""Figure generation from fixtures — CLAUDE.md §30.3 (see test_fig1_cu_surface.py's module
docstring for the categorization rationale). Exercises
analysis/figures/fig5_bytes_on_wire.py against the real corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from diloco_measured.analysis.figures import fig5_bytes_on_wire
from diloco_measured.analysis.filter import apply
from diloco_measured.analysis.load import load_run_results
from diloco_measured.measurement.wire import predict as wire_predict

CORPUS_DIR = Path(__file__).parents[1] / "fixtures" / "run_results"


@pytest.fixture
def kept_v1_records() -> list[dict]:
    records = load_run_results(CORPUS_DIR)
    kept, _ = apply(records, harness_version="v1")
    return kept


@pytest.mark.integration_cpu
def test_build_returns_a_figure_with_two_lines(kept_v1_records):
    fig = fig5_bytes_on_wire.build(kept_v1_records, algorithm="diloco", harness_version="v1")
    lines = fig.axes[0].get_lines()
    assert len(lines) == 2  # measured, predicted


@pytest.mark.integration_cpu
def test_measured_is_solid_predicted_is_dashed(kept_v1_records):
    """CLAUDE.md §18's presentation convention, checked as code."""
    fig = fig5_bytes_on_wire.build(kept_v1_records, algorithm="diloco", harness_version="v1")
    lines = {line.get_label(): line for line in fig.axes[0].get_lines()}
    assert lines["measured"].get_linestyle() == "-"
    assert lines["predicted"].get_linestyle() == "--"


@pytest.mark.integration_cpu
def test_bytes_per_token_decreases_with_h(kept_v1_records):
    """The entire point of the H sweep, per methods/wire_model.md §2: communication volume
    per token is O(1/H). If this regresses, either the fixture generator or fig5 itself broke
    the relationship.
    """
    fig = fig5_bytes_on_wire.build(kept_v1_records, algorithm="diloco", harness_version="v1")
    for line in fig.axes[0].get_lines():
        ys = list(line.get_ydata())
        assert ys == sorted(ys, reverse=True), f"{line.get_label()} is not decreasing in H"


@pytest.mark.integration_cpu
def test_measured_values_match_wire_predict_times_known_overhead_factor(kept_v1_records):
    """Ties the figure's output back to the actual wire.py::predict() implementation, not
    just internally-consistent fixture arithmetic — catches drift if either changes without
    the other (see tests/fixtures/generate_run_result_corpus.py::_wire_overrides_for_diloco).
    """
    fig = fig5_bytes_on_wire.build(kept_v1_records, algorithm="diloco", harness_version="v1")
    predicted_line = next(
        line for line in fig.axes[0].get_lines() if line.get_label() == "predicted"
    )
    xs, ys = predicted_line.get_xdata(), predicted_line.get_ydata()

    tokens_per_rank_per_step = 2 * 1024 * 4  # matches the fixture generator's constant
    for H, plotted_bytes_per_token in zip(xs, ys, strict=True):
        spec = {"algorithm": "diloco", "world_size": 4, "H": int(H)}
        expected = wire_predict(spec, model_params=1_000_000_000, dtype_bytes=4)
        expected_rounded = round(expected)  # fixtures round to int bytes before dividing
        assert plotted_bytes_per_token == pytest.approx(
            expected_rounded / tokens_per_rank_per_step, rel=1e-6
        )


@pytest.mark.integration_cpu
def test_pools_across_bandwidth_levels_at_fixed_h(kept_v1_records):
    """H=32 has three cu_grid records at different bandwidths (1g x2 repeats, 200m x1) — all
    three must contribute to the H=32 point, since bytes-on-wire is bandwidth-independent.
    """
    fig = fig5_bytes_on_wire.build(kept_v1_records, algorithm="diloco", harness_version="v1")
    assert "6 contributing runs" in fig.axes[0].get_title()  # H=8,32(x3),128,512


@pytest.mark.integration_cpu
def test_raises_when_no_matching_records_exist(kept_v1_records):
    with pytest.raises(ValueError, match="nothing to plot"):
        fig5_bytes_on_wire.build(kept_v1_records, algorithm="nonexistent_algorithm")
