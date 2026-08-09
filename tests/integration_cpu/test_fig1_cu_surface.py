"""Figure generation from fixtures — CLAUDE.md §30.3 "Aggregation pipeline | Fixture records |
Figures generate from fixtures with no GPU, no network." Exercises
analysis/figures/fig1_cu_surface.py against the real corpus (tests/fixtures/run_results/),
through the same load -> filter -> figure pipeline `make figures` will eventually run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from diloco_measured.analysis.figures import fig1_cu_surface
from diloco_measured.analysis.filter import apply
from diloco_measured.analysis.load import load_run_results

CORPUS_DIR = Path(__file__).parents[1] / "fixtures" / "run_results"


@pytest.fixture
def kept_v1_records() -> list[dict]:
    records = load_run_results(CORPUS_DIR)
    kept, _ = apply(records, harness_version="v1")
    return kept


@pytest.mark.integration_cpu
def test_build_returns_a_figure_with_one_axes(kept_v1_records):
    fig = fig1_cu_surface.build(kept_v1_records, algorithm="diloco", harness_version="v1")
    assert len(fig.axes) == 1


@pytest.mark.integration_cpu
def test_diloco_produces_one_line_per_h_per_series(kept_v1_records):
    """4 H values (8, 32, 128, 512) present in the cu_grid-phase diloco fixtures, x3 series
    (measured, analytic_link, analytic_achieved) = 12 lines. H=32's two bandwidth points
    (1g and 200m, one of which has 2 repeats) collapse onto one line each, not two.
    """
    fig = fig1_cu_surface.build(kept_v1_records, algorithm="diloco", harness_version="v1")
    lines = fig.axes[0].get_lines()
    assert len(lines) == 12


@pytest.mark.integration_cpu
def test_measured_series_is_solid_analytic_series_are_not(kept_v1_records):
    """CLAUDE.md §18: "measured series are solid; analytic/simulated series are dashed... this
    single convention carries the project's entire visual argument" — this is the one
    invariant in this whole figure that must never silently break.
    """
    fig = fig1_cu_surface.build(kept_v1_records, algorithm="diloco", harness_version="v1")
    lines = fig.axes[0].get_lines()

    for line in lines:
        label = line.get_label()
        if "measured" in label:
            assert line.get_linestyle() == "-", f"{label} must be solid"
        else:
            assert line.get_linestyle() in ("--", ":"), f"{label} must not be solid"


@pytest.mark.integration_cpu
def test_ddp_excludes_the_unshaped_record(kept_v1_records):
    """5 DDP fixtures exist (unshaped/5g/1g/200m/50m) but only 4 have a defined bandwidth —
    the unshaped one must not appear on a log-bandwidth x-axis.
    """
    fig = fig1_cu_surface.build(kept_v1_records, algorithm="ddp", harness_version="v1")
    lines = fig.axes[0].get_lines()
    assert len(lines) == 3  # H=1 only, x3 series
    measured_line = next(line for line in lines if "measured" in line.get_label())
    expected_bandwidths = [50_000_000, 200_000_000, 1_000_000_000, 5_000_000_000]
    assert sorted(measured_line.get_xdata()) == expected_bandwidths


@pytest.mark.integration_cpu
def test_convergence_phase_records_do_not_leak_into_the_cu_grid_figure(kept_v1_records):
    """cu_grid-diloco-1b-h32-bw200m-r0 (cu_measured=0.55) and
    convergence-diloco-130m-h32-bw200m-notreached-r0 (cu_measured=0.85, the factory default)
    share the same (algorithm, H, bandwidth) — if phase weren't filtered, aggregate_repeats
    would silently average across two different experiments (§10.2, methods/cu_model.md).
    """
    fig = fig1_cu_surface.build(kept_v1_records, algorithm="diloco", harness_version="v1")
    measured_h32 = next(
        line for line in fig.axes[0].get_lines()
        if line.get_label() == "H=32, measured"
    )
    ys = dict(zip(measured_h32.get_xdata(), measured_h32.get_ydata(), strict=True))
    # Only the cu_grid record's value (0.55) should appear at bw=200m — not an average with
    # the convergence record's 0.85.
    assert ys[200_000_000] == pytest.approx(0.55)


@pytest.mark.integration_cpu
def test_title_states_harness_version_and_run_count(kept_v1_records):
    fig = fig1_cu_surface.build(kept_v1_records, algorithm="diloco", harness_version="v1")
    title = fig.axes[0].get_title()
    assert "harness_version=v1" in title
    assert "6 contributing runs" in title  # H=8,32(r0),32(r1),128,512,32@200m


@pytest.mark.integration_cpu
def test_raises_when_no_matching_records_exist(kept_v1_records):
    with pytest.raises(ValueError, match="nothing to plot"):
        fig1_cu_surface.build(kept_v1_records, algorithm="nonexistent_algorithm")


@pytest.mark.integration_cpu
def test_harness_version_none_labels_as_mixed(kept_v1_records):
    fig = fig1_cu_surface.build(kept_v1_records, algorithm="diloco", harness_version=None)
    assert "harness_version=mixed/unspecified" in fig.axes[0].get_title()
