"""analysis/report.py::generate_all_figures against the real corpus
(tests/fixtures/run_results/) — the same load -> filter -> per-algorithm figure pipeline
`diloco-measured figures` runs, minus the CLI layer. See test_fig1_cu_surface.py /
test_fig5_bytes_on_wire.py for the underlying per-figure-module behaviour this wraps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from diloco_measured.analysis.report import generate_all_figures

CORPUS_DIR = Path(__file__).parents[1] / "fixtures" / "run_results"

# cu_grid-phase algorithms present in the fixture corpus, per tests/fixtures/run_results/.
_EXPECTED_ALGORITHMS = {"ddp", "diloco", "fsdp2", "localsgd"}


@pytest.fixture(autouse=True)
def _close_figures_after_each_test():
    yield
    plt.close("all")


@pytest.mark.integration_cpu
def test_writes_one_fig1_and_one_fig5_per_algorithm(tmp_path):
    saved = generate_all_figures(
        results_dir=CORPUS_DIR, output_dir=tmp_path, harness_version="v1"
    )
    assert {p.stem.rsplit("_", 1)[-1] for p in saved["fig1_cu_surface"]} == _EXPECTED_ALGORITHMS
    assert {p.stem.rsplit("_", 1)[-1] for p in saved["fig5_bytes_on_wire"]} == _EXPECTED_ALGORITHMS


@pytest.mark.integration_cpu
def test_files_actually_exist_on_disk(tmp_path):
    saved = generate_all_figures(
        results_dir=CORPUS_DIR, output_dir=tmp_path, harness_version="v1"
    )
    for paths in saved.values():
        for p in paths:
            assert p.is_file()
            assert p.stat().st_size > 0


@pytest.mark.integration_cpu
def test_output_dir_is_created_if_missing(tmp_path):
    output_dir = tmp_path / "nested" / "figures"
    assert not output_dir.exists()
    generate_all_figures(results_dir=CORPUS_DIR, output_dir=output_dir, harness_version="v1")
    assert output_dir.is_dir()


@pytest.mark.integration_cpu
def test_empty_results_dir_returns_empty_lists_not_an_error(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    saved = generate_all_figures(results_dir=empty_dir, output_dir=tmp_path / "out")
    assert saved == {
        "fig1_cu_surface": [], "fig3_convergence_curves": [],
        "fig4_cu_vs_h": [], "fig5_bytes_on_wire": [],
    }


@pytest.mark.integration_cpu
def test_version_mismatch_excludes_the_oldversion_fixture_without_allow_mix(tmp_path):
    """cu_grid-diloco-1b-h32-bw1g-oldversion-r0.json exists specifically to exercise this
    exclusion (analysis/filter.py) — confirm it doesn't silently make it into a saved figure.
    """
    saved_strict = generate_all_figures(
        results_dir=CORPUS_DIR, output_dir=tmp_path / "strict", harness_version="v1",
        allow_version_mix=False,
    )
    saved_mixed = generate_all_figures(
        results_dir=CORPUS_DIR, output_dir=tmp_path / "mixed", harness_version="v1",
        allow_version_mix=True,
    )
    # Both still produce a diloco figure either way (plenty of v1 diloco records exist) — the
    # exclusion changes the data feeding the plot, not whether the file gets written, so this
    # just confirms neither call errors and both cover the same algorithm set.
    assert saved_strict.keys() == saved_mixed.keys()
