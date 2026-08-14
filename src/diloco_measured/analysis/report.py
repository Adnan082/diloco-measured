"""Top-level orchestration for `make figures` / `diloco-measured figures`: load committed
results, filter, and regenerate every figure this project currently has a module for.

Lives here, not in `cli.py`, because CLAUDE.md §19.1 restricts `cli.py` to argument parsing
and thin dispatch — "which algorithms get a figure" is itself a decision this function makes
(from what's actually in the loaded corpus), not something `cli.py` should decide.

Pure orchestration over already-implemented pieces (`analysis/load.py`, `analysis/filter.py`,
`analysis/figures/*.py`) — opens no socket, no CUDA context, no AWS credential, matching
FR-11's promise that this whole path runs on a reviewer's laptop.
"""

from __future__ import annotations

from pathlib import Path

from diloco_measured.analysis.figures import fig1_cu_surface, fig4_cu_vs_h, fig5_bytes_on_wire
from diloco_measured.analysis.filter import apply as filter_apply
from diloco_measured.analysis.load import load_run_results

FIGURE_MODULES = ("fig1_cu_surface", "fig4_cu_vs_h", "fig5_bytes_on_wire")


def generate_all_figures(
    results_dir: Path | str = "results/raw",
    output_dir: Path | str = "results/figures",
    harness_version: str | None = None,
    allow_version_mix: bool = False,
) -> dict[str, list[Path]]:
    """Regenerate every figure this project has a module for, from committed `results/raw/`.

    Returns `{figure_module_name: [saved file paths]}` — one file per `algorithm` found with
    matching `phase == "cu_grid"` data (both `fig1_cu_surface` and `fig5_bytes_on_wire`
    currently require that). An algorithm with no shaped `cu_grid` data is skipped, not an
    error — `ValueError` from a figure module's `build()` (its documented "nothing to plot"
    signal) is caught here specifically for that reason.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_run_results(results_dir)
    kept, _report = filter_apply(
        records, harness_version=harness_version, allow_version_mix=allow_version_mix
    )

    algorithms = sorted(
        {r["spec"]["algorithm"] for r in kept if r["spec"].get("phase") == "cu_grid"}
    )

    saved: dict[str, list[Path]] = {name: [] for name in FIGURE_MODULES}

    for algorithm in algorithms:
        try:
            fig1 = fig1_cu_surface.build(kept, algorithm=algorithm, harness_version=harness_version)
        except ValueError:
            pass
        else:
            path = output_dir / f"fig1_cu_surface_{algorithm}.png"
            fig1.savefig(path, dpi=150)
            saved["fig1_cu_surface"].append(path)

        try:
            fig5 = fig5_bytes_on_wire.build(
                kept, algorithm=algorithm, harness_version=harness_version
            )
        except ValueError:
            pass
        else:
            path = output_dir / f"fig5_bytes_on_wire_{algorithm}.png"
            fig5.savefig(path, dpi=150)
            saved["fig5_bytes_on_wire"].append(path)

        # fig4 needs one plot per (algorithm, bandwidth level) found in the corpus — a
        # single-bandwidth H-sweep (e.g. an unshaped baseline) is exactly the case
        # fig1_cu_surface can't cover (it needs >=2 bandwidth levels to have an x-axis).
        bandwidth_levels = sorted(
            {
                r["spec"].get("bandwidth_requested_bps")
                for r in kept
                if r["spec"].get("phase") == "cu_grid" and r["spec"]["algorithm"] == algorithm
            },
            key=lambda v: (v is not None, v),
        )
        for bw in bandwidth_levels:
            try:
                fig4 = fig4_cu_vs_h.build(
                    kept, algorithm=algorithm, bandwidth_requested_bps=bw,
                    harness_version=harness_version,
                )
            except ValueError:
                continue
            bw_tag = "unshaped" if bw is None else str(bw)
            path = output_dir / f"fig4_cu_vs_h_{algorithm}_bw{bw_tag}.png"
            fig4.savefig(path, dpi=150)
            saved["fig4_cu_vs_h"].append(path)

    return saved
