"""CLI entrypoint for `diloco-measured`.

RULE (CLAUDE.md §19.1): this module contains argument parsing, config resolution, and exit
codes ONLY. No measurement logic, no computation, no decision logic. Every command below is a
thin dispatch to `measurement/` or `analysis/`. If you find yourself writing an `if` that isn't
about argument validity, it belongs in the module you're calling, not here.

Full CLI surface and contract: CLAUDE.md §17.1.

STATUS: `analyze` and `figures` are real — they dispatch to `analysis/`, which is complete
and GPU-free. The unified measurement commands exit with an explicit unsupported
message and code 2. Existing campaigns use the experiment scripts documented in
docs/REPRODUCTION.md; these wrappers do not provision or measure infrastructure.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="diloco-measured",
    help="Measured, not simulated: bandwidth-controlled semi-sync LLM training. See CLAUDE.md.",
    no_args_is_help=True,
)

network_app = typer.Typer(help="FR-01: network characterization.")
app.add_typer(network_app, name="network")


@network_app.command("characterize")
def network_characterize(
    profile_id: str = typer.Option(None, "--profile-id"),
    levels: str = typer.Option("5g,1g,200m,50m", "--levels"),
) -> None:
    """FR-01. Full network characterization: iperf3 all-pairs + NCCL BW curve + burst-decay probe.

    Requires: cluster up, sudo on nodes. Writes: results/network/<profile_id>.json.
    """
    # from diloco_measured.measurement import netshape, probe
    typer.echo(
        "Unsupported unified command: use the documented experiment scripts in "
        "docs/REPRODUCTION.md. No infrastructure was changed.",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("run")
def run(
    spec: str = typer.Option(..., "--spec", help="Path to an ExperimentSpec YAML"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """FR-03. One instrumented run.

    Writes results/raw/<run_id>.json + results/steps/<run_id>.parquet.
    Idempotency: run_id includes repeat_index; re-running creates a NEW record, never
    overwrites.
    """
    typer.echo(
        "Unsupported unified command: use the documented experiment scripts in "
        "docs/REPRODUCTION.md. No infrastructure was changed.",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("grid")
def grid(
    config: str = typer.Option(..., "--config"),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    """Execute a campaign of runs. A failed point is logged; the grid CONTINUES (§19.4)."""
    typer.echo(
        "Unsupported unified command: use the documented experiment scripts in "
        "docs/REPRODUCTION.md. No infrastructure was changed.",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("converge")
def converge(
    spec: str = typer.Option(..., "--spec"),
) -> None:
    """FR-06. Fixed-token-budget run with periodic eval, computing time-to-target-loss."""
    typer.echo(
        "Unsupported unified command: use the documented experiment scripts in "
        "docs/REPRODUCTION.md. No infrastructure was changed.",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("plan")
def plan(
    probe: bool = typer.Option(False, "--probe"),
    bandwidth: int = typer.Option(None, "--bandwidth", help="bps, alternative to --probe"),
    model: str = typer.Option(..., "--model"),
    gpus: int = typer.Option(None, "--gpus"),
) -> None:
    """FR-07. Recommend H for a measured or given bandwidth. Requires a fitted PredictorModel.

    Extrapolation outside the calibration domain is NEVER silent (US-05).
    """
    typer.echo(
        "Unsupported unified command: use the documented experiment scripts in "
        "docs/REPRODUCTION.md. No infrastructure was changed.",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command("analyze")
def analyze(
    phase: str = typer.Option(None, "--phase", help="restrict to one ExperimentSpec phase"),
    results_dir: str = typer.Option("results/raw", "--results-dir"),
    harness_version: str = typer.Option(
        None, "--harness-version", help="if set, excludes any other harness_version"
    ),
    allow_version_mix: bool = typer.Option(False, "--allow-version-mix"),
) -> None:
    """Aggregate committed records. GPU-free (FR-11).

    Loads + schema-validates every record under --results-dir, applies the FR-13 exclusion
    rules (crashed/diverged, version mismatch, loader-bound, reconciliation residual), and
    prints the resulting counts. All the actual logic lives in analysis/load.py and
    analysis/filter.py — this command is the dispatch only.
    """
    from diloco_measured.analysis.filter import apply as filter_apply
    from diloco_measured.analysis.load import SchemaValidationError, load_run_results

    try:
        records = load_run_results(results_dir)
    except SchemaValidationError as e:
        typer.echo(f"refusing to load: {e}", err=True)
        raise typer.Exit(code=1) from e

    if phase is not None:
        records = [r for r in records if r["spec"].get("phase") == phase]

    kept, report = filter_apply(
        records, harness_version=harness_version, allow_version_mix=allow_version_mix
    )

    typer.echo(f"loaded {report.total} record(s), kept {report.kept}")
    if report.excluded_total:
        typer.echo(
            "excluded "
            f"{report.excluded_total}: crashed={report.excluded_crashed} "
            f"diverged={report.excluded_diverged} "
            f"other_status={report.excluded_other_status} "
            f"version_mismatch={report.excluded_version_mismatch} "
            f"loader_bound={report.excluded_loader_bound} "
            f"reconciliation_failed={report.excluded_reconciliation_failed}"
        )
    for r in kept:
        typer.echo(
            f"  {r['run_id']}  algorithm={r['spec']['algorithm']} H={r['spec']['H']} "
            f"bandwidth_requested_bps={r['spec'].get('bandwidth_requested_bps')}"
        )


@app.command("figures")
def figures(
    only: str = typer.Option(
        None, "--only", help="restrict to one figure module, e.g. fig1_cu_surface"
    ),
    results_dir: str = typer.Option("results/raw", "--results-dir"),
    output_dir: str = typer.Option("results/figures", "--output-dir"),
    harness_version: str = typer.Option(None, "--harness-version"),
    allow_version_mix: bool = typer.Option(False, "--allow-version-mix"),
) -> None:
    """FR-11. Regenerate figures from committed data. GPU-free, network-free, credential-free.

    Dispatches to analysis/report.py::generate_all_figures(), which decides which
    algorithms have data and calls each figures/*.py::build() in turn — that decision logic
    does not belong in this module (CLAUDE.md §19.1).
    """
    from diloco_measured.analysis.load import SchemaValidationError
    from diloco_measured.analysis.report import generate_all_figures

    try:
        saved = generate_all_figures(
            results_dir=results_dir,
            output_dir=output_dir,
            harness_version=harness_version,
            allow_version_mix=allow_version_mix,
        )
    except SchemaValidationError as e:
        typer.echo(f"refusing to load: {e}", err=True)
        raise typer.Exit(code=1) from e

    if only is not None:
        saved = {k: v for k, v in saved.items() if k == only}

    paths = [p for group in saved.values() for p in group]
    if not paths:
        typer.echo(
            f"no figures generated — no completed, phase='cu_grid' records found under "
            f"{results_dir}"
        )
        return
    for p in paths:
        typer.echo(f"wrote {p}")


if __name__ == "__main__":
    app()
