"""CLI entrypoint for `diloco-measured`.

RULE (CLAUDE.md §19.1): this module contains argument parsing, config resolution, and exit
codes ONLY. No measurement logic, no computation, no decision logic. Every command below is a
thin dispatch to `measurement/` or `analysis/`. If you find yourself writing an `if` that isn't
about argument validity, it belongs in the module you're calling, not here.

Full CLI surface and contract: CLAUDE.md §17.1.

STATUS: [PROPOSED] scaffold. Every command below raises NotImplementedError until the
corresponding module (§14 repo structure) exists — this is intentional (Architecture
Principle #5: fail loud, fail early) rather than a silent no-op.
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
    raise NotImplementedError("Phase 0/1 — see methods/network_protocol.md")


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
    raise NotImplementedError("Phase 0/1 — see CLAUDE.md §10.1 run lifecycle")


@app.command("grid")
def grid(
    config: str = typer.Option(..., "--config"),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    """Execute a campaign of runs. A failed point is logged; the grid CONTINUES (§19.4)."""
    raise NotImplementedError("Phase 3")


@app.command("converge")
def converge(
    spec: str = typer.Option(..., "--spec"),
) -> None:
    """FR-06. Fixed-token-budget run with periodic eval, computing time-to-target-loss."""
    raise NotImplementedError("Phase 4")


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
    raise NotImplementedError("Phase 5 — see methods/statistics.md, PLAYBOOK.md")


@app.command("analyze")
def analyze(
    phase: str = typer.Option(None, "--phase"),
    allow_version_mix: bool = typer.Option(False, "--allow-version-mix"),
) -> None:
    """Aggregate committed records. GPU-free (FR-11)."""
    raise NotImplementedError("Phase 0 analysis-side work")


@app.command("figures")
def figures(
    only: str = typer.Option(None, "--only", help="e.g. fig1"),
) -> None:
    """FR-11. Regenerate figures from committed data. GPU-free, network-free, credential-free."""
    raise NotImplementedError("Phase 0 analysis-side work")


if __name__ == "__main__":
    app()
