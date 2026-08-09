"""One-off generator for tests/fixtures/run_results/*.json — the synthetic RunResult corpus
CLAUDE.md §30.6 calls for ("~20 synthetic RunResult records covering every status, used by
all analysis tests").

Run manually to regenerate the corpus after a schema change:

    python tests/fixtures/generate_run_result_corpus.py

Not a test file (no `test_` prefix, so pytest never collects it) and not run automatically —
the corpus is committed, static data, matching the project's "results are committed JSON"
philosophy (ADR-004). Re-run and re-commit deliberately, don't regenerate silently in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

from factories import make_convergence_curve, make_cu_observation, make_fault_event, make_run_result

OUT_DIR = Path(__file__).parent / "run_results"


def _reconciled_cu(**overrides) -> dict:
    return make_cu_observation(**overrides)


def build_corpus() -> dict[str, dict]:
    """Returns {run_id: RunResult dict}. Keys double as filenames (run_id.json)."""
    records: dict[str, dict] = {}

    def add(run_id: str, **kwargs) -> None:
        records[run_id] = make_run_result(run_id, **kwargs)

    # --- DDP baseline (H=1) across every bandwidth level, all completed --------------------
    for bw_label, bw_bps in [
        ("unshaped", None), ("5g", 5_000_000_000), ("1g", 1_000_000_000),
        ("200m", 200_000_000), ("50m", 50_000_000),
    ]:
        add(
            f"cu_grid-ddp-1b-h1-bw{bw_label}-r0",
            spec_overrides={
                "algorithm": "ddp", "implementation": "reference", "H": 1,
                "bandwidth_requested_bps": bw_bps,
            },
        )

    # --- DiLoCo across the H sweep at 1g, all completed -------------------------------------
    for H in (8, 32, 128, 512):
        add(
            f"cu_grid-diloco-1b-h{H}-bw1g-r0",
            spec_overrides={
                "algorithm": "diloco", "H": H, "bandwidth_requested_bps": 1_000_000_000,
            },
        )
    # A second repeat at one grid point, for aggregate.py's repeat-handling tests.
    add(
        "cu_grid-diloco-1b-h32-bw1g-r1",
        spec_overrides={
            "algorithm": "diloco", "H": 32, "bandwidth_requested_bps": 1_000_000_000,
            "repeat_index": 1,
        },
        cu_overrides={"cu_measured": 0.81},
    )
    add(
        "cu_grid-diloco-1b-h32-bw200m-r0",
        spec_overrides={"algorithm": "diloco", "H": 32, "bandwidth_requested_bps": 200_000_000},
        cu_overrides={"cu_measured": 0.55, "cu_analytic_link": 0.70, "cu_analytic_achieved": 0.60},
    )

    # --- LocalSGD and FSDP2 ablations --------------------------------------------------------
    add(
        "cu_grid-localsgd-1b-h32-bw1g-r0",
        spec_overrides={"algorithm": "localsgd", "implementation": "torchft", "H": 32},
    )
    add(
        "cu_grid-fsdp2-1b-h1-bw1g-r0",
        spec_overrides={"algorithm": "fsdp2", "implementation": "reference", "H": 1},
    )

    # --- Non-completed statuses (crashed/diverged/aborted_shaping/oom) — every status the
    # schema enumerates, each with partial/absent telemetry as the real run lifecycle would
    # produce (CLAUDE.md §15.2 Run/RunResult state transitions). ----------------------------
    add(
        "cu_grid-diloco-1b-h32-bw1g-r2",
        status="crashed", cu=None, wire=None, throughput=None,
        notes="rank 2 died unexpectedly at step 140 (not injected)",
    )
    add(
        "cu_grid-diloco-1b-h512-bw50m-r0",
        status="diverged",
        spec_overrides={
            "algorithm": "diloco", "H": 512, "bandwidth_requested_bps": 50_000_000,
        },
        cu_overrides={"cu_measured": 0.20},
        notes="loss NaN at step 88 — divergence at large H is a legitimate finding, not a bug",
    )
    add(
        "cu_grid-diloco-1b-h32-bw1g-r3",
        status="aborted_shaping", cu=None, wire=None, throughput=None,
        notes=(
            "shaping verification failed at 38% error against a 10% tolerance, "
            "retried once, still failed"
        ),
    )
    add(
        "cu_grid-diloco-1b-h32-bw1g-r4",
        status="oom", cu=None, wire=None, throughput=None,
        notes="OOM on rank 1 at micro_batch_size=2, seq_len=1024",
    )

    # --- Records that ARE completed but must still be excluded by analysis/filter.py --------
    add(
        "cu_grid-diloco-1b-h32-bw1g-loaderbound-r0",
        loader_bound_warning=True,
        notes="dataloader stall exceeded 5% of step time — flagged, not silently included",
    )
    add(
        "cu_grid-diloco-1b-h32-bw1g-oldversion-r0",
        harness_version="v0",  # everything else in this corpus is v1
        fingerprint_overrides={"harness_version": "v0"},
        notes="pre-dates a measurement-path fix; must not be pooled with v1 records (ADR-006)",
    )
    add(
        "cu_grid-diloco-1b-h32-bw1g-reconciliationfail-r0",
        cu_overrides={
            # compute+sync+optimizer+loader = 15, total_s=100 -> 85% residual, way over the
            # 5% tolerance (methods/cu_model.md §5) — must be excluded by filter.py.
            "compute_s": 10.0, "sync_blocked_s": 3.0, "optimizer_s": 1.0, "loader_stall_s": 1.0,
            "total_s": 100.0,
        },
        notes="instrumentation gap: components don't reconcile to total_s",
    )

    # --- Convergence runs (Phase B / FR-06) --------------------------------------------------
    add(
        "convergence-ddp-130m-h1-bw1g-reference-r0",
        spec_overrides={
            "phase": "convergence", "algorithm": "ddp", "H": 1, "world_size": 1,
            "budget_type": "tokens", "budget_value": 400_000_000,
            "model_config": "configs/models/130m.toml",
        },
        convergence=make_convergence_curve(),
    )
    add(
        "convergence-diloco-130m-h32-bw200m-notreached-r0",
        spec_overrides={
            "phase": "convergence", "algorithm": "diloco", "H": 32,
            "budget_type": "tokens", "budget_value": 400_000_000,
            "bandwidth_requested_bps": 200_000_000, "model_config": "configs/models/130m.toml",
        },
        convergence=make_convergence_curve(
            tttl_s=None, tttl_smoothed_s=None, reached_target=False, final_loss=2.9,
        ),
        notes="target loss never reached in budget — tttl_s is null, not a big number",
    )

    # --- Compression ablation (FR-10 / G6) ---------------------------------------------------
    add(
        "compression-diloco-130m-h32-bw200m-int8ef-r0",
        spec_overrides={
            "algorithm": "diloco", "H": 32, "compression": "int8_ef",
            "bandwidth_requested_bps": 200_000_000, "model_config": "configs/models/130m.toml",
        },
    )

    # --- Fault injection (FR-09 / G7) --------------------------------------------------------
    add(
        "faults-diloco-130m-h512-bw1g-r0",
        spec_overrides={
            "phase": "faults", "algorithm": "diloco", "H": 512,
            "fault_schedule": [{"rank": 3, "t_s": 600.0}],
        },
        faults=[make_fault_event()],
    )
    add(
        "faults-ddp-130m-h1-bw1g-r0",
        spec_overrides={
            "phase": "faults", "algorithm": "ddp", "H": 1,
            "fault_schedule": [{"rank": 3, "t_s": 600.0}],
        },
        faults=[
            make_fault_event(outcome="hung", detected_at_s=None, resumed_at_s=None, steps_lost=0)
        ],
        notes="DDP hangs on a worker death rather than recovering — expected, per FR-09",
    )

    return records


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for existing in OUT_DIR.glob("*.json"):
        existing.unlink()

    corpus = build_corpus()
    for run_id, record in corpus.items():
        path = OUT_DIR / f"{run_id}.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")

    print(f"Wrote {len(corpus)} RunResult fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
