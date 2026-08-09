"""Factory helpers for building schema-valid `RunResult` fixtures.

Centralizes the "valid RunResult skeleton" pattern so the static fixture corpus
(tests/fixtures/run_results/*.json — CLAUDE.md §30.6: "a fixture corpus of ~20 synthetic
RunResult records covering every status, used by all analysis tests") and any test that needs
a one-off variant build from the same source of truth. Not itself a test file — imported by
tests and by scripts/generate_run_result_corpus.py.
"""

from __future__ import annotations

from typing import Any


def make_experiment_spec(**overrides: Any) -> dict:
    base = {
        "spec_id": "s0", "phase": "cu_grid", "algorithm": "diloco", "implementation": "reference",
        "H": 32, "model_config": "configs/models/1b.toml",
        "bandwidth_requested_bps": 1_000_000_000,
        "world_size": 4, "micro_batch_size": 2, "seq_len": 1024, "grad_accum": 4,
        "budget_type": "steps", "budget_value": 200, "warmup_steps": 20,
        "compression": None, "seed": 0, "repeat_index": 0, "fault_schedule": None,
    }
    base.update(overrides)
    return base


def make_fingerprint(**overrides: Any) -> dict:
    base = {
        "harness_git_sha": "abc123", "harness_dirty": False, "harness_version": "v1",
        "pytorch_version": "2.13", "nccl_version": "2.20", "cuda_version": "12.4",
        "driver_version": "550.0", "instance_types": ["g6e.2xlarge"], "az": "us-east-1a",
        "gpu_clocks_locked": True, "kernel_version": "6.8.0",
        "dataset_shard_checksum": "deadbeef", "seed": 0,
    }
    base.update(overrides)
    return base


def make_cu_observation(**overrides: Any) -> dict:
    base = {
        "cu_measured": 0.85,
        "cu_analytic_link": 0.90,
        "cu_analytic_achieved": 0.88,
        "nccl_bw_used_bps": 900_000_000,
        "nccl_bw_interpolated": False,
        "discrepancy_link": 0.90 / 0.85,
        "discrepancy_achieved": 0.88 / 0.85,
        "compute_s": 80.0, "sync_blocked_s": 15.0, "optimizer_s": 3.0, "loader_stall_s": 2.0,
        "total_s": 100.0,
    }
    base.update(overrides)
    return base


def make_wire_account(**overrides: Any) -> dict:
    base = {
        "predicted_bytes": 1_000_000, "measured_bytes": 1_100_000, "overhead_ratio": 1.1,
        "bytes_per_training_token_predicted": 10.0,
        "bytes_per_training_token_measured": 11.0,
        "idle_baseline_bytes": 0,
    }
    base.update(overrides)
    return base


def make_throughput_summary(**overrides: Any) -> dict:
    base = {
        "tokens_per_s": 12_000.0, "mfu": 0.42,
        "step_time_p50_ms": 120.0, "step_time_p90_ms": 150.0, "step_time_p99_ms": 200.0,
        "peak_memory_bytes": 20_000_000_000,
    }
    base.update(overrides)
    return base


def make_convergence_curve(**overrides: Any) -> dict:
    base = {
        "points": [
            {"tokens": 100_000_000, "wall_s": 300.0, "train_loss": 3.2, "val_loss": 3.3},
            {"tokens": 200_000_000, "wall_s": 600.0, "train_loss": 2.8, "val_loss": 2.9},
            {"tokens": 400_000_000, "wall_s": 1200.0, "train_loss": 2.4, "val_loss": 2.5},
        ],
        "target_loss": 2.5,
        "tttl_s": 1200.0,
        "tttl_smoothed_s": 1180.0,
        "final_loss": 2.4,
        "reached_target": True,
    }
    base.update(overrides)
    return base


def make_fault_event(**overrides: Any) -> dict:
    base = {
        "injected_at_s": 600.0, "rank": 3,
        "detected_at_s": 602.5, "resumed_at_s": 615.0,
        "steps_lost": 4, "outcome": "recovered",
    }
    base.update(overrides)
    return base


def make_run_result(
    run_id: str = "r0",
    *,
    spec_overrides: dict | None = None,
    fingerprint_overrides: dict | None = None,
    cu_overrides: dict | None = None,
    wire_overrides: dict | None = None,
    throughput_overrides: dict | None = None,
    convergence: dict | None = None,
    faults: list | None = None,
    **overrides: Any,
) -> dict:
    """A schema-valid `RunResult` (schemas/run_result.v1.json), `status="completed"` by
    default. Pass `status=...` (and anything else) via `**overrides` to build the other
    branches (`crashed`, `diverged`, etc.) — see `tests/fixtures/run_results/README.md` for
    the corpus this backs.
    """
    base: dict = {
        "run_id": run_id,
        "spec": make_experiment_spec(**(spec_overrides or {})),
        "fingerprint": make_fingerprint(**(fingerprint_overrides or {})),
        "network_profile_id": "np0",
        "harness_version": "v1",
        "status": "completed",
        "started_at": "2026-08-09T00:00:00Z",
        "ended_at": "2026-08-09T00:05:00Z",
        "cu": make_cu_observation(**(cu_overrides or {})),
        "wire": make_wire_account(**(wire_overrides or {})),
        "throughput": make_throughput_summary(**(throughput_overrides or {})),
        "faults": faults or [],
        "loader_bound_warning": False,
        "notes": "",
    }
    if convergence is not None:
        base["convergence"] = convergence
    base.update(overrides)

    # schemas/run_result.v1.json does not accept `null` for cu/wire/throughput/convergence —
    # they're optional-but-typed (a crashed run has no telemetry, so the key is ABSENT, not
    # null; CLAUDE.md §15.2's RunResult entity marks them nullable in prose, but the schema
    # expresses "nullable" as "omittable"). A caller passing e.g. `cu=None` to represent
    # "no telemetry for this non-completed run" means exactly that — drop the key.
    for optional_field in ("cu", "wire", "throughput", "convergence"):
        if base.get(optional_field, "__present__") is None:
            del base[optional_field]

    return base
