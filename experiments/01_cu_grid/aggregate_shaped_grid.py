"""Aggregate the shaped bandwidth x H grid (`run_shaped_grid.py`'s output) into schema-valid
`RunResult` records under `results/raw/`.

Unlike `aggregate_results.py` (the earlier unshaped slice, ADR-034), this campaign has a real
per-point `ShapingVerification` (FR-02) and a real *shaped* bandwidth level, so:
  - `cu_analytic_link` is fed the FR-02-VERIFIED measured rate for THIS point (not a shared
    unshaped baseline) -- this is "the papers' naive assumption" (nominal achieved
    point-to-point bandwidth) applied per-point, per ADR-007/ADR-015.
  - `cu_analytic_achieved` is fed a bandwidth DERIVED FROM THE REAL MEASURED outer-sync time
    of this exact run (`bytes_synced * 8 / mean_single_sync_s`), not an interpolated NCCL
    curve -- we have the real number directly (StepTimer's `sync_blocked_ms`, summed over the
    kept window and divided by the count of steps that actually did an outer sync), so using
    it is more direct than interpolating a curve measured under different conditions. This is
    a deliberate methodological difference from the unshaped campaign's
    interpolate-a-precomputed-NCCL-curve approach, recorded here rather than silently reused.

Points with `status == "aborted_shaping"` produce NO RunResult (CLAUDE.md invariant: "passed
== false => no RunResult may exist for this run") -- they are reported in RESULTS.md /
NOTES.md narrative only. Points with `status == "crashed"` likewise produce no analysis-
eligible RunResult here (none occurred in this campaign as of writing).

Reads `run_shaped_grid.py`'s output at `shaped_grid_run_logs/_summary.json` (next to this
file) by default; override with `DILOCO_SHAPED_GRID_SUMMARY` to point at a different run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from diloco_measured.analysis.cu import analytic as cu_analytic
from diloco_measured.analysis.cu import measured as cu_measured

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = Path(
    os.environ.get(
        "DILOCO_SHAPED_GRID_SUMMARY",
        str(Path(__file__).resolve().parent / "shaped_grid_run_logs" / "_summary.json"),
    )
)
RAW_TELEMETRY_OUT_DIR = Path(__file__).resolve().parent / "raw_step_telemetry_shaped"

# Same fingerprint shape as the unshaped campaign (ADR-034) -- same harness commit lineage,
# same cluster topology/AMI, new instance IDs (relaunched cluster, ADR-035). Captured via
# measurement/fingerprint.py::capture() on the rank-0 node for the first shaped point and
# reused across the campaign (bootstrap is identical across all 4 GPU nodes by construction --
# same setup_node.sh run, same pinned deps -- so a single capture is representative; CLAUDE.md
# FR-08 requires a fingerprint per record, not a per-record capture of physically-identical
# environments).
FINGERPRINT = {
    "harness_git_sha": "PLACEHOLDER",  # filled in by main() from local git HEAD
    "harness_dirty": False,
    "harness_version": "0.1.0",
    "pytorch_version": "2.13.0+cu130",
    "nccl_version": "2.29.7",
    "cuda_version": "13.0",
    "driver_version": "595.71.05",
    "instance_types": ["g6e.2xlarge"],
    "az": "us-east-1b",
    "placement_group_id": "pg-0ee059f5ef7da671b",
    "gpu_clocks_locked": True,
    "nccl_env": {"NCCL_SOCKET_IFNAME": "enp39s0"},
    "kernel_version": "6.17.0-1019-aws",
    "dataset_shard_checksum": "e1a2eea5c5c8b7a0c0d17c760a8eacf0be6861828332efb6218919b9e70a141e",
    "seed": 42,
    "torchtitan_version": "0.2.2",
}


def build_experiment_spec(
    H: int, bandwidth_bps: int, steps: int, warmup_steps: int, micro_batch_size: int
) -> dict:
    return {
        "spec_id": f"cu_grid-diloco-30m-h{H}-bw{bandwidth_bps}-r0",
        "phase": "cu_grid",
        "algorithm": "diloco",
        "implementation": "reference",
        "H": H,
        "model_config": "30m-realvocab",
        "world_size": 4,
        "bandwidth_requested_bps": bandwidth_bps,
        "micro_batch_size": micro_batch_size,
        "seq_len": 512,
        "grad_accum": 1,
        "budget_type": "steps",
        "budget_value": steps,
        "warmup_steps": warmup_steps,
        "compression": None,
        "seed": 42,
        "repeat_index": 0,
        "fault_schedule": None,
    }


def aggregate_one(point: dict) -> dict:
    raw = point["raw"]
    H = raw["H"]
    n_params = raw["n_params"]
    steps = raw["steps"]
    warmup_steps = raw["warmup_steps"]
    step_records = raw["step_records"]
    sv = point["shaping_verification"]

    cu_m = cu_measured(step_records, warmup=warmup_steps)

    kept = step_records[warmup_steps:]
    total_compute_s = sum(r["compute_time_ms"] for r in kept) / 1000
    total_sync_s = sum(r["sync_blocked_ms"] for r in kept) / 1000
    total_optimizer_s = sum(r["optimizer_time_ms"] for r in kept) / 1000
    total_loader_s = sum(r["loader_stall_ms"] for r in kept) / 1000
    total_wall_s = sum(r["wall_time_ms"] for r in kept) / 1000
    mean_compute_s_per_step = total_compute_s / len(kept)

    bytes_synced = n_params * 4  # fp32 pseudo-gradient, methods/wire_model.md §2
    spec_for_cu = {"H": H}

    # cu_analytic_link: fed the FR-02-VERIFIED achieved rate for THIS point (the naive
    # papers' assumption -- nominal point-to-point bandwidth -- applied per shaped level).
    cu_link = cu_analytic(
        spec_for_cu, t_compute_s=mean_compute_s_per_step,
        bytes_synced=bytes_synced, bandwidth_bps=int(sv["measured_bps"]),
    )

    # cu_analytic_achieved: fed a bandwidth DERIVED FROM THE REAL MEASURED sync time of this
    # exact run, not an interpolated curve -- see module docstring.
    num_outer_syncs = sum(1 for r in kept if r["did_outer_step"])
    if num_outer_syncs > 0 and total_sync_s > 0:
        mean_single_sync_s = total_sync_s / num_outer_syncs
        achieved_bw_bps = bytes_synced * 8 / mean_single_sync_s
    else:
        achieved_bw_bps = None

    if achieved_bw_bps is not None:
        cu_achieved = cu_analytic(
            spec_for_cu, t_compute_s=mean_compute_s_per_step,
            bytes_synced=bytes_synced, bandwidth_bps=int(achieved_bw_bps),
        )
    else:
        cu_achieved = None

    reconciliation_residual_pct = (
        abs(total_wall_s - (total_compute_s + total_sync_s + total_optimizer_s + total_loader_s))
        / total_wall_s * 100
        if total_wall_s > 0 else 0.0
    )
    assert reconciliation_residual_pct < 5.0, (
        f"reconciliation residual {reconciliation_residual_pct:.2f}% exceeds 5% tolerance "
        f"for {point['run_id']}"
    )

    cu_observation = {
        "cu_measured": cu_m,
        "cu_analytic_link": cu_link,
        "cu_analytic_achieved": cu_achieved,
        "nccl_bw_used_bps": int(achieved_bw_bps) if achieved_bw_bps is not None else None,
        "nccl_bw_interpolated": False,  # derived from THIS run's real measurement, not a curve
        "discrepancy_link": cu_link / cu_m,
        "discrepancy_achieved": (cu_achieved / cu_m) if cu_achieved is not None else None,
        "compute_s": total_compute_s,
        "sync_blocked_s": total_sync_s,
        "optimizer_s": total_optimizer_s,
        "loader_stall_s": total_loader_s,
        "total_s": total_wall_s,
    }

    tokens_per_step_per_rank = raw["micro_batch_size"] * raw["seq_len"]
    global_tokens = tokens_per_step_per_rank * raw["world_size"] * len(kept)
    tokens_per_s = global_tokens / total_wall_s if total_wall_s > 0 else 0.0
    wall_times_sorted = sorted(r["wall_time_ms"] for r in kept)

    def pctile(data: list[float], p: float) -> float:
        idx = min(len(data) - 1, int(len(data) * p))
        return data[idx]

    throughput = {
        "tokens_per_s": tokens_per_s,
        "step_time_p50_ms": pctile(wall_times_sorted, 0.50),
        "step_time_p90_ms": pctile(wall_times_sorted, 0.90),
        "step_time_p99_ms": pctile(wall_times_sorted, 0.99),
    }

    shaping = {
        "requested_bps": sv["requested_bps"],
        "measured_bps": sv["measured_bps"],
        "error_pct": sv["error_pct"],
        "tolerance_pct": sv["tolerance_pct"],
        "passed": sv["passed"],
        "attempts": sv["attempts"],
        "iperf_raw": "",  # not retained by run_shaped_grid.py -- see its own docstring
        "qdisc_dump": "",
    }

    return {
        "run_id": point["run_id"],
        "spec": build_experiment_spec(
            H, point["bandwidth_requested_bps"], steps, warmup_steps, raw["micro_batch_size"]
        ),
        "fingerprint": FINGERPRINT,
        "shaping": shaping,
        "network_profile_id": "phase1-us-east-1b-20260814",
        "harness_version": "0.1.0",
        "status": "completed",
        "started_at": raw["started_at"],
        "ended_at": raw["ended_at"],
        "cu": cu_observation,
        "throughput": throughput,
        "faults": [],
        "loader_bound_warning": False,
        "notes": (
            f"Shaped bandwidth grid, 2026-08-14 (see CLAUDE.md ADR-035). Bandwidth level "
            f"'{point['bandwidth_label']}' requested={sv['requested_bps']}bps "
            f"measured={sv['measured_bps']:.0f}bps (FR-02 gate passed, "
            f"error={sv['error_pct']:.2f}%). Same model/driver/campaign lineage as the "
            f"unshaped H-sweep (ADR-034), relaunched cluster, same 30.8M-param model, "
            f"real FineWeb-Edu data. cu_analytic_achieved derived from this run's own "
            f"measured outer-sync time, not an interpolated NCCL curve (see "
            f"aggregate_shaped_grid.py docstring for why)."
        ),
    }


def main() -> None:
    import subprocess

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    FINGERPRINT["harness_git_sha"] = git_sha

    points = json.loads(SUMMARY_PATH.read_text())
    out_dir = REPO_ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    RAW_TELEMETRY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_completed = n_aborted = n_other = 0
    for point in points:
        if point["status"] == "completed":
            result = aggregate_one(point)
            out_path = out_dir / f"{result['run_id']}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            # Preserve raw per-step telemetry alongside, same policy as ADR-034.
            raw_out = RAW_TELEMETRY_OUT_DIR / f"{point['run_id']}.json"
            with open(raw_out, "w") as f:
                json.dump(point["raw"], f)
            cu = result["cu"]
            print(
                f"{point['run_id']}: cu_measured={cu['cu_measured']:.4f} "
                f"cu_analytic_link={cu['cu_analytic_link']:.4f} "
                f"cu_analytic_achieved={cu['cu_analytic_achieved']:.4f} -> wrote {out_path}"
            )
            n_completed += 1
        elif point["status"] == "aborted_shaping":
            print(
                f"{point['run_id']}: ABORTED_SHAPING (no RunResult written, per CLAUDE.md "
                f"invariant) -- requested={point['shaping_verification']['requested_bps']} "
                f"measured={point['shaping_verification']['measured_bps']:.0f} "
                f"error={point['shaping_verification']['error_pct']:.1f}%"
            )
            n_aborted += 1
        else:
            print(f"{point['run_id']}: status={point['status']} (no RunResult written)")
            n_other += 1

    print(f"\n{n_completed} completed, {n_aborted} aborted_shaping, {n_other} other")


if __name__ == "__main__":
    main()
