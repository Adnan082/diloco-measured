"""Aggregate the real DDP + LocalSGD grids (`run_ddp_localsgd_grid.py`'s output) into
schema-valid `RunResult` records under `results/raw/`. Same fingerprint lineage as the
DiLoCo shaped campaign (ADR-035), relaunched cluster (new placement group, same instance
types/AZ/AMI/bootstrap -- see FINGERPRINT below), same 30.8M-param model, same re-staged
FineWeb-Edu shards.

Two algorithms, two different derivations of `cu.sync_blocked_s`:

  - **LocalSGD**: same shape as DiLoCo's aggregator (aggregate_shaped_grid.py) -- `sync_
    blocked_ms` is a DIRECT StepTimer measurement (`mark_compute_done()` after the inner
    step's backward, `mark_sync_done()` after the explicit outer all-reduce). No imputation.

  - **DDP**: `sync_blocked_ms` is NOT a direct measurement. DDP overlaps its gradient
    all-reduce with backward() by design, so a naive "compute" phase marker right after
    backward() would fold communication time into compute and CU would read ~100% even
    under severe bandwidth scarcity -- exactly the failure mode this project exists to
    catch, not commit. `train_driver_ddp.py` instead uses a CALIBRATION-PROBE methodology:
    a compute-only baseline is measured once during warmup via `ddp_model.no_sync()` (skips
    gradient communication), median-reduced over 3 samples; every real step then computes
    `compute_ms = min(baseline, synced_backward_ms)` and `sync_ms = max(0, synced_backward_ms
    - baseline)`. This means DDP's `cu.compute_s`/`cu.sync_blocked_s` split is an IMPUTED
    decomposition of one directly-measured quantity (synced backward wall time), not two
    independently-measured quantities -- flagged explicitly in every DDP record's `notes`
    field, not silently presented as equivalent in provenance to LocalSGD's or DiLoCo's.
    Two real calibration bugs were found and fixed getting this right (Triton JIT
    contaminating first the synced-path warmup, then the no_sync()-path warmup separately --
    see git history accd18d/c6d8889 and the ADR for this campaign) -- both fixes were
    verified against real hardware before any grid point was trusted.

Points with `status in {"aborted_shaping", "crashed"}` produce NO RunResult (CLAUDE.md
invariant). None occurred in this campaign's final state (one DDP point required a manual
recovery of an already-completed-but-late-arriving result -- see the ADR for the full
teardown-hang story; the underlying RunResult is unaffected, since the training itself
completed and reconciled normally).

Reads `run_ddp_localsgd_grid.py`'s output at `ddp_localsgd_run_logs/_summary.json` (next to
this file) by default; override with `DILOCO_DDP_LOCALSGD_SUMMARY`.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from diloco_measured.analysis.cu import analytic as cu_analytic
from diloco_measured.analysis.cu import measured as cu_measured

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = Path(
    os.environ.get(
        "DILOCO_DDP_LOCALSGD_SUMMARY",
        str(Path(__file__).resolve().parent / "ddp_localsgd_run_logs" / "_summary.json"),
    )
)
RAW_TELEMETRY_OUT_DIR = Path(__file__).resolve().parent / "raw_step_telemetry_ddp_localsgd"

# Relaunched cluster (2026-08-16), same instance types/AZ/AMI/bootstrap as ADR-035's shaped
# DiLoCo campaign -- only the placement group (and therefore instance IDs) changed. Confirmed
# live via `aws ec2 describe-instances`/`describe-placement-groups` this session, not assumed.
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
    "placement_group_id": "pg-0f8e8774a16b3b7e7",
    "gpu_clocks_locked": True,
    "nccl_env": {"NCCL_SOCKET_IFNAME": "enp39s0"},
    "kernel_version": "6.17.0-1019-aws",
    "dataset_shard_checksum": "e1a2eea5c5c8b7a0c0d17c760a8eacf0be6861828332efb6218919b9e70a141e",
    "seed": 42,
    "torchtitan_version": "0.2.2",
}

# Same real iperf3/NCCL measurements as ADR-034/aggregate_results.py -- reused for the DDP
# unshaped point (the only point in this campaign with no ShapingVerification), same
# rationale: this cluster's physical link characteristics (g6e.2xlarge/ENA in us-east-1b) are
# what was measured, not an instance-ID-specific property, so re-use across relaunches is
# representative, not a fabricated stand-in (CLAUDE.md §33.2.6 does not apply -- this is a
# real, previously-measured value about the same physical network class, reused deliberately
# and documented, not invented).
IPERF_PAIRS_GBIT_S = [
    9.530112832232550, 9.530236592782026, 9.530058498673240, 9.530009406517925,
    9.530048805505130, 9.529940456024270, 9.530022752802435, 9.530117438349810,
    9.530129986071130, 9.530076292195415, 9.529998283838156, 9.530011788966310,
]
LINK_BANDWIDTH_BPS = int((sum(IPERF_PAIRS_GBIT_S) / len(IPERF_PAIRS_GBIT_S)) * 1e9)
NCCL_CURVE = [
    (1048576, 8509005895.14442),
    (4194304, 15846303757.348486),
    (16777216, 14261874406.802505),
    (67108864, 14883281252.425484),
    (268435456, 14983550538.748339),
    (1073741824, 14829704862.217632),
]


def interpolate_nccl_bw(msg_bytes: int) -> tuple[float, bool]:
    points = sorted(NCCL_CURVE)
    if msg_bytes <= points[0][0]:
        return points[0][1], msg_bytes != points[0][0]
    if msg_bytes >= points[-1][0]:
        return points[-1][1], msg_bytes != points[-1][0]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x0 <= msg_bytes <= x1:
            if msg_bytes == x0:
                return y0, False
            if msg_bytes == x1:
                return y1, False
            frac = (msg_bytes - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0), True
    raise AssertionError("unreachable")


def build_experiment_spec(
    algorithm: str, H: int, bandwidth_bps: int | None,
    steps: int, warmup_steps: int, micro_batch_size: int, run_id: str,
) -> dict:
    return {
        "spec_id": run_id,
        "phase": "cu_grid",
        "algorithm": algorithm,
        "implementation": "reference",  # our own driver, not torchft -- US-06 still not run
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


def aggregate_one(point: dict, algorithm: str) -> dict:
    raw = point["raw"]
    H = raw["H"]
    n_params = raw["n_params"]
    steps = raw["steps"]
    warmup_steps = raw["warmup_steps"]
    step_records = raw["step_records"]
    sv = point.get("shaping_verification")  # None for the DDP unshaped point

    cu_m = cu_measured(step_records, warmup=warmup_steps)

    kept = step_records[warmup_steps:]
    total_compute_s = sum(r["compute_time_ms"] for r in kept) / 1000
    total_sync_s = sum(r["sync_blocked_ms"] for r in kept) / 1000
    total_optimizer_s = sum(r["optimizer_time_ms"] for r in kept) / 1000
    total_loader_s = sum(r["loader_stall_ms"] for r in kept) / 1000
    total_wall_s = sum(r["wall_time_ms"] for r in kept) / 1000
    mean_compute_s_per_step = total_compute_s / len(kept)

    bytes_synced = n_params * 4  # fp32, full tensor either way (pseudo-grad / raw params / grad)
    spec_for_cu = {"H": H}
    num_outer_syncs = sum(1 for r in kept if r["did_outer_step"])

    if sv is not None:
        # Shaped point: link = FR-02-verified rate; achieved = derived from this run's own
        # measured (DDP: imputed) sync time, per ADR-007/ADR-015 and aggregate_shaped_grid.py.
        cu_link = cu_analytic(
            spec_for_cu, t_compute_s=mean_compute_s_per_step,
            bytes_synced=bytes_synced, bandwidth_bps=int(sv["measured_bps"]),
        )
        if num_outer_syncs > 0 and total_sync_s > 0:
            mean_single_sync_s = total_sync_s / num_outer_syncs
            achieved_bw_bps = bytes_synced * 8 / mean_single_sync_s
        else:
            achieved_bw_bps = None
        nccl_interpolated = False
    else:
        # Unshaped point: link = real iperf3 mean; achieved = real NCCL curve, interpolated.
        cu_link = cu_analytic(
            spec_for_cu, t_compute_s=mean_compute_s_per_step,
            bytes_synced=bytes_synced, bandwidth_bps=LINK_BANDWIDTH_BPS,
        )
        achieved_bw_bps, nccl_interpolated = interpolate_nccl_bw(bytes_synced)

    cu_achieved = (
        cu_analytic(
            spec_for_cu, t_compute_s=mean_compute_s_per_step,
            bytes_synced=bytes_synced, bandwidth_bps=int(achieved_bw_bps),
        )
        if achieved_bw_bps is not None else None
    )

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
        "nccl_bw_interpolated": nccl_interpolated,
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

    shaping = None
    if sv is not None:
        shaping = {
            "requested_bps": sv["requested_bps"],
            "measured_bps": sv["measured_bps"],
            "error_pct": sv["error_pct"],
            "tolerance_pct": sv["tolerance_pct"],
            "passed": sv["passed"],
            "attempts": sv.get("attempts", 1),
            "iperf_raw": "",
            "qdisc_dump": "",
        }

    if algorithm == "ddp":
        note = (
            f"Real DDP training, 2026-08-16 grid (see CLAUDE.md ADR for this campaign). "
            f"H=1 by definition (syncs every step). IMPORTANT: cu.compute_s/cu.sync_blocked_s "
            f"here are an IMPUTED split, not two independent direct measurements -- DDP "
            f"overlaps its all-reduce with backward() by design, so `train_driver_ddp.py` uses "
            f"a calibration-probe methodology (median of 3 no_sync() compute-only samples "
            f"taken during warmup; per-step compute_ms=min(baseline,synced_backward_ms), "
            f"sync_ms=max(0,synced_backward_ms-baseline)). Two real Triton-JIT-contamination "
            f"bugs in this probe were found and fixed on real hardware before this grid ran "
            f"(see git history). Bandwidth level '{point['bandwidth_label']}'"
            + (
                f" requested={sv['requested_bps']}bps measured={sv['measured_bps']:.0f}bps "
                f"(FR-02 gate passed, error={sv['error_pct']:.2f}%)."
                if sv is not None else " (unshaped, no tc shaping applied)."
            )
            + (
                " NOTE: this point's ssh session hung after training completed (a real, "
                "unresolved teardown-hang mechanism reproducing at extreme bandwidth scarcity "
                "-- see force_cleanup_remote()'s docstring in run_ddp_localsgd_grid.py); the "
                "training itself finished normally and wrote a valid, complete result before "
                "the hang, which is what this record reflects."
                if point.get("note") else ""
            )
        )
    else:
        note = (
            f"Real LocalSGD training, 2026-08-16 grid (see CLAUDE.md ADR for this campaign). "
            f"No-outer-optimizer ablation against DiLoCo -- parameter averaging every H steps "
            f"(LocalSGDTrainer, measurement/localsgd.py), no pseudo-gradient/Nesterov outer "
            f"step. sync_blocked_ms is a DIRECT StepTimer measurement (explicit outer "
            f"all-reduce), not imputed. Bandwidth level '{point['bandwidth_label']}'"
            + (
                f" requested={sv['requested_bps']}bps measured={sv['measured_bps']:.0f}bps "
                f"(FR-02 gate passed, error={sv['error_pct']:.2f}%)."
                if sv is not None else "."
            )
        )

    return {
        "run_id": point["run_id"],
        "spec": build_experiment_spec(
            algorithm, H, point["bandwidth_requested_bps"], steps, warmup_steps,
            raw["micro_batch_size"], point["run_id"],
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
        "notes": note,
    }


def main() -> None:
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    FINGERPRINT["harness_git_sha"] = git_sha

    summary = json.loads(SUMMARY_PATH.read_text())
    out_dir = REPO_ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    RAW_TELEMETRY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_completed = n_aborted = n_other = 0
    grids = (("ddp", summary["ddp_points"]), ("localsgd", summary["localsgd_points"]))
    for algorithm, points in grids:
        for point in points:
            if point["status"] == "completed":
                result = aggregate_one(point, algorithm)
                out_path = out_dir / f"{result['run_id']}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
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
                sv = point["shaping_verification"]
                print(
                    f"{point['run_id']}: ABORTED_SHAPING (no RunResult written) -- "
                    f"requested={sv['requested_bps']} measured={sv['measured_bps']:.0f} "
                    f"error={sv['error_pct']:.1f}%"
                )
                n_aborted += 1
            else:
                print(f"{point['run_id']}: status={point['status']} (no RunResult written)")
                n_other += 1

    print(f"\n{n_completed} completed, {n_aborted} aborted_shaping, {n_other} other")


if __name__ == "__main__":
    main()
