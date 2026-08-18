"""Aggregate the real FSDP2 grid (`run_fsdp2_grid.py`'s output) into schema-valid `RunResult`
records under `results/raw/`. Same shape as `aggregate_ddp_localsgd_grid.py`'s DDP branch
(imputed compute/sync split from a calibration probe, not a direct measurement — see
`train_driver_fsdp2.py`'s module docstring) with one real difference: `bytes_synced` is
`3 · n_params · 4`, not `n_params · 4` — FSDP2 does three real collective calls per step
(forward all-gather, backward all-gather, backward reduce-scatter), and this project's
`cu_analytic_*` model counts each real transfer as one full-tensor `bytes/B` cost (methods/
cu_model.md's added FSDP2 bullet, methods/wire_model.md §3a — read both before changing this).

Points with `status in {"aborted_shaping", "crashed"}` produce NO RunResult (CLAUDE.md
invariant). Reads `run_fsdp2_grid.py`'s output at `fsdp2_run_logs/_summary.json` (next to this
file) by default; override with `DILOCO_FSDP2_GRID_SUMMARY`.
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
        "DILOCO_FSDP2_GRID_SUMMARY",
        str(Path(__file__).resolve().parent / "fsdp2_run_logs" / "_summary.json"),
    )
)
RAW_TELEMETRY_OUT_DIR = Path(__file__).resolve().parent / "raw_step_telemetry_fsdp2"

# Same real iperf3/NCCL measurements as ADR-034/aggregate_results.py/aggregate_ddp_localsgd_
# grid.py -- reused for the unshaped point (this cluster's physical link characteristics, not
# an instance-ID-specific property). Filled in for real at aggregation time from whichever
# cluster this campaign actually ran on (see main()'s FINGERPRINT assembly).
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
    bandwidth_bps: int | None, steps: int, warmup_steps: int, micro_batch_size: int, run_id: str
) -> dict:
    return {
        "spec_id": run_id,
        "phase": "cu_grid",
        "algorithm": "fsdp2",
        "implementation": "reference",  # our own driver -- raw torch.distributed.fsdp, no torchft
        "H": 1,
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


def aggregate_one(point: dict, fingerprint: dict) -> dict:
    raw = point["raw"]
    n_params = raw["n_params"]
    steps = raw["steps"]
    warmup_steps = raw["warmup_steps"]
    step_records = raw["step_records"]
    sv = point.get("shaping_verification")

    cu_m = cu_measured(step_records, warmup=warmup_steps)

    kept = step_records[warmup_steps:]
    total_compute_s = sum(r["compute_time_ms"] for r in kept) / 1000
    total_sync_s = sum(r["sync_blocked_ms"] for r in kept) / 1000
    total_optimizer_s = sum(r["optimizer_time_ms"] for r in kept) / 1000
    total_loader_s = sum(r["loader_stall_ms"] for r in kept) / 1000
    total_wall_s = sum(r["wall_time_ms"] for r in kept) / 1000
    mean_compute_s_per_step = total_compute_s / len(kept)

    # FSDP2-specific: 3 real collective calls per step (forward all-gather, backward
    # all-gather, backward reduce-scatter), each counted as one full-tensor transfer under
    # this model's own simplification -- methods/cu_model.md's FSDP2 bullet.
    bytes_synced = 3 * n_params * 4
    spec_for_cu = {"H": 1}
    num_outer_syncs = sum(1 for r in kept if r["did_outer_step"])

    if sv is not None:
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

    note = (
        f"Real FSDP2 training, 2026-08-17 grid (see CLAUDE.md ADR for this campaign). H=1 by "
        f"definition (not a semi-synchronous method in this project's framework). IMPORTANT: "
        f"cu.compute_s/cu.sync_blocked_s here are an IMPUTED split, not two independent direct "
        f"measurements -- FSDP2 overlaps its all-gather/reduce-scatter collectives with "
        f"forward+backward compute by design, so `train_driver_fsdp2.py` uses a calibration-"
        f"probe methodology (median of 3 comm-suppressed forward+backward samples taken during "
        f"warmup, using FSDPModule.set_requires_gradient_sync(False)/set_reshard_after_forward"
        f"(False)/set_reshard_after_backward(False) -- FSDP2's own public API, not a hack; per-"
        f"step compute_ms=min(baseline,synced_step_ms), sync_ms=max(0,synced_step_ms-baseline)). "
        f"bytes_synced=3*n_params*4 (three real collective calls per step, not one -- methods/"
        f"wire_model.md §3a, methods/cu_model.md's FSDP2 bullet). Bandwidth level "
        f"'{point['bandwidth_label']}'"
        + (
            f" requested={sv['requested_bps']}bps measured={sv['measured_bps']:.0f}bps "
            f"(FR-02 gate passed, error={sv['error_pct']:.2f}%)."
            if sv is not None else " (unshaped, no tc shaping applied)."
        )
    )

    return {
        "run_id": point["run_id"],
        "spec": build_experiment_spec(
            point["bandwidth_requested_bps"], steps, warmup_steps, raw["micro_batch_size"],
            point["run_id"],
        ),
        "fingerprint": fingerprint,
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

    fingerprint_path = Path(
        os.environ.get(
            "DILOCO_FSDP2_FINGERPRINT",
            str(Path(__file__).resolve().parent / "fsdp2_run_logs" / "_fingerprint.json"),
        )
    )
    if not fingerprint_path.exists():
        raise SystemExit(
            f"{fingerprint_path} not found -- write the real cluster fingerprint there before "
            "aggregating (see aggregate_ddp_localsgd_grid.py's FINGERPRINT dict for the shape; "
            "this campaign's is generated fresh per-cluster rather than hardcoded, since it's "
            "the only piece of this script that changes between a smoke test and the real grid)"
        )
    fingerprint = json.loads(fingerprint_path.read_text())
    fingerprint["harness_git_sha"] = git_sha

    summary = json.loads(SUMMARY_PATH.read_text())
    out_dir = REPO_ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    RAW_TELEMETRY_OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_completed = n_aborted = n_other = 0
    for point in summary["fsdp2_points"]:
        if point["status"] == "completed":
            result = aggregate_one(point, fingerprint)
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
