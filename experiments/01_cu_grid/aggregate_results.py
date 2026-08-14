"""Aggregate the 4 real H-sweep training runs (`raw_step_telemetry/result_h*.json`, produced
by `train_driver.py`) into schema-valid `RunResult` records under `results/raw/`.

This is analysis-layer work (pure arithmetic over already-measured StepRecord-shaped data,
CLAUDE.md §11.2/§14.2) even though it lives under `experiments/` rather than `analysis/` —
it belongs here because it is a one-off aggregation of one specific campaign's raw telemetry
into the canonical record shape, not a reusable analysis-layer function. Rerunning this script
overwrites the four `results/raw/cu_grid-diloco-30m-h{1,8,32,128}-bwunshaped-r0.json` files
with the same numbers (deterministic given the same inputs) — it does not touch anything else
in `results/raw/`.

Run from the repository root with the project's venv (needs `diloco_measured` importable):
    python experiments/01_cu_grid/aggregate_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

from diloco_measured.analysis.cu import analytic as cu_analytic
from diloco_measured.analysis.cu import measured as cu_measured

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_TELEMETRY_DIR = Path(__file__).resolve().parent / "raw_step_telemetry"

# Real fingerprint captured via measurement/fingerprint.py::capture() on the actual GPU node
# that was global rank 0 for every run in this sweep, 2026-08-14. Scrubbed per CLAUDE.md §23
# (no account IDs, ARNs, or private/public IPs) before being hardcoded here.
FINGERPRINT = {
    "harness_git_sha": "bfd808ec075e8595b97076ec21526970a32efaf3",
    "harness_dirty": True,
    "harness_version": "0.1.0",
    "pytorch_version": "2.13.0+cu130",
    "nccl_version": "2.29.7",
    "cuda_version": "13.0",
    "driver_version": "595.71.05",
    "instance_types": ["g6e.2xlarge"],
    "az": "us-east-1b",
    "placement_group_id": "pg-04ac04963de1615d8",
    "gpu_clocks_locked": True,
    "nccl_env": {"NCCL_SOCKET_IFNAME": "enp39s0"},
    "kernel_version": "6.17.0-1019-aws",
    "dataset_shard_checksum": "a7b9db02a2f15e720016ec1ccc9c1ed300ace2a3680471c8e86c53f40143a90c",
    "seed": 42,
    "torchtitan_version": "0.2.2",
}

# Real iperf3 all-pairs mean (results/network/phase1-us-east-1b-20260814.json), used as the
# "link bandwidth" input for cu_analytic_link (ADR-007/ADR-015: the papers' naive assumption
# -- nominal achievable point-to-point bandwidth, not the optimized NCCL collective).
IPERF_PAIRS_GBIT_S = [
    9.530112832232550, 9.530236592782026, 9.530058498673240, 9.530009406517925,
    9.530048805505130, 9.529940456024270, 9.530022752802435, 9.530117438349810,
    9.530129986071130, 9.530076292195415, 9.529998283838156, 9.530011788966310,
]
LINK_BANDWIDTH_BPS = int((sum(IPERF_PAIRS_GBIT_S) / len(IPERF_PAIRS_GBIT_S)) * 1e9)

# Real NCCL curve (same NetworkProfile), used to interpolate the ACHIEVED bandwidth at the
# actual pseudo-gradient message size (ADR-007: separates "the model is wrong" from "the
# model's input assumption is wrong").
NCCL_CURVE = [
    (1048576, 8509005895.14442),
    (4194304, 15846303757.348486),
    (16777216, 14261874406.802505),
    (67108864, 14883281252.425484),
    (268435456, 14983550538.748339),
    (1073741824, 14829704862.217632),
]


def interpolate_nccl_bw(msg_bytes: int) -> tuple[float, bool]:
    """Linear interpolation between the two nearest measured NCCL curve points.
    Returns (achieved_bps, was_interpolated).
    """
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


def build_experiment_spec(H: int, steps: int, warmup_steps: int, micro_batch_size: int) -> dict:
    return {
        "spec_id": f"cu_grid-diloco-30m-h{H}-bwunshaped-r0",
        "phase": "cu_grid",
        "algorithm": "diloco",
        "implementation": "reference",
        "H": H,
        "model_config": "30m-realvocab",
        "world_size": 4,
        "bandwidth_requested_bps": None,
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


def aggregate_one(raw_path: Path) -> dict:
    with open(raw_path) as f:
        raw = json.load(f)

    H = raw["H"]
    n_params = raw["n_params"]
    steps = raw["steps"]
    warmup_steps = raw["warmup_steps"]
    step_records = raw["step_records"]

    # --- CU measured (real, from StepTimer data, warmup discarded) ---
    cu_m = cu_measured(step_records, warmup=warmup_steps)

    kept = step_records[warmup_steps:]
    total_compute_s = sum(r["compute_time_ms"] for r in kept) / 1000
    total_sync_s = sum(r["sync_blocked_ms"] for r in kept) / 1000
    total_optimizer_s = sum(r["optimizer_time_ms"] for r in kept) / 1000
    total_loader_s = sum(r["loader_stall_ms"] for r in kept) / 1000
    total_wall_s = sum(r["wall_time_ms"] for r in kept) / 1000
    mean_compute_s_per_step = total_compute_s / len(kept)

    # --- CU analytic (both variants, one shared code path per FR-04) ---
    bytes_synced = n_params * 4  # fp32 pseudo-gradient, full tensor (methods/wire_model.md §2)
    spec_for_cu = {"H": H}

    cu_link = cu_analytic(
        spec_for_cu, t_compute_s=mean_compute_s_per_step,
        bytes_synced=bytes_synced, bandwidth_bps=LINK_BANDWIDTH_BPS,
    )
    achieved_bw_bps, was_interpolated = interpolate_nccl_bw(bytes_synced)
    cu_achieved = cu_analytic(
        spec_for_cu, t_compute_s=mean_compute_s_per_step,
        bytes_synced=bytes_synced, bandwidth_bps=int(achieved_bw_bps),
    )

    reconciliation_residual_pct = (
        abs(total_wall_s - (total_compute_s + total_sync_s + total_optimizer_s + total_loader_s))
        / total_wall_s * 100
        if total_wall_s > 0 else 0.0
    )

    cu_observation = {
        "cu_measured": cu_m,
        "cu_analytic_link": cu_link,
        "cu_analytic_achieved": cu_achieved,
        "nccl_bw_used_bps": int(achieved_bw_bps),
        "nccl_bw_interpolated": was_interpolated,
        "discrepancy_link": cu_link / cu_m,
        "discrepancy_achieved": cu_achieved / cu_m,
        "compute_s": total_compute_s,
        "sync_blocked_s": total_sync_s,
        "optimizer_s": total_optimizer_s,
        "loader_stall_s": total_loader_s,
        "total_s": total_wall_s,
    }
    assert reconciliation_residual_pct < 5.0, (
        f"reconciliation residual {reconciliation_residual_pct:.2f}% exceeds 5% tolerance "
        f"(CLAUDE.md §15.2 CUObservation invariant) for {raw_path}"
    )

    # --- throughput ---
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

    return {
        "run_id": raw["run_id"],
        "spec": build_experiment_spec(H, steps, warmup_steps, raw["micro_batch_size"]),
        "fingerprint": FINGERPRINT,
        "shaping": None,
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
            f"First real training runs, 2026-08-14. Model: torchtitan debugmodel dims "
            f"(dim=256,n_layers=6,n_heads=16) with vocab_size overridden to 50257 (gpt2) -- "
            f"{n_params} real params. Data: real FineWeb-Edu (sample-10BT), gpt2-tokenized, "
            f"one disjoint shard per rank. Reference DiLoCoTrainer (measurement/diloco.py), "
            f"NOT torchft's DiLoCo path (US-06 equivalence test still not run -- see ADR-032). "
            f"Unshaped baseline (no tc shaping applied). Run via train_driver.py directly "
            f"(torchrun), not through measurement/train.py::run()'s full orchestration -- "
            f"see ADR-034 for exactly what that means and doesn't mean."
        ),
    }


def main() -> None:
    out_dir = REPO_ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    for h_val, fname in [
        (1, "result_h1.json"), (8, "result_h8.json"),
        (32, "result_h32.json"), (128, "result_h128.json"),
    ]:
        raw_path = RAW_TELEMETRY_DIR / fname
        result = aggregate_one(raw_path)
        out_path = out_dir / f"{result['run_id']}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        cu = result["cu"]
        print(
            f"H={h_val}: cu_measured={cu['cu_measured']:.4f} "
            f"cu_analytic_link={cu['cu_analytic_link']:.4f} "
            f"cu_analytic_achieved={cu['cu_analytic_achieved']:.4f} "
            f"-> wrote {out_path}"
        )


if __name__ == "__main__":
    main()
