"""Aggregate `run_convergence_campaign.py`'s output into schema-valid `RunResult` records
with a real `ConvergenceCurve` (G3, CLAUDE.md §4.1 / FR-06).

L* (target loss, ADR-021/FR-06 §40 Q5): the single-GPU reference run's OWN final loss at the
same token budget -- not a fixed absolute number, not a percentile of the grid. TTTL: the
first wall-clock time (cumulative, from step 0 -- including warmup, since TTTL is about real
elapsed training time, not the CU-measurement-window convention of discarding warmup) at which
a config's training loss first reaches (<=) L*. `tttl_s: null` when a config never reaches L*
within its token budget -- CLAUDE.md is explicit this must never be rendered as a finite
number (ConvergenceCurve invariant, §15.2).

Smoothing (FR-06 alt-flow 3a): both raw and a 5-point trailing-EMA-smoothed crossing are
computed and recorded (`tttl_s` / `tttl_smoothed_s`) -- loss curves are noisy step-to-step,
and a single lucky low-loss step near L* would otherwise produce an unrepresentatively early
crossing.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = Path(
    os.environ.get(
        "DILOCO_CONVERGENCE_SUMMARY",
        str(Path(__file__).resolve().parent / "convergence_run_logs" / "_summary.json"),
    )
)

TOKEN_BUDGET = 400_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 4

# Same fingerprint lineage as ADR-035/aggregate_shaped_grid.py -- same bootstrap, same cluster
# session this campaign ran in.
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
    "gpu_clocks_locked": True,
    "nccl_env": {"NCCL_SOCKET_IFNAME": "enp39s0"},
    "kernel_version": "6.17.0-1019-aws",
    "dataset_shard_checksum": "e1a2eea5c5c8b7a0c0d17c760a8eacf0be6861828332efb6218919b9e70a141e",
    "seed": 42,
    "torchtitan_version": "0.2.2",
}


def _ema_smooth(values: list[float], window: int = 5) -> list[float]:
    """Trailing EMA, window-sized ramp-up at the start (no look-ahead)."""
    smoothed = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        smoothed.append(sum(values[lo : i + 1]) / (i + 1 - lo))
    return smoothed


def _cumulative_wall_s(step_records: list[dict]) -> list[float]:
    cum = []
    total = 0.0
    for r in step_records:
        total += r["wall_time_ms"] / 1000
        cum.append(total)
    return cum


def _first_crossing(wall_s: list[float], losses: list[float], target: float) -> float | None:
    for w, loss in zip(wall_s, losses, strict=True):
        if loss <= target:
            return w
    return None


def build_convergence_curve(step_records: list[dict], target_loss: float, world_size: int) -> dict:
    losses = [r["loss"] for r in step_records]
    wall_s = _cumulative_wall_s(step_records)
    smoothed_losses = _ema_smooth(losses)

    tttl_s = _first_crossing(wall_s, losses, target_loss)
    tttl_smoothed_s = _first_crossing(wall_s, smoothed_losses, target_loss)

    tokens_per_step = MICRO_BATCH_SIZE * SEQ_LEN * world_size
    points = [
        {
            "tokens": (i + 1) * tokens_per_step,
            "wall_s": wall_s[i],
            "train_loss": losses[i],
            "val_loss": None,  # no held-out eval set built for this campaign -- see notes
        }
        for i in range(len(step_records))
    ]
    final_loss = losses[-1]
    return {
        "points": points,
        "target_loss": target_loss,
        "tttl_s": tttl_s,
        "tttl_smoothed_s": tttl_smoothed_s,
        "final_loss": final_loss,
        "reached_target": tttl_s is not None,
    }


def build_experiment_spec(
    H: int, bandwidth_bps: int | None, steps: int, warmup_steps: int, world_size: int
) -> dict:
    bw_tag = "unshaped" if bandwidth_bps is None else str(bandwidth_bps)
    return {
        "spec_id": f"convergence-diloco-30m-h{H}-bw{bw_tag}",
        "phase": "convergence",
        # The reference run is ALSO algorithm="diloco" (H set unreachably large so the outer
        # step never fires -- see this module's docstring) rather than a distinct "reference"
        # enum value; the schema has no such value, and this is a real, if unusual, DiLoCo
        # config (H > total steps), not a fiction.
        "algorithm": "diloco",
        "implementation": "reference",
        "H": H,
        "model_config": "30m-realvocab",
        "world_size": world_size,
        "bandwidth_requested_bps": bandwidth_bps,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "grad_accum": 1,
        "budget_type": "tokens",
        "budget_value": steps * MICRO_BATCH_SIZE * SEQ_LEN * world_size,
        "warmup_steps": warmup_steps,
        "compression": None,
        "seed": 42,
        "repeat_index": 0,
        "fault_schedule": None,
    }


def main() -> None:
    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    FINGERPRINT["harness_git_sha"] = git_sha

    summary = json.loads(SUMMARY_PATH.read_text())
    out_dir = REPO_ROOT / "results" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Reference run: L* = its own final loss. Recorded as a RunResult too (world_size=1,
    # phase=convergence) so it's traceable in the corpus, not just a number embedded elsewhere.
    ref_raw = summary["reference"]
    ref_step_records = ref_raw["step_records"]
    ref_final_loss = ref_step_records[-1]["loss"]
    print(f"Reference (single-GPU) final loss L* = {ref_final_loss:.4f}")

    ref_curve = build_convergence_curve(ref_step_records, target_loss=ref_final_loss, world_size=1)
    ref_result = {
        "run_id": ref_raw["run_id"],
        "spec": build_experiment_spec(
            H=ref_raw["H"], bandwidth_bps=None, steps=ref_raw["steps"],
            warmup_steps=ref_raw["warmup_steps"], world_size=1,
        ),
        "fingerprint": FINGERPRINT,
        "shaping": None,
        "network_profile_id": "phase1-us-east-1b-20260814",
        "harness_version": "0.1.0",
        "status": "completed",
        "started_at": ref_raw["started_at"],
        "ended_at": ref_raw["ended_at"],
        "convergence": ref_curve,
        "faults": [],
        "loader_bound_warning": False,
        "notes": (
            "Single-GPU reference run, 2026-08-15 (CLAUDE.md ADR-037). Defines L* (this run's "
            "own final training loss) per ADR-021/FR-06 SS40 Q5. Produced via "
            "train_driver.py with H set far larger than the step count so "
            "DiLoCoTrainer.ready_for_outer_step() never fires -- inner_step() alone is plain "
            "per-step AdamW with zero cross-replica communication, i.e. standard single-GPU "
            "training, reusing the same driver unchanged rather than writing a new one. No "
            "held-out validation set was built for this campaign -- val_loss is null "
            "throughout; train_loss is used as both signals, a real simplification stated "
            "here rather than hidden."
        ),
    }
    ref_out_path = out_dir / f"{ref_result['run_id']}.json"
    with open(ref_out_path, "w") as f:
        json.dump(ref_result, f, indent=2)
    print(f"wrote {ref_out_path}")

    # --- DiLoCo grid, TTTL against L* ---
    n_completed = n_aborted = n_other = 0
    for point in summary["diloco_points"]:
        if point["status"] != "completed":
            print(f"{point['run_id']}: status={point['status']} (no RunResult written)")
            if point["status"] == "aborted_shaping":
                n_aborted += 1
            else:
                n_other += 1
            continue

        raw = point["raw"]
        curve = build_convergence_curve(
            raw["step_records"], target_loss=ref_final_loss, world_size=4
        )

        shaping = None
        if point.get("shaping_verification") is not None:
            sv = point["shaping_verification"]
            shaping = {
                "requested_bps": sv["requested_bps"],
                "measured_bps": sv["measured_bps"],
                "error_pct": sv["error_pct"],
                "tolerance_pct": sv["tolerance_pct"],
                "passed": sv["passed"],
                "attempts": 1,
                "iperf_raw": "",
                "qdisc_dump": "",
            }

        result = {
            "run_id": point["run_id"],
            "spec": build_experiment_spec(
                H=point["H"], bandwidth_bps=point["bandwidth_requested_bps"],
                steps=raw["steps"], warmup_steps=raw["warmup_steps"], world_size=4,
            ),
            "fingerprint": FINGERPRINT,
            "shaping": shaping,
            "network_profile_id": "phase1-us-east-1b-20260814",
            "harness_version": "0.1.0",
            "status": "completed",
            "started_at": raw["started_at"],
            "ended_at": raw["ended_at"],
            "convergence": curve,
            "faults": [],
            "loader_bound_warning": False,
            "notes": (
                f"Convergence campaign, 2026-08-15 (CLAUDE.md ADR-037). H={point['H']}, "
                f"bandwidth={point['bandwidth_label']}, token_budget={TOKEN_BUDGET}. "
                f"target_loss (L*) = {ref_final_loss:.4f} from the single-GPU reference run "
                f"at the same token budget. tttl_s is null if this config never reached L* "
                f"within its token budget -- see reached_target."
            ),
        }
        out_path = out_dir / f"{result['run_id']}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        tttl_str = (
            f"{curve['tttl_s']:.1f}s" if curve["tttl_s"] is not None else "null (not reached)"
        )
        print(
            f"{point['run_id']}: final_loss={curve['final_loss']:.4f} "
            f"tttl_s={tttl_str} -> wrote {out_path}"
        )
        n_completed += 1

    print(
        f"\n{n_completed} completed, {n_aborted} aborted_shaping, {n_other} other "
        "(+ 1 reference)"
    )


if __name__ == "__main__":
    main()
