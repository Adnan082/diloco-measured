"""G3: "≥10 completed convergence runs with TTTL against a single-GPU reference" (CLAUDE.md
§4.1). Runs a single-GPU reference (defines L*, ADR-021/FR-06 §40 Q5: L* = the reference's
final loss at the same token budget) followed by a DiLoCo grid across `H` and bandwidth, all
to the SAME token budget -- then `aggregate_convergence.py` (this directory) computes TTTL per
config from each run's own loss-vs-wall-clock curve.

Reuses `experiments/01_cu_grid/train_driver.py` UNCHANGED for both the reference and the
DiLoCo runs -- no new training driver was written. The single-GPU reference is produced by
setting `--H` far larger than the run's total step count: `DiLoCoTrainer.ready_for_outer_step()`
returns `_h_count >= H`, so with H unreachably large it never fires, and `inner_step()` alone
(plain per-step AdamW on the live model, zero cross-replica communication) is exactly a
standard single-GPU training loop -- see this module's own investigation, not asserted.

Cluster config from `DILOCO_NODES`/`DILOCO_SSH_KEY` env vars, same convention as
`run_shaped_grid.py` (see that module's docstring for why -- CLAUDE.md §23 private-IP
discipline applies to any committed file).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from diloco_measured.measurement import netshape  # noqa: E402

KEY_FILE = os.environ.get("DILOCO_SSH_KEY", os.path.expanduser("~/.ssh/diloco-measured-key.pem"))
OUT_DIR = Path(__file__).resolve().parent / "convergence_run_logs"
OUT_DIR.mkdir(exist_ok=True)

TOKEN_BUDGET = 400_000
SEQ_LEN = 512
MICRO_BATCH_SIZE = 4
REFERENCE_H = 999_999  # unreachably large -- ready_for_outer_step() never fires (see docstring)
REFERENCE_WARMUP = 10
DILOCO_WARMUP = 5

# DiLoCo grid: same H values as the CU grid (ADR-035), 3 of its 4 bandwidth levels (skipping
# 50m to keep total campaign wall-time bounded -- H=1 at 50m alone would need ~49 steps *
# ~20s/step of sync time, see run_shaped_grid.py's own per-step-time comments).
DILOCO_BANDWIDTHS = [
    ("unshaped", None),
    ("1g", 1_000_000_000),
    ("200m", 200_000_000),
]
H_VALUES = [1, 8, 32, 128]


def _load_nodes() -> list[netshape.Node]:
    raw = os.environ.get("DILOCO_NODES", "")
    if not raw:
        raise SystemExit("DILOCO_NODES not set -- see run_shaped_grid.py's docstring for format.")
    nodes = []
    for pair in raw.split(","):
        public_ip, private_ip = pair.strip().split(":")
        nodes.append(netshape.Node(host=public_ip, private_ip=private_ip, ssh_key_file=KEY_FILE))
    if len(nodes) != 4:
        raise SystemExit(f"DILOCO_NODES must list exactly 4 nodes, got {len(nodes)}")
    return nodes


NODES = _load_nodes()
RDZV_ENDPOINT = f"{NODES[0].private_ip}:29500"
REMOTE_OUTPUT = "/home/ubuntu/diloco-measured/convergence_out.json"
REMOTE_DATA_DIR = "/opt/dlami/nvme/dataset"


def _ssh_argv(host: str, remote_cmd: str) -> list[str]:
    return [
        "ssh", "-i", KEY_FILE, "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", f"ubuntu@{host}", remote_cmd,
    ]


def _clean_remote_output(nodes: list[netshape.Node]) -> None:
    for node in nodes:
        netshape.ssh_run(node, ["rm", "-f", REMOTE_OUTPUT], timeout_s=10)


def run_reference() -> dict:
    """Single-node, single-GPU, standalone torchrun (no multi-node rendezvous needed)."""
    steps = TOKEN_BUDGET // (MICRO_BATCH_SIZE * SEQ_LEN)
    run_id = "convergence-reference-30m-singlegpu"
    print(f"\n=== {run_id} (steps={steps}, world_size=1) ===")
    node = NODES[0]
    _clean_remote_output([node])

    remote_cmd = (
        "cd diloco-measured && source $HOME/.local/bin/env && "
        f".venv/bin/torchrun --standalone --nproc-per-node=1 "
        f"experiments/01_cu_grid/train_driver.py --H {REFERENCE_H} --steps {steps} "
        f"--warmup-steps {REFERENCE_WARMUP} --micro-batch-size {MICRO_BATCH_SIZE} "
        f"--run-id {run_id} --output {REMOTE_OUTPUT} "
        f"--data-path {REMOTE_DATA_DIR}/shard_0000.npy"
    )
    log_path = OUT_DIR / f"{run_id}.log"
    t_start = time.time()
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            _ssh_argv(node.host, remote_cmd), stdout=log_f, stderr=subprocess.STDOUT
        )
    print(f"  wall time: {time.time() - t_start:.1f}s, ssh exit={result.returncode}")

    local_path = OUT_DIR / f"{run_id}.json"
    scp_argv = [
        "scp", "-i", KEY_FILE, "-o", "StrictHostKeyChecking=accept-new",
        f"ubuntu@{node.host}:{REMOTE_OUTPUT}", str(local_path),
    ]
    scp_result = subprocess.run(scp_argv, capture_output=True, text=True, timeout=30)
    if scp_result.returncode != 0 or not local_path.exists():
        raise RuntimeError(f"reference run produced no output file: {scp_result.stderr}")
    return json.loads(local_path.read_text())


def launch_training(H: int, steps: int, warmup: int, run_id: str) -> list[subprocess.Popen]:
    procs = []
    for i, node in enumerate(NODES):
        data_path = f"{REMOTE_DATA_DIR}/shard_{i:04d}.npy"
        remote_cmd = (
            "cd diloco-measured && source $HOME/.local/bin/env && "
            f".venv/bin/torchrun --nnodes=4 --nproc-per-node=1 --rdzv-backend=c10d "
            f"--rdzv-endpoint={RDZV_ENDPOINT} --rdzv-id={run_id} "
            f"experiments/01_cu_grid/train_driver.py --H {H} --steps {steps} "
            f"--warmup-steps {warmup} --micro-batch-size {MICRO_BATCH_SIZE} "
            f"--run-id {run_id} --output {REMOTE_OUTPUT} --data-path {data_path}"
        )
        log_path = OUT_DIR / f"{run_id}_node{i}.log"
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            _ssh_argv(node.host, remote_cmd), stdout=log_f, stderr=subprocess.STDOUT
        )
        procs.append(proc)
    return procs


def fetch_result(run_id: str) -> dict | None:
    for i, node in enumerate(NODES):
        local_path = OUT_DIR / f"{run_id}.json"
        scp_argv = [
            "scp", "-i", KEY_FILE, "-o", "StrictHostKeyChecking=accept-new",
            f"ubuntu@{node.host}:{REMOTE_OUTPUT}", str(local_path),
        ]
        result = subprocess.run(scp_argv, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and local_path.exists():
            print(f"  fetched result from node{i} ({node.host})")
            return json.loads(local_path.read_text())
    return None


def run_diloco_point(H: int, bw_label: str, bw_bps: int | None) -> dict:
    run_id = f"convergence-diloco-30m-h{H}-bw{bw_label}"
    steps = TOKEN_BUDGET // (MICRO_BATCH_SIZE * SEQ_LEN * len(NODES))
    print(f"\n=== {run_id} (steps={steps}) ===")

    handle = netshape.apply(bw_bps, NODES) if bw_bps is not None else None
    verification: netshape.ShapingVerification | None = None
    try:
        if handle is not None:
            for attempt in range(2):
                verification = netshape.verify(handle, tolerance_pct=10.0, duration_s=15)
                print(
                    f"  shaping verify attempt {attempt+1}: requested={bw_bps} "
                    f"measured={verification.measured_bps:.0f} "
                    f"error={verification.error_pct:.1f}% passed={verification.passed}"
                )
                if verification.passed:
                    break
            assert verification is not None  # range(2) always executes >=1 iteration
            if not verification.passed:
                print(f"  ABORT {run_id}: shaping verification failed after retry")
                return {
                    "run_id": run_id, "status": "aborted_shaping", "H": H,
                    "bandwidth_label": bw_label, "bandwidth_requested_bps": bw_bps,
                }

        _clean_remote_output(NODES)
        t_start = time.time()
        procs = launch_training(H, steps, DILOCO_WARMUP, run_id)
        for i, proc in enumerate(procs):
            rc = proc.wait(timeout=900)
            if rc != 0:
                print(f"  WARNING: node{i} ssh/torchrun exited {rc} (see log)")
        print(f"  training wall time: {time.time() - t_start:.1f}s")

        raw = fetch_result(run_id)
        if raw is None:
            print(f"  ABORT {run_id}: no node produced an output file (crashed?)")
            return {
                "run_id": run_id, "status": "crashed", "H": H,
                "bandwidth_label": bw_label, "bandwidth_requested_bps": bw_bps,
            }

        result: dict = {
            "run_id": run_id, "status": "completed", "H": H,
            "bandwidth_label": bw_label, "bandwidth_requested_bps": bw_bps, "raw": raw,
        }
        if verification is not None:
            result["shaping_verification"] = {
                "requested_bps": verification.requested_bps,
                "measured_bps": verification.measured_bps,
                "error_pct": verification.error_pct,
                "tolerance_pct": verification.tolerance_pct,
                "passed": verification.passed,
            }
        return result
    finally:
        if handle is not None:
            netshape.restore(handle)
            print(f"  shaping restored for {run_id}")


def main() -> None:
    summary: dict = {"reference": None, "diloco_points": []}
    summary_path = OUT_DIR / "_summary.json"

    summary["reference"] = run_reference()
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"reference final loss: {summary['reference']['step_records'][-1]['loss']:.4f}")

    for H in H_VALUES:
        for bw_label, bw_bps in DILOCO_BANDWIDTHS:
            point = run_diloco_point(H, bw_label, bw_bps)
            summary["diloco_points"].append(point)
            summary_path.write_text(json.dumps(summary, indent=2))

    completed = sum(1 for p in summary["diloco_points"] if p["status"] == "completed")
    total = len(summary["diloco_points"])
    print(f"\n=== CONVERGENCE CAMPAIGN DONE: {completed}/{total} DiLoCo points completed ===")


if __name__ == "__main__":
    main()
