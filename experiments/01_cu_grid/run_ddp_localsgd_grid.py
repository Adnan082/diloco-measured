"""Orchestrate the real DDP and LocalSGD grids -- extends the CU-grid measurement path beyond
DiLoCo-only (CLAUDE.md ADR-035/037 covered DiLoCo; this covers the other two algorithms this
project has a real training driver for). Same FR-02 apply->verify->(retry once)->run->restore
sequence as `run_shaped_grid.py`, same `DILOCO_NODES`/`DILOCO_SSH_KEY` env-var convention
(CLAUDE.md §23 private-IP discipline -- see that module's docstring for why).

DDP grid: bandwidth only (H is always 1 by definition, CLAUDE.md ExperimentSpec invariant) --
5 points (unshaped + the same 4 shaped levels DiLoCo/LocalSGD use).
LocalSGD grid: the SAME H x bandwidth grid as DiLoCo's shaped campaign (ADR-035) -- 16 points,
same step/warmup table, for direct comparability.
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
OUT_DIR = Path(__file__).resolve().parent / "ddp_localsgd_run_logs"
OUT_DIR.mkdir(exist_ok=True)


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
REMOTE_OUTPUT = "/home/ubuntu/diloco-measured/grid_out.json"
REMOTE_DATA_DIR = "/opt/dlami/nvme/dataset"

# DDP: bandwidth only, H=1 always. Step counts sized the same way as DiLoCo's H=1 sweep
# (run_shaped_grid.py) since DDP's per-step all-reduce moves the same bytes (full gradient,
# same tensor size as DiLoCo's pseudo-gradient) every step -- same sync-cost-vs-bandwidth
# shape expected. DDP's driver requires warmup_steps >= 5 (1 JIT-warmup + 3 calibration + 1).
DDP_GRID = [
    ("unshaped", None, 100, 10),
    ("5g", 5_000_000_000, 100, 10),
    ("1g", 1_000_000_000, 100, 10),
    ("200m", 200_000_000, 35, 8),
    ("50m", 50_000_000, 15, 6),
]

# LocalSGD: same H x bandwidth grid as DiLoCo's shaped campaign (ADR-035), same step/warmup
# table -- direct comparability is the point.
LOCALSGD_GRID = [
    (1, "5g", 5_000_000_000, 100, 10),
    (1, "1g", 1_000_000_000, 100, 10),
    (1, "200m", 200_000_000, 35, 5),
    (1, "50m", 50_000_000, 12, 2),
    (8, "5g", 5_000_000_000, 150, 10),
    (8, "1g", 1_000_000_000, 150, 10),
    (8, "200m", 200_000_000, 150, 10),
    (8, "50m", 50_000_000, 80, 8),
    (32, "5g", 5_000_000_000, 150, 10),
    (32, "1g", 1_000_000_000, 150, 10),
    (32, "200m", 200_000_000, 150, 10),
    (32, "50m", 50_000_000, 150, 10),
    (128, "5g", 5_000_000_000, 200, 20),
    (128, "1g", 1_000_000_000, 200, 20),
    (128, "200m", 200_000_000, 200, 20),
    (128, "50m", 50_000_000, 200, 20),
]


def clean_remote_output() -> None:
    for node in NODES:
        netshape.ssh_run(node, ["rm", "-f", REMOTE_OUTPUT], timeout_s=10)


def _ssh_argv(host: str, remote_cmd: str) -> list[str]:
    return [
        "ssh", "-i", KEY_FILE, "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", f"ubuntu@{host}", remote_cmd,
    ]


def launch_ddp(steps: int, warmup: int, run_id: str) -> list[subprocess.Popen]:
    procs = []
    for i, node in enumerate(NODES):
        data_path = f"{REMOTE_DATA_DIR}/shard_{i:04d}.npy"
        remote_cmd = (
            "cd diloco-measured && source $HOME/.local/bin/env && "
            f".venv/bin/torchrun --nnodes=4 --nproc-per-node=1 --rdzv-backend=c10d "
            f"--rdzv-endpoint={RDZV_ENDPOINT} --rdzv-id={run_id} "
            f"experiments/01_cu_grid/train_driver_ddp.py --steps {steps} "
            f"--warmup-steps {warmup} --micro-batch-size 4 "
            f"--run-id {run_id} --output {REMOTE_OUTPUT} --data-path {data_path}"
        )
        log_path = OUT_DIR / f"{run_id}_node{i}.log"
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            _ssh_argv(node.host, remote_cmd), stdout=log_f, stderr=subprocess.STDOUT
        )
        procs.append(proc)
    return procs


def launch_localsgd(H: int, steps: int, warmup: int, run_id: str) -> list[subprocess.Popen]:
    procs = []
    for i, node in enumerate(NODES):
        data_path = f"{REMOTE_DATA_DIR}/shard_{i:04d}.npy"
        remote_cmd = (
            "cd diloco-measured && source $HOME/.local/bin/env && "
            f".venv/bin/torchrun --nnodes=4 --nproc-per-node=1 --rdzv-backend=c10d "
            f"--rdzv-endpoint={RDZV_ENDPOINT} --rdzv-id={run_id} "
            f"experiments/01_cu_grid/train_driver_localsgd.py --H {H} --steps {steps} "
            f"--warmup-steps {warmup} --micro-batch-size 4 "
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


def run_one_point(
    launch_fn, run_id: str, bw_label: str, bw_bps: int | None
) -> dict:
    """FR-02's apply -> VERIFY -> (retry once) -> run -> restore sequence. `launch_fn` is
    called with no arguments (a closure capturing whatever per-point args it needs) and must
    return the list of Popen procs.
    """
    print(f"\n=== {run_id} ===")
    handle = netshape.apply(bw_bps, NODES) if bw_bps is not None else None
    verification = None
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
            assert verification is not None
            if not verification.passed:
                print(f"  ABORT {run_id}: shaping verification failed after retry")
                return {
                    "run_id": run_id, "status": "aborted_shaping",
                    "bandwidth_label": bw_label, "bandwidth_requested_bps": bw_bps,
                }

        clean_remote_output()
        t_start = time.time()
        procs = launch_fn()
        for i, proc in enumerate(procs):
            rc = proc.wait(timeout=1500)
            if rc != 0:
                print(f"  WARNING: node{i} ssh/torchrun exited {rc} (see log)")
        print(f"  training wall time: {time.time() - t_start:.1f}s")

        raw = fetch_result(run_id)
        if raw is None:
            print(f"  ABORT {run_id}: no node produced an output file (crashed?)")
            return {
                "run_id": run_id, "status": "crashed",
                "bandwidth_label": bw_label, "bandwidth_requested_bps": bw_bps,
            }

        result: dict = {
            "run_id": run_id, "status": "completed",
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
    summary: dict = {"ddp_points": [], "localsgd_points": []}
    summary_path = OUT_DIR / "_summary.json"

    for bw_label, bw_bps, steps, warmup in DDP_GRID:
        run_id = f"cu_grid-ddp-30m-bw{bw_label}-r0"
        point = run_one_point(
            lambda steps=steps, warmup=warmup, run_id=run_id: launch_ddp(steps, warmup, run_id),
            run_id, bw_label, bw_bps,
        )
        summary["ddp_points"].append(point)
        summary_path.write_text(json.dumps(summary, indent=2))

    for H, bw_label, bw_bps, steps, warmup in LOCALSGD_GRID:
        run_id = f"cu_grid-localsgd-30m-h{H}-bw{bw_label}-r0"
        point = run_one_point(
            lambda H=H, steps=steps, warmup=warmup, run_id=run_id: launch_localsgd(
                H, steps, warmup, run_id
            ),
            run_id, bw_label, bw_bps,
        )
        summary["localsgd_points"].append(point)
        summary_path.write_text(json.dumps(summary, indent=2))

    ddp_completed = sum(1 for p in summary["ddp_points"] if p["status"] == "completed")
    localsgd_completed = sum(1 for p in summary["localsgd_points"] if p["status"] == "completed")
    print(
        f"\n=== GRID DONE: DDP {ddp_completed}/{len(DDP_GRID)}, "
        f"LocalSGD {localsgd_completed}/{len(LOCALSGD_GRID)} ==="
    )


if __name__ == "__main__":
    main()
