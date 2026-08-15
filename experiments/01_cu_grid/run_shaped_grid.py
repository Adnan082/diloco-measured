"""Orchestrate the shaped bandwidth x H grid: for each point, apply real `tc` shaping on all
4 GPU nodes, verify via `iperf3` (FR-02 gate, exactly one retry), launch `torchrun` across all
4 nodes over SSH, fetch the rank-0 output, restore shaping (always, in a `finally`), then move
to the next point. This is the script that actually produced the 16 shaped
`results/raw/cu_grid-diloco-30m-h*-bw*-r0.json` records on 2026-08-14/15 (CLAUDE.md ADR-035).

Runs LOCALLY on the operator's machine (needs the SSH private key and `diloco_measured`
importable) -- matches `netshape.py`'s own design (`Node.host` = the public IP the OPERATOR
SSHes from, not something a cloud node would have).

**Cluster config comes from the `DILOCO_NODES` env var, never hardcoded** -- CLAUDE.md §23
requires fingerprints to exclude private IPs, and the same discipline applies to any committed
file: EC2 IPs are ephemeral and specific to one operator's one live cluster, so this script
takes them at run time instead of embedding a point-in-time snapshot that would be stale (and
would read as a real infrastructure identifier) the moment that cluster is torn down. Set it
to 4 comma-separated `public_ip:private_ip` pairs, e.g.:
    DILOCO_NODES="13.1.2.3:172.31.1.1,13.1.2.4:172.31.1.2,13.1.2.5:172.31.1.3,13.1.2.6:172.31.1.4"
(the historical run that produced `results/raw/cu_grid-diloco-30m-h*-bw*-r0.json` used the
`us-east-1b` cluster launched in this session, per CLAUDE.md ADR-035 -- those IPs are gone
now that the cluster is torn down, which is exactly why they don't belong in this file).
`DILOCO_SSH_KEY` overrides `KEY_FILE` similarly (default matches `launch_cluster.sh`'s own
default key path).

**Honesty note (same posture as `train_driver.py`):** this script drives the real, validated
`netshape.py` apply/verify/restore primitives (FR-02's actual gate, exercised for real, not
mocked or bypassed) but is NOT `measurement/train.py::run()`'s own orchestration -- there is
no automated precondition gate here, and no in-process fingerprinting (that was done
separately and hand-embedded into `aggregate_shaped_grid.py`). `measurement/train.py::run()`'s
full FR-03 orchestration (which would take a `Node` list from a real cluster-inventory
mechanism rather than an env var) is still the intended long-term replacement for this script.

`DILOCO_REPEAT_INDEX` (default `0`) selects which repeat this invocation produces --
`run_id`s and output paths are suffixed `-r{index}` accordingly, so re-running with index 1,
then 2, against a (re-)bootstrapped cluster is how CLAUDE.md §40 Q6's repeat requirement (G1:
"3 repeats each") gets satisfied without ever overwriting an earlier repeat's output.
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
REPEAT_INDEX = int(os.environ.get("DILOCO_REPEAT_INDEX", "0"))
OUT_DIR = Path(__file__).resolve().parent / "shaped_grid_run_logs"
OUT_DIR.mkdir(exist_ok=True)


def _load_nodes() -> list[netshape.Node]:
    raw = os.environ.get("DILOCO_NODES", "")
    if not raw:
        raise SystemExit(
            "DILOCO_NODES not set -- see this module's docstring for the expected format "
            "(4 comma-separated public_ip:private_ip pairs, one per GPU node)."
        )
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

# (H, bandwidth_label, bandwidth_bps, steps, warmup_steps) -- step counts sized so total
# added pseudo-gradient sync time per point stays roughly under ~3-5 min wall clock, given
# bytes_synced = 30,846,720*4 = 123,386,880 bytes and sync_time ~= bytes*8/bandwidth_bps.
GRID = [
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


def launch_training(H: int, steps: int, warmup: int, run_id: str) -> list[subprocess.Popen]:
    procs = []
    for i, node in enumerate(NODES):
        data_path = f"{REMOTE_DATA_DIR}/shard_{i:04d}.npy"
        remote_cmd = (
            "cd diloco-measured && source $HOME/.local/bin/env && "
            f".venv/bin/torchrun --nnodes=4 --nproc-per-node=1 --rdzv-backend=c10d "
            f"--rdzv-endpoint={RDZV_ENDPOINT} --rdzv-id={run_id} "
            f"experiments/01_cu_grid/train_driver.py --H {H} --steps {steps} "
            f"--warmup-steps {warmup} --micro-batch-size 4 --run-id {run_id} "
            f"--output {REMOTE_OUTPUT} --data-path {data_path}"
        )
        ssh_argv = [
            "ssh", "-i", KEY_FILE, "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
            f"ubuntu@{node.host}", remote_cmd,
        ]
        log_path = OUT_DIR / f"{run_id}_node{i}.log"
        log_f = open(log_path, "w")
        proc = subprocess.Popen(ssh_argv, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append(proc)
    return procs


def fetch_result(run_id: str) -> dict | None:
    """Only rank 0 (assigned by c10d connection order, not deterministic per node) writes
    REMOTE_OUTPUT -- try every node and return whichever one actually has it.
    """
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


def run_one_point(H: int, bw_label: str, bw_bps: int, steps: int, warmup: int) -> dict:
    """FR-02's apply -> VERIFY -> (retry once) -> run -> restore sequence, exactly. `restore()`
    runs unconditionally in `finally`, including on the abort path, matching CLAUDE.md §25.3.
    """
    run_id = f"cu_grid-diloco-30m-h{H}-bw{bw_label}-r{REPEAT_INDEX}"
    print(f"\n=== {run_id} (steps={steps} warmup={warmup}) ===")

    handle = netshape.apply(bw_bps, NODES)
    verification: netshape.ShapingVerification | None = None
    try:
        for attempt in range(2):  # exactly one retry, FR-02
            verification = netshape.verify(handle, tolerance_pct=10.0, duration_s=15)
            print(
                f"  shaping verify attempt {attempt+1}: requested={bw_bps} "
                f"measured={verification.measured_bps:.0f} error={verification.error_pct:.1f}% "
                f"passed={verification.passed}"
            )
            if verification.passed:
                break
        assert verification is not None  # range(2) always executes >=1 iteration
        if not verification.passed:
            print(f"  ABORT {run_id}: shaping verification failed after retry")
            return {
                "run_id": run_id, "status": "aborted_shaping",
                "H": H, "bandwidth_label": bw_label, "bandwidth_requested_bps": bw_bps,
                "shaping_verification": {
                    "requested_bps": verification.requested_bps,
                    "measured_bps": verification.measured_bps,
                    "error_pct": verification.error_pct,
                    "passed": verification.passed,
                },
            }

        clean_remote_output()
        t_start = time.time()
        procs = launch_training(H, steps, warmup, run_id)
        for i, proc in enumerate(procs):
            rc = proc.wait(timeout=900)
            if rc != 0:
                print(f"  WARNING: node{i} ssh/torchrun exited {rc} (see log)")
        elapsed = time.time() - t_start
        print(f"  training wall time: {elapsed:.1f}s")

        raw = fetch_result(run_id)
        if raw is None:
            print(f"  ABORT {run_id}: no node produced an output file (crashed?)")
            return {
                "run_id": run_id, "status": "crashed",
                "H": H, "bandwidth_label": bw_label, "bandwidth_requested_bps": bw_bps,
            }

        return {
            "run_id": run_id, "status": "completed",
            "H": H, "bandwidth_label": bw_label, "bandwidth_requested_bps": bw_bps,
            "shaping_verification": {
                "requested_bps": verification.requested_bps,
                "measured_bps": verification.measured_bps,
                "error_pct": verification.error_pct,
                "tolerance_pct": verification.tolerance_pct,
                "passed": verification.passed,
                "attempts": attempt + 1,
            },
            "raw": raw,
        }
    finally:
        netshape.restore(handle)
        print(f"  shaping restored for {run_id}")


def main() -> None:
    summary = []
    summary_path = OUT_DIR / f"_summary_r{REPEAT_INDEX}.json"
    for H, bw_label, bw_bps, steps, warmup in GRID:
        point_result = run_one_point(H, bw_label, bw_bps, steps, warmup)
        summary.append(point_result)
        # Written incrementally (not just at the end) so a crash partway through the campaign
        # doesn't lose the points that already completed.
        summary_path.write_text(json.dumps(summary, indent=2))

    completed = sum(1 for p in summary if p["status"] == "completed")
    print(f"\n=== GRID DONE (repeat {REPEAT_INDEX}): {completed}/{len(GRID)} completed ===")
    for p in summary:
        print(f"  {p['run_id']}: {p['status']}")


if __name__ == "__main__":
    main()
