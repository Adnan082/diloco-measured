"""Orchestrate the real FSDP2 grid -- the last of the original three-algorithm ask (DDP/FSDP2/
LocalSGD; ADR-039 covered DDP and LocalSGD). Same FR-02 apply->verify->(retry once)->run->
restore sequence as `run_shaped_grid.py`/`run_ddp_localsgd_grid.py`, same `DILOCO_NODES`/
`DILOCO_SSH_KEY` env-var convention (CLAUDE.md §23 private-IP discipline), and the SAME
orchestration robustness fixes `run_ddp_localsgd_grid.py` needed for real (ADR-039: shared-
deadline polling instead of per-node sequential timeouts, force-kill stragglers, always
attempt a result fetch regardless of clean ssh exit, resume-awareness, defensive remote
process cleanup between every point) -- ported here directly rather than re-discovering the
same failure class a third time.

FSDP2 grid: bandwidth only (H is always 1 by definition -- FSDP2 is not a semi-synchronous
method in this project's framework, same as DDP; `measurement/spec.py`'s validator now
enforces this for both algorithms) -- 5 points (unshaped + the same 4 shaped levels DDP/
DiLoCo/LocalSGD use), matching DDP's exact grid table for direct comparability.
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
OUT_DIR = Path(__file__).resolve().parent / "fsdp2_run_logs"
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

# Same table as DDP's grid (run_ddp_localsgd_grid.py::DDP_GRID) for direct comparability --
# FSDP2 moves 1.5x DDP's per-step bytes (methods/wire_model.md §3a), so if timing at the
# lowest bandwidth turns out to badly exceed DDP's real wall-clock (the 50m DDP point took
# ~793s for 15 steps, ADR-039), these step counts are the first thing to shrink -- not done
# preemptively, since guessing at a correction before seeing a real number would just be a
# different unverified guess.
FSDP2_GRID = [
    ("unshaped", None, 100, 10),
    ("5g", 5_000_000_000, 100, 10),
    ("1g", 1_000_000_000, 100, 10),
    ("200m", 200_000_000, 35, 8),
    ("50m", 50_000_000, 15, 6),
]


def clean_remote_output() -> None:
    for node in NODES:
        netshape.ssh_run(node, ["rm", "-f", REMOTE_OUTPUT], timeout_s=10)


def force_cleanup_remote() -> None:
    """Same defensive cleanup as run_ddp_localsgd_grid.py's identically-named function -- see
    that module's docstring for the real teardown-hang mechanism this guards against."""
    for node in NODES:
        netshape.ssh_run(
            node, ["pkill", "-9", "-f", "train_driver_fsdp2.py"], timeout_s=10
        )
        netshape.ssh_run(node, ["pkill", "-9", "-f", "torchrun"], timeout_s=10)


def _ssh_argv(host: str, remote_cmd: str) -> list[str]:
    return [
        "ssh", "-i", KEY_FILE, "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", f"ubuntu@{host}", remote_cmd,
    ]


def launch_fsdp2(steps: int, warmup: int, run_id: str) -> list[subprocess.Popen]:
    procs = []
    for i, node in enumerate(NODES):
        data_path = f"{REMOTE_DATA_DIR}/shard_{i:04d}.npy"
        remote_cmd = (
            "cd diloco-measured && source $HOME/.local/bin/env && "
            f".venv/bin/torchrun --nnodes=4 --nproc-per-node=1 --rdzv-backend=c10d "
            f"--rdzv-endpoint={RDZV_ENDPOINT} --rdzv-id={run_id} "
            f"experiments/01_cu_grid/train_driver_fsdp2.py --steps {steps} "
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


def wait_for_procs_or_recover(
    procs: list[subprocess.Popen], run_id: str, max_wait_s: float
) -> bool:
    """Ported verbatim (in behavior) from run_ddp_localsgd_grid.py -- see that module's
    docstring for the real incident that made this necessary (ADR-039)."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline and any(p.poll() is None for p in procs):
        time.sleep(5)
    all_clean = all(p.poll() is not None for p in procs)
    for i, proc in enumerate(procs):
        if proc.poll() is None:
            print(f"  WARNING: node{i} ssh session still running after {max_wait_s:.0f}s "
                  f"deadline -- killing it and checking for a completed output file anyway")
            proc.kill()
            proc.wait(timeout=10)
        elif proc.returncode != 0:
            print(f"  WARNING: node{i} ssh/torchrun exited {proc.returncode} (see log)")
    return all_clean


def run_one_point(
    launch_fn, run_id: str, bw_label: str, bw_bps: int | None, max_wait_s: float = 2400
) -> dict:
    """Same FR-02 sequence + resume-awareness as run_ddp_localsgd_grid.py::run_one_point()."""
    cached_path = OUT_DIR / f"{run_id}.json"
    if cached_path.exists():
        print(f"\n=== {run_id} === (skipping -- cached result on disk)")
        raw = json.loads(cached_path.read_text())
        return {
            "run_id": run_id, "status": "completed",
            "bandwidth_label": bw_label, "bandwidth_requested_bps": bw_bps, "raw": raw,
        }

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
        all_clean = wait_for_procs_or_recover(procs, run_id, max_wait_s)
        print(f"  training wall time: {time.time() - t_start:.1f}s"
              f"{' (forced -- see WARNING above)' if not all_clean else ''}")

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
        if not all_clean:
            result["note"] = (
                "one or more ssh sessions had to be force-killed after the deadline; the "
                "training itself completed and wrote a valid output file before that"
            )
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
        force_cleanup_remote()


def main() -> None:
    summary: dict = {"fsdp2_points": []}
    summary_path = OUT_DIR / "_summary.json"

    for bw_label, bw_bps, steps, warmup in FSDP2_GRID:
        run_id = f"cu_grid-fsdp2-30m-bw{bw_label}-r0"
        point = run_one_point(
            lambda steps=steps, warmup=warmup, run_id=run_id: launch_fsdp2(steps, warmup, run_id),
            run_id, bw_label, bw_bps,
        )
        summary["fsdp2_points"].append(point)
        summary_path.write_text(json.dumps(summary, indent=2))

    n_completed = sum(1 for p in summary["fsdp2_points"] if p["status"] == "completed")
    print(f"\n=== GRID DONE: FSDP2 {n_completed}/{len(FSDP2_GRID)} ===")


if __name__ == "__main__":
    main()
