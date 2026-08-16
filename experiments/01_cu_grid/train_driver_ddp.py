"""Real DDP training driver — the standard baseline the DiLoCo comparisons are measured
against. Uses `torch.nn.parallel.DistributedDataParallel` (real PyTorch DDP, not a DiLoCo
degenerate case) over the same `torchtitan` debugmodel architecture + real gpt2 vocab as
`train_driver.py` (ADR-034), so the model itself is identical across every algorithm this
project measures.

**A real methodological problem this driver had to solve, not gloss over:** DDP overlaps
gradient all-reduce with backward-pass computation by design (that's DDP's whole efficiency
trick) — a plain "mark_compute_done() right after backward()" would silently fold
communication time INTO compute_time_ms. That would make `cu_measured` for DDP stay near
100% even at catastrophic bandwidth scarcity, since almost all wall time collapses into one
undifferentiated bucket regardless of how slow the network is — exactly the kind of misleading
metric CLAUDE.md's measurement-integrity principle exists to prevent, and it would defeat the
entire point of this project's headline metric for the one algorithm (DDP) most people
actually compare DiLoCo against.

**The fix: an explicit calibration probe, run once during warmup, not on every step.** After a
couple of warmup iterations (to let CUDA kernels/allocator settle), one extra backward pass is
run inside `ddp_model.no_sync()` (skips gradient communication entirely) to measure a
compute-only backward time baseline. For every real (non-calibration) step afterward:
    compute_time_ms = min(calibrated_compute_only_ms, this_step_synced_backward_ms)
    sync_blocked_ms = max(0, this_step_synced_backward_ms - calibrated_compute_only_ms)
This makes `sync_blocked_ms` track the REAL, bandwidth-dependent synced backward time (which
DOES grow under `tc` shaping, since NCCL's bucketed all-reduce is real network traffic) while
`compute_time_ms` stays anchored to a real, once-measured, communication-free baseline. This is
an imputed split from ONE calibration measurement per run, not a per-step direct measurement
the way DiLoCo's separate, unoverlapped outer sync is — stated explicitly here and in every
record's `notes` field, not left for a reader to discover.

**A real bug this caught on the first real smoke test, fixed before any real grid point ran:**
the calibration probe's FIRST sample measured 37,955ms against ~21ms for the other two --
Triton's FlexAttention backward kernel JIT-compiles on its first-ever invocation in the
process (the same one-time cost `train_driver.py`/ADR-032 already knew about at the *step*
level, but this driver's calibration probe IS that first invocation), so the naive average was
dominated by a one-time compilation cost, not steady-state compute. That inflated baseline
would have made `sync_blocked_ms` compute to 0 on every real step forever (since the real
per-step time is always less than an ~12-second "baseline"), silently defeating the entire
point of this calibration mechanism. Fixed with one throwaway, fully-synced warmup step BEFORE
the calibration loop starts, so Triton compilation completes before any timing measurement
begins -- `N_JIT_WARMUP_STEPS` below.
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from diloco_measured.measurement.telemetry import StepTimer

VOCAB_SIZE = 50257  # gpt2 -- matches configs/models/30m-realvocab.toml
SEQ_LEN = 512
N_JIT_WARMUP_STEPS = 1  # throwaway, fully-synced -- lets Triton JIT-compile before timing
N_CALIBRATION_SAMPLES = 3  # averaged, taken during warmup, never inside the measured window


def build_model():
    """Identical construction to train_driver.py's build_model() -- same architecture, same
    real gpt2-vocab override, so DDP and DiLoCo are measuring the same model.
    """
    from torchtitan.models.llama3 import model_registry

    spec = model_registry("debugmodel")
    cfg = spec.model
    cfg.vocab_size = VOCAB_SIZE
    cfg.tok_embeddings.num_embeddings = VOCAB_SIZE
    cfg.lm_head.out_features = VOCAB_SIZE
    return cfg.build().cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--data-path", type=str, default="/opt/dlami/nvme/dataset/local_shard.npy"
    )
    args = parser.parse_args()
    min_warmup = N_JIT_WARMUP_STEPS + N_CALIBRATION_SAMPLES + 1
    if args.warmup_steps < min_warmup:
        raise ValueError(
            f"--warmup-steps must be >= {min_warmup} to fit the JIT-warmup step and the "
            f"calibration probe before the measured window starts, got {args.warmup_steps}"
        )

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42 + rank)

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)

    arr = np.load(args.data_path)
    data = torch.from_numpy(arr.astype(np.int64)).cuda()
    n_seq = data.shape[0]
    positions = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(
        args.micro_batch_size, -1
    )

    def loss_fn(batch: torch.Tensor) -> torch.Tensor:
        masks = model.get_attention_masks(positions)
        logits = ddp_model(batch, attention_masks=masks, positions=positions)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_targets = batch[:, 1:].reshape(-1)
        return F.cross_entropy(shift_logits, shift_targets)

    def sample_batch() -> torch.Tensor:
        idx = torch.randint(0, n_seq, (args.micro_batch_size,), device="cuda")
        return data[idx]

    # --- JIT warmup: one fully-synced, throwaway step BEFORE any timing starts. Triton's
    # FlexAttention backward kernel JIT-compiles on its first-ever invocation in this process
    # (seconds, not milliseconds) -- without this, that one-time cost lands inside the
    # calibration probe below and silently wrecks it (see module docstring's "real bug" note).
    for _ in range(N_JIT_WARMUP_STEPS):
        batch = sample_batch()
        loss = loss_fn(batch)
        loss.backward()
        torch.cuda.synchronize()
        optimizer.zero_grad()

    # --- Calibration probe: measure compute-only (no_sync) backward time, once, during
    # warmup. Excluded from the measured window entirely -- gradients from these calibration
    # backward passes are discarded (zero_grad() immediately after) so they never affect the
    # real training trajectory.
    calibration_samples_ms = []
    for _ in range(N_CALIBRATION_SAMPLES):
        batch = sample_batch()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        with ddp_model.no_sync():
            loss = loss_fn(batch)
            loss.backward()
        t1.record()
        torch.cuda.synchronize()
        calibration_samples_ms.append(t0.elapsed_time(t1))
        optimizer.zero_grad()
    calibrated_compute_only_ms = sum(calibration_samples_ms) / len(calibration_samples_ms)
    if rank == 0:
        print(
            f"calibration: compute-only backward = {calibrated_compute_only_ms:.2f}ms "
            f"(samples: {[round(x, 2) for x in calibration_samples_ms]})",
            flush=True,
        )

    step_records = []
    started_at = datetime.now(UTC)
    t_wall_start = time.perf_counter()
    remaining_warmup = args.warmup_steps - N_JIT_WARMUP_STEPS - N_CALIBRATION_SAMPLES

    for step in range(args.steps):
        with StepTimer() as timer:
            batch = sample_batch()
            timer.mark_loader_done()

            backward_t0 = torch.cuda.Event(enable_timing=True)
            backward_t1 = torch.cuda.Event(enable_timing=True)
            optimizer.zero_grad()
            backward_t0.record()
            loss = loss_fn(batch)
            loss.backward()  # real DDP gradient all-reduce, overlapped with this call
            backward_t1.record()
            torch.cuda.synchronize()
            synced_backward_ms = backward_t0.elapsed_time(backward_t1)

            compute_ms = min(calibrated_compute_only_ms, synced_backward_ms)
            sync_ms = max(0.0, synced_backward_ms - calibrated_compute_only_ms)
            # StepTimer's own marks would double-count the backward pass (it already
            # happened above, timed via cuda.Event directly for the calibration-adjusted
            # split) -- mark_compute_done()/mark_sync_done() here just place the *boundary*
            # StepTimer uses for wall_time_ms reconciliation; the actual compute/sync SPLIT
            # comes from synced_backward_ms above, overridden into the StepTiming below.
            timer.mark_compute_done()
            timer.mark_sync_done()

            optimizer.step()
            timer.mark_optimizer_done()

        raw_timing = timer.result()
        # Override compute/sync with the calibration-adjusted split; keep wall_time_ms,
        # optimizer_time_ms, loader_stall_ms as StepTimer measured them directly.
        wall_ms = raw_timing.wall_time_ms
        optimizer_ms = raw_timing.optimizer_time_ms
        loader_ms = raw_timing.loader_stall_ms

        step_records.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "did_outer_step": True,  # DDP syncs every step by definition (H=1)
                "wall_time_ms": wall_ms,
                "compute_time_ms": compute_ms,
                "sync_blocked_ms": sync_ms,
                "optimizer_time_ms": optimizer_ms,
                "loader_stall_ms": loader_ms,
            }
        )
        if rank == 0 and step % 20 == 0:
            print(
                f"step {step}: loss={float(loss.detach()):.4f} "
                f"wall_ms={wall_ms:.2f} compute_ms={compute_ms:.2f} sync_ms={sync_ms:.2f}",
                flush=True,
            )

    ended_at = datetime.now(UTC)
    total_wall_s = time.perf_counter() - t_wall_start

    if rank == 0:
        result = {
            "run_id": args.run_id,
            "algorithm": "ddp",
            "H": 1,
            "world_size": world_size,
            "n_params": n_params,
            "steps": args.steps,
            "warmup_steps": remaining_warmup,  # calibration steps excluded, not double counted
            "micro_batch_size": args.micro_batch_size,
            "seq_len": SEQ_LEN,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "total_wall_s": total_wall_s,
            "calibrated_compute_only_ms": calibrated_compute_only_ms,
            "calibration_samples_ms": calibration_samples_ms,
            "step_records": step_records,
        }
        with open(args.output, "w") as f:
            json.dump(result, f)
        print(f"RUN_COMPLETE run_id={args.run_id} n_params={n_params} wrote={args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
