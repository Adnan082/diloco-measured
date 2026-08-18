"""Real FSDP2 training driver — the last of the original three-algorithm ask (DDP/FSDP2/
LocalSGD, ADR-039 covered the first two). Uses `torch.distributed.fsdp.fully_shard` (real
PyTorch FSDP2, the composable per-parameter-DTensor sharding API — NOT FSDP1's wrapper class),
wrapping the SAME `torchtitan` debugmodel architecture + real gpt2 vocab used by every other
driver in this project, so the model itself stays identical across every algorithm measured.

**Why this needed its own calibration methodology, not just a copy of DDP's:** DDP's only
per-step communication is a single backward-time all-reduce, so its calibration probe only
needed to isolate `backward()`. FSDP2 is a fundamentally different shape: because parameters
are SHARDED at rest (that's the entire memory-saving point of FSDP2), it does real
communication in FORWARD too — before each wrapped sub-module runs, FSDP2 all-gathers that
module's full parameters from all ranks. With this project's `reshard_after_forward=True`
wrapping (the default for non-root modules with no pipeline parallelism — see
`torchtitan/models/llama3/infra/parallelize.py::apply_fsdp()`, mirrored here directly rather
than pulled in via torchtitan's much heavier `ParallelDims`/`JobConfig` orchestration layer,
consistent with how `train_driver_ddp.py` uses raw `torch.nn.parallel.DistributedDataParallel`
rather than torchtitan's `apply_ddp()`), a full step does THREE real collectives: an all-gather
in forward, a second all-gather in backward (params were freed after forward), and a
reduce-scatter of gradients after backward. All three are real, `tc`-shaped network traffic,
and all three are overlapped with compute by FSDP2's design — so, exactly as with DDP, a naive
`mark_compute_done()` right after `backward()` would silently fold real communication time into
`compute_time_ms`.

**The fix: a genuinely communication-free calibration baseline, using FSDP2's own public
`FSDPModule` API rather than a hack.** `set_requires_gradient_sync(False)` is FSDP2's
documented equivalent of DDP's `no_sync()` (it disables the backward reduce-scatter).
Combined with `set_reshard_after_forward(False)` and `set_reshard_after_backward(False)`,
params stay materialized (unsharded) across iterations once first all-gathered, so — unlike
DDP, where `no_sync()` alone is sufficient because DDP's forward never communicates — FSDP2
needs all three knobs off together to reach a truly communication-free steady state. The
calibration loop is run with all three off; every real (measured) step runs with FSDP2's
normal settings restored (`reshard_after_forward=True`, gradient sync on), so the actual
`bytes_synced`-worth of network traffic really happens on every measured step, exactly as it
would in an unmodified training run.

Per DDP's own documented lesson (`train_driver_ddp.py`'s two real Triton-JIT-contamination
bugs — the synced path and the comm-suppressed path JIT-compile separately, and only fixing
one left the other silently broken across an entire completed grid): this driver warms up
BOTH the normal path and the comm-suppressed path once each, defensively, BEFORE calibration
starts, rather than waiting to discover the same failure class a third time. Median (not mean)
of 3 calibration samples, same defensive reasoning as `train_driver_ddp.py`.

**Wire-model note** (see `methods/wire_model.md` §3, FSDP2 row — now `[CONFIRMED]`, derived
analytically from the collective pattern above and torchtitan's own default wrapping policy,
not yet cross-checked against a real `/proc/net/dev` measurement — that gap is the same
project-wide one `fig5_bytes_on_wire` already documents as unresolved): FSDP2 moves
`3 · N · (P−1)/P` bytes per rank per step under ring collectives (2 all-gathers + 1
reduce-scatter, each costing the same `N(P−1)/P` a ring all-reduce's half does) — 1.5× DDP's
`2 · N · (P−1)/P`. For the simpler `cu_analytic_*` model (`methods/cu_model.md` §3, which
deliberately treats each real transfer as one undifferentiated `bytes/B` cost rather than
accounting for ring parallelism — the same simplification already applied to every other
algorithm in this project), the aggregator uses `bytes_synced = 3 · N` (three separate
full-tensor logical transfers), not `1.5 · N` — the CU model's convention is per-transfer-
event, not per-ring-collective-efficiency, so FSDP2's three separate collective calls each
count once, the same way DDP's one all-reduce call counts once as `N`.

`H` has no meaning for FSDP2 in this project's framework — like DDP, it is not a
semi-synchronous method (there is no inner/outer loop to control a synchronization interval),
so every grid point here fixes `H=1`, matching DDP's grid exactly (bandwidth only).
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
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from diloco_measured.measurement.telemetry import StepTimer

VOCAB_SIZE = 50257  # gpt2 -- matches configs/models/30m-realvocab.toml
SEQ_LEN = 512
N_JIT_WARMUP_STEPS = 1  # throwaway, per path -- lets Triton JIT-compile before timing
N_CALIBRATION_SAMPLES = 3  # median-reduced, taken during warmup, never inside the measured window


def build_model():
    """Identical construction to every other driver in this project -- same architecture, same
    real gpt2-vocab override, so FSDP2 is measuring the same model DDP/DiLoCo/LocalSGD are.
    Built directly on GPU (not meta-device) -- the model is tiny (30.8M params), so
    torchtitan's meta-device-then-materialize pattern (for multi-GB models where the full,
    unsharded model wouldn't fit in one rank's memory) buys nothing here and would just be
    extra moving parts, consistent with CLAUDE.md's "simplicity over generality" principle.
    """
    from torchtitan.models.llama3 import model_registry

    spec = model_registry("debugmodel")
    cfg = spec.model
    cfg.vocab_size = VOCAB_SIZE
    cfg.tok_embeddings.num_embeddings = VOCAB_SIZE
    cfg.lm_head.out_features = VOCAB_SIZE
    return cfg.build().cuda()


def apply_fsdp2(model: torch.nn.Module, mesh) -> None:
    """Mirrors `torchtitan/models/llama3/infra/parallelize.py::apply_fsdp()`'s wrapping order
    exactly (same model class), with `reshard_after_forward=True` passed explicitly everywhere
    rather than relying on `fully_shard`'s root-vs-non-root default (`None` -> True for
    non-root, False for root) -- explicit here because this project's wire-model derivation
    (module docstring above) assumes every real parameter-holding group reshards after
    forward, and leaving it implicit would make that assumption silently dependent on which
    calls torchtitan's own default logic happens to treat as "root".
    """
    fully_shard(model.tok_embeddings, mesh=mesh, reshard_after_forward=True)
    for transformer_block in model.layers.values():
        fully_shard(transformer_block, mesh=mesh, reshard_after_forward=True)
    fully_shard([model.norm, model.output], mesh=mesh, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)


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
    min_warmup = 2 * N_JIT_WARMUP_STEPS + N_CALIBRATION_SAMPLES + 1
    if args.warmup_steps < min_warmup:
        raise ValueError(
            f"--warmup-steps must be >= {min_warmup} to fit both JIT-warmup passes and the "
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
    mesh = init_device_mesh("cuda", (world_size,))
    apply_fsdp2(model, mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    arr = np.load(args.data_path)
    data = torch.from_numpy(arr.astype(np.int64)).cuda()
    n_seq = data.shape[0]
    positions = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(
        args.micro_batch_size, -1
    )

    def loss_fn(batch: torch.Tensor) -> torch.Tensor:
        masks = model.get_attention_masks(positions)
        logits = model(batch, attention_masks=masks, positions=positions)
        shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
        shift_targets = batch[:, 1:].reshape(-1)
        return F.cross_entropy(shift_logits, shift_targets)

    def sample_batch() -> torch.Tensor:
        idx = torch.randint(0, n_seq, (args.micro_batch_size,), device="cuda")
        return data[idx]

    # --- JIT warmup, normal (fully-synced) path: one throwaway forward+backward with FSDP2's
    # real default settings (all-gather x2, reduce-scatter). Times both Triton compilation AND
    # NCCL's lazy first-call setup for these specific collectives.
    for _ in range(N_JIT_WARMUP_STEPS):
        batch = sample_batch()
        optimizer.zero_grad()
        loss = loss_fn(batch)
        loss.backward()
        torch.cuda.synchronize()

    # --- JIT warmup, comm-suppressed path: per DDP's own documented lesson (module docstring),
    # warm this path separately and explicitly rather than assuming the normal-path warmup
    # above covers it.
    model.set_requires_gradient_sync(False)
    model.set_reshard_after_forward(False)
    model.set_reshard_after_backward(False)
    for _ in range(N_JIT_WARMUP_STEPS):
        batch = sample_batch()
        optimizer.zero_grad()
        loss = loss_fn(batch)
        loss.backward()
        torch.cuda.synchronize()

    # --- Calibration probe: compute-only forward+backward time, comm suppressed by the three
    # FSDPModule settings above (params stay unsharded across iterations once first
    # materialized by the JIT-warmup pass immediately above, so none of these samples should
    # trigger any further all-gather/reduce-scatter). MEDIAN, not mean -- same defensive
    # reasoning as train_driver_ddp.py.
    calibration_samples_ms = []
    for _ in range(N_CALIBRATION_SAMPLES):
        batch = sample_batch()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        optimizer.zero_grad()
        t0.record()
        loss = loss_fn(batch)
        loss.backward()
        t1.record()
        torch.cuda.synchronize()
        calibration_samples_ms.append(t0.elapsed_time(t1))
    sorted_samples = sorted(calibration_samples_ms)
    mid = len(sorted_samples) // 2
    calibrated_compute_only_ms = (
        sorted_samples[mid] if len(sorted_samples) % 2 == 1
        else (sorted_samples[mid - 1] + sorted_samples[mid]) / 2
    )
    if rank == 0:
        print(
            f"calibration: compute-only fwd+bwd (median) = {calibrated_compute_only_ms:.2f}ms "
            f"(samples: {[round(x, 2) for x in calibration_samples_ms]})",
            flush=True,
        )

    # Restore FSDP2's real per-step behavior before any measured step runs.
    model.set_requires_gradient_sync(True)
    model.set_reshard_after_forward(True)
    model.set_reshard_after_backward(True)

    step_records = []
    started_at = datetime.now(UTC)
    t_wall_start = time.perf_counter()
    remaining_warmup = args.warmup_steps - 2 * N_JIT_WARMUP_STEPS - N_CALIBRATION_SAMPLES

    for step in range(args.steps):
        with StepTimer() as timer:
            batch = sample_batch()
            timer.mark_loader_done()

            step_t0 = torch.cuda.Event(enable_timing=True)
            step_t1 = torch.cuda.Event(enable_timing=True)
            optimizer.zero_grad()
            step_t0.record()
            loss = loss_fn(batch)  # real forward all-gather, inside this call
            loss.backward()  # real backward all-gather + reduce-scatter, inside this call
            step_t1.record()
            torch.cuda.synchronize()
            synced_step_ms = step_t0.elapsed_time(step_t1)

            compute_ms = min(calibrated_compute_only_ms, synced_step_ms)
            sync_ms = max(0.0, synced_step_ms - calibrated_compute_only_ms)
            # See train_driver_ddp.py's identical comment: these marks place StepTimer's wall-
            # time reconciliation boundary; the actual compute/sync split comes from
            # synced_step_ms above, overridden into the StepTiming below.
            timer.mark_compute_done()
            timer.mark_sync_done()

            optimizer.step()
            timer.mark_optimizer_done()

        raw_timing = timer.result()
        wall_ms = raw_timing.wall_time_ms
        optimizer_ms = raw_timing.optimizer_time_ms
        loader_ms = raw_timing.loader_stall_ms

        step_records.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "did_outer_step": True,  # FSDP2 syncs every step by definition (H=1)
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
            "algorithm": "fsdp2",
            "H": 1,
            "world_size": world_size,
            "n_params": n_params,
            "steps": args.steps,
            "warmup_steps": remaining_warmup,
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
