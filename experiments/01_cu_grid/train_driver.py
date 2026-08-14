"""Real DiLoCo training driver: torchtitan's `debugmodel` architecture (real gpt2 vocab
override, see `configs/models/30m-realvocab.toml`) + the reference `DiLoCoTrainer`
(`measurement/diloco.py`, ADR-003/D3), run across 4 real GPU nodes via `torchrun`.

This is the script that actually produced `results/raw/cu_grid-diloco-30m-h{1,8,32,128}-
bwunshaped-r0.json` on 2026-08-14 (see CLAUDE.md ADR-034). It is committed here, not left in
a local scratchpad, so the run is reproducible from the repository per CLAUDE.md §14.1 (this
directory is exactly where "what actually happened" belongs) and FR-03's spirit even though
this run did not go through `measurement/train.py::run()`'s full orchestration.

**Honesty note (read before trusting this as "the" run lifecycle):** `measurement/train.py`
implements FR-03's real preconditions, shaping gate, and fingerprinting (ADR-028), but its
own step 6 (actual torchrun launch of a training loop) was deliberately left unimplemented at
the time because torchtitan/torchft's real API surface wasn't validated yet (ADR-009 was
still `[PROPOSED]`). By the time GPU hardware existed and validation completed (ADR-032),
building a `Trainer.Config` adapter for torchtitan's `model_registry()` composition system was
real remaining work its own docstring flagged as not done. Rather than block the first real
training measurement on that adapter layer, this script was written as a direct, hand-driven
`torchrun` entrypoint that uses the SAME correctness-critical primitives FR-03 depends on
(`DiLoCoTrainer`, `StepTimer`) but skips `train.py`'s precondition checks, shaping gate, and
in-process fingerprinting — those were done manually and separately for this run (real
`fingerprint.py::capture()` output, hand-copied into `aggregate_results.py`; unshaped, so no
shaping gate applies; preconditions — dataset present, GPU clocks, etc. — were checked by hand
during bootstrap, not by an automated gate). This is a real gap versus the fully-automated
FR-03 path, not a substitute for it, and should be closed before running a shaped (`tc`-gated)
campaign, where the shaping verification gate is not optional (FR-02/ADR-002).

Usage (once run.sh's cluster context is set up — see run.sh in this directory):
    torchrun --nnodes=4 --nproc-per-node=1 --rdzv-backend=c10d \\
        --rdzv-endpoint=<control-or-rank0-node>:<port> \\
        train_driver.py --H <H> --steps <N> --warmup-steps 10 --micro-batch-size 4 \\
        --run-id <run_id> --output <path> --data-path <path-to-local-shard.npy>
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

from diloco_measured.measurement.diloco import DiLoCoTrainer
from diloco_measured.measurement.telemetry import StepTimer

VOCAB_SIZE = 50257  # gpt2 — matches configs/models/30m-realvocab.toml
SEQ_LEN = 512


def build_model():
    """torchtitan's `debugmodel` dims (dim=256, n_layers=6, n_heads=16 — real, validated on
    GPU hardware in ADR-032) with vocab_size overridden to a real tokenizer's vocab
    (configs/models/30m-realvocab.toml). Real parameter count: 30,846,720.
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
    parser.add_argument("--H", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--data-path", type=str, default="/opt/dlami/nvme/dataset/local_shard.npy"
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(42 + rank)

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())

    trainer = DiLoCoTrainer(
        model=model,
        inner_optimizer_cfg={"name": "adamw", "lr": 3e-4},
        outer_optimizer_cfg={"name": "nesterov_sgd", "lr": 0.7, "momentum": 0.9},
        H=args.H,
    )

    # Real, disjoint, gpt2-tokenized FineWeb-Edu shard for this rank (infra/prepare_dataset.py,
    # ADR-030) — one .npy per node, pre-staged onto local NVMe, NOT shared across ranks.
    arr = np.load(args.data_path)
    data = torch.from_numpy(arr.astype(np.int64)).cuda()
    n_seq = data.shape[0]
    positions = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(
        args.micro_batch_size, -1
    )

    step_records = []
    started_at = datetime.now(UTC)
    t_wall_start = time.perf_counter()

    for step in range(args.steps):
        with StepTimer() as timer:
            idx = torch.randint(0, n_seq, (args.micro_batch_size,), device="cuda")
            batch = data[idx]
            timer.mark_loader_done()

            def loss_fn(batch=batch):
                masks = model.get_attention_masks(positions)
                logits = model(batch, attention_masks=masks, positions=positions)
                shift_logits = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)
                shift_targets = batch[:, 1:].reshape(-1)
                return F.cross_entropy(shift_logits, shift_targets)

            inner_loss = trainer.inner_step(loss_fn)
            torch.cuda.synchronize()
            timer.mark_compute_done()

            did_outer = trainer.ready_for_outer_step()
            if did_outer:
                trainer.outer_step()
            torch.cuda.synchronize()
            timer.mark_sync_done()
            timer.mark_optimizer_done()

        timing = timer.result()
        step_records.append(
            {
                "step": step,
                "loss": inner_loss,
                "did_outer_step": did_outer,
                "wall_time_ms": timing.wall_time_ms,
                "compute_time_ms": timing.compute_time_ms,
                "sync_blocked_ms": timing.sync_blocked_ms,
                "optimizer_time_ms": timing.optimizer_time_ms,
                "loader_stall_ms": timing.loader_stall_ms,
            }
        )
        if rank == 0 and step % 20 == 0:
            print(
                f"step {step}: loss={inner_loss:.4f} outer={did_outer} "
                f"wall_ms={timing.wall_time_ms:.2f} compute_ms={timing.compute_time_ms:.2f} "
                f"sync_ms={timing.sync_blocked_ms:.2f}",
                flush=True,
            )

    ended_at = datetime.now(UTC)
    total_wall_s = time.perf_counter() - t_wall_start

    if rank == 0:
        result = {
            "run_id": args.run_id,
            "H": args.H,
            "world_size": world_size,
            "n_params": n_params,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "micro_batch_size": args.micro_batch_size,
            "seq_len": SEQ_LEN,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "total_wall_s": total_wall_s,
            "step_records": step_records,
        }
        with open(args.output, "w") as f:
            json.dump(result, f)
        print(f"RUN_COMPLETE run_id={args.run_id} n_params={n_params} wrote={args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
