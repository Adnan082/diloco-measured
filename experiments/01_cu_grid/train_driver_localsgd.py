"""Real LocalSGD training driver -- the no-outer-optimizer ablation against DiLoCo (CLAUDE.md
§43 Glossary). Same model, same data, same StepTimer usage as `train_driver.py` (DiLoCo) and
`train_driver_ddp.py` (DDP) -- LocalSGDTrainer's outer step is a plain, EXPLICIT all-reduce
(not fused into backward() the way DDP's is), so this driver needs none of
`train_driver_ddp.py`'s calibration-probe machinery: `mark_compute_done()` after the inner
step's backward, `mark_sync_done()` after the explicit outer all-reduce, exactly like DiLoCo.
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

from diloco_measured.measurement.localsgd import LocalSGDTrainer
from diloco_measured.measurement.telemetry import StepTimer

VOCAB_SIZE = 50257  # gpt2 -- matches configs/models/30m-realvocab.toml
SEQ_LEN = 512


def build_model():
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

    trainer = LocalSGDTrainer(model, optimizer_cfg={"name": "adamw", "lr": 3e-4}, H=args.H)

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
            "algorithm": "localsgd",
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
