# diloco-measured

**Measured, Not Simulated — A Bandwidth-Controlled Evaluation of Semi-Synchronous LLM Training
on Commodity Ethernet.**

> Every published bandwidth-vs-synchronization curve in this literature is simulated. This
> repository contains measured ones, the code that produced them, and the discrepancy between
> the two.

**Document status:** `[CONFIRMED]` — a real, shaped, multi-bandwidth compute-utilization grid
has run on real GPU hardware (2026-08-14/15, see `CLAUDE.md` ADR-035). The figure and numbers
below are real, not placeholders. Scope: DiLoCo only (no DDP/FSDP2/LocalSGD driver yet), one
30.8M-parameter model, one repeat per point. See "What's real so far" below for the precise
scope and what's still open.

👉 **Read [`PRIOR_ART.md`](PRIOR_ART.md) first.** It states exactly what is and is not novel here.

---

## The headline result

Real DiLoCo training, 4× `g6e.2xlarge` (L40S, TCP-only interconnect, no NVLink/EFA), 30.8M
real parameters, real FineWeb-Edu data, real Linux `tc` bandwidth shaping verified against a
real `iperf3` measurement before every run (FR-02) — `H ∈ {1, 8, 32, 128}` × bandwidth
`∈ {50, 200, 1000, 5000} Mbit/s`, 16 points, zero shaping-gate failures, zero crashes:

![Compute utilization vs. bandwidth — measured vs. analytic, DiLoCo, 4 values of H](report/assets/fig1_cu_surface_diloco.png)

`cu_measured` (compute utilization — fraction of wall-clock time spent computing rather than
blocked on the network):

| H | 50 Mbit/s | 200 Mbit/s | 1 Gbit/s | 5 Gbit/s |
| --- | --- | --- | --- | --- |
| 1 | 0.07% | 0.32% | 1.74% | 7.76% |
| 8 | 0.55% | 2.48% | 12.35% | 40.24% |
| 32 | 2.41% | 10.01% | 36.01% | 68.10% |
| 128 | 11.55% | 36.09% | 71.52% | 87.00% |

This is DiLoCo's core value proposition, measured directly: at `H=1` (sync every step, like
DDP) compute utilization **collapses** under bandwidth scarcity — the GPUs sit almost entirely
idle waiting on the network. At `H=128`, the same 50 Mbit/s link sustains **11.55%** CU —
roughly **165× better utilization** from amortizing the same sync cost over 128 steps instead
of 1. Measured CU is below the naive analytic model at every one of the 16 points
(`discrepancy_link` 1.08×–1.92×, worst at the low-bandwidth/low-`H` corner) — real hardware
underperforms the literature's simulated model everywhere tested, and the size of the gap
itself varies with the operating point rather than being a single constant factor. Full
writeup: `CLAUDE.md` ADR-035.

**What this is not yet:** DiLoCo only — no training driver exists yet for DDP/FSDP2/LocalSGD,
so this is not the full 4-algorithm `phase_a.yaml` comparison. One repeat per point (no
variance estimate). A 30.8M-parameter model, not the 1B `phase_a.yaml` specifies. See
`experiments/01_cu_grid/NOTES.md` and ADR-035's "Not resolved" section for the complete,
honest list.

## What's real so far

| Piece | Status |
| --- | --- |
| 4× `g6e.2xlarge` + 1× `c7i.2xlarge` cluster, real AWS hardware | ✅ launched, validated, torn down after use |
| `torchft-nightly` + `torchtitan`, pinned and validated on a real L40S | ✅ ADR-032 |
| FR-01 network characterization (`iperf3` all-pairs, NCCL bandwidth curve, burst-decay probe) | ✅ `results/network/phase1-us-east-1b-20260814.json` |
| Real FineWeb-Edu data, gpt2-tokenized, staged per-node | ✅ ADR-034 |
| First real DiLoCo training measurement (`H` sweep, unshaped) | ✅ ADR-034 |
| **Shaped, multi-bandwidth DiLoCo grid** (real `tc` shaping + FR-02 gate, 16 points) | ✅ `results/raw/cu_grid-diloco-30m-h*-bw*-r0.json`, ADR-035 |
| DDP / FSDP2 / LocalSGD drivers (the rest of `phase_a.yaml`'s 4-algorithm comparison) | ⬜ not yet built |
| Convergence / time-to-target-loss runs | ⬜ not yet run |
| Fault injection, predictor validation | ⬜ not yet run |

## What this is

Four single-GPU EC2 nodes (`g6e.2xlarge`), connected only by TCP over ENA — no NVLink, no PCIe
peer-to-peer, no EFA. A Linux `tc` shaper converts interconnect bandwidth from a fixed hardware
property into a swept, independently *verified* experimental variable. On that rig we run DDP,
FSDP2, LocalSGD, and DiLoCo across synchronization intervals `H` and bandwidth levels, and
compare measured compute utilization against the literature's analytic model — through one
shared code path, so the comparison is defensible.

Full specification: [`CLAUDE.md`](CLAUDE.md) (the project's engineering brain — read before
changing anything).

## What this is NOT

- Not a new distributed-training algorithm.
- Not a reproduction of DiLoCo's quality results at scale.
- Not a production library. See `CLAUDE.md` §4.5 for the full non-goals list.

## Three-command reproduction (no GPU required)

```bash
git clone <repo>
cd diloco-measured
make figures   # regenerates every report figure from committed results/raw/
```

`make figures` runs with no GPU, no network access, and no AWS credentials (FR-11), and
produces the figure above (among others) from the committed `results/raw/` records — nothing
in it is hand-drawn.

## Repository map

| Path | What it is |
| --- | --- |
| `CLAUDE.md` | The master engineering specification. Read this first. |
| `PRIOR_ART.md` | What is and isn't novel here. |
| `LIMITATIONS.md` | Every known confound, stated by us. |
| `PLAYBOOK.md` | Practitioner-facing lookup: "N GPUs at X Mbit/s → use H=…" |
| `RESULTS.md` | Every run, including nulls, crashes, and abandoned lines. |
| `methods/` | How every number is computed, with every assumption listed. |
| `experiments/01_cu_grid/` | The scripts and raw telemetry behind the headline result above. |
| `src/diloco_measured/measurement/` | Needs GPUs + AWS. Never imported by analysis. |
| `src/diloco_measured/analysis/` | Pure. No GPU, no network, no credentials. |
| `results/` | Committed, append-only measurement corpus. |

## Status

Phase 2 in progress (`CLAUDE.md` v0.1). Network characterization (Phase 1) and a real shaped,
multi-bandwidth DiLoCo grid (Phase 2/3, `M3`/`M4` — the project's headline construct) are done
and committed, DiLoCo-only. Extending the same grid to DDP/FSDP2/LocalSGD, more repeats, and a
larger model are the main remaining work toward the full `phase_a.yaml` scope. See
`CLAUDE.md` §35 for the full phase plan and §40 for remaining open questions.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
