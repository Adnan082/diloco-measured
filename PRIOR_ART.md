# Prior Art

**Status:** `[PROPOSED]` — draft skeleton. Per ADR-008 this file must be completed and reviewed
*before* any measurement code is trusted, and it must be linked from the first screenful of
`README.md` (US-07).

Its job is narrow: state precisely what is already solved, by whom, and what this repository
adds on top of it. If a claim here cannot be traced to a source, it does not belong here.

---

## 1. What is NOT novel here

- **The DiLoCo algorithm itself.** Inner AdamW / outer Nesterov SGD over a pseudo-gradient is
  DeepMind's design (DiLoCo, and its descendants Streaming DiLoCo / Decoupled DiLoCo).
- **The scaling-law claims** connecting `H`, bandwidth, and compute utilization — those are the
  papers' analytic/simulated results, not ours.
- **LocalSGD** as a parameter-averaging baseline.
- **`torchft`'s fault-tolerant process groups** (lighthouse, HSDP, semi-sync training paths).
- **`torchtitan`'s** model definitions and FSDP2/DDP training loop.

## 2. What this repository claims as novel

> A controlled, verified, real-hardware bandwidth sweep — on real NCCL over commodity TCP,
> with bandwidth as an independent experimental variable — measuring compute utilization,
> bytes-on-wire, and time-to-target-loss, and comparing the result against the literature's
> analytic model through one shared code path.

That is the entire claim. See §2.3 and §2.7 of `CLAUDE.md` for the falsifiable hypothesis this
project tests.

## 3. Source-by-source positioning

`[CONFIRMED — from literature review; see CLAUDE.md §2.3]`

| Source | What it reports | Evidence type | How this project differs |
| --- | --- | --- | --- |
| Scaling Laws for DiLoCo (DeepMind, arXiv 2503.09799) | CU across bandwidth × H | Simulated / idealized | Measured on real NCCL/TCP |
| Eager Updates for Overlapped Communication (arXiv 2502.12996) | ~95% CU at 1–5 Gbit/s | Simulated (FLOPs rule at assumed 60% MFU) | Measured step time, not modeled |
| Decoupled DiLoCo (DeepMind, 2026) | Bandwidth-efficiency, goodput-under-failure | Simulated (charts explicitly stated as simulated, except the ML-quality chart) | Real fault injection (FR-09), real bandwidth sweep |
| Decoupled DiLoCo bandwidth-requirement table | Gbit/s for 50/75/90/95/99% CU | Model-derived | Same targets, measured |
| OpenDiLoCo (Prime Intellect, arXiv 2407.07852) | 90–95% CU, 4 workers | Measured, but *uncontrolled* natural links (127–935 Mbit/s), one configuration | Bandwidth is a *controlled, swept, verified* variable here |
| PyTorch/torchft on L40S | Throughput at a couple of sync intervals | Measured, unshaped, not a sweep | Full sweep with verification gate |

## 4. The gap, stated once

**Nobody has published a controlled experiment where interconnect bandwidth is the swept
independent variable, on real hardware, with real NCCL, validating those curves.**

## 5. Adjacent work not yet reviewed

`[UNKNOWN]` — to be filled in during Phase 0 literature pass: any additional 2025–2026
semi-synchronous training papers, any other bandwidth-shaping methodology precedent, any
prior `tc`/`tbf`-based ML networking benchmarks. Do not claim completeness of this list until
that pass is done.

## 6. History note

`[CONFIRMED]` The project originally under consideration (auto-tuned speculative decoding) was
abandoned after a prior-art review showed it was already solved and shipped by the vLLM team
(ADR-008). This file exists specifically so that mistake is not repeated silently a second time.
