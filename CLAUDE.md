# CLAUDE.md

**Project:** Measured, Not Simulated — A Bandwidth-Controlled Evaluation of Semi-Synchronous LLM Training on Commodity Ethernet
**Repository (intended):** `diloco-measured`
**Document status:** v0.1 — early implementation (Phase 1/2 in progress; first real training measurement landed, see ADR-034)
**Last updated:** 2026-08-14
**Owner:** Project author (solo engineer)

---

## READ THIS FIRST

This file is the **brain of the project**. It is not a README and it is not marketing copy. It is a living engineering specification.

Any future Claude session working on this repository must read this file before writing, changing, or deleting anything.

Three rules govern this document:

1. **Nothing here is invented and presented as fact.** Every statement is tagged `[CONFIRMED]`, `[PROPOSED]`, `[RECOMMENDATION]`, or `[UNKNOWN]`. If you cannot find a tag, treat the statement as `[PROPOSED]`.
2. **This is a measurement project, not a product.** The deliverable is *trustworthy numbers* plus the code that produced them. Correctness of measurement outranks features, speed, and elegance.
3. **Silent changes to the measurement path are the single worst thing you can do.** See §33 (Claude Coding Rules) and §44 (Change Management).

### Status tag legend

| Tag | Meaning |
| --- | --- |
| `[CONFIRMED]` | Explicitly established by the project owner, or a hard external constraint that has been verified |
| `[PROPOSED]` | A reasonable engineering proposal that has NOT been confirmed; safe to build on, must be flagged if it becomes load-bearing |
| `[RECOMMENDATION]` | Claude's engineering advice; the owner may overrule |
| `[UNKNOWN]` | Genuinely undecided or unmeasurable until we are on the hardware. Must not be guessed. Listed in §40 Open Questions |

---

# 1. Project Master Overview

## 1.1 Master Project Map

```text
PROJECT: diloco-measured
│
├── PROBLEM
│     The DiLoCo / semi-synchronous training literature reports
│     compute-utilization-vs-bandwidth curves that are SIMULATED.
│     Nobody has measured them on real hardware over a real network.
│
├── CLAIM UNDER TEST
│     "Bandwidth B is sufficient to reach compute utilization CU
│      at synchronization interval H"  ← derived analytically in the papers
│
├── METHOD
│     4 single-GPU nodes on commodity Ethernet
│     + Linux tc traffic shaping (bandwidth becomes an INDEPENDENT VARIABLE)
│     + a verification gate (never trust a rate you did not measure)
│     + measured CU vs the papers' analytic CU, computed side by side
│
├── ARTIFACTS
│     ├── measurement rig (code)
│     ├── raw result corpus (committed JSON/Parquet)
│     ├── calibrated H-predictor (CLI tool)
│     ├── figures + technical report
│     └── PRIOR_ART.md (honest positioning)
│
├── HARDWARE (hard constraint)
│     32 vCPU AWS GPU quota. NON-NEGOTIABLE. Cannot be raised.
│     → 4× g6e.2xlarge (8 vCPU, 1× L40S each) = exactly 32 vCPU
│     → + 1× CPU-only node (consumes zero GPU quota)
│
├── AUDIENCE
│     1. The author (portfolio / interview defence)
│     2. External reproducers (researchers, practitioners)
│     3. Future Claude sessions
│
└── SUCCESS
      A reader can regenerate every figure from committed raw data
      with zero GPUs, and every number traces to a verified measurement.
```

## 1.2 One-paragraph description `[CONFIRMED]`

We build a controlled measurement rig for semi-synchronous distributed LLM training and use it to test claims the field currently asserts from simulation. Four single-GPU EC2 nodes are connected only by TCP over ENA — no NVLink, no PCIe peer-to-peer, no EFA. A Linux `tc` traffic shaper on each node's egress converts interconnect bandwidth from a fixed hardware property into a swept experimental variable, verified before and after every run with `iperf3` and a NCCL all-reduce probe. On that rig we run DDP, FSDP2, LocalSGD, and DiLoCo at synchronization intervals `H ∈ {1, 8, 32, 128, 512}` across five bandwidth levels, measuring three things nobody reports together: achieved compute utilization, actual bytes on the wire per training token, and wall-clock time-to-target-loss on real convergence runs. We compare the measured compute-utilization surface against the analytic model the literature uses, quantify the discrepancy, and ship a **calibrated** predictor that maps a measured bandwidth to the synchronization interval minimizing time-to-target-loss.

## 1.3 What this project is NOT

- Not a new distributed training algorithm.
- Not a new model.
- Not a framework or a library others should depend on in production.
- Not a claim that DiLoCo is good or bad.
- Not a reproduction of DiLoCo's *quality* results at scale (we cannot afford that).

---

# 2. Problem Statement

## 2.1 The problem

Distributed LLM training has bifurcated. One branch runs on NVLink-and-InfiniBand superclusters. The other — growing rapidly — runs across cheap, poorly connected, heterogeneous, failure-prone capacity: multi-datacenter, multi-region, neocloud, and spot instances.

In the second branch, the binding constraint is not FLOPs. It is the interconnect. Synchronous data parallelism exchanges gradients every step and collapses when bandwidth is scarce. The established remedy is **semi-synchronous training**: each replica trains locally for `H` steps and synchronizes infrequently (LocalSGD, DiLoCo, and their descendants).

Every capacity-planning decision in that world reduces to one question:

> **Given the network I actually have, how often should I synchronize?**

The literature answers this question. But it answers it from a model, not from a measurement.

## 2.2 Why the problem exists

Running a controlled bandwidth sweep requires either (a) physically varying a real interconnect, which large labs cannot do on production clusters, or (b) an artificial shaping layer, which nobody has bothered to build and validate for this purpose. Simulation is cheaper and, for the papers' purposes (arguing that DiLoCo scales), sufficient. The gap persists because closing it is low-glamour infrastructure work that no lab is incentivised to do — and because most people who *could* do it have fast networks and therefore no reason to care.

## 2.3 Evidence that the gap is real `[CONFIRMED — from literature review, see PRIOR_ART.md]`

| Source | The bandwidth/H claim | Evidence type |
| --- | --- | --- |
| Scaling Laws for DiLoCo (DeepMind, arXiv 2503.09799) | Compute utilization across a range of bandwidth and synchronization cadences H; idealized wall-clock time under networks of varying bandwidth | **Simulated / idealized** |
| Eager Updates for Overlapped Communication (arXiv 2502.12996) | ~95% compute utilization at 1–5 Gbit/s for 1B/10B/100B models | **Simulated** (step time estimated from a FLOPs rule at an assumed 60% MFU) |
| Decoupled DiLoCo (DeepMind, 2026) | Bandwidth-efficiency and goodput-under-failure charts | **Simulated** (the source states the first two charts are based on simulated runs; only the ML-quality chart is from real runs) |
| Decoupled DiLoCo, bandwidth-requirement table | Gbit/s required to reach 50/75/90/95/99% compute utilization | **Model-derived** |
| OpenDiLoCo (Prime Intellect, arXiv 2407.07852) | 90–95% compute utilization across 4 workers | **Measured**, but on *uncontrolled* natural links (127–935 Mbit/s), one configuration |
| PyTorch/torchft on L40S | Throughput at a couple of sync intervals, TCP-only cluster | **Measured**, unshaped network, not a sweep |

**Nobody has published a controlled experiment where interconnect bandwidth is the swept independent variable, on real hardware, with real NCCL, validating those curves.** That is the gap.

## 2.4 Who experiences the problem

| Stakeholder | Pain today |
| --- | --- |
| Practitioner with a cheap multi-node cluster | Picks `H` by copying a number from a paper; has no idea if it is right for their network |
| Capacity planner at a lab doing multi-DC training | Budgets WAN bandwidth from a simulated table; risk of a costly over- or under-provision |
| Researcher proposing a new LoCo variant | Compares against a simulated baseline, so the comparison inherits the simulation's assumptions |
| The project author | Needs a defensible, honest, technically serious portfolio artifact for ML-systems / distributed-training roles |

## 2.5 What happens today

Practitioners either (a) copy `H = 30` / `H = 100` / `H = 500` from a paper, (b) run a small ad-hoc sweep with no bandwidth control and no verification, or (c) do not use semi-synchronous training at all because they cannot justify it.

## 2.6 What happens if we do not solve it

Nothing catastrophic — this is a research/portfolio project, not a system with users at risk. The cost of *not* doing it is: the simulated numbers remain unchecked, and the author does not have the artifact.

## 2.7 The falsifiable hypothesis `[PROPOSED]`

> Measured compute utilization on commodity Ethernet is materially lower than the analytic model predicts at the same nominal bandwidth, because the model assumes achievable bandwidth equals link bandwidth, homogeneous workers, and non-overlapped-but-instantaneous synchronization.

Three specific mechanisms we expect to observe and will attempt to attribute:

1. **NCCL over TCP does not achieve link bandwidth.** Expect roughly 50–75% of nominal for the relevant message sizes, varying as NCCL switches between ring and tree protocols. `[PROPOSED — must be measured]`
2. **Synchronization is a blocking barrier over stragglers.** You pay `max` over workers, not `mean`. Simulations assume homogeneous workers; EC2 instances are not perfectly homogeneous. `[PROPOSED]`
3. **Burstable ENA credits.** `g6e.2xlarge` networking is rated "up to 20 Gigabit" — that is burst, not sustained. A large all-reduce every `H` steps is exactly the pattern that drains credits, so *effective* bandwidth may itself depend on `H`. `[PROPOSED — this would be a novel second-order finding if confirmed]`

**The null result is also a result.** If measured ≈ simulated, we have produced the first empirical validation of widely cited curves. The project is designed so that no experimental outcome yields nothing. This framing is pre-committed here, before any data exists, precisely so it cannot be constructed post-hoc.

---

# 3. Product Vision

The "product" is a **research instrument plus its output**, in four parts:

1. **The rig** — a reproducible, verifiable measurement harness for semi-synchronous training over shaped networks.
2. **The corpus** — committed raw measurements that anyone can re-analyse without a GPU.
3. **The tool** — `diloco-measured plan`, a calibrated predictor that probes a live network and recommends `H`.
4. **The report** — figures and prose positioning the measurements against the simulated literature.

## 3.1 Core value proposition

> Every published bandwidth-vs-synchronization curve in this literature is simulated. This repository contains measured ones, the code that produced them, and the discrepancy between the two.

## 3.2 The product from four perspectives

### End user (a practitioner with a slow multi-node cluster)
Runs `diloco-measured plan --probe`. The tool measures their actual inter-node bandwidth, applies the calibrated model, and prints a recommended `H` with expected tokens/s, expected compute utilization, and expected bytes-on-wire per hour. Alternatively they read `PLAYBOOK.md` and look up their situation in a table.

### Reproducer / reviewer (a researcher or a skeptical interviewer)
Clones the repo, runs `make figures` on a laptop with no GPU, and regenerates every figure in the report from `results/raw/`. Opens `results/network/` and confirms that every claimed bandwidth was independently verified by `iperf3` before the run. Reads `PRIOR_ART.md` and sees exactly what is and is not novel.

### Developer (future Claude sessions, or a collaborator)
Reads this file. Understands the module boundaries, the immutability rules around `results/raw/`, the harness-versioning discipline, and the fact that measurement code and analysis code must never be entangled.

### System / operator (the author during the compute week)
Runs `make cluster-up`, `make network-characterize`, `make grid`, `make converge`, `make cluster-down`. Watches a Grafana dashboard. Gets a loud failure if a shaping verification gate does not pass, rather than a quietly wrong number.

---

# 4. Goals

## 4.1 Primary goals `[CONFIRMED]`

| # | Goal | Success measure |
| --- | --- | --- |
| G1 | Produce a **measured** compute-utilization surface over (bandwidth × H × algorithm) on real hardware | A populated grid with ≥4 verified bandwidth levels and ≥4 H values, 3 repeats each |
| G2 | Quantify the discrepancy vs the literature's analytic model | A discrepancy factor `F` reported at 50/75/90/95% CU, with confidence intervals |
| G3 | Measure **time-to-target-loss** (not just throughput) for the main algorithms | ≥10 completed convergence runs at a fixed token budget against a single-GPU reference loss |
| G4 | Ship a calibrated `H`-predictor validated on a held-out configuration | Predicted vs measured optimal `H`, with wall-clock regret reported |
| G5 | Full reproducibility without GPUs | `make figures` regenerates 100% of report figures from committed raw data |

## 4.2 Secondary goals `[PROPOSED]`

| # | Goal |
| --- | --- |
| G6 | Measure the additional wire reduction and loss cost of int8 pseudo-gradient compression with error feedback |
| G7 | Measure the cost of a worker failure as a function of `H` (detect→resume time, wasted steps, loss delta) |
| G8 | Publish the NCCL-over-TCP achieved-bandwidth-vs-message-size characterization as a standalone artifact (useful independent of the rest) |
| G9 | Upstream any bug found in `torchft`'s experimental LocalSGD/DiLoCo paths as a GitHub issue or PR |

## 4.3 "Business" goals (portfolio) `[CONFIRMED]`

- The author can defend every number in an interview without hedging.
- The repository reads as senior work: prior art first, claims scoped, negative results published.
- Two or three resume bullets with real measured figures.

## 4.4 Technical goals

- Measurement integrity: no unverified value ever enters a result record.
- Zero-GPU analysis path.
- Two independent implementations of the correctness-critical algorithm (DiLoCo) that agree.

## 4.5 Non-goals `[CONFIRMED]`

Explicitly out of scope for v1:

| Non-goal | Why |
| --- | --- |
| Inventing a new semi-synchronous algorithm | Solved space; we are measuring, not inventing |
| Beating published loss numbers | We have 4 GPUs for a week |
| Models above ~1B parameters | At 50 Mbit/s a per-step all-reduce of 8B params measures only the shaper |
| Multi-node-per-replica (FSDP *inside* a DiLoCo worker) | Each worker is one GPU; the two-level hierarchy is untestable here. Must be stated as a limitation, not hidden |
| A production-grade library, packaging to PyPI, or a stable public API | Research instrument, not a dependency |
| A web UI, user accounts, multi-tenancy, or a hosted service | No such users exist |
| Real WAN emulation with realistic latency/jitter/loss as a *primary* axis | `[PROPOSED]` — a `netem` sub-experiment is a stretch goal; the primary sweep is bandwidth-only, and this limitation must be stated prominently |
| Cross-cloud or cross-region measurement | Cost and time |
| Asynchronous / decoupled DiLoCo variants | Scope; note as future work |

---

# 5. Constraints

## 5.1 Hard constraints `[CONFIRMED]`

| Constraint | Value | Consequence |
| --- | --- | --- |
| AWS GPU vCPU quota | **32 vCPU, cannot be increased** | Determines the entire hardware topology. Any design requiring more GPU vCPUs is invalid |
| Compute window | ~7 days | Forces a fixed experiment budget; no open-ended exploration |
| Budget | ~$650–800 on-demand target | Blocks continuous 168-hour cluster uptime; requires spin-up/spin-down discipline |
| No NVLink, no PCIe P2P, no EFA at these instance sizes | — | All inter-GPU traffic is NCCL over TCP. This is the point of the project, not a limitation to work around |
| Single engineer | — | No parallel workstreams; the plan must be strictly sequential |

## 5.2 Derived hardware topology `[CONFIRMED]`

```text
GPU FLEET — 4 × g6e.2xlarge  (4 × 8 = 32 vCPU, exactly at quota)
  per node: 8 vCPU AMD EPYC 7R13 | 64 GiB RAM
            1 × NVIDIA L40S (45 GiB usable, Ada sm89, no NVLink)
            450 GB local NVMe
            ENA, "up to 20 Gigabit" (BURSTABLE — must be measured)
  ~$2.24/hr each on-demand (us-east-1) → ~$8.97/hr for the fleet

CONTROL NODE — 1 × c7i.2xlarge  (CPU-only → consumes ZERO GPU quota)
  torchft lighthouse | torchrun rendezvous | Prometheus/Grafana
  dataset tokenization | result aggregation
  ~$0.36/hr
```

`[RECOMMENDATION]` Place all five instances in one **cluster placement group**, one AZ, one subnet, with a security group that allows *all* traffic between members of the group (not just SSH). The most common multi-node NCCL failure is a security group that only opens port 22.

`[UNKNOWN]` Which region/AZ has `g6e` capacity at the time of the run. Must be checked before Day 1. See §40 Q1.

## 5.3 What the constraint buys us

The 32-vCPU cap is normally a disadvantage. Here it is the enabling condition: it forces four *separate* single-GPU hosts on commodity Ethernet, which is precisely the regime the DiLoCo literature simulates and never measures. **Do not treat this constraint as something to escape.** It is the experimental apparatus.

---

# 6. Requirements

Requirements use the format: ID, statement, actor, preconditions, trigger, main flow, alternative flows, failure conditions, expected result, data involved, status.

## 6.1 Functional requirements — Measurement layer

---

### FR-01 — Network characterization

**Statement:** The system shall characterize the achievable inter-node network before any training experiment runs.
**Actor:** Operator (via CLI), System
**Preconditions:** Cluster is up; all nodes reachable; `iperf3` installed on all nodes
**Trigger:** `make network-characterize` or `diloco-measured network characterize`

**Main flow:**
1. Operator invokes network characterization.
2. System runs `iperf3` between all ordered node pairs, both directions, 60 s each.
3. System runs a NCCL all-reduce probe across all 4 ranks, sweeping message size log-spaced from 1 MiB to 4 GiB.
4. System records, for each configured shaping rate, requested rate, measured `iperf3` rate, and the NCCL achieved-bandwidth curve.
5. System runs a 10-minute sustained transfer at the unshaped rate to detect ENA burst-credit decay.
6. System writes a `NetworkProfile` record to `results/network/`.

**Alternative flows:**
- 4a. A node is unreachable → abort, report which node, do not write a partial profile.
- 5a. Sustained throughput decays > 20% over 10 minutes → record `burst_decay_detected: true` and the decay curve. This is a finding, not an error.

**Failure conditions:** any pair fails `iperf3`; NCCL probe cannot initialize; shaping cannot be applied.
**Expected result:** A committed `NetworkProfile` JSON, which is a precondition for every subsequent experiment.
**Data involved:** `NetworkProfile`, `NcclBandwidthCurve`, `ShapingFidelityRecord`
**Status:** `[CONFIRMED]`

---

### FR-02 — Bandwidth shaping with a verification gate

**Statement:** The system shall set egress bandwidth on every node to a requested rate, and shall verify the achieved rate before allowing an experiment to proceed.
**Actor:** System
**Preconditions:** Root/sudo on each node; `tc` available; a `NetworkProfile` exists
**Trigger:** Start of any run whose spec includes a shaped bandwidth

**Main flow:**
1. System applies `tc qdisc ... tbf rate <R> burst <B> latency <L>` on `ens5` egress on all 4 nodes.
2. System runs a short `iperf3` (≥15 s) between two nodes.
3. System asserts `|measured − requested| / requested ≤ tolerance` (`[PROPOSED]` tolerance = 10%).
4. On pass, System records the **measured** rate into the run record and proceeds.
5. On run completion, System restores the original qdisc.

**Alternative flows:**
- 3a. Assertion fails → retry shaping once. On second failure, **abort the run** and write a `ShapingFailure` record. Never proceed with an unverified rate.
- 5a. Restore fails → mark the node dirty; subsequent runs on that node abort until restore succeeds.

**Failure conditions:** `tc` unavailable; permission denied; shaping unstable across retries.
**Expected result:** Every run record contains `bandwidth_measured_bps`, never only `bandwidth_requested_bps`.
**Data involved:** `ShapingRequest`, `ShapingVerification`, `Run`
**Status:** `[CONFIRMED]` — this is the central integrity mechanism of the project.

---

### FR-03 — Instrumented training run

**Statement:** The system shall execute a distributed training run under a given (algorithm, H, model, bandwidth) specification and emit per-step telemetry plus an aggregated result.
**Actor:** Operator, System
**Preconditions:** Cluster up; shaping verified (FR-02); tokenized dataset present on every node's local NVMe
**Trigger:** `diloco-measured run --spec <path>`

**Main flow:**
1. System validates the `ExperimentSpec` against schema.
2. System resolves and records the full environment fingerprint (§6.3 FR-08).
3. System launches `torchrun` across 4 nodes via the control node's rendezvous.
4. Each rank executes the configured algorithm for the configured number of steps or tokens.
5. Each rank records per-step: wall time, compute time, sync-blocked time, optimizer time, dataloader stall, loss, tokens processed, peak memory.
6. Rank 0 aggregates, computes derived metrics, and writes a `RunResult` to `results/raw/`.
7. System restores network state.

**Alternative flows:**
- 4a. A rank dies unexpectedly (not injected) → mark run `status: crashed`, preserve partial telemetry, do not emit a `RunResult` usable for analysis.
- 5a. Dataloader stall exceeds `[PROPOSED]` 5% of step time → set `loader_bound_warning: true` in the result. Analysis must exclude or flag these.

**Failure conditions:** OOM; NCCL timeout; rendezvous failure; spec invalid.
**Expected result:** One schema-valid `RunResult` JSON plus a per-step Parquet file.
**Data involved:** `ExperimentSpec`, `StepRecord`, `RunResult`, `EnvironmentFingerprint`
**Status:** `[CONFIRMED]`

---

### FR-04 — Compute utilization: measured and simulated, side by side

**Statement:** For every run, the system shall compute compute utilization by direct measurement AND by the literature's analytic model, using the same input schema, and record both.
**Actor:** System
**Preconditions:** A completed run with per-step telemetry
**Trigger:** Run aggregation

**Main flow:**
1. Measured: `CU_measured = Σ compute_time / Σ total_step_time` over the measurement window (post-warmup).
2. Analytic: `CU_analytic = f(H, t_compute, bytes_synced, bandwidth)` using the model form documented in `methods/cu_model.md`, with all assumptions listed.
3. System computes both a **link-bandwidth** variant (papers' assumption) and an **achieved-bandwidth** variant (fed the measured NCCL bandwidth for the relevant message size).
4. System records all three values plus the discrepancy ratios.

**Alternative flows:**
- 3a. The message size falls outside the measured NCCL curve → interpolate, and set `nccl_bw_interpolated: true`.

**Failure conditions:** Missing NCCL curve for this bandwidth level → analytic-achieved variant is `null`, not guessed.
**Expected result:** Every result row carries `cu_measured`, `cu_analytic_link`, `cu_analytic_achieved`.
**Data involved:** `CUObservation`, `NcclBandwidthCurve`
**Status:** `[CONFIRMED]` — this comparison *is* the project's headline contribution.

> **Design note.** Computing both through one code path with a shared input schema is deliberate. It removes the "you compared apples to oranges" objection, which is the first thing a serious reviewer will raise.

---

### FR-05 — Bytes-on-wire accounting

**Statement:** The system shall predict and independently measure the bytes transferred per training token.
**Actor:** System
**Preconditions:** Run in progress
**Trigger:** Run start and run end

**Main flow:**
1. Before the measurement window, System snapshots `/proc/net/dev` on every node.
2. System computes the analytic prediction from first principles (ring all-reduce `2N(P−1)/P` bytes per rank per sync; frequency `1/H`).
3. After the window, System snapshots again and differences.
4. System records predicted bytes, measured bytes, ratio, and per-token normalizations.

**Failure conditions:** Other traffic on the interface pollutes the counter → mitigate by running nothing else on the nodes during a measurement window, and by recording the idle-baseline drift.
**Expected result:** `wire_bytes_predicted`, `wire_bytes_measured`, `wire_overhead_ratio`.
**Data involved:** `WireAccount`
**Status:** `[CONFIRMED]`

> **Design note.** When prediction and measurement agree, you have demonstrated understanding of the collectives. When they diverge, the gap is TCP/IP headers, NCCL protocol overhead, and retransmits — and quantifying it is itself a result.

---

### FR-06 — Convergence run and time-to-target-loss

**Statement:** The system shall run training to a fixed token budget and compute wall-clock time to reach a reference target loss.
**Actor:** Operator, System
**Preconditions:** A single-GPU reference run exists defining the target loss
**Trigger:** `diloco-measured converge --spec <path>`

**Main flow:**
1. System runs the single-GPU reference at the same token budget; records final loss `L*` and the full curve.
2. For each configuration, System trains to the same token budget, recording loss vs tokens and loss vs wall clock.
3. System computes `TTTL` = wall-clock time at which validation loss first reaches `L*`.
4. If `L*` is never reached, System records `TTTL: null` and `final_loss`, and analysis must report it as "did not reach target" rather than as a large number.

**Alternative flows:**
- 3a. Loss curve is noisy near `L*` → use a smoothed curve (`[PROPOSED]` EMA over the last 5 evals) and record both raw and smoothed crossings.

**Failure conditions:** Divergence (loss NaN/spike) → record `status: diverged` with the step at which it happened. Divergence at large `H` is a legitimate finding.
**Expected result:** `ConvergenceCurve` + `TTTL` per configuration.
**Data involved:** `ConvergenceCurve`, `RunResult`
**Status:** `[CONFIRMED]`

---

### FR-07 — Calibrated H-predictor

**Statement:** The system shall fit a predictor mapping (measured bandwidth, model size, local step time) to a recommended `H`, and shall be validatable on held-out configurations.
**Actor:** End user, System
**Preconditions:** Phase A and Phase B result corpora exist
**Trigger:** `diloco-measured plan --probe` or `--bandwidth <bps> --model <cfg>`

**Main flow:**
1. With `--probe`, System measures live bandwidth (FR-01 subset).
2. System evaluates the calibrated model over candidate `H` values.
3. System returns recommended `H`, expected tokens/s, expected CU, expected bytes/hour, and a confidence qualifier.
4. System states the calibration domain and warns loudly if the request is outside it (`[CONFIRMED]` requirement — extrapolation must never be silent).

**Failure conditions:** Request outside the calibration domain → return the recommendation *with an explicit extrapolation warning*, never silently.
**Expected result:** A recommendation plus its provenance.
**Data involved:** `PredictorModel`, `NetworkProfile`
**Status:** `[CONFIRMED]`

---

### FR-08 — Environment fingerprinting

**Statement:** Every result record shall embed a complete environment fingerprint.
**Actor:** System
**Trigger:** Run start

**Captured:** harness git SHA + dirty flag, `harness_version`, PyTorch/torchtitan/torchft/NCCL/CUDA/driver versions, instance types, AZ, placement group ID, `nvidia-smi topo -m`, locked clock settings, `NCCL_*` environment variables, kernel version, `tc` qdisc dump, dataset shard checksum, random seeds.
**Expected result:** A result cannot be written without a complete fingerprint.
**Status:** `[CONFIRMED]`

---

### FR-09 — Fault injection `[PROPOSED — secondary goal G7]`

**Statement:** The system shall kill a designated worker at a scheduled time and measure recovery.
**Main flow:** schedule → `SIGKILL` rank *k* at time *t* → observe lighthouse reconfiguration → record detect→resume latency, steps lost since last sync, and post-recovery loss trajectory.
**Failure conditions:** The job hangs instead of recovering → record `recovery: hung` with a timeout. For DDP this is the expected outcome and is the point of the comparison.
**Status:** `[PROPOSED]`

---

### FR-10 — Pseudo-gradient compression `[PROPOSED — secondary goal G6]`

**Statement:** The system shall support pluggable compression of the pseudo-gradient prior to all-reduce, with fp16, int8-with-error-feedback, and top-k codecs.
**Invariant:** the error-feedback residual accumulator must persist across outer steps and must be included in checkpoints.
**Status:** `[PROPOSED]`

## 6.2 Functional requirements — Analysis layer

---

### FR-11 — GPU-free reproducible analysis

**Statement:** All analysis and figure generation shall run on a machine with no GPU, no network access, and no AWS credentials, reading only from `results/raw/` and `results/network/`.
**Trigger:** `make figures`
**Expected result:** Byte-stable or numerically-stable regeneration of every figure in the report.
**Status:** `[CONFIRMED]` — this is what makes the work checkable, and therefore credible.

---

### FR-12 — Result schema validation

**Statement:** Every record written to `results/` shall validate against a versioned JSON Schema, and analysis shall refuse to load records that do not.
**Status:** `[CONFIRMED]`

---

### FR-13 — Provenance in figures

**Statement:** Every generated figure shall embed (in caption or metadata) the harness version, the number of runs contributing, and whether values are measured, analytic, or interpolated.
**Status:** `[RECOMMENDATION]` — cheap, and it eliminates a whole class of reviewer doubt.

## 6.3 Non-functional requirements

| ID | Requirement | Status |
| --- | --- | --- |
| NFR-01 | No result may contain a network rate that was requested but not verified | `[CONFIRMED]` |
| NFR-02 | `results/raw/` is append-only and immutable; records are never edited or deleted, only superseded by a new harness version | `[CONFIRMED]` |
| NFR-03 | Measurement code and analysis code live in separate modules with no analysis→measurement imports | `[CONFIRMED]` |
| NFR-04 | Any change to a measurement code path increments `harness_version`; results across versions are never silently pooled | `[CONFIRMED]` |
| NFR-05 | Total cloud spend must stay under the budget ceiling; the harness logs cumulative cluster-hours | `[PROPOSED]` |
| NFR-06 | A single Phase-A grid point must complete in under ~5 minutes so the grid fits an overnight block | `[PROPOSED]` |
| NFR-07 | No AWS credentials, tokens, or private endpoints in the repository or in any result record | `[CONFIRMED]` |
| NFR-08 | GPU clocks locked for all timed runs; the lock state recorded in the fingerprint | `[CONFIRMED]` |
| NFR-09 | Warmup steps discarded from every measurement window; the discard count recorded | `[CONFIRMED]` |

---

# 7. Actors and Roles

This project has **no end-user accounts, no authentication, and no multi-tenancy.** Roles are operational, not authorization-based. A permission matrix over application features would be invented complexity; the meaningful matrix is over *infrastructure capability*.

| Actor | Description |
| --- | --- |
| **Operator** | The author, running experiments. Has AWS credentials and SSH to all nodes |
| **Analyst** | Anyone (including the author, later) analysing committed results. Needs no credentials, no GPU |
| **Reproducer** | External party re-running experiments on their own hardware |
| **Reviewer** | Reads the report and inspects raw data. Read-only, no execution |
| **CI** | Automated. Runs unit and CPU-integration tests. Never touches AWS |
| **Future Claude session** | Reads this file, modifies code under §33 rules |

## 7.1 Capability matrix

| Capability | Operator | Analyst | Reproducer | Reviewer | CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Launch/terminate GPU cluster | ✓ | ✗ | ✓ (own account) | ✗ | ✗ |
| Apply `tc` shaping | ✓ | ✗ | ✓ | ✗ | ✗ |
| Execute training runs | ✓ | ✗ | ✓ | ✗ | ✗ |
| Write to `results/raw/` | ✓ | ✗ | ✓ (own fork) | ✗ | ✗ |
| Read `results/raw/` | ✓ | ✓ | ✓ | ✓ | ✓ |
| Regenerate figures | ✓ | ✓ | ✓ | ✓ | ✓ |
| Run unit + CPU integration tests | ✓ | ✓ | ✓ | ✓ | ✓ |
| Modify measurement code | ✓ | ✗ | ✓ (own fork) | ✗ | ✗ |
| Requires AWS credentials | ✓ | ✗ | ✓ | ✗ | ✗ |
| Requires a GPU | ✓ | ✗ | ✓ | ✗ | ✗ |

---

# 8. User Stories and Acceptance Criteria

### US-01 — Trustworthy bandwidth manipulation
> As the **Operator**, I want the harness to refuse to run when the requested bandwidth was not actually achieved, so that no unverified number ever reaches a figure.

**Acceptance criteria**
```text
GIVEN a run spec requesting 1 Gbit/s
WHEN shaping is applied and iperf3 measures 0.62 Gbit/s (38% error, tolerance 10%)
THEN the harness retries shaping once
AND on repeated failure aborts the run
AND writes a ShapingFailure record
AND writes NO RunResult
```
```text
GIVEN shaping verification passes at 0.96 Gbit/s for a 1 Gbit/s request
WHEN the run completes
THEN the RunResult records bandwidth_measured_bps = 0.96e9
AND bandwidth_requested_bps = 1.0e9
AND all analysis keys on the MEASURED value
```

---

### US-02 — The headline comparison
> As the **Operator**, I want measured and analytic compute utilization computed from one code path, so that the comparison is defensible.

**Acceptance criteria**
```text
GIVEN a completed run at H=32, 1 Gbit/s, 1B model
WHEN aggregation runs
THEN the result contains cu_measured, cu_analytic_link, cu_analytic_achieved
AND all three derive from the same ExperimentSpec + telemetry inputs
AND methods/cu_model.md documents every assumption in cu_analytic_*
```

---

### US-03 — Convergence, not just throughput
> As the **Analyst**, I want time-to-target-loss for each algorithm, so that I can compare speed and quality together instead of separately.

**Acceptance criteria**
```text
GIVEN a single-GPU reference run defining target loss L*
WHEN DiLoCo H=32 at 200 Mbit/s completes its token budget
THEN TTTL is computed from the wall-clock axis
AND if L* was never reached, TTTL is null and final_loss is reported
AND the analysis renders "did not reach target" rather than an inflated time
```

---

### US-04 — Reproduction without hardware
> As a **Reviewer**, I want to regenerate every figure on my laptop, so that I can check the claims without renting GPUs.

**Acceptance criteria**
```text
GIVEN a fresh clone on a machine with no GPU and no AWS credentials
WHEN I run `make figures`
THEN every figure in the report is produced from results/raw/
AND no step attempts network access or GPU initialization
AND the run completes in under 5 minutes
```

---

### US-05 — Actionable recommendation
> As an **End user** with a slow cluster, I want a recommended `H` for my measured network, so that I do not have to copy a number from a paper.

**Acceptance criteria**
```text
GIVEN a live 4-node cluster with unknown bandwidth
WHEN I run `diloco-measured plan --probe --model 1b`
THEN the tool measures bandwidth, prints a recommended H,
     expected tokens/s, expected CU, and expected bytes/hour
AND if my bandwidth is outside the calibration domain,
     the output carries an explicit extrapolation warning
```

---

### US-06 — Algorithm correctness confidence
> As a **Developer**, I want two independent DiLoCo implementations that agree, so that a bug in one does not silently corrupt the entire study.

**Acceptance criteria**
```text
GIVEN the same seed, model, data order, and H
WHEN the in-repo diloco.py and the torchft path each run 200 steps
THEN the loss curves agree within a documented tolerance
AND a divergence beyond tolerance FAILS CI
```

---

### US-07 — Honest positioning
> As a **Reviewer**, I want to know immediately what is novel and what is not, so that I can trust the rest of the document.

**Acceptance criteria**
```text
GIVEN the repository root
WHEN I open README.md
THEN PRIOR_ART.md is linked in the first screenful
AND PRIOR_ART.md names DiLoCo, Streaming DiLoCo, Eager Updates,
    OpenDiLoCo, Decoupled DiLoCo and the Scaling Laws work
AND states plainly that the algorithm and the scaling laws are not ours
AND states precisely what is: the measured bandwidth sweep
```

---

### US-08 — Failure cost as a function of H `[PROPOSED]`
> As the **Analyst**, I want the measured cost of a worker death at each `H`, so that the bandwidth/robustness tradeoff is quantified rather than argued.

**Acceptance criteria**
```text
GIVEN DiLoCo running at H=512
WHEN rank 3 is SIGKILLed at t=10min
THEN the harness records detect→resume latency,
     the number of inner steps lost, and the post-recovery loss delta
AND the same experiment at H=32 produces a comparable record
```

---

# 9. User Journeys

## 9.1 Journey A — Operator executes the primary experiment (the critical path)

```text
Operator (laptop, no GPU)
   │  writes/validates ExperimentSpec files
   ▼
make cluster-up
   │  ├─ create cluster placement group
   │  ├─ launch 4× g6e.2xlarge + 1× c7i.2xlarge
   │  ├─ security group: all traffic within group
   │  └─ wait for SSH + nvidia-smi on all nodes
   ▼
make bootstrap
   │  ├─ install deps (pinned), lock GPU clocks
   │  ├─ sync tokenized shards S3 → each node's NVMe
   │  └─ verify checksums
   ▼
make network-characterize                   ◄── FR-01
   │  iperf3 all-pairs · NCCL BW curve · burst-decay probe
   │  writes results/network/<profile>.json
   ▼
make smoke                                   ◄── E2E gate
   │  4 nodes · tiny model · 20 steps · valid RunResult emitted
   ▼
make grid PHASE=A                            ◄── FR-02, FR-03, FR-04, FR-05
   │  for each (bandwidth, algorithm, H):
   │     shape → VERIFY → run → aggregate → restore
   │     ├─ verification fails → abort point, log, continue grid
   │     └─ run crashes       → mark crashed, continue grid
   ▼
make converge PHASE=B                        ◄── FR-06
   │  reference run first, then configurations, overnight
   ▼
make cluster-down                            ◄── cost control
   ▼
make figures                                 ◄── FR-11 (no GPU needed)
   ▼
Report + PLAYBOOK.md + resume bullets
```

**Data changes:** append-only writes to `results/raw/`, `results/network/`, `results/environment/`.
**External calls:** AWS EC2/S3, Hugging Face Hub (dataset, once, from the control node).
**Success state:** a populated grid plus convergence curves, all schema-valid.
**Failure states and recovery:** see §29 Reliability and §38 Risks.

## 9.2 Journey B — Reviewer verifies a claim

```text
Reviewer
  ▼ reads README headline claim ("90% CU needed F× more bandwidth than modelled")
  ▼ opens PRIOR_ART.md → confirms what is and isn't novel
  ▼ opens results/network/ → confirms every bandwidth was iperf3-verified
  ▼ opens methods/cu_model.md → checks the analytic model's assumptions
  ▼ clones, runs `make figures` on a laptop
  ▼ figures match the report → claim is checkable → trust established
```

## 9.3 Journey C — End user gets a recommendation

```text
Practitioner with N GPU nodes on unknown Ethernet
  ▼ pip install -e . (or docker run)
  ▼ diloco-measured plan --probe --model 1b --gpus 4
  ▼ tool measures bandwidth (~15s)
  ▼ tool evaluates calibrated model
  ▼ prints: recommended H=64 · ~41,200 tok/s · 92% CU · 0.9 GB/hr on wire
  ▼ (if outside calibration domain) prints EXTRAPOLATION WARNING with the domain bounds
```

## 9.4 Journey D — Future Claude extends the work

```text
Claude session opens the repo
  ▼ reads CLAUDE.md §33 (Coding Rules) and §44 (Change Management)
  ▼ identifies whether the task touches the MEASUREMENT path or the ANALYSIS path
  ├─ MEASUREMENT → must bump harness_version, must not pool old results,
  │                must state the impact on already-collected data
  └─ ANALYSIS    → free to iterate; results/raw/ remains untouched
  ▼ implements, tests, updates §41 Decision Log and §40 Open Questions
```

---

# 10. Functional Workflows

## 10.1 The run lifecycle (the single most important flow)

```text
ExperimentSpec (YAML)
      │
      ▼
[1] Schema validation ───────────► invalid → ABORT, no side effects
      │
      ▼
[2] Preconditions
      ├─ NetworkProfile exists for this bandwidth level?
      ├─ Dataset shards present + checksum OK on all nodes?
      ├─ GPU clocks locked?
      └─ No dirty qdisc left by a previous run?
      │                                    any fail → ABORT
      ▼
[3] Apply tc shaping on all nodes
      │
      ▼
[4] VERIFICATION GATE  (iperf3 + NCCL probe)
      │        │
      │        └─ outside tolerance → retry once → still bad → ABORT
      │                                            + ShapingFailure record
      ▼ pass (record MEASURED rate)
[5] Environment fingerprint captured
      │
      ▼
[6] torchrun launch across 4 nodes (rendezvous on control node)
      │
      ▼
[7] Warmup steps (discarded, count recorded)
      │
      ▼
[8] Measurement window
      ├─ per-step CUDA-event decomposition
      ├─ /proc/net/dev snapshots (start/end)
      ├─ periodic loss eval (convergence runs only)
      └─ DCGM sampling
      │
      ▼
[9] Aggregation on rank 0
      ├─ CU measured
      ├─ CU analytic (link) + CU analytic (achieved)
      ├─ wire bytes predicted + measured
      ├─ throughput, MFU, step-time percentiles
      └─ schema validation of the output record
      │
      ▼
[10] Write RunResult + StepRecords  (APPEND ONLY)
      │
      ▼
[11] Restore network state ──────► restore fails → mark node dirty
      │
      ▼
   DONE
```

## 10.2 The analysis flow

```text
results/raw/*.json + results/network/*.json
      │
      ▼
Loader (schema-validated, refuses invalid records)
      │
      ▼
Filter (exclude: crashed, loader_bound_warning, harness_version mismatch)
      │
      ▼
Aggregate across repeats (median + IQR; never mean-only)
      │
      ├──► CU surface (measured vs analytic)  → Fig 1 (headline)
      ├──► Required-bandwidth table            → Table 1
      ├──► NCCL BW vs message size             → Fig 2 (mechanism)
      ├──► TTTL vs bandwidth                   → Fig 3
      ├──► Loss@budget vs H + throughput vs H  → Fig 4
      ├──► Bytes-on-wire per token             → Fig 5
      └──► Predictor: predicted vs measured H  → Fig 6
```

## 10.3 The DiLoCo inner/outer loop (algorithmic reference)

```text
θ_outer  ← initial weights (broadcast to all replicas)
for round r = 0, 1, 2, ...:
    θ_inner ← θ_outer                       # each replica starts from the global model
    for h = 1..H:                           # INNER: purely local, zero communication
        θ_inner ← AdamW_step(θ_inner, local_batch)
    Δ ← θ_outer − θ_inner                   # pseudo-gradient (a.k.a. outer gradient)
    Δ̄ ← all_reduce_mean(Δ)                  # OUTER: the ONLY cross-replica traffic
    θ_outer ← NesterovSGD_step(θ_outer, Δ̄)
    broadcast θ_outer to all replicas
```

Key invariants that MUST hold and MUST be tested:
- Inner optimizer state **persists across rounds** (this is what distinguishes DiLoCo from naive FedOpt).
- After the outer step, all replicas hold **bit-identical** `θ_outer` (within all-reduce determinism tolerance).
- Communication volume per round is `O(N)` bytes, independent of `H`; per *step* it is `O(N/H)`.
- With compression enabled, the error-feedback residual persists across rounds and is checkpointed.

---

# 11. System Architecture

## 11.1 Architecture diagram

```text
                        ┌──────────────────────────────────────┐
                        │      OPERATOR LAPTOP (no GPU)        │
                        │  specs · analysis · figures · report │
                        └──────────────┬───────────────────────┘
                                       │ ssh / aws cli / git
                                       ▼
              ┌────────────────────────────────────────────────────┐
              │   CONTROL NODE — c7i.2xlarge (0 GPU quota used)    │
              │  ┌──────────────┐ ┌───────────────┐ ┌───────────┐  │
              │  │ torchrun     │ │ torchft       │ │Prometheus │  │
              │  │ rendezvous   │ │ lighthouse    │ │ + Grafana │  │
              │  └──────────────┘ └───────────────┘ └───────────┘  │
              │  dataset tokenization · result aggregation · S3     │
              └───────┬────────────┬────────────┬────────────┬─────┘
                      │            │            │            │
        ══════════════╪════════════╪════════════╪════════════╪══════════
          CLUSTER PLACEMENT GROUP · ONE SUBNET · SG allows all intra-group
        ══════════════╪════════════╪════════════╪════════════╪══════════
                      │            │            │            │
              ┌───────▼──┐  ┌──────▼───┐  ┌─────▼────┐  ┌────▼─────┐
              │  NODE 0  │  │  NODE 1  │  │  NODE 2  │  │  NODE 3  │
              │ g6e.2xl  │  │ g6e.2xl  │  │ g6e.2xl  │  │ g6e.2xl  │
              │ 1× L40S  │  │ 1× L40S  │  │ 1× L40S  │  │ 1× L40S  │
              │ 8 vCPU   │  │ 8 vCPU   │  │ 8 vCPU   │  │ 8 vCPU   │
              │ 450G NVMe│  │ 450G NVMe│  │ 450G NVMe│  │ 450G NVMe│
              ├──────────┤  ├──────────┤  ├──────────┤  ├──────────┤
              │ tc/tbf   │  │ tc/tbf   │  │ tc/tbf   │  │ tc/tbf   │
              │ shaper   │  │ shaper   │  │ shaper   │  │ shaper   │
              │ on ens5  │  │ on ens5  │  │ on ens5  │  │ on ens5  │
              └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
                   └─────────────┴─────────────┴─────────────┘
                        NCCL over TCP  (no NVLink · no EFA)
                        ▲
                        └── THIS LINK IS THE INSTRUMENT, NOT THE PLUMBING
```

## 11.2 Software layer diagram

```text
┌───────────────────────────────────────────────────────────────┐
│  CLI  (diloco-measured …)                                     │
│  network | run | converge | grid | plan | analyze | figures    │
└───────────────┬───────────────────────────────┬───────────────┘
                │                               │
   ┌────────────▼─────────────┐   ┌─────────────▼──────────────┐
   │   MEASUREMENT PACKAGE    │   │    ANALYSIS PACKAGE        │
   │   (needs GPUs + AWS)     │   │    (pure, GPU-free)        │
   │                          │   │                            │
   │  netshape  · probe       │   │  load · validate · filter  │
   │  train     · diloco      │   │  aggregate · fit · figures │
   │  compress  · wire        │   │  predictor calibration     │
   │  faults    · fingerprint │   │                            │
   └────────────┬─────────────┘   └─────────────▲──────────────┘
                │  WRITES (append-only)         │  READS (never writes)
                ▼                               │
   ┌──────────────────────────────────────────────────────────┐
   │              RESULT STORE  (filesystem + git)            │
   │  results/raw/  results/network/  results/environment/    │
   │  JSON (records) + Parquet (per-step telemetry)           │
   │  Versioned by JSON Schema; immutable; append-only        │
   └──────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────┐
   │  SUBSTRATE (vendored/pinned, not ours)                   │
   │  PyTorch · torchtitan · torchft · NCCL · torchao         │
   └──────────────────────────────────────────────────────────┘

HARD RULE: analysis never imports measurement. measurement never imports analysis.
```

## 11.3 Key architectural decisions

Each decision below is expanded in §41 (Decision Log). Summary form:

### D1 — Four single-GPU nodes rather than one multi-GPU node
**Choice:** 4× `g6e.2xlarge`. **Why:** 32 vCPU quota is hard; small instances maximize GPUs per vCPU, and the TCP-only topology *is* the experimental regime. **Alternatives:** 1× `g6e.8xlarge` (1 GPU, no distribution — kills the project); `g6e.12xlarge` (4 GPUs, PCIe — needs 48 vCPU, unavailable). **Trade-offs:** gain the target regime and 4 replicas; lose intra-node bandwidth, lose the ability to test FSDP-inside-a-replica, and get only 8 vCPU per node for data loading. **Status:** `[CONFIRMED]`

### D2 — `tc`/`tbf` egress shaping as the manipulation, with a verification gate
**Choice:** Linux traffic control on each node's egress interface. **Why:** it is the only way to make bandwidth an independent variable without renting geographically distributed capacity. **Alternatives:** rent nodes in different regions (uncontrolled, expensive, confounded by latency); `netem rate` (similar, different queueing model); do not sweep (destroys the project). **Trade-offs:** gain control and repeatability; lose realism — `tbf` models a bandwidth ceiling only, with no added latency, jitter, or loss. **This limitation must be stated prominently in the report; it makes our measured discrepancy a *lower bound* on real-WAN discrepancy.** **Status:** `[CONFIRMED]`

### D3 — Two DiLoCo implementations
**Choice:** an in-repo `diloco.py` reference implementation AND `torchft` as the system under test, cross-validated. **Why:** `torchft`'s LocalSGD/DiLoCo are marked experimental; a silent bug there would invalidate everything. **Alternatives:** trust torchft alone (fragile); write our own alone (loses the fault-tolerance machinery). **Trade-offs:** a few hours of extra work buys the strongest correctness argument available and a working fallback. **Status:** `[CONFIRMED]`

### D4 — Filesystem + git result store, no database
**Choice:** JSON records + Parquet telemetry, committed to git. **Why:** the corpus is small (thousands of records), the access pattern is batch, and committing it is what makes the work reproducible without infrastructure. **Alternatives:** SQLite (adds an index but hides the data from a reader browsing GitHub); Postgres (absurd for this scale); W&B (external dependency, non-reproducible for third parties). **Trade-offs:** lose query convenience; gain zero-dependency reproduction and reviewability. `[RECOMMENDATION]` a derived SQLite index may be *generated* from JSON for convenience, but JSON remains the source of truth. **Status:** `[PROPOSED]`

### D5 — Strict measurement/analysis separation
**Choice:** two packages, no cross-imports, analysis is a pure function of committed data. **Why:** it is what makes FR-11 (GPU-free reproduction) achievable, and it prevents the classic research-code failure where a figure silently depends on a live cluster. **Status:** `[CONFIRMED]`

### D6 — torchtitan as the training substrate
**Choice:** use torchtitan's Llama-3 model definitions, FSDP2/DDP wiring, and checkpointable data loading. **Why:** it is PyTorch-native, it is what the reference L40S semi-sync work used, and it removes a day of undifferentiated work. **Alternatives:** raw PyTorch (more control, more time); nanoGPT-style minimal (simpler but diverges from the reference setup); Megatron-LM (overkill, NVIDIA-centric). **Trade-offs:** gain speed and comparability; inherit torchtitan's API churn. **Status:** `[PROPOSED]` — must be validated on Day 0/1 against a pinned version.

### D7 — Verification gate aborts rather than warns
**Choice:** a failed shaping verification aborts the run. **Why:** a warning in a log is a number in a figure six days later. **Trade-offs:** loses grid points; gains the ability to say "every bandwidth in this repo was measured." **Status:** `[CONFIRMED]`

### D8 — Both CU variants (link-bandwidth and achieved-bandwidth) are computed
**Choice:** report `cu_analytic_link` (the papers' assumption) *and* `cu_analytic_achieved` (fed measured NCCL bandwidth). **Why:** it separates "the model is wrong" from "the model's *input* is wrong," which is the difference between a naive result and a useful one. **Status:** `[CONFIRMED]`

---

# 12. Architecture Principles

These are the rules implementation must follow. They are ordered by precedence — when two conflict, the higher one wins.

1. **Measurement integrity outranks everything.** If a value was not measured, it does not go in a record. If a precondition was not verified, the run does not start. Never soften a gate to save a grid point.
2. **Raw data is immutable.** `results/raw/` is append-only. Corrections are made by superseding records with a new `harness_version`, never by editing.
3. **Measurement and analysis are strictly separated.** No analysis module imports a measurement module. Analysis runs with no GPU, no network, no credentials.
4. **Everything is fingerprinted.** A result without a complete environment fingerprint is invalid and must be rejected at write time.
5. **Fail loud, fail early, fail cheap.** Preconditions are checked before a GPU is touched. A misconfigured run should die in seconds, not after twenty minutes.
6. **Two implementations for correctness-critical logic.** The DiLoCo loop and the wire accounting each have an independent cross-check.
7. **Honest labelling in code, not just prose.** Fields are named `cu_measured` / `cu_analytic_link` / `cu_analytic_achieved`, never just `cu`. Interpolated values carry a flag.
8. **Simplicity over generality.** This is a seven-day instrument, not a framework. No plugin systems, no abstract base classes with one implementation, no configuration DSLs.
9. **Cost is a first-class constraint.** Cluster-hours are logged. Anything that can run on CPU runs on CPU. The cluster is not a development environment.
10. **The limitation section is part of the deliverable.** Every known confound (no WAN latency, 4 replicas, one GPU generation, small models, short budgets) is documented in the report, not discovered by a reviewer.

---

# 13. Technology Stack

Nothing here is chosen by default. Each row states purpose, rationale, alternatives, trade-offs, and status.

## 13.1 Core

| Technology | Purpose | Why | Alternatives | Trade-offs | Status |
| --- | --- | --- | --- | --- | --- |
| **Python 3.11+** | Everything | The ML ecosystem; matches PyTorch tooling | Rust for the harness | Rust would be faster and less fragile for the orchestration layer, but the training loop must be Python anyway and a split-language project is not worth it at this size | `[PROPOSED]` |
| **PyTorch (pinned)** | Training, `torch.distributed` | Non-negotiable substrate for FSDP2/DDP/NCCL | JAX | JAX has no torchft/torchtitan equivalent for this exact reference setup | `[CONFIRMED]` |
| **torchtitan (pinned SHA)** | Llama-3 model defs, FSDP2/DDP, data loading | PyTorch-native; matches the reference L40S semi-sync work; ships checkpointable C4 loading | Raw PyTorch, nanoGPT, Megatron-LM | Gains a day; inherits API churn — hence a pinned SHA, not a range | `[PROPOSED]` |
| **torchft (pinned `torchft-nightly` PyPI version, not a source SHA)** | LocalSGD / DiLoCo / Streaming DiLoCo, lighthouse, fault-tolerant process groups | It is the PyTorch-native implementation of exactly our algorithms, and its fault-tolerance machinery gives us G7 nearly free | OpenDiLoCo (unmaintained; superseded), Prime's `prime` stack (heavier, FSDP2-centric), own implementation only | Its semi-sync paths are marked **experimental** — this is the single largest technical risk (§38 R2). Mitigated by D3. Has a Rust extension (`maturin`/PyO3) — pin the prebuilt `manylinux` nightly wheel, not a source build, or every node needs a Rust + `protobuf-compiler` toolchain (§40 Q2, researched 2026-08-10) | `[PROPOSED]` |
| **NCCL over TCP** | Collectives | Only transport available at these instance sizes | gloo (slower, but a valid fallback for CPU tests) | gloo is used deliberately for CPU-only CI tests | `[CONFIRMED]` |

## 13.2 Measurement and orchestration

| Technology | Purpose | Why | Alternatives | Status |
| --- | --- | --- | --- | --- |
| **Linux `tc` (`tbf` qdisc)** | Egress bandwidth shaping | Standard, deterministic, available on any EC2 Linux AMI | `netem rate` (different queueing behaviour — keep as fallback), hardware rate limits (unavailable) | `[CONFIRMED]` |
| **`iperf3`** | Independent bandwidth verification | Trusted, ubiquitous, independent of NCCL — which is exactly why it is the gate | `nuttcp`, raw sockets | `[CONFIRMED]` |
| **`/proc/net/dev`** | Bytes-on-wire ground truth | Kernel-level counter, independent of the framework's own accounting | `nload`, eBPF (more precise, more setup) | `[CONFIRMED]` |
| **`torch.cuda.Event`** | Per-step compute/sync decomposition | Low overhead, GPU-timeline accurate | `torch.profiler` (richer but too heavy for every step) | `[CONFIRMED]` |
| **`torch.profiler`** | Deep traces at a few representative points only | Explains *why*, not *how much* | Nsight Systems (better, heavier setup) | `[PROPOSED]` |
| **DCGM exporter** | GPU utilization, power, memory | Standard, Prometheus-native | `nvidia-smi` polling (coarser) | `[PROPOSED]` |
| **Prometheus + Grafana** | Live operator visibility during runs | Operator situational awareness; NOT the source of truth for results | Plain logs | `[PROPOSED]` — explicitly optional; results must never depend on it |
| **`torchrun`** | Multi-node launch + rendezvous | Standard PyTorch elastic launcher | Slurm (heavy), MPI (extra dependency), manual | `[PROPOSED]` |

## 13.3 Data and analysis

| Technology | Purpose | Why | Status |
| --- | --- | --- | --- |
| **JSON + JSON Schema** | Result records + contract enforcement | Human-readable in a GitHub diff; schema gives a hard contract; no runtime dependency to read | `[CONFIRMED]` |
| **Parquet (pyarrow)** | Per-step telemetry | Thousands of steps per run × hundreds of runs; JSON would bloat the repo | `[PROPOSED]` |
| **pandas / polars** | Aggregation | Standard | `[PROPOSED]` — pandas unless volume demands otherwise |
| **matplotlib** | Figures | Deterministic, scriptable, no JS runtime, renders in papers | `[CONFIRMED — in use since ADR-017]`, `Agg` backend, headless by construction |
| **pydantic** | Spec + record validation in Python | Schema-as-code, good errors, generates JSON Schema | `[PROPOSED]` |
| **Hugging Face `datasets`** | Dataset acquisition (control node only) | Standard access to FineWeb-Edu / C4 | `[PROPOSED]` |

## 13.4 Infrastructure and tooling

| Technology | Purpose | Why | Status |
| --- | --- | --- | --- |
| **AWS EC2 + S3** | Compute and dataset staging | The available capacity; the quota constraint is defined in its terms | `[CONFIRMED]` |
| **Bash + AWS CLI** for cluster lifecycle | `launch_cluster.sh` etc. | 5 instances, one week, one operator — Terraform would be ceremony | `[RECOMMENDATION]` — revisit only if the cluster is rebuilt more than ~10 times |
| **`uv` with a lockfile** | Reproducible envs, bare metal (no Docker) | Version drift between nodes would silently corrupt measurements; Docker was rejected — see below | `[CONFIRMED — ADR-020]` |
| **Docker** | Node environment | Rejected for Day 1: reduces version drift (R15) but adds NCCL networking configuration surface on top of R1, already the higher-impact risk. Fewest moving parts wins while debugging NCCL for the first time | `[CONFIRMED — REJECTED, ADR-020]` — revisit only if drift is actually observed |
| **pytest** | Tests | Standard | `[PROPOSED]` |
| **GitHub Actions** | CI: unit + CPU integration only | Never touches AWS or GPUs | `[PROPOSED]` |
| **Weights & Biases** | Experiment tracking | **Deliberately excluded from the critical path.** External, non-reproducible for third parties, and would undermine FR-11. May be used as a *mirror*, never as the source of truth | `[RECOMMENDATION — do not adopt as source of truth]` |

## 13.5 Explicitly rejected

| Rejected | Reason |
| --- | --- |
| Any web frontend | No users, no interactive workflow. Figures are static; the report is markdown |
| Any database server | Corpus is small and batch-accessed; a DB would hide data from reviewers |
| Kubernetes | Five instances, one week, one operator |
| Terraform/Pulumi | Same; bash + AWS CLI is proportionate |
| Multi-cloud abstraction | The quota constraint is AWS-specific and central |
| EFA / InfiniBand paths | Unavailable at these instance sizes, and irrelevant to the research question |

---

# 14. Repository Structure

```text
diloco-measured/
├── README.md                     # Headline claim, headline figure, 3-command repro, links
├── PRIOR_ART.md                  # READ FIRST. What is solved, by whom, and what is not.
├── CLAUDE.md                     # ← this file. The master specification.
├── RESULTS.md                    # Every run, including nulls, crashes, and abandoned lines
├── PLAYBOOK.md                   # Practitioner-facing: "N GPUs at X Mbit/s → use H=…"
├── LIMITATIONS.md                # Every known confound, stated by us before a reviewer finds it
├── LICENSE                       # Apache-2.0
├── Makefile                      # cluster-up | bootstrap | network-characterize | smoke
│                                 # | grid | converge | figures | test | cluster-down
├── pyproject.toml                # deps + entrypoint `diloco-measured`
├── uv.lock                       # PINNED. Environment drift = corrupted measurements.
│
├── methods/                      # The scientific method, written down. Reviewed like code.
│   ├── network_protocol.md       # Shaping + verification gate, in full, reproducible
│   ├── cu_model.md               # The analytic CU model: form, source, EVERY assumption
│   ├── wire_model.md             # Bytes-on-wire derivation per algorithm
│   ├── diloco.md                 # Inner/outer derivation, hyperparameters, why Nesterov
│   ├── measurement_windows.md    # Warmup discard, repeat policy, outlier policy
│   └── statistics.md             # Median vs mean, IQR, how many repeats and why
│
├── src/diloco_measured/
│   ├── __init__.py
│   ├── cli.py                    # Thin argparse/typer layer. NO logic here.
│   │
│   ├── measurement/              # ── needs GPUs + AWS. Never imported by analysis. ──
│   │   ├── netshape.py           # tc set/verify/restore + iperf3 assertion gate  (FR-02)
│   │   ├── probe.py              # NCCL all-reduce BW vs message size             (FR-01)
│   │   ├── fingerprint.py        # Environment capture                            (FR-08)
│   │   ├── train.py              # torchtitan-based loop, fixed-token-budget mode (FR-03)
│   │   ├── diloco.py             # OUR reference inner/outer implementation       (D3)
│   │   ├── compress.py           # fp16 / int8+error-feedback / top-k codecs      (FR-10)
│   │   ├── wire.py               # /proc/net/dev accounting + analytic prediction (FR-05)
│   │   ├── faults.py             # Scheduled SIGKILL injector + recovery timer    (FR-09)
│   │   └── telemetry.py          # CUDA-event step decomposition
│   │
│   ├── analysis/                 # ── pure. No GPU, no network, no credentials. ──
│   │   ├── load.py               # Schema-validated loading; refuses invalid records
│   │   ├── filter.py             # Exclusion rules (crashed, loader-bound, version mismatch)
│   │   ├── cu.py                 # CU: measured AND analytic, one input schema     (FR-04)
│   │   ├── aggregate.py          # Repeats → median + IQR
│   │   ├── predictor.py          # Calibration + held-out validation               (FR-07)
│   │   └── figures/              # One module per figure; each is a pure function
│   │
│   └── schemas/                  # Versioned JSON Schemas — THE CONTRACT
│       ├── experiment_spec.v1.json
│       ├── run_result.v1.json
│       ├── network_profile.v1.json
│       └── step_record.v1.json
│
├── configs/
│   ├── models/                   # debugmodel.toml (smoke test) · 130m.toml · 500m.toml · 1b.toml
│   ├── algorithms/               # ddp · fsdp2 · localsgd · diloco · diloco_int8
│   ├── bandwidths/               # unshaped · 5g · 1g · 200m · 50m
│   └── grids/                    # phase_a.yaml · phase_b.yaml · phase_c.yaml · phase_d.yaml
│
├── infra/
│   ├── launch_cluster.sh         # placement group + SG + 4 GPU nodes + control node
│   ├── setup_node.sh             # deps, clock lock, NVMe mount, dataset sync
│   ├── prepare_dataset.py        # stream + tokenize + shard + checksum, control node → S3
│   ├── torchrun_multinode.sh
│   ├── teardown.sh               # idempotent; MUST leave nothing billable
│   └── cost_report.sh            # cumulative cluster-hours and spend
│
├── experiments/                  # One directory per experiment campaign
│   ├── 00_network_characterization/{run.sh, spec.yaml, NOTES.md}
│   ├── 01_cu_grid/
│   ├── 02_convergence/
│   ├── 03_compression/
│   ├── 04_faults/
│   └── 05_predictor_validation/
│       └── NOTES.md              # ← what broke, what changed, what you'd redo. MANDATORY.
│
├── results/                      # APPEND-ONLY. Committed. The credibility layer.
│   ├── raw/                      # RunResult JSON (one per run)
│   ├── steps/                    # Per-step Parquet (one per run)
│   ├── network/                  # Every iperf3 + NCCL probe output, per shaping level
│   ├── environment/              # Fingerprints, nvidia-smi topo, qdisc dumps
│   └── figures/                  # Generated. Deleted and regenerated by `make figures`.
│
├── notebooks/analysis.ipynb      # Exploration only. Never the source of a published figure.
├── report/                       # The technical report / blog post + its assets
└── tests/
    ├── unit/                     # cost models, wire math, codecs, schema validation
    ├── integration_cpu/          # 4-process gloo DiLoCo equivalence; netshape on loopback
    └── e2e/                      # smoke: 4 nodes, tiny model, 20 steps, valid RunResult
```

## 14.1 Directory rules

| Directory | Belongs here | Does NOT belong here |
| --- | --- | --- |
| `methods/` | The scientific protocol, prose, with every assumption | Code, results |
| `src/.../measurement/` | Anything that touches a GPU, a NIC, or AWS | Plotting, aggregation, any analysis |
| `src/.../analysis/` | Pure functions over committed records | Anything requiring a GPU, network, or credentials |
| `src/.../schemas/` | Versioned schemas only | Python logic |
| `configs/` | Declarative specs | Executable logic, secrets |
| `results/raw/` | Immutable, append-only records | Edited values, hand-corrected numbers, figures |
| `results/figures/` | Generated artifacts (safe to delete) | Anything not regenerable by `make figures` |
| `experiments/*/NOTES.md` | What actually happened, including mistakes | Sanitized narrative |
| `notebooks/` | Exploration | Anything a published figure depends on |
| `infra/` | Cluster lifecycle | Experiment logic, credentials |

## 14.2 Dependency rules

```text
cli ──► measurement ──┬──► substrate (torch, torchtitan, torchft)
 │                     └──► schemas   (ExperimentSpec preflight validation, e.g. spec.py)
 └────► analysis ──► schemas
                └──► results/  (READ ONLY)

FORBIDDEN EDGES:
  analysis   ──X──► measurement
  measurement ──X──► analysis
  figures    ──X──► anything that opens a socket or a CUDA context
  notebooks  ──X──► being a dependency of a published figure

Note: schemas/ (including schemas/registry.py, the shared referencing.Registry builder) is a
neutral dependency both measurement and analysis may import — the forbidden edges above are
specifically analysis<->measurement, not either package's edge to schemas/ (ADR-013).
```

## 14.3 Naming conventions

- Metric fields carry their provenance: `cu_measured`, `cu_analytic_link`, `cu_analytic_achieved`. A bare `cu` is a bug.
- Requested vs measured is always explicit: `bandwidth_requested_bps` vs `bandwidth_measured_bps`.
- Units are in the name: `_bps`, `_bytes`, `_ms`, `_s`, `_tokens`. No unitless numerics in a record.
- Booleans that flag uncertainty read as warnings: `nccl_bw_interpolated`, `loader_bound_warning`, `burst_decay_detected`.
- Run IDs: `{phase}-{algorithm}-{model}-h{H}-bw{rate}-r{repeat}-{shortsha}`.

---

# 15. Domain Model

There is no user domain here. The domain is **experiments and measurements**. Entities are records, not rows in a transactional database, but they still have lifecycles, invariants, and state transitions.

## 15.1 Entity map

```text
NetworkProfile ──── characterizes ────► the cluster at a point in time
      │
      │ (precondition for)
      ▼
ExperimentSpec ────► produces ────► Run ────► RunResult
      │                              │            │
      │                              │            ├── CUObservation
      │                              │            ├── WireAccount
      │                              │            ├── ThroughputSummary
      │                              │            └── ConvergenceCurve (convergence runs)
      │                              │
      │                              ├── EnvironmentFingerprint  (1:1, mandatory)
      │                              ├── ShapingVerification     (1:1, mandatory if shaped)
      │                              ├── StepRecord[]            (per step, Parquet)
      │                              └── FaultEvent[]            (0..n)
      │
      └────► belongs to ────► GridCampaign

RunResult[] ────► calibrates ────► PredictorModel ────► Recommendation
```

## 15.2 Entities

---

### `NetworkProfile`
**Purpose:** The measured state of the cluster network. A precondition for every experiment.

| Field | Type | Notes |
| --- | --- | --- |
| `profile_id` | str | |
| `captured_at` | datetime | |
| `cluster_id` | str | Placement group / launch ID |
| `iperf_pairs` | list[PairMeasurement] | ordered pair, direction, Gbit/s, duration |
| `nccl_curve` | list[{msg_bytes, achieved_bps}] | per shaping level |
| `shaping_fidelity` | list[{requested_bps, measured_bps, error_pct}] | |
| `burst_decay_detected` | bool | from the 10-min sustained probe |
| `burst_decay_curve` | list[{t_s, bps}] | nullable |
| `topology` | str | `nvidia-smi topo -m` output |

**Invariants:** `nccl_curve` must be non-empty for every shaping level used by any run referencing this profile.
**Lifecycle:** created → committed → referenced (immutable thereafter).

---

### `ExperimentSpec`
**Purpose:** The complete, declarative description of a run. The only input to the measurement path.

| Field | Type | Notes |
| --- | --- | --- |
| `spec_id` | str | |
| `phase` | enum | `network` \| `cu_grid` \| `convergence` \| `compression` \| `faults` \| `predictor` |
| `algorithm` | enum | `ddp` \| `fsdp2` \| `localsgd` \| `diloco` |
| `implementation` | enum | `torchft` \| `reference` — which DiLoCo code path |
| `H` | int | `1` for DDP |
| `model_config` | str | ref to `configs/models/*` |
| `bandwidth_requested_bps` | int \| null | null = unshaped |
| `world_size` | int | 4 |
| `micro_batch_size`, `seq_len`, `grad_accum` | int | |
| `budget_type` | enum | `steps` \| `tokens` |
| `budget_value` | int | |
| `warmup_steps` | int | discarded from the measurement window |
| `compression` | enum \| null | `fp16` \| `int8_ef` \| `topk` |
| `seed` | int | |
| `repeat_index` | int | |
| `fault_schedule` | list[{rank, t_s}] \| null | |

**Invariants:** `H == 1` iff `algorithm == ddp`; `compression` only valid with `localsgd`/`diloco`; `budget_type == tokens` required for convergence phase.
**Validation:** schema + cross-field rules, enforced before any resource is allocated.

---

### `ShapingVerification`
**Purpose:** Proof that the requested bandwidth was achieved. Without it, a shaped run is invalid.

| Field | Type |
| --- | --- |
| `requested_bps` | int |
| `measured_bps` | int |
| `error_pct` | float |
| `tolerance_pct` | float |
| `passed` | bool |
| `attempts` | int |
| `iperf_raw` | str (path) |
| `qdisc_dump` | str |

**Invariant:** `passed == false` ⇒ **no `RunResult` may exist for this run.**

---

### `Run` / `RunResult`
**Purpose:** One execution and its aggregated outcome.

| Field | Type | Notes |
| --- | --- | --- |
| `run_id` | str | see naming convention |
| `spec` | ExperimentSpec | embedded, not referenced — records are self-contained |
| `fingerprint` | EnvironmentFingerprint | mandatory |
| `shaping` | ShapingVerification \| null | null only when unshaped |
| `network_profile_id` | str | |
| `harness_version` | str | **critical for cross-version pooling rules** |
| `status` | enum | `completed` \| `crashed` \| `diverged` \| `aborted_shaping` \| `oom` |
| `started_at`, `ended_at` | datetime | |
| `cu` | CUObservation | |
| `wire` | WireAccount | |
| `throughput` | ThroughputSummary | |
| `convergence` | ConvergenceCurve \| null | |
| `faults` | list[FaultEvent] | |
| `loader_bound_warning` | bool | |
| `notes` | str | free text from the operator |

**State transitions:**
```text
pending → preconditions_ok → shaped → verified → running → aggregating → completed
   │            │                │                  │
   │            │                └─► aborted_shaping│
   │            └─► aborted_preconditions           ├─► crashed
   └─► invalid_spec                                 ├─► oom
                                                    └─► diverged
```
**Invariant:** only `completed` runs may enter analysis aggregations. All other statuses are recorded and reported in `RESULTS.md` but excluded from figures (and their exclusion is counted and stated).

---

### `CUObservation`

| Field | Type | Notes |
| --- | --- | --- |
| `cu_measured` | float | `Σ compute / Σ total`, post-warmup |
| `cu_analytic_link` | float | model fed nominal link bandwidth (the papers' assumption) |
| `cu_analytic_achieved` | float \| null | model fed measured NCCL bandwidth at the relevant message size |
| `nccl_bw_used_bps` | int \| null | |
| `nccl_bw_interpolated` | bool | |
| `discrepancy_link` | float | `cu_analytic_link / cu_measured` |
| `discrepancy_achieved` | float \| null | |
| `compute_s`, `sync_blocked_s`, `optimizer_s`, `loader_stall_s`, `total_s` | float | must sum to `total_s` within a recorded residual |

**Invariant:** the component times must reconcile to `total_s`; the residual is recorded, never hidden. A residual above `[PROPOSED]` 5% invalidates the observation.

---

### `WireAccount`

| Field | Type |
| --- | --- |
| `predicted_bytes` | int |
| `measured_bytes` | int |
| `overhead_ratio` | float |
| `bytes_per_training_token_predicted` | float |
| `bytes_per_training_token_measured` | float |
| `idle_baseline_bytes` | int |

---

### `ConvergenceCurve`

| Field | Type | Notes |
| --- | --- | --- |
| `points` | list[{tokens, wall_s, train_loss, val_loss}] | |
| `target_loss` | float | from the single-GPU reference |
| `tttl_s` | float \| null | **null when the target was never reached** |
| `tttl_smoothed_s` | float \| null | |
| `final_loss` | float | |
| `reached_target` | bool | |

**Invariant:** `tttl_s == null` must never be rendered as a large finite number in any figure.

---

### `FaultEvent`

| Field | Type |
| --- | --- |
| `injected_at_s` | float |
| `rank` | int |
| `detected_at_s` | float \| null |
| `resumed_at_s` | float \| null |
| `steps_lost` | int |
| `outcome` | enum: `recovered` \| `hung` \| `job_died` |

---

### `PredictorModel`

| Field | Type | Notes |
| --- | --- | --- |
| `model_id`, `fitted_at` | str/datetime | |
| `training_run_ids` | list[str] | full provenance |
| `form` | str | functional form, human-readable |
| `params` | dict | |
| `calibration_domain` | dict | bandwidth range, model-size range, H range |
| `holdout_validation` | {predicted_H, measured_H, regret_pct} | |

**Invariant:** a `Recommendation` produced outside `calibration_domain` must carry `extrapolation_warning: true`.

---

# 16. Data Architecture (in place of a database)

`[CONFIRMED]` **There is no database.** The data layer is a versioned, append-only, git-committed file store. This is a deliberate architectural decision (D4), not an omission.

## 16.1 Layout and formats

```text
results/
├── raw/<run_id>.json              # RunResult — self-contained, schema-validated
├── steps/<run_id>.parquet         # StepRecord[] — one row per training step
├── network/<profile_id>.json      # NetworkProfile
├── network/<profile_id>/iperf/*   # raw iperf3 JSON output — the audit trail
├── environment/<run_id>.json      # EnvironmentFingerprint (also embedded in RunResult)
└── index.sqlite                   # DERIVED, gitignored, regenerable. Never authoritative.
```

**Why records are self-contained:** each `RunResult` embeds its full spec and fingerprint rather than referencing them. It costs disk (trivially) and buys the property that any single file is independently interpretable years later, by someone with no repository.

## 16.2 Schema versioning and migration

- Schemas are versioned in the filename: `run_result.v1.json`.
- A schema change means a **new version file**, never an edit to an existing one.
- Old records are never rewritten. The loader supports all versions it has ever seen.
- Analysis declares which versions it accepts; unknown versions are rejected loudly.

## 16.3 Immutability and the harness-version rule

`[CONFIRMED]` These two rules are the backbone of the project's credibility:

1. **`results/raw/` is append-only.** No edits. No deletions. A wrong record is superseded by a new record, and the old one stays, marked in `RESULTS.md`.
2. **Any change to a measurement code path bumps `harness_version`.** Results from different harness versions are **never pooled in one figure** unless an explicit, documented equivalence argument exists in `methods/`. The loader enforces this: mixing versions requires an explicit override flag that is recorded in the figure metadata.

> **Why this matters more than it looks.** The most common way research measurements become quietly wrong is: someone fixes a timing bug on day 5, does not re-run days 1–4, and pools everything. This rule makes that impossible by default.

## 16.4 Retention

- Everything in `results/` is committed permanently. It is the artifact.
- `results/figures/` is generated and may be deleted at any time.
- `[CONFIRMED — ADR-023, formerly §40 Q10]` Per-step Parquet is published in full only for a
  representative subset of runs; every run still gets its aggregated `RunResult` JSON.
  Exactly which runs count as "representative" is `[PROPOSED]`, to be picked once real Phase A
  data volume is known (candidates: one repeat per grid point, or the specific runs any
  published figure's caption cites) — but the *policy* (subset-in-full + aggregates-for-all,
  not aggregates-only and not everything) is decided. The selection and any resulting
  downsampling are themselves recorded, per the existing sentence below.
- If the repository still exceeds `[PROPOSED]` 500 MB after that, per-step Parquet may be
  further downsampled; the downsampling is itself recorded.

---

# 17. Interface Architecture (CLI and module contracts, in place of an HTTP API)

`[CONFIRMED]` There is no network API. There are no endpoints, no authentication, no rate limiting, and no versioned HTTP contract, because there is no client-server relationship anywhere in this system. Specifying one would be invented complexity.

The real interfaces are: **the CLI**, **the module boundaries**, and — most importantly — **the result schemas**, which are the contract between the measurement half and the analysis half.

## 17.1 CLI surface

```text
diloco-measured network characterize [--profile-id ID] [--levels 5g,1g,200m,50m]
    Purpose:   FR-01. Full network characterization.
    Requires:  cluster up, sudo on nodes
    Writes:    results/network/<profile_id>.json + raw iperf outputs
    Errors:    unreachable node · tc unavailable · NCCL init failure
    Side effects: applies and restores shaping on all nodes

diloco-measured run --spec configs/... [--dry-run]
    Purpose:   FR-03. One instrumented run.
    Requires:  a NetworkProfile covering the requested bandwidth
    Writes:    results/raw/<run_id>.json, results/steps/<run_id>.parquet
    Errors:    invalid spec · precondition failure · shaping verification failure
               · OOM · NCCL timeout · divergence
    Side effects: shapes and restores the network; writes append-only records
    Idempotency: run_id includes repeat_index; re-running the same spec
                 creates a NEW record, never overwrites

diloco-measured grid --config configs/grids/phase_a.yaml [--resume]
    Purpose:   Execute a campaign of runs.
    Behaviour: a failed point is logged and the grid CONTINUES.
               --resume skips points with an existing completed record.

diloco-measured converge --spec ...
    Purpose:   FR-06. Fixed-token-budget run with periodic eval.

diloco-measured plan (--probe | --bandwidth BPS) --model CFG [--gpus N]
    Purpose:   FR-07. Recommendation.
    Requires:  a fitted PredictorModel
    Output:    recommended H, expected tok/s, expected CU, expected bytes/hr,
               calibration domain, and an EXTRAPOLATION WARNING if outside it
    Errors:    no fitted model · probe failure

diloco-measured analyze [--phase A] [--allow-version-mix]
    Purpose:   Aggregate committed records. GPU-free.

diloco-measured figures [--only fig1]
    Purpose:   FR-11. Regenerate figures from committed data. GPU-free, network-free.
```

## 17.2 Module contracts (the real "API")

```python
# measurement/netshape.py
def apply(rate_bps: int | None, nodes: list[Node]) -> ShapingHandle: ...
def verify(handle: ShapingHandle, tolerance_pct: float) -> ShapingVerification: ...
def restore(handle: ShapingHandle) -> None: ...
# CONTRACT: verify() NEVER returns a passing result it did not measure.
#           restore() is idempotent and must be safe to call twice.

# measurement/wire.py
def snapshot(nodes) -> WireSnapshot: ...
def predict(spec: ExperimentSpec, model_params: int) -> int: ...
def account(before: WireSnapshot, after: WireSnapshot, predicted: int) -> WireAccount: ...

# analysis/cu.py
def measured(steps: StepRecords, warmup: int) -> float: ...
def analytic(spec: ExperimentSpec, t_compute_s: float,
             bytes_synced: int, bandwidth_bps: int) -> float: ...
# CONTRACT: analytic() takes bandwidth as an EXPLICIT parameter so the caller
#           must decide, visibly, whether it is passing link or achieved bandwidth.
#           There is no default. This is intentional friction.
```

## 17.3 Error contract

Every failure produces a structured record, never a silent skip:

```text
{ "error_class": "shaping_verification_failed",
  "run_id": "...", "requested_bps": 1e9, "measured_bps": 6.2e8,
  "tolerance_pct": 10, "attempts": 2, "action_taken": "run_aborted" }
```

---

# 18. Frontend Architecture

`[CONFIRMED]` **Not applicable.** There is no frontend, no browser client, no routing, no component library, no state management, and no accessibility surface, because there are no interactive users.

The three presentation surfaces are:

| Surface | Technology | Notes |
| --- | --- | --- |
| **Figures** | matplotlib → PNG/SVG in `results/figures/` | Deterministic, regenerated by `make figures` |
| **Report** | Markdown in `report/` | The narrative artifact; renders on GitHub and as a blog post |
| **Grafana dashboard** | Grafana JSON in `dashboards/` | **Operator situational awareness during runs only.** `[PROPOSED]`. No result may ever depend on it |

Presentation principles that do apply:
- Every figure states, in-caption, whether values are measured, analytic, or interpolated.
- Measured series are solid; analytic/simulated series are dashed, in matching colours. This single convention carries the project's entire argument visually.
- No figure exists that cannot be regenerated from `results/raw/`.

---

# 19. Execution Architecture (in place of "backend")

## 19.1 Where logic belongs

| Layer | Contains | Must NOT contain |
| --- | --- | --- |
| `cli.py` | Argument parsing, config file resolution, exit codes | Any measurement, computation, or decision logic |
| `measurement/*` | Device, network, and process orchestration; telemetry capture | Plotting, statistics, curve fitting |
| `analysis/*` | Statistics, fitting, figure generation | Anything that opens a socket or a CUDA context |
| `schemas/*` | Contracts | Behaviour |
| `configs/*` | Declarative parameters | Executable logic |

## 19.2 Run orchestration

`[PROPOSED]` The control node is the orchestrator. It does not participate in training. It:
1. Resolves the spec and checks preconditions.
2. Issues shaping commands to all GPU nodes over SSH, then runs the verification gate.
3. Hosts the `torchrun` rendezvous endpoint and the `torchft` lighthouse.
4. Launches training on the 4 GPU nodes.
5. Collects telemetry and writes the aggregated record.

Rationale for putting orchestration on a CPU-only node: it consumes zero GPU quota, it keeps orchestration overhead off the measured hosts, and it survives a GPU node dying — which matters directly for the fault-injection experiments.

## 19.3 Concurrency model

`[CONFIRMED]` **Runs are strictly serial.** Never two experiments on the cluster at once. Concurrent runs would contend for the shaped link and destroy every timing measurement. The grid runner is a sequential loop with resume support, not a scheduler.

## 19.4 Failure isolation

- A failed grid point does not abort the campaign; it is recorded and the loop continues.
- A dirty qdisc on any node blocks subsequent runs on that node until restore succeeds.
- A crashed run yields preserved partial telemetry but **no** analysis-eligible `RunResult`.

---

# 20. Async and Background Processing

`[CONFIRMED]` There is no job queue, no worker pool, no message broker, and no retry-with-backoff infrastructure. Adding one would be textbook over-engineering: the workload is a serial sequence of long-running jobs executed by one operator over seven days.

What *does* exist:

| Concern | Mechanism |
| --- | --- |
| Long-running campaigns | `diloco-measured grid` — a sequential loop, resumable via `--resume` |
| Unattended overnight execution | `tmux`/`nohup` on the control node + campaign log |
| Retry policy | **Exactly one** retry, and only for shaping application (FR-02). Training runs are never auto-retried — a retry would mask nondeterministic failure, which is itself data |
| Dead-letter equivalent | Failed points recorded with `status` and an `error_class`, surfaced in `RESULTS.md` |
| Idempotency | `--resume` skips points that already have a completed record with a matching `harness_version` |
| Deduplication | `run_id` includes `repeat_index`; identical specs intentionally produce distinct records |

```text
Campaign spec
   ↓
Enumerate points  ─────► skip points already completed at this harness_version
   ↓
For each point (SERIAL):
   ├─ preconditions → shape → VERIFY → run → aggregate → write → restore
   ├─ on failure: record error, continue
   └─ append to campaign log
   ↓
Campaign summary (counts by status) → RESULTS.md
```

---

# 21. External Integrations

| System | Purpose | Auth | Data exchanged | Failure behaviour | Local dev / testing |
| --- | --- | --- | --- | --- | --- |
| **AWS EC2** | Provision/terminate 4 GPU + 1 control node | IAM via env/profile — **never in the repo** | Instance lifecycle calls | Capacity error → abort, report, retry another AZ. Never silently downsize | `--dry-run` prints the plan without calling AWS |
| **AWS S3** | Stage tokenized dataset shards | IAM | Dataset shards, optional result backup | Sync failure → abort bootstrap; checksums verified before any run | LocalStack or a local directory `[PROPOSED]` |
| **Hugging Face Hub** | Download FineWeb-Edu / C4 (control node only, once) | Anonymous or read token | Dataset files | Failure → abort dataset prep. Never partially tokenize | Cached fixture for tests |
| **Prometheus / DCGM** | Live operator metrics | None (private subnet) | GPU telemetry | Failure → runs continue. **No result may depend on it** | Optional |
| **GitHub** | Code, results, issues | SSH key | Repository | — | — |
| **`torchft` lighthouse** | Replica-group coordination | None (private subnet) | Membership/heartbeat | Failure → job stalls; this is expected in fault experiments and is recorded | 4-process local test |

**Rate limits:** only Hugging Face is realistically rate-limited; mitigated by downloading once to S3 and never re-downloading from the GPU nodes.

**Timeouts (all `[PROPOSED]`, to be tuned on Day 1):** `iperf3` 60 s; NCCL init 300 s; NCCL collective 600 s (must exceed the worst-case sync at 50 Mbit/s — miscalibrating this will cause spurious failures at low bandwidth, so compute it, don't guess); S3 sync 1800 s.

---

# 22. Authentication and Authorization

`[CONFIRMED]` **Not applicable at the application level.** There are no user accounts, no sessions, no tokens, no roles enforced in code, and no multi-tenancy. Building any of it would be invented complexity.

Access control exists only at the infrastructure layer:

| Boundary | Mechanism |
| --- | --- |
| Who can launch/terminate the cluster | AWS IAM credentials held by the operator |
| Who can reach the GPU nodes | SSH key + security group; SSH restricted to the operator's IP `[RECOMMENDATION]` |
| Node-to-node traffic | Security group permits all traffic **within the group only** — required for NCCL, and the most common cause of multi-node failures when misconfigured |
| Control-node services (lighthouse, Prometheus) | Bound to the private subnet; **never** exposed to `0.0.0.0` |
| Who can write to `results/` | Git commit access to the repository |

---

# 23. Security

The threat model here is small and honest: the assets worth protecting are **AWS credentials**, **the integrity of the measurement corpus**, and **the cost ceiling**. There is no user data, no PII, no authentication surface, and no untrusted input from third parties.

| Concern | Control | Status |
| --- | --- | --- |
| **Credential leakage** | No credentials in the repo, in configs, in result records, or in log output. IAM via environment/instance profile. Pre-commit secret scanning | `[CONFIRMED]` |
| **Least privilege** | IAM policy scoped to EC2 lifecycle + the specific S3 bucket; no wildcard admin | `[RECOMMENDATION]` |
| **Network exposure** | Only SSH (port 22) from the operator IP is public. Everything else is private-subnet only. Lighthouse and Prometheus never bind publicly | `[CONFIRMED]` |
| **Fingerprint scrubbing** | Environment fingerprints are committed publicly — they must exclude account IDs, private IPs, ARNs, key names, and bucket names. A scrubber runs before write | `[CONFIRMED]` — easy to forget, embarrassing to fix later |
| **Cost as a security concern** | Idempotent `teardown.sh`; a billing alarm; a cluster-hours ledger. **An orphaned 4-GPU cluster is the most likely real "incident" in this project** | `[RECOMMENDATION]` |
| **Supply chain** | Pinned lockfile; `torchtitan`/`torchft` pinned to SHAs; no unpinned `main` installs | `[CONFIRMED]` |
| **Privilege on nodes** | `tc` requires root. Shaping commands are a fixed, parameterized allowlist — no shell interpolation of user input into `tc` invocations | `[CONFIRMED]` |
| **Result integrity** | Append-only convention + git history + schema validation. Not cryptographic; the threat model does not include a malicious insider | `[CONFIRMED]` |

Not applicable, and deliberately so: CSRF, XSS, SQL injection, SSRF, file-upload handling, password storage, session management. There is no web surface.

---

# 24. Privacy and Data Handling

| Question | Answer |
| --- | --- |
| What personal data is collected? | **None.** No users, no accounts, no telemetry from third parties |
| What data is collected? | Machine performance measurements, network measurements, training loss curves |
| Training data | Public web-derived corpora (FineWeb-Edu, C4). Used for language-model pretraining measurement only |
| Model weights | Randomly initialized, trained on public data, **not released**. No provenance or licensing concern |
| Where is data stored? | Node NVMe (transient), S3 (dataset), git repository (results) |
| Who can access it? | Results are public by design. Dataset staging is private |
| Retention | Results retained permanently. Cluster storage destroyed at teardown |
| Sensitive data in results | Only the infrastructure identifiers noted in §23 — removed by the scrubber |
| Licensing | `[CONFIRMED]` FineWeb-Edu and C4 are both ODC-BY v1.0, both subject to Common Crawl's ToU (which includes an AI/ML-use indemnification clause, §40 Q7/ADR-019, `LIMITATIONS.md`). This repo redistributes neither raw nor tokenized dataset content — only checksums and download scripts |

---

# 25. Error Handling

## 25.1 Error taxonomy

| Class | Example | Logged | Recorded | Retry | Aborts run |
| --- | --- | --- | --- | --- | --- |
| `spec_invalid` | Schema violation, `H>1` with DDP | ✓ | ✗ (no run created) | ✗ | ✓ (before resources) |
| `precondition_failed` | Missing NetworkProfile, dataset checksum mismatch, clocks unlocked | ✓ | ✓ | ✗ | ✓ |
| `shaping_apply_failed` | `tc` permission denied | ✓ | ✓ | ✓ ×1 | ✓ on repeat |
| `shaping_verification_failed` | Measured rate outside tolerance | ✓ | ✓ | ✓ ×1 | ✓ on repeat |
| `nccl_init_failed` | Rendezvous/SG misconfiguration | ✓ | ✓ | ✗ | ✓ |
| `nccl_timeout` | Collective exceeded timeout | ✓ | ✓ | ✗ | ✓ |
| `oom` | GPU out of memory | ✓ | ✓ | ✗ | ✓ |
| `diverged` | Loss NaN/spike | ✓ | ✓ | ✗ | ✓ (**this is data, not a bug** — divergence at large `H` is a finding) |
| `crashed` | Unexpected rank death | ✓ | ✓ (partial telemetry) | ✗ | ✓ |
| `restore_failed` | qdisc not restored | ✓ | ✓ | ✓ ×2 | node marked dirty |
| `reconciliation_failed` | Step-time components don't sum within tolerance | ✓ | ✓ | ✗ | observation invalid |
| `schema_write_violation` | Output record fails validation | ✓ | ✓ | ✗ | ✓ — **never write an invalid record** |

## 25.2 Error flow

```text
Error raised
   ↓
Classify (error_class)
   ↓
Structured log entry (run_id, phase, class, context) — never a bare traceback
   ↓
Persist a failure record to results/raw/ with status ≠ completed
   ↓
Restore network state (ALWAYS — even on abort)
   ↓
Retry? ── only shaping_apply / shaping_verification / restore ── else no
   ↓
Grid continues to the next point; campaign summary counts by class
   ↓
Surfaced in RESULTS.md — failures are published, not hidden
```

## 25.3 Principles

- **A silent skip is a bug.** Every failure produces a persisted record.
- **Never write a partially valid record.** Validate before write; on failure, write a failure record instead.
- **Restore is unconditional.** Network state is restored on every exit path, including `SIGINT`.
- **Divergence and hangs are outcomes, not exceptions.** They are recorded with the same rigour as successes.

---

# 26. Observability

## 26.1 The distinction that matters

| Purpose | System | Authoritative? |
| --- | --- | --- |
| **Results** | Structured records in `results/` | **Yes — the only source of truth** |
| **Operator awareness during a run** | Prometheus + Grafana + DCGM | No. May be absent entirely |
| **Debugging** | Structured logs, `torch.profiler` traces, `NCCL_DEBUG=INFO` | No |

`[CONFIRMED]` No figure, table, or claim may derive from Prometheus. Monitoring is for the human watching the run, not for the paper.

## 26.2 Logging

**Format:** structured JSON lines. **Correlation:** every line carries `run_id`, `campaign_id`, `rank`, `harness_version`.

| Must be logged | Must NOT be logged |
| --- | --- |
| Run lifecycle transitions | AWS credentials, tokens, key material |
| Every shaping apply/verify/restore with measured values | Account IDs, private IPs, bucket ARNs (scrub before write) |
| Precondition check outcomes | Full tensors or model weights |
| Every error with its class and context | Raw dataset content |
| Cumulative cluster-hours and estimated spend | Anything at per-step granularity in the *text* log (that belongs in Parquet) |

## 26.3 Metrics

**Per-step (Parquet):** wall time, compute time, sync-blocked time, optimizer time, loader stall, tokens, loss, peak memory, current `H`, whether this step was a sync step.

**Per-run (JSON):** everything in §15's `RunResult`.

**Live (Prometheus, optional):** GPU utilization, power, memory, PCIe/network counters, step rate, current loss.

## 26.4 Health checks and alerts

| Check | Cadence | Action |
| --- | --- | --- |
| All 4 nodes reachable and GPU visible | Before every run | Abort |
| Clocks locked | Before every run | Abort |
| Dataset checksums match | Before every campaign | Abort |
| No dirty qdisc | Before every run | Abort |
| Step-time component reconciliation | Per run | Invalidate observation |
| Loader stall < threshold | Per run | Set `loader_bound_warning` |
| Cluster-hours vs budget | Hourly | Warn the operator at 80% |
| Idle cluster (no run for 30 min) | Continuous | **Warn loudly — this is how money is wasted** `[RECOMMENDATION]` |

---

# 27. Performance

Note the inversion: for this project, "performance" is mostly about **measurement fidelity and cost**, not about making the system fast. Making the training loop faster than it naturally is would actively corrupt the experiment.

| Target | Value | Status |
| --- | --- | --- |
| Phase A grid point wall time | ≤ 5 min so the grid fits an overnight block | `[PROPOSED]` |
| Phase A grid total | ≤ 6 cluster-hours | `[PROPOSED]` |
| Convergence run (130M, 400M tokens, 4× L40S) | ~40 min | `[PROPOSED — must be measured on Day 1; this estimate drives the entire Phase B plan]` |
| Phase B total | ≤ 9 h, one overnight block | `[PROPOSED]` |
| Instrumentation overhead | < 1% of step time | `[PROPOSED]` — **must be measured**: CUDA-event sync points can be surprisingly expensive |
| `make figures` on a laptop | ≤ 5 min | `[PROPOSED]` |
| Shaping verification | ≤ 30 s per level | `[PROPOSED]` |
| Total cloud spend | ≤ $800 | `[PROPOSED]` |
| Repository size | ≤ 500 MB | `[PROPOSED]` |

`[UNKNOWN]` Achieved tokens/s per L40S for each model size; achieved NCCL bandwidth over ENA; whether 8 vCPU can feed one L40S at seq len 1024. **All three must be measured on Day 1 and will invalidate the Phase B schedule if the estimates are badly wrong.** Do not fabricate these numbers.

## 27.1 Measurement-fidelity requirements

These are not optimizations; they are correctness conditions:
- GPU clocks locked for every timed run (`nvidia-smi -lgc`).
- Warmup discarded, count recorded.
- Median and IQR over ≥3 repeats for throughput; never mean-only.
- Nothing else running on the nodes during a measurement window.
- Instrumentation overhead measured once and reported.

---

# 28. Scalability

`[CONFIRMED]` **This system does not need to scale, and designing for scale would be a mistake.** Expected lifetime scale: 5 instances, one operator, one week, a few hundred runs, a few hundred MB of results.

What is deliberately *not* built: horizontal scaling, autoscaling, sharded storage, caching layers, CDN, partitioning, read replicas.

The one dimension where scaling matters is the **experiment grid**, and it is handled by:
- The memory/feasibility filter, which prunes infeasible points before execution.
- `--resume`, so an interrupted campaign is cheap to continue.
- Prioritized ordering: the most informative grid points run first, so a truncated campaign still yields a publishable result.

If someone later wants 16 nodes, the changes are: `world_size` in the spec, the launch script, and re-fitting the predictor. Nothing in the architecture blocks it. That is sufficient forward-compatibility; anything more is speculation.

---

# 29. Reliability

## 29.1 Failure domains

| Domain | Failure | Blast radius | Mitigation |
| --- | --- | --- | --- |
| A GPU node | Instance failure, spot reclaim | One run | Grid continues; the run is recorded as crashed and re-queued |
| Control node | Lighthouse/rendezvous loss | All in-flight runs | On-demand (never spot) for the control node; cheap to replace |
| Network shaping | qdisc stuck | Subsequent runs on that node | Node marked dirty; explicit reset path |
| Dataset | Corrupt/missing shard | All runs | Checksums verified at bootstrap and before each campaign |
| AWS capacity | `g6e` unavailable | The whole project | Check capacity Day 0; have a second AZ and the `g6.2xlarge`/L4 fallback identified |
| Human | Cluster left running | Budget | Idle alarm + billing alarm + idempotent teardown |

## 29.2 Recovery

- **Campaign level:** `--resume` is the recovery mechanism. Completed records are the checkpoint.
- **Convergence runs:** periodic model checkpoints so a multi-hour run is not lost to a spot reclaim `[PROPOSED]`.
- **Network state:** restore on every exit path, including signals; a dedicated `make network-reset` for manual recovery.
- **Cluster:** `launch_cluster.sh` is idempotent; a full rebuild should take under 20 minutes and is the sanctioned response to a confusing environment.

## 29.3 Consistency

Append-only records with no cross-record mutation means there are no transactions and no consistency problems. A record either exists and is valid, or it does not exist. That property is worth more here than any convenience a database would add.

## 29.4 Instance market policy `[CONFIRMED — ADR-025, supersedes ADR-022/§40 Q9]`

**Everything on-demand. No spot, anywhere, for now.** Checked against the actual account
(2026-08-10): the Spot "G and VT" vCPU quota is 8 — under the 32 vCPU the 4-node Phase A fleet
needs simultaneously — and live `g6e.2xlarge` spot pricing (~$2.15–2.24/hr) was essentially
identical to on-demand (~$2.24/hr). Spot bought no real discount and couldn't run the full
fleet at once, so it was dropped everywhere rather than partially adopted. Revisit only if a
spot-quota increase is granted **and** spot pricing later shows a genuine gap versus on-demand
— re-check both before trusting this note.

---

# 30. Testing Strategy

## 30.1 The pyramid, adapted

```text
                    ┌──────────────────────────┐
                    │  E2E (cluster required)  │   1 test: `make smoke`
                    │  4 nodes · tiny · 20 stp │   Runs on Day 1 and before each campaign
                    └──────────────────────────┘
              ┌────────────────────────────────────┐
              │  Integration — CPU only (CI)       │   gloo 4-proc DiLoCo equivalence
              │  no GPU, no AWS                    │   netshape on loopback
              └────────────────────────────────────┘
        ┌────────────────────────────────────────────────┐
        │  Unit                                          │   cost models · wire math
        │  fast, deterministic, every commit             │   codecs · schemas · filters
        └────────────────────────────────────────────────┘
```

## 30.2 Unit tests

| Target | What is asserted |
| --- | --- |
| `analysis/cu.analytic` | Known-input/known-output cases hand-computed from `methods/cu_model.md`; H=1 reduces to the DDP case; monotonicity in bandwidth and in H |
| `wire.predict` | Ring all-reduce byte counts for known (N, P, H); DDP vs DiLoCo ratio equals H |
| `compress` codecs | Round-trip error bounds; **error-feedback residual accumulates and is not dropped across rounds** (the invariant most likely to be silently broken) |
| Schema validation | Valid records pass; every documented invalid case is rejected with a useful message |
| `filter` | Crashed/loader-bound/version-mismatch records are excluded; exclusion counts are reported |
| `predictor` | Extrapolation outside `calibration_domain` sets the warning flag |
| `ConvergenceCurve` | `tttl == null` never becomes a finite number downstream |
| Spec cross-field validation | `H>1` with DDP rejected; compression with DDP rejected |

## 30.3 Integration tests (CPU, in CI)

| Test | Method | Asserts |
| --- | --- | --- |
| **DiLoCo cross-implementation equivalence** | `gloo`, 4 processes, tiny model, fixed seed, H=4 | Reference `diloco.py` and the torchft path produce loss curves agreeing within a documented tolerance. **A divergence fails CI.** (US-06) |
| DiLoCo invariants | Same harness | All replicas hold identical `θ_outer` after each outer step; inner optimizer state persists across rounds; loss decreases |
| `netshape` gate | Loopback or two cheap instances | Requesting a rate and measuring a different one produces `passed: false` and no `RunResult` |
| Restore idempotency | Loopback | `restore()` twice is safe; qdisc returns to the original state |
| Aggregation pipeline | Fixture records | Figures generate from fixtures with no GPU, no network |
| Fingerprint completeness | Fixture | A record missing any fingerprint field is rejected at write time |

## 30.4 E2E test

`make smoke` — 4 real nodes, ~1M-parameter model, 20 steps, one shaped bandwidth level. Asserts: cluster reachable, shaping verified, run completes, a schema-valid `RunResult` is emitted, network restored. **This is the gate before every campaign.** If smoke fails, nothing else runs.

## 30.5 Statistical / methodological tests

| Check | Why |
| --- | --- |
| Repeat variance within tolerance | High variance invalidates single-run conclusions |
| Warmup sufficiency | Compare 10/20/30-step discard; if the answer changes, warmup is too short |
| Instrumentation overhead | Instrumented vs uninstrumented run at the same config |
| Idle-baseline network drift | Confirms `/proc/net/dev` accounting isn't polluted |
| Straggler spread across nodes | Quantifies how much of the sync stall is heterogeneity rather than bandwidth |

## 30.6 Test data and fixtures

- A fixture corpus of ~20 synthetic `RunResult` records covering every status, used by all analysis tests.
- A ~1M-parameter model config for smoke and CPU tests.
- A small tokenized shard committed for CPU integration tests.
- **No mocks in the measurement path.** A mocked `iperf3` would defeat the purpose of the gate; that path is tested on real interfaces only.

## 30.7 Coverage expectations `[PROPOSED]`

- `analysis/` and `schemas/`: ≥90% line coverage. These are pure and cheap to test, and an error here corrupts every conclusion.
- `measurement/`: coverage is not a meaningful target; correctness is established by the E2E smoke test, the cross-implementation check, and the reconciliation invariants.

## 30.8 CI behaviour

Runs on every push: lint, type-check, unit, CPU integration. **Never** touches AWS, never requires a GPU. A CI run must complete in under 10 minutes or people stop trusting it.

---

# 31. Development Workflow

```text
Requirement / question
   ↓
[1] Classify: does this touch the MEASUREMENT path or the ANALYSIS path?
   ↓        (this classification determines everything that follows)
[2] Check CLAUDE.md — is the decision already made? Is it PROPOSED or CONFIRMED?
   ↓
[3] If UNKNOWN → add to §40 Open Questions. DO NOT GUESS.
   ↓
[4] Design the smallest change that answers the requirement
   ↓
[5] Write the test first where the logic is pure (analysis, schemas, codecs)
   ↓
[6] Implement
   ↓
[7] MEASUREMENT PATH? → bump harness_version, state the impact on existing results
   ↓
[8] Run unit + CPU integration locally
   ↓
[9] If cluster work: `make smoke` before any campaign
   ↓
[10] Update CLAUDE.md: §41 Decision Log, §40 Open Questions, §39 Technical Debt
   ↓
[11] Review the diff for scope creep and for accidental measurement-path changes
   ↓
[12] Commit with a message stating whether harness_version changed
```

## 31.1 Two development modes

The project alternates between two very different modes, and confusing them is the main way time gets wasted.

| | **Offline mode** (most of the time) | **Cluster mode** (expensive, timeboxed) |
| --- | --- | --- |
| Cost | ~$0 | ~$9.33/hr |
| Activities | Writing code, tests, schemas, analysis, figures, docs | Running campaigns, characterizing the network, debugging real NCCL |
| Rule | Everything that *can* be done here *must* be done here | Nothing is written from scratch here |
| Discipline | — | **The cluster is not a development environment.** Arrive with working code |

`[CONFIRMED]` Day 0 exists precisely to maximize offline mode. Every hour of code written before the cluster launches is an hour of GPU spend saved and a de-risked Day 1.

## 31.2 Branching `[PROPOSED]`

Solo project: work on `main` with disciplined commits. Use a branch only for a change that could break the measurement path mid-campaign. Tag the harness version at the start of each campaign (`harness-v3-phaseA`) so results are traceable to an exact tree.

---

# 32. Feature Development Protocol

Every change follows these eight steps. For this project, step 3 has a project-specific shape.

### Step 1 — Understand
Read the relevant requirement (§6) and the method document in `methods/`. If the change affects how a number is produced, the method doc is the specification, not the code.

### Step 2 — Explore
Find the existing module. Check whether an equivalent function already exists. **Search before creating.** This codebase is small enough that duplication is always avoidable.

### Step 3 — Impact analysis (project-specific)

| Dimension | Question |
| --- | --- |
| **Measurement path?** | Does this change *any* number that enters a record? If yes → `harness_version` bump, and existing results cannot be pooled with new ones without a documented equivalence argument |
| **Schema?** | Does the record shape change? → new schema version, loader must handle both |
| **Analysis?** | Do existing figures change? → they must be regenerated and the change noted in `RESULTS.md` |
| **Method doc?** | Does an assumption in `methods/` change? → update it in the same commit |
| **Cost?** | Does this add cluster time? → estimate it against the remaining budget |
| **Reproducibility?** | Can a reviewer still run `make figures` with no GPU? |
| **Prior art?** | Does this change what we claim is novel? → `PRIOR_ART.md` |

### Step 4 — Design
The smallest design that satisfies the requirement. If the design introduces an abstraction, it must have at least two concrete users today — not "later."

### Step 5 — Implement
Follow existing patterns. Keep measurement and analysis separated. No logic in `cli.py`.

### Step 6 — Test
Pure logic → unit test. Distributed logic → CPU gloo integration test. Cluster-only behaviour → note explicitly that it is covered by `make smoke` and nothing else.

### Step 7 — Verify
Lint, type-check, unit, CPU integration. If measurement code changed, `make smoke` on the cluster before trusting anything.

### Step 8 — Review
Read the diff. Ask: did I accidentally change a measurement path? Did I widen scope? Did I add an abstraction with one user? Did I update `CLAUDE.md`?

---

# 33. Claude Coding Rules

**These are operational rules for every future Claude session on this repository.**

## 33.1 Claude MUST

1. **Read this file before any change.** Especially §12 (Principles), §16.3 (immutability + harness versioning), and §41 (Decision Log).
2. **Classify every task as measurement-path or analysis-path** before writing code, and state the classification in the response.
3. **Bump `harness_version` for any measurement-path change**, and explicitly tell the operator which existing results are now non-poolable.
4. **Treat `results/raw/` as read-only.** Always. No exceptions. Corrections are new records.
5. **Preserve the verification gate.** If a task appears to require relaxing or bypassing shaping verification, refuse and explain — that gate is the project's integrity.
6. **Label uncertainty.** Use `[CONFIRMED]` / `[PROPOSED]` / `[UNKNOWN]` in code comments and docs. Never present a guess as a measurement.
7. **Keep field names provenance-explicit.** `cu_measured`, not `cu`. `bandwidth_measured_bps`, not `bandwidth`.
8. **Write the method doc alongside the code** when a change alters how a number is computed.
9. **Add tests for pure logic**, always. Pure logic here is cheap to test and expensive to get wrong.
10. **Ask before spending cluster time.** Any suggestion that implies GPU hours must state the estimated cost.
11. **Report negative results** rather than quietly dropping them.
12. **Update §40 Open Questions** when an unknown is encountered, and §41 when a decision is made.
13. **Prefer deleting over abstracting.** If code is unused, remove it.
14. **State limitations in the artifact**, not just in conversation.

## 33.2 Claude MUST NOT

1. **Never edit or delete anything in `results/raw/`, `results/network/`, or `results/environment/`.**
2. **Never pool results across `harness_version` values** without an explicit, documented equivalence argument approved by the operator.
3. **Never record a requested value where a measured value is required.** No fallback to "the rate we asked for" when verification is unavailable.
4. **Never soften or skip a precondition or gate to make a run succeed.** A failed gate is the correct outcome.
5. **Never let analysis code import measurement code**, or introduce a GPU/network/credential dependency into the figure path.
6. **Never invent a number.** Not a bandwidth, not a throughput, not a token count, not a cost. If it is not measured, it is `[UNKNOWN]`.
7. **Never claim novelty beyond §2.3.** The algorithm is not ours. The scaling laws are not ours. Only the controlled measurement is.
8. **Never introduce a database, a queue, a web framework, a container orchestrator, or an external experiment tracker** as a load-bearing dependency. All were considered and rejected (§13.5).
9. **Never add an abstraction with a single implementation.**
10. **Never hardcode credentials, account IDs, ARNs, bucket names, or private IPs** — including in fingerprints and logs.
11. **Never auto-retry a training run.** A nondeterministic failure is data.
12. **Never render `tttl == null` as a finite number**, and never render a `crashed`/`diverged` run as if it were `completed`.
13. **Never modify a shared measurement code path mid-campaign** without stopping the campaign first.
14. **Never leave the cluster running** at the end of a work session.

## 33.3 The single most dangerous action in this repository

> Fixing a bug in the timing or accounting code on day 5, and then plotting days 1–5 together.

This produces a figure that is wrong in a way nobody can detect from the figure. The `harness_version` rule exists solely to prevent it. If you are ever tempted to bypass that rule to save a re-run, **stop and ask the operator.**

---

# 34. Definition of Done

## 34.1 For a code change

```text
[ ] Requirement identified in §6 (or added, with a status tag)
[ ] Classified: measurement path or analysis path — stated explicitly
[ ] Design is the smallest that satisfies the requirement
[ ] Implemented, following existing patterns
[ ] Measurement/analysis separation preserved
[ ] harness_version bumped if the measurement path changed
[ ] Impact on existing results stated (poolable / not poolable)
[ ] Schema updated + versioned if the record shape changed
[ ] Unit tests added for pure logic
[ ] CPU integration test added/updated if distributed logic changed
[ ] methods/ doc updated if a computation's definition changed
[ ] Lint, type-check, unit, CPU integration all pass
[ ] make smoke passes (if measurement code changed and a cluster is up)
[ ] CLAUDE.md updated: §40 Open Questions, §41 Decision Log, §39 Debt
[ ] Diff reviewed for scope creep and accidental measurement changes
```

## 34.2 For an experiment campaign

```text
[ ] make smoke passed before the campaign started
[ ] A NetworkProfile exists covering every bandwidth level used
[ ] Every completed run carries a passing ShapingVerification (or is unshaped)
[ ] Every record is schema-valid
[ ] Every record has a complete environment fingerprint
[ ] Repeat count met; median + IQR computed
[ ] Failed/crashed/diverged points recorded and counted, not deleted
[ ] Step-time reconciliation within tolerance for all included runs
[ ] Loader-bound runs flagged and excluded (with the count reported)
[ ] Results committed to results/raw/
[ ] experiments/<campaign>/NOTES.md written, including what went wrong
[ ] RESULTS.md updated with the campaign summary
[ ] Cluster torn down; cost logged
```

## 34.3 For the project

```text
[ ] G1 measured CU surface populated (≥4 bandwidth levels × ≥4 H values × 3 repeats)
[ ] G2 discrepancy factor F reported at 50/75/90/95% CU with uncertainty
[ ] G3 ≥10 convergence runs with TTTL against a single-GPU reference
[ ] G4 predictor fitted and validated on a held-out configuration
[ ] G5 `make figures` regenerates every report figure with no GPU
[ ] PRIOR_ART.md complete and linked from the first screenful of README
[ ] LIMITATIONS.md complete (no WAN latency, 4 replicas, one GPU generation,
      small models, short budgets, single-GPU replicas)
[ ] PLAYBOOK.md written for practitioners
[ ] Report/blog written; every claim traceable to a committed record
[ ] Resume bullets drafted with real measured numbers (no placeholders left)
[ ] Interview-defence pass: every figure survives "how do you know?"
[ ] Cluster terminated; final cost recorded
```

---

# 35. Implementation Phases

## Phase 0 — Discovery and offline construction (Day 0, ~$0)

**Objective:** arrive at the cluster with nothing left to write.

- Resolve §40 Q1 (region/AZ capacity), Q2 (torchft version). Q4 (Docker or not) is already
  resolved — ADR-020, bare metal.
- Write `PRIOR_ART.md` **first** — it forces precision about the gap and prevents scope drift all week.
- Implement: `diloco.py`, `compress.py`, `cu.py` (both paths), `netshape.py`, `wire.py`, schemas, spec validation.
- Test: 4-process gloo DiLoCo equivalence and invariants; `netshape` gate on loopback/two `t3.micro`s.
- Pre-tokenize FineWeb-Edu on the control node → S3 (hours of wall clock, pennies of cost — deferring this to mid-week is the single most expensive scheduling mistake available).
- Write `launch_cluster.sh`, `setup_node.sh`, `Makefile`; dry-run the launcher.

**Exit criteria:** unit + CPU integration tests green; two DiLoCo implementations agree on CPU; dataset staged in S3; launcher dry-runs cleanly.

---

## Phase 1 — Cluster bring-up and network characterization (Day 1)

**Objective:** a working 4-node distributed environment and a trustworthy network profile. *This phase is a deliverable in its own right* — the NCCL-over-TCP characterization is publishable independently (G8).

- Launch cluster; verify NCCL over TCP with a 4-node DDP step.
- `iperf3` all-pairs; NCCL bandwidth curve; shaping fidelity at all 5 levels; burst-decay probe.
- Measure the three unknowns that drive the schedule: tokens/s per model size, achieved NCCL bandwidth, whether 8 vCPU can feed one L40S.
- `make smoke`.

**Exit criteria:** a 4-node run completes; requesting 1 Gbit/s measures 1 Gbit/s within tolerance; a committed `NetworkProfile`; **the Phase B time estimate is now measured, not assumed.**

---

## Phase 2 — Algorithm validation and first CU slice (Day 2)

**Objective:** know by tonight whether the headline discrepancy is real.

- Get the torchft DiLoCo path running on GPUs; cross-validate against the reference implementation at a fixed seed.
- Run a ~12-point CU slice at one bandwidth level.

**Exit criteria:** implementations agree on GPU; `cu_measured` vs `cu_analytic_link` plotted for one bandwidth level. **This is the earliest possible signal on the central hypothesis, and it is deliberately scheduled early so a pivot is still cheap.**

---

## Phase 3 — The CU grid (Day 3)

**Objective:** G1 and G2.

- Full Phase A grid, largely unattended. Analyse as points land. Write `predictor.py`.

**Exit criteria:** the headline surface exists; the discrepancy factor `F` is computable with uncertainty.

---

## Phase 4 — Convergence (Days 4–5)

**Objective:** G3.

- Day 4: validate the 130M config; launch the Phase B block overnight. During the day (offline): implement/unit-test compression; draft the methods section while network details are fresh.
- Day 5: loss curves in; TTTL computed; run compression variants.

**Exit criteria:** TTTL-vs-bandwidth figure exists; ≥10 convergence runs recorded.

---

## Phase 5 — Faults and predictor validation (Day 6)

**Objective:** G4, G7.

- Phase C fault injection at H=32 and H=512, plus the DDP baseline.
- Phase D held-out predictor validation.

**Exit criteria:** recovery numbers recorded; predicted-vs-measured `H` plotted with regret.

---

## Phase 6 — Close-out (Day 7)

**Objective:** G5 and the artifact.

- Final headline re-runs on a quiet, on-demand cluster.
- **Tear down the cluster before writing anything.**
- Regenerate all figures from committed data; write README, PRIOR_ART, LIMITATIONS, PLAYBOOK, RESULTS, report; draft resume bullets; record the demo.

**Exit criteria:** §34.3 satisfied; cluster terminated; final cost recorded.

---

# 36. Milestones

### M1 — Offline foundation `(Phase 0)`
**Deliverables:** repo skeleton; `PRIOR_ART.md`; reference DiLoCo; codecs; CU module (both paths); netshape + gate; schemas; test suite; dataset in S3; launcher.
**Dependencies:** none.
**Acceptance:** CI green; gloo DiLoCo equivalence passes; launcher dry-run clean.
**Risks:** underestimating tokenization time (R7).
**Exit:** nothing on Day 1 requires writing new code.

### M2 — Trustworthy rig `(Phase 1)`
**Deliverables:** running 4-node cluster; committed `NetworkProfile`; verified shaping at all levels; passing smoke test; measured baseline throughput.
**Dependencies:** M1.
**Acceptance:** US-01 acceptance criteria demonstrably satisfied on real hardware.
**Risks:** R1 (NCCL bring-up), R6 (capacity), R3 (shaping instability).
**Exit:** every subsequent number rests on a verified network.

### M3 — Hypothesis signal `(Phase 2)`
**Deliverables:** GPU-validated DiLoCo (both implementations); a 12-point CU slice.
**Dependencies:** M2.
**Acceptance:** implementations agree; measured and analytic CU plotted together.
**Risks:** R2 (torchft experimental).
**Exit:** **go/no-go on the framing.** If measured ≈ analytic, the report reframes to "first empirical validation" — decided here, not later.

### M4 — The headline result `(Phase 3)`
**Deliverables:** full CU surface; discrepancy factor `F`; required-bandwidth table; NCCL BW-vs-message-size figure.
**Dependencies:** M3.
**Acceptance:** G1 and G2 met; ≥3 repeats per point.
**Exit:** the project has its central figure. **From here, everything else is additive** — this is the point after which the project cannot fail outright.

### M5 — Convergence evidence `(Phase 4)`
**Deliverables:** reference run; ≥10 convergence runs; TTTL; loss-vs-H; compression ablation.
**Dependencies:** M4 (informs which H values are worth the expensive runs).
**Acceptance:** G3 met; divergences recorded rather than hidden.
**Risks:** R4 (runs slower than estimated) — mitigated by shrinking the model and budget, not by cutting repeats.

### M6 — Tool and robustness `(Phase 5)`
**Deliverables:** fitted predictor + held-out validation; fault-injection records.
**Dependencies:** M4, M5.
**Acceptance:** G4 met; extrapolation warning behaves per US-05.

### M7 — Publishable artifact `(Phase 6)`
**Deliverables:** all docs; all figures regenerated from committed data; demo; resume bullets.
**Dependencies:** M4 minimum; M5/M6 desirable.
**Acceptance:** §34.3.
**Exit:** a reviewer with a laptop can check the central claim.

---

# 37. Dependency Graph

```text
                    §40 Open Questions Q1,Q2,Q4 resolved
                                  │
                                  ▼
                         M1 Offline foundation
                    (code · tests · schemas · dataset)
                                  │
                                  ▼
                      Cluster launch + bootstrap
                                  │
                                  ▼
                   M2 Network characterization + smoke
                   (NOTHING downstream is valid without this)
                                  │
                                  ▼
                    M3 DiLoCo validated + CU slice
                          │            │
                          │            └──────► GO/NO-GO on framing
                          ▼
                    M4 CU grid  ◄── THE HEADLINE RESULT
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        M5 Convergence  M6 Predictor  G8 NCCL characterization
              │           │            (independently publishable)
              └─────┬─────┘
                    ▼
              M7 Artifact
                    │
                    ▼
              Teardown + report
```

**Critical path:** M1 → M2 → M3 → M4 → M7. Everything else is parallel or optional.
**Hard gate:** M2. No result produced before a committed `NetworkProfile` is admissible.

---

# 38. Risks

## 38.1 Technical risks

| ID | Risk | Prob | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| **R1** | Multi-node NCCL will not establish (security group, `NCCL_SOCKET_IFNAME`, rendezvous) | Medium | **Critical** | Cluster placement group + SG allowing all intra-group traffic; `NCCL_DEBUG=INFO`; timebox to Day 1 22:00; **fallback: drop to 2 nodes** (`2× g6e.4xlarge`, still 32 vCPU) — RQ1 survives intact because the analytic model is per-link, not per-replica-count |
| **R2** | `torchft` LocalSGD/DiLoCo experimental APIs broken or shifted | **High** | Medium | D3: our own `diloco.py` written on Day 0. This risk is *pre-mitigated by design*. File the bug upstream (G9) |
| **R3** | `tc` shaping unstable, non-monotone, or confounded by ENA burst credits | Medium | High | Verification gate catches it; fall back to `netem rate`; reduce to 3 verified levels. **Never report an unverified rate.** Burst-credit interaction may itself be a finding |
| **R4** | Convergence runs slower than the ~40 min estimate | Medium | High | The estimate is measured on Day 1, not assumed. Fallback: 60M model, 200M tokens, and state the scale limitation |
| **R5** | Step-time components do not reconcile (instrumentation overhead or overlap accounting) | Medium | High | Reconciliation is an explicit invariant with a recorded residual; fall back to the two-measurement method (synchronous vs normal collectives) |
| **R6** | `g6e` capacity unavailable in the chosen AZ | **MATERIALIZED, 2026-08-11 — see ADR-031** | **Critical (currently blocking)** | Both documented fallbacks (2× `g6e.4xlarge`, `g6.2xlarge`/L4) also tried for real and also exhausted, across every AZ each type is offered in (a/b/c/d, +f for `g6`). Remaining options are outside this repo's control: wait for AWS capacity, or request quota in a new region. Not a script or config bug — every attempt's control node and permission checks succeeded; only GPU capacity was refused |
| **R7** | 8 vCPU per node cannot feed the GPU; runs are loader-bound | Medium | Medium | Pre-tokenized flat shards on NVMe; `loader_bound_warning`; measure and report loader stall |
| **R8** | Instrumentation overhead distorts the measurement it makes | Low | High | Measure it explicitly (instrumented vs uninstrumented at one config) and report it |

## 38.2 Scientific / result risks

| ID | Risk | Prob | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| **R9** | Measured ≈ simulated; no discrepancy | Medium | Low | **Pre-committed reframing** (§2.7): the null result becomes "first empirical validation." The project is designed so no outcome yields nothing |
| **R10** | Straggler heterogeneity across EC2 instances confounds CU | Medium | Medium | Measure per-rank step-time spread; report it as a separate component; it is a legitimate part of the real-hardware story |
| **R11** | Conclusions do not generalize beyond 4 replicas / 1B params / one GPU generation | **High** | Medium | **Accepted, not mitigated.** Stated prominently in `LIMITATIONS.md`. Overclaiming here is the real risk |
| **R12** | Novelty is challenged ("SmartSpec-style objection": this is already known) | Medium | High | `PRIOR_ART.md` written *first*, positioned precisely; claim only the controlled measurement |

## 38.3 Operational risks

| ID | Risk | Prob | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| **R13** | Cluster left running; budget blown | Medium | High | Idle alarm; billing alarm; idempotent teardown; cluster-hours ledger. **The most likely real incident in this project** |
| **R14** | ~~Spot reclaim mid-convergence-run~~ — moot as of ADR-025 (everything is on-demand now); retained as a general instance-failure risk (host maintenance, hardware fault) | Low | Medium | Periodic checkpoints (still good practice regardless of instance market) |
| **R15** | Environment drift between nodes | Low | **Critical** | Pinned lockfile; fingerprint on every record; bootstrap verifies version equality across nodes |
| **R16** | Seven days is not enough for all six phases | **High** | Medium | Phase ordering is by value: M4 is the headline and lands on Day 3. M5/M6 are additive. Truncation degrades gracefully |
| **R17** | Credentials or account identifiers committed | Low | High | Scrubber before write; pre-commit secret scan |

---

# 39. Technical Debt

Debt accepted deliberately, recorded so it is never mistaken for an oversight.

| ID | Debt | Why accepted | Trigger to repay |
| --- | --- | --- | --- |
| TD-1 | Bash + AWS CLI instead of IaC | 5 instances, one week, one operator | If the cluster is rebuilt >10 times or a second operator joins |
| TD-2 | No database; JSON + Parquet on disk | Small corpus; reviewability is worth more than queryability | If the corpus exceeds ~10k records |
| TD-3 | `tbf` shaping only; no latency/jitter/loss | Bandwidth is the variable under test; adding three more axes explodes the grid | If a WAN-realism claim is ever made |
| TD-4 | Only 4 replicas | Hard quota constraint | If more quota becomes available |
| TD-5 | Single-GPU replicas; no FSDP-inside-replica | Hardware; the two-level hierarchy is untestable here | If a multi-GPU node becomes available |
| TD-6 | Small models (≤1B) and short budgets (400M tokens) | Compute budget | If a larger grant of compute appears |
| TD-7 | One seed per convergence configuration | Cost; repeats spent on throughput instead | If a convergence conclusion turns out to be seed-sensitive |
| TD-8 | Prometheus/Grafana optional and untested | Not authoritative for any result | Never — deliberate |
| TD-9 | No packaging/publication to PyPI | Research instrument, not a dependency | If someone asks to depend on it |
| TD-10 | `tool.mypy.python_version = "3.12"` while `requires-python = ">=3.11"` | numpy>=2.5's stubs use a `type X = ...` statement mypy only parses under a 3.12+ target; doesn't change the actual minimum supported Python | Revisit if a genuine 3.11-only incompatibility needs catching |

**Not reader-facing** (TD-10 is a dev-tooling quirk, not a project limitation): only TD-1 through TD-9 belong in `LIMITATIONS.md`. **Every item in that range must appear there** in reader-facing language. Debt hidden from the reader is dishonesty; debt stated plainly is rigour. Debt hidden from the reader is dishonesty; debt stated plainly is rigour.

---

# 40. Open Questions / Decisions Required

> Every unresolved decision lives here so it is not rediscovered. Resolve, then move it to §41.

---


---

**Q2 — Which `torchft` and `torchtitan` commits do we pin?**
*Why it matters:* both move quickly; `torchft`'s semi-sync paths are experimental (R2). Version drift mid-week would silently invalidate comparisons.

*Research done 2026-08-10 (not a full resolution — the actual pin has to happen at Day 0, since both projects publish continuously and a version picked today would already be stale; this is de-risking HOW to pin, not WHAT to pin):*

- **The repo moved:** `torchft` now lives at `github.com/meta-pytorch/torchft` (renamed from the `pytorch` org — old links likely redirect, but write launch scripts against the new URL).
- **`torchft` is NOT pure Python** — it has a Rust "lighthouse" coordination component built via `maturin`/PyO3. Confirmed the hard way: `pip install git+https://github.com/meta-pytorch/torchft.git@<sha>` fails on a machine without a Rust toolchain + `protobuf-compiler` (`cargo metadata` error). **But** `torchft-nightly` publishes prebuilt `manylinux_x86_64` wheels **daily** (verified `2026.8.8`, `2026.8.9`, `2026.8.10` all present, cp39–cp313) — so on the actual Ubuntu GPU nodes, `pip install torchft-nightly==<date>` needs no Rust/protobuf toolchain at all. This is a better pin than a source-built SHA: it's an immutable published artifact, not a build step with its own toolchain dependency, which fits ADR-020's "fewest moving parts" reasoning even better than the original three options anticipated.
  → **This changes the option set.** The real choice is no longer "latest release tag vs. a source-built SHA" — `torchft` has no GitHub Releases at all (0 published). It's: **pin a specific `torchft-nightly` dated version from PyPI** (recommended — no toolchain dependency on nodes) vs. build a specific commit from source (only if a fix landed on `main` that hasn't reached a nightly build yet).
- **`torchft` requires PyTorch 2.7 RC+ or Nightly** — comfortably satisfied by any current PyTorch (the dev venv already runs 2.13).
- Checked 8 open GitHub issues tagged `diloco`/`localsgd` (as of 2026-08-10): all feature requests (bucketized averaging, federated extension, constant global batch size, etc.), **none describe a correctness bug**. Reassuring, but "no known bug" is not "verified correct" — the gloo cross-validation test (ADR-003/D3, currently skipped pending this exact pin) is still required, not optional.
- **`torchtitan` is a plain pure-Python wheel** (`pip install torchtitan` works cleanly, no compiled extension, no toolchain issue at all). Latest PyPI release: `0.2.2`. Latest `main` commit: `e061028`, 2026-08-09 (very active — ~1900 commits total). The PyPI release cadence lags `main` substantially (`0.2.0` shipped Oct 2025; only patch bumps since, against ~10 months of continued `main` development), so unlike `torchft`, pinning a `torchtitan` **git SHA from `main`** (not the PyPI release) is likely the better call — to be confirmed on Day 0 against whatever features are actually needed.
- **Model-size gap confirmed, then closed:** `torchtitan`'s shipped model configs are Llama 3.1 at 8B/70B/405B only — no small (<1B) preset exists. `configs/models/{130m,500m,1b}.toml`'s `[UNKNOWN]` dims were replaced with real values computed against `torchtitan`'s actual `TransformerModelArgs`/`FeedForward` formulas the same session — see ADR-026.

*Recommendation (updated):* `torchtitan` — pin a specific `main` SHA, validated on Day 0. `torchft` — pin a specific `torchft-nightly` dated PyPI version (not a source SHA), validated with the gloo equivalence test on Day 0; keep the reference-only path (skip `torchft`) as the live fallback if the pinned nightly's DiLoCo path doesn't check out.
*Decision:* **RESOLVED 2026-08-14 — see ADR-032.** `torchtitan @ git+https://github.com/pytorch/torchtitan@624c312b67` (note: `pytorch/torchtitan`, not `meta-pytorch/torchtitan` — that org rename applies to `torchft` only, a real bug in the previously-documented candidate). `torchft-nightly==2026.8.14`. Both installed and validated for real on a live `g6e.2xlarge`: `torchft`'s own DiLoCo/LocalSGD test suite passes 8/8; `torchtitan`'s debugmodel builds and runs a real forward+backward pass on an L40S with an exact parameter-count match to ADR-026 (6,163,712). The reference-vs-torchft cross-implementation equivalence test (US-06) itself is still NOT run — that remains open, tracked in ADR-032's "Not resolved" section.

---

**Q6 — How many repeats, and how many seeds for convergence?**
*Why it matters:* directly trades cost against statistical strength; under-repeating invalidates conclusions, over-repeating burns the budget.
*Options:* (1) 3 repeats throughput / 1 seed convergence; (2) 5/2; (3) adaptive based on observed variance.
*Recommendation:* option 1 as the plan, with option 3 as a rule: if throughput IQR exceeds a threshold, add repeats for that configuration only.
*Decision:* **PENDING — resolve after Day 1 variance is known.**

---

**Q8 — Is a `netem` WAN-emulation sub-experiment in scope?**
*Why it matters:* it would substantially strengthen the realism claim, but it multiplies the grid.
*Options:* (1) out of scope, stated as a limitation; (2) a single point (80 ms RTT, 0.05% loss) at one bandwidth as a spot check; (3) a full axis.
*Recommendation:* option 2 if and only if M4 lands early. Option 1 otherwise.
*Decision:* **PENDING.**

---

---

# 41. Architectural Decision Log

---
**ADR-001 — Four single-GPU nodes over commodity Ethernet**
**Status:** Accepted · **Date:** 2026-08-08
**Context:** Hard 32-vCPU GPU quota that cannot be raised. Options were 1× `g6e.8xlarge` (1 GPU), 4× `g6e.2xlarge` (4 GPUs, TCP only), or 4× `g6.2xlarge` (4× L4, cheaper, weaker).
**Decision:** 4× `g6e.2xlarge` plus a CPU-only control node.
**Reason:** it is the only configuration within quota that yields four replicas, and the TCP-only topology *is* the experimental regime the literature simulates and never measures. The constraint is the apparatus.
**Trade-offs:** no NVLink, no EFA, 8 vCPU per node, no ability to test FSDP-inside-a-replica (TD-5).

---
**ADR-002 — `tc`/`tbf` shaping with a hard verification gate**
**Status:** Accepted · **Date:** 2026-08-08
**Context:** Bandwidth must be an independent variable; renting geographically distributed nodes gives uncontrolled, confounded links.
**Decision:** shape egress on every node; verify with `iperf3` + a NCCL probe; abort on failure; record only measured rates.
**Reason:** control and repeatability, plus an auditable integrity guarantee.
**Trade-offs:** `tbf` adds no latency/jitter/loss, so measured discrepancy is a lower bound on real-WAN discrepancy (TD-3). Aborting costs grid points.

---
**ADR-003 — Two DiLoCo implementations**
**Status:** Accepted · **Date:** 2026-08-08
**Context:** `torchft`'s LocalSGD/DiLoCo are marked experimental; a silent bug would invalidate the study.
**Decision:** an in-repo reference implementation cross-validated against `torchft`, written before cluster time begins.
**Reason:** strongest available correctness argument; doubles as the fallback for R2.
**Trade-offs:** a few hours of duplicated effort.

---
**ADR-004 — Filesystem result store, no database**
**Status:** Accepted · **Date:** 2026-08-08
**Context:** A few hundred self-contained records, batch access, public reviewability as a primary goal.
**Decision:** schema-validated JSON + Parquet, committed to git, append-only. A derived SQLite index may exist but is never authoritative.
**Reason:** zero-dependency reproduction; records readable in a GitHub diff.
**Trade-offs:** no ad-hoc querying (TD-2).

---
**ADR-005 — Strict measurement/analysis separation**
**Status:** Accepted · **Date:** 2026-08-08
**Decision:** two packages, no cross-imports; analysis runs with no GPU, no network, no credentials.
**Reason:** it is what makes FR-11 achievable, and it prevents figures from silently depending on a live cluster.
**Trade-offs:** minor duplication of small helpers.

---
**ADR-006 — `harness_version` gating on result pooling**
**Status:** Accepted · **Date:** 2026-08-08
**Context:** The most common way research measurements go quietly wrong is fixing a timing bug late and pooling everything.
**Decision:** any measurement-path change bumps `harness_version`; the loader refuses to mix versions without an explicit override that is recorded in figure metadata.
**Reason:** makes the failure mode impossible by default rather than by discipline.
**Trade-offs:** occasional forced re-runs.

---
**ADR-007 — Both analytic CU variants are computed**
**Status:** Accepted · **Date:** 2026-08-08
**Decision:** report `cu_analytic_link` and `cu_analytic_achieved` alongside `cu_measured`.
**Reason:** separates "the model is wrong" from "the model's input assumption is wrong" — the difference between a naive result and a useful one.
**Trade-offs:** requires a NCCL bandwidth curve per shaping level (extra Phase 1 time).

---
**ADR-008 — Prior art documented before any code**
**Status:** Accepted · **Date:** 2026-08-08
**Context:** The originally considered project (auto-tuned speculative decoding) was abandoned after prior-art review showed it was already solved and shipped by the vLLM team. Overclaiming novelty is a fatal interview risk.
**Decision:** `PRIOR_ART.md` is written first and linked in the first screenful of the README; the claim is scoped to the controlled measurement only.
**Reason:** precision about the gap prevents scope drift and establishes credibility.

---
**ADR-009 — torchtitan as training substrate**
**Status:** **Accepted** (validated for real 2026-08-14, see ADR-032) · **Date:** 2026-08-08
**Decision:** use torchtitan for model definitions, FSDP2/DDP wiring, and data loading, pinned to a SHA.
**Reason:** PyTorch-native, matches the reference L40S semi-sync setup, saves ~a day.
**Trade-offs:** API churn risk — confirmed real (ADR-032 found the model config surface changed substantially between 2026-08-10 and 2026-08-14), but the core model-construction path itself is validated working on real GPU hardware, matching ADR-026's parameter counts exactly.

---
**ADR-010 — No external experiment tracker as source of truth**
**Status:** Accepted · **Date:** 2026-08-08
**Decision:** W&B and similar may mirror, never own, results.
**Reason:** an external, account-gated dependency would break FR-11 and make third-party reproduction impossible.

---
**ADR-011 — Model sizes 130M / 500M / 1B**
**Status:** **Proposed** · **Date:** 2026-08-08
**Decision:** ~130M for convergence, ~500M for a scale check, 1B for throughput/CU.
**Reason:** 130M permits ~13 convergence runs overnight; 1B matches the reference L40S setup for comparability; the 130M→500M jump tests the direction of H-scaling over one decade.
**Trade-offs:** one decade is not five; must be stated (TD-6, R11).
**Depends on:** Day 1 throughput measurement.

---
**ADR-012 — Reference DiLoCo trainer is model-agnostic plain-PyTorch, θ_outer held as a separate parameter set**
**Status:** Accepted · **Date:** 2026-08-09
**Context:** `measurement/diloco.py` (ADR-003's independent cross-check) needed an implementation before torchtitan/torchft are pinned (§40 Q2, still PENDING) — Phase 0 is meant to maximize offline-mode work (§31.1), and the reference trainer has no actual dependency on either.
**Decision:** `DiLoCoTrainer` operates on any `nn.Module` via plain `torch.optim.AdamW` (inner) / `torch.optim.SGD(nesterov=True)` (outer). θ_outer is kept as a separate `nn.Parameter` list, never aliased to the model's live parameters, so `theta_inner <- theta_outer` at the top of each round (methods/diloco.md §1) is an explicit copy, and the outer step applies the pseudo-gradient via `.grad` assignment rather than a hand-rolled update rule (reuses PyTorch's own Nesterov-momentum implementation instead of re-deriving it). `outer_step()` calls `dist.all_reduce` only if `torch.distributed.is_initialized()`, so the same code path runs single-process (unit tests) and multi-rank (CPU integration test, and later the real 4-GPU job) without a mode flag.
**Verification:** methods/diloco.md §3 invariants 1 and 2 are now directly tested — invariant 1 (inner optimizer state persists across rounds) as a single-process unit test asserting AdamW's state dict is untouched by `outer_step()`; invariant 2 (bit-identical θ_outer across replicas) as a 2-process CPU `gloo` integration test using `torch.multiprocessing.spawn` with TCP rendezvous on a free localhost port (chosen over file-store rendezvous for cross-platform reliability) and `dist.gather_object` to compare θ_outer across ranks in the test process.
**Trade-offs:** the reference-vs-torchft equivalence test (US-06) remains skipped — it needs a pinned torchft (§40 Q2) and running it against unpinned `main` would violate the no-unpinned-installs rule (§33.2.8) while producing a meaningless pass/fail. `compress.py`'s three codecs (fp16, int8 error-feedback, top-k) were implemented alongside and unit-tested for the error-feedback residual-persistence invariant (methods/diloco.md §3 invariant 4) specifically, since CLAUDE.md §30.2 flags that as the invariant most likely to be silently broken.

---
**ADR-013 — `schemas/registry.py`: a shared, neutral schema-resolution helper for both measurement and analysis**
**Status:** Accepted · **Date:** 2026-08-09
**Context:** `measurement/spec.py` needed to validate an incoming `ExperimentSpec` against `experiment_spec.v1.json` (run lifecycle step 1, §10.1) — the same `$ref`-resolution machinery `analysis/load.py` already used to validate `RunResult`/`NetworkProfile` records. Duplicating the `referencing.Registry` construction in both places was the alternative, but §14.2's dependency diagram only showed `analysis ──► schemas`, leaving it ambiguous whether `measurement` was allowed the same edge.
**Decision:** factor the registry/validator construction into `schemas/registry.py` (`load_registry()`, `validator_for()`, `format_errors()`) and have both `analysis/load.py` and `measurement/spec.py` import it. The forbidden edges in §11.2/§14.2 are specifically `analysis <-> measurement`, not either package's edge to `schemas/` — a schema is a contract, not measurement or analysis logic. §14.2's diagram is updated accordingly.
**Reason:** avoids duplicating non-trivial `$ref`-resolution code (CLAUDE.md §32 Step 2: "search before creating"); the two-caller rule for introducing an abstraction (Architecture Principle #8, §33.2.9) is satisfied immediately, not speculatively.
**Also in this change:** `measurement/spec.py::validate_experiment_spec()` closes a gap this document's own test suite had flagged (a JSON-Schema-only test proving `H == 1 iff algorithm == "ddp"` cannot be expressed in the schema alone) by enforcing that invariant plus the other two documented `ExperimentSpec` cross-field rules (compression only with `localsgd`/`diloco`; `budget_type == "tokens"` required for the `convergence` phase) explicitly. `measurement/fingerprint.py::capture()` was also implemented: best-effort local capture (git SHA/dirty flag, installed package versions, EC2 IMDS when reachable, `nvidia-smi` when present) with an explicit `"unknown"` sentinel — never a silent `None` — for fields the schema requires as non-nullable strings but that aren't determinable off-cluster; `scrub()` (account IDs, ARNs, private IPv4s) is applied unconditionally before `capture()` returns.

---
**ADR-014 — `StepTimer` has two backends (CUDA-event, `perf_counter`), auto-selected, same API**
**Status:** Accepted (perf_counter backend) / **Proposed — unverified on real hardware** (CUDA-event backend) · **Date:** 2026-08-09
**Context:** CLAUDE.md §13.2 specifies `torch.cuda.Event` for per-step compute/sync decomposition, but no GPU exists in the dev environment to implement or test that backend against real kernels, and Phase 0 must not block on Phase 1 hardware (§31.1).
**Decision:** `measurement/telemetry.py::StepTimer` picks a `_CudaEventBackend` when `torch.cuda.is_available()` and a `_PerfCounterBackend` otherwise, behind one phase-marker API (`mark_loader_done()`/`mark_compute_done()`/`mark_sync_done()`/`mark_optimizer_done()`). Reconciliation (`methods/cu_model.md §5`) is deliberately NOT enforced by construction: an unmarked phase boundary collapses to zero duration rather than raising, so a genuine instrumentation gap (e.g. a forgotten `mark_optimizer_done()`) surfaces as real residual on `StepTiming.reconciliation_residual_pct` instead of being silently absorbed.
**Verification:** the `perf_counter` backend is fully unit-tested (exact reconciliation, unmarked-phase handling, the residual calculation itself). The CUDA-event backend is implemented but **has never run against a real GPU** — it is explicitly marked `[PROPOSED — UNVERIFIED ON REAL HARDWARE]` in its docstring, and `make smoke` (Phase 1, §30.4) is the first real test of it. Do not trust a GPU-backend number before that gate passes.
**Also in this change:** `measure_instrumentation_overhead()` implemented — runs `step_fn` with and without a `StepTimer` and returns the relative wall-time overhead (target `< 1%`, §27/R8; this function measures the number, it does not assert against the target). It refuses to trust a baseline step faster than 1ms, since Python call overhead alone dominates below that and the ratio becomes meaningless.

---
**ADR-015 — Analytic CU model form (formerly §40 Q3): Option 1, the non-overlapped blocking-sync form**
**Status:** Accepted · **Date:** 2026-08-09 · **Decided by:** project owner (explicit choice, not a Claude default)
**Context:** `analysis/cu.py::analytic()` — FR-04's headline comparison — cannot exist without picking which analytic CU model to attribute to "the literature." Three candidates were on file in `methods/cu_model.md` §2 with no decision (§40 Q3, PENDING). A previous session had written an explicit in-code guardrail into `analytic()` ("must not be implemented against a guessed form... do not fill this in until Q3 is resolved") specifically to prevent implementing this under a self-authored rationalization instead of a real decision — that guardrail held; this ADR records the actual decision once asked for.
**Decision:** `CU = H · t_compute / (H · t_compute + bytes_synced · 8 / B)` (methods/cu_model.md §2 Option 1) — compute time accumulated over `H` inner steps, divided by that plus the wall time of one non-overlapped, instantaneous-once-initiated blocking synchronization at bandwidth `B`.
**Reason:** it's the simplest form that captures the mechanism under test (H amortizes a fixed sync cost), matches the functional shape implicit in the papers surveyed in `PRIOR_ART.md` closely enough to serve as "the literature's" baseline, and — critically — is the only one of the three candidates with an actual specified formula; Option 2 (partial overlap) has never had a functional form written down anywhere in this repo, and Option 3 (per-paper reproduction) requires transcribing multiple papers' equations, both real follow-up work rather than a same-session decision.
**Trade-offs / what's NOT resolved by this ADR:** the §6 sensitivity analysis (re-running the discrepancy factor under Options 2/3 to check whether the headline conclusion survives the choice of model form) is explicitly still owed before publication and needs real Phase 3 grid data — this ADR unblocks implementation, it does not close the "apples to oranges" objection by itself. Option 2's functional form must be derived/sourced and written into `methods/cu_model.md` §2 before it can be implemented; guessing at it would violate §33.2.6.
**Also in this change:** `methods/cu_model.md` updated to `[CONFIRMED]` on the form (§2) with every assumption enumerated (§3), including the previously-`[UNKNOWN]` question of what `t_compute_s` means under rank heterogeneity (flagged as the most likely source of a future "the model's input is wrong, not the model" finding). `analysis/cu.py::analytic()` implemented and unit-tested (known-value cases, H=1 DDP-reduction check, monotonicity in `H` and in bandwidth, boundary/error handling). `analysis/cu.py::measured()` was implemented in the same session but is unrelated to Q3 — it's pure StepRecord arithmetic with no model-form dependency.

---
**ADR-016 — Synthetic `RunResult` fixture corpus, generated from factories, not hand-written**
**Status:** Accepted · **Date:** 2026-08-09
**Context:** CLAUDE.md §30.6 calls for "a fixture corpus of ~20 synthetic RunResult records covering every status, used by all analysis tests" — this didn't exist yet, so `analysis/filter.py` and `analysis/aggregate.py` had only ever been exercised against small inline dicts per-test, never together as a pipeline.
**Decision:** `tests/fixtures/factories.py` (schema-shape builder functions) + `tests/fixtures/generate_run_result_corpus.py` (a one-off, manually-rerun script, NOT executed at test time) produce 25 committed JSON files under `tests/fixtures/run_results/`, covering every `RunResult` status reachable in practice (`completed`, `crashed`, `diverged`, `aborted_shaping`, `oom` — `invalid_spec`/`aborted_preconditions` are correctly absent, since neither ever produces a `RunResult` per the §15.2 state machine), plus three `completed`-but-must-be-excluded cases (loader-bound, version-mismatched, reconciliation-failed) and the convergence/compression/fault-injection branches.
**Bug found while building this:** `experiment_spec.v1.json`'s `world_size` was `const: 4`, which would have rejected every valid single-GPU reference run FR-06 requires — caught because the reference-run fixture failed schema validation the first time the corpus was generated. Fixed to `minimum: 1`. This is exactly the value of building the fixture corpus now rather than waiting for real Phase 1 data to hit it.
**Reason for "generated, not hand-written":** 25 files × ~30 schema fields each would drift from the schema the moment either changes if maintained by hand; the factory is the single source of truth, and the JSON files are still committed static data (ADR-004) — the script is rerun and the diff re-committed deliberately, not regenerated silently in CI.
**Verification:** `tests/integration_cpu/test_aggregation_pipeline.py` — 8 tests running `load_run_results()` → `filter.apply()` → `aggregate_repeats()` against the real corpus, with exact expected exclusion counts per category (not just "some were excluded").

---
**ADR-017 — `fig1_cu_surface.py`: the headline figure, built and tested against the synthetic corpus**
**Status:** Accepted · **Date:** 2026-08-09
**Context:** `analysis/cu.py` (FR-04) and the fixture corpus (ADR-016) existed; nothing yet exercised them as an actual figure. `CLAUDE.md` §10.2 names this "CU surface (measured vs analytic) → Fig 1 (headline)" and §18 sets one non-negotiable presentation rule: measured series solid, analytic series dashed, matching colours — "this single convention carries the project's entire visual argument."
**Decision:** `analysis/figures/fig1_cu_surface.py::build(records, algorithm, harness_version=None)` groups already-filtered `RunResult` records by `H` and `bandwidth_requested_bps`, restricted to `phase == "cu_grid"` and one `algorithm` (both required, no silent default — mixing algorithms or phases on one curve would conflate different communication patterns or different experiments at the same nominal (H, bandwidth) point). `matplotlib.use("Agg")` at import time makes the module unconditionally headless, satisfying FR-11 (`make figures` runs on a reviewer's laptop, no display).
**Bug found while building this:** the fixture corpus's compression-ablation record was missing an explicit `phase` field and defaulted to `"cu_grid"` — which would have silently averaged a compression-ablation run's CU into the main grid's numbers. Caught the same way as ADR-016's `world_size` bug: writing the actual grouping/plotting code against the fixtures immediately surfaced it, rather than it lying latent until real Phase 3 data arrived. Fixed the fixture; added `phase == "cu_grid"` as an explicit filter in `_group_by_h_and_bandwidth()`, plus a regression test (`test_convergence_phase_records_do_not_leak_into_the_cu_grid_figure`) asserting the two records' `cu_measured` values (0.55 vs. the unrelated 0.85 default) are never blended.
**Verification:** `tests/integration_cpu/test_fig1_cu_surface.py` — 8 tests against the real corpus, including a direct check that every line whose label contains "measured" has `linestyle == "-"` and every other line does not (the §18 invariant, checked as code, not just as a comment).

---
**ADR-018 — `fig5_bytes_on_wire.py`: pools across bandwidth levels at fixed H (opposite of fig1's grouping)**
**Status:** Accepted · **Date:** 2026-08-09
**Context:** `wire.py::predict()`/`account()` (implemented earlier) had never been exercised end-to-end into a figure. CLAUDE.md §10.2 names this "Bytes-on-wire per token → Fig 5."
**Decision:** `analysis/figures/fig5_bytes_on_wire.py::build(records, algorithm, harness_version=None)` groups by `H` only, deliberately POOLING every bandwidth level at each `H` — the opposite of fig1_cu_surface's per-bandwidth grouping. This is correct, not an oversight: per methods/wire_model.md §2, bytes-on-wire per training token is `O(N/H)` and does not depend on bandwidth at all (bandwidth affects sync *time*, not the *byte count*), so pooling across bandwidth increases the effective repeat count for the median/IQR at each `H` rather than artificially fragmenting the same underlying quantity into separate series.
**Verification:** to make this figure's H-trend test meaningful (rather than checking a flat placeholder), the fixture corpus's `wire` fields for the DiLoCo H-sweep records were regenerated using the REAL `wire.py::predict()` formula (`tests/fixtures/generate_run_result_corpus.py::_wire_overrides_for_diloco()`), not hand-set numbers — `test_measured_values_match_wire_predict_times_known_overhead_factor` asserts the plotted values trace back to that exact function, so the two can't silently drift apart. `test_bytes_per_token_decreases_with_h` checks the `O(1/H)` trend directly. `test_measured_is_solid_predicted_is_dashed` re-checks the §18 convention, same pattern as ADR-017.
**Also in this change:** fixed a schema-adjacent bug the regenerated fixture data exposed immediately — `predicted_bytes`/`measured_bytes` are typed `integer` in `run_result.v1.json`, but the raw formula output is a float; the fixture generator now rounds before writing (this was never hit before because the old placeholder values happened to already be integers).

---
**ADR-019 — Dataset licensing verified (formerly §40 Q7): FineWeb-Edu and C4 are both ODC-BY, subject to Common Crawl's ToU**
**Status:** Resolved (research finding, not a design choice — nothing here was a judgment call) · **Date:** 2026-08-10
**Context:** §40 Q7 required verifying FineWeb-Edu's and C4's license terms before publication, since the repository is public.
**Finding:** both datasets are released under the **Open Data Commons Attribution License (ODC-BY) v1.0**, and both are explicitly subject to **Common Crawl's Terms of Use** (both are Common-Crawl-derived). One clause is worth recording rather than discovering later: Common Crawl's ToU §9 requires users to indemnify Common Crawl for claims arising from "use of Crawled Content in connection with artificial intelligence, machine learning, or other similar technologies... including... large language models" — exactly this project's use. This is a standard indemnification clause, not a usage prohibition; using Common-Crawl-derived corpora for LLM pretraining is standard, widespread, published practice (both datasets are built and published for exactly this purpose). Not legal advice; this project carries no commercial stakes; recorded because a careful reviewer may ask.
**Decision:** per the original §40 Q7 recommendation (now acted on, not merely proposed): this repository redistributes neither raw nor tokenized dataset content — only tokenized-shard checksums and download/tokenization scripts, so a reproducer fetches the corpus directly from its original source under the same license terms.
**Also in this change:** `LIMITATIONS.md` "Dataset licensing" section written with full citations; §24's Privacy and Data Handling "Licensing" row updated from `[UNKNOWN]` to `[CONFIRMED]`.

Sources: [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [C4 dataset card](https://huggingface.co/datasets/allenai/c4), [Common Crawl Terms of Use](https://commoncrawl.org/terms-of-use).

---
**ADR-020 — Bare metal + `uv` lockfile on the GPU nodes, no Docker (formerly §40 Q4)**
**Status:** Accepted · **Date:** 2026-08-10 · **Decided by:** project owner, explicit choice (recommended option accepted as-is)
**Context:** Docker reduces cross-node version drift (R15) but adds container networking (`--network=host` or equivalent) as one more layer that can break NCCL rendezvous — and R1 (multi-node NCCL failing to establish) is already the highest-impact technical risk in the project (§38.1).
**Decision:** bare metal + a `uv` lockfile on every node. No Docker, no NGC container, no custom AMI for Day 1.
**Reason:** fewest moving parts while debugging NCCL for the first time on real hardware — when something breaks on Day 1, "is it Docker's networking or is it NCCL" is a diagnosis question this choice removes entirely. Revisit only if actual environment drift is observed (the risk Docker would have mitigated) — don't pre-solve a problem that may not occur.
**Trade-offs:** environment drift between the 4 GPU nodes becomes possible in principle; mitigated by the lockfile plus the per-run environment fingerprint (FR-08, `measurement/fingerprint.py`) — a drift, if it happens, is at least visible after the fact, not silent.

---
**ADR-021 — TTTL target loss `L*` = the single-GPU reference run's final loss at the same token budget (formerly §40 Q5)**
**Status:** Accepted · **Date:** 2026-08-10 · **Decided by:** project owner, explicit choice
**Context:** FR-06 already described this exact definition when it was written — Q5 formally ratifies what was implicitly assumed throughout §6/§8/§43, rather than introducing something new.
**Decision:** `L*` is the single-GPU reference run's final loss, at the same token budget as the configuration under test (not a fixed absolute number, not a percentile of the grid's best run).
**Reason:** most defensible choice available — it's tied to an actual measured baseline rather than an arbitrary threshold — and it's what makes "did not reach target" (`tttl_s: null`) a meaningful, non-arbitrary category rather than a threshold nobody agreed on.
**No other sections required updating:** FR-06, US-03, and the Glossary (§43 `TTTL`) already stated this definition; this ADR closes the formal PENDING status, it doesn't change any spec text.

---
**ADR-022 — Spot instances for Phase A only; on-demand everywhere else (formerly §40 Q9)**
**Status:** **Superseded by ADR-025** (same day — real account data invalidated the premise before any run used this policy) · **Date:** 2026-08-10 · **Decided by:** project owner, explicit choice
**Decision:** Phase A (CU grid) repeats run on spot. Phase B (convergence), the Day 7 headline re-runs, and the control node are always on-demand.
**Reason:** Phase A points are short (`[PROPOSED]` ≤5 min/point, NFR-06) and re-runnable — a spot reclaim costs a re-run, not a corrupted result. A reclaim mid-convergence-run or mid-headline-re-run is pure, uncheckpointed loss (R14) in a context where timing stability is the entire point of the run.
**Also in this change:** §29.4 updated from `[RECOMMENDATION]` to `[CONFIRMED]` — no other section needed changes, since §29.4 already stated exactly this policy; §40 Q9 was just the last formally-open reference to it.
**Why superseded:** this decision was made from the *general* spot-vs-on-demand tradeoff reasoning, before the account's actual spot quota and current spot pricing had been checked. See ADR-025 — both turned out to undercut the reasoning above.

---
**ADR-023 — Per-step Parquet: full for a representative subset, aggregates for every run (formerly §40 Q10)**
**Status:** Accepted (policy) / `[PROPOSED]` (exact subset-selection rule) · **Date:** 2026-08-10 · **Decided by:** project owner, explicit choice
**Context:** Publishing every run's full per-step Parquet risks the `[PROPOSED]` ≤500MB repo-size target once hundreds of runs accumulate (NFR); publishing aggregates only would mean a reviewer can never inspect the raw step-by-step timing behind any specific figure.
**Decision:** every run gets its aggregated `RunResult` JSON (as already required, NFR/§16.1); a *representative subset* additionally gets its full per-step Parquet published in full.
**Not resolved by this ADR:** exactly which runs count as "representative" — candidates on the table are one repeat per grid point, or specifically the runs any published figure's caption cites — is deferred to when real Phase A data volume is known (§16.4), same as the downsampling threshold below it. This ADR settles the *shape* of the policy (subset-in-full + aggregates-for-all), not its exact parameters.
**Also in this change:** §16.4 updated with the confirmed policy and the still-open selection-rule note.

---
**ADR-024 — Region confirmed: `us-east-1`, no viable fallback (formerly §40 Q1)**
**Status:** Accepted (region + quota fact) · AZ choice `[PROPOSED]`, to be re-confirmed at actual launch · **Date:** 2026-08-10
**Context:** Q1 asked which region/AZ has `g6e` capacity, with `us-east-2` and `us-west-2` listed as candidate fallbacks alongside `us-east-1`. AWS access was connected this session (a fresh IAM user, `dilico`, needed a scoped policy attached before any read call worked at all — see the account's IAM console for the attached inline policy) and the account's real quotas and instance-type offerings were checked directly, not assumed.
**Finding:** the account's **On-Demand "G and VT" vCPU quota is 32 in `us-east-1` and 0 in both `us-east-2` and `us-west-2`.** This isn't "capacity varies by region" — those two regions cannot run a single `g6e` instance at all under this account today. `us-east-1` is not a preference, it's the only option, unless a quota increase is separately requested and granted in another region. `g6e.2xlarge` is offered (AWS will accept the instance type) in all 4 of `us-east-1`'s AZs (a/b/c/d); `c7i.2xlarge` (control node) is offered in 5 (a/b/c/d/f) — so any of a/b/c/d works for both.
**Decision:** region = `us-east-1`, confirmed. AZ: primary `us-east-1a`, backup `us-east-1b` (arbitrary choice among four equally-offered options — no signal favored one over another). **"Offered" is not "has physical capacity right now"** — that is genuinely only knowable by attempting a real launch, which is real spend and stays a Phase 1 Day 1 action (§35), not something this read-only check can promise. If `us-east-1a` fails with `InsufficientInstanceCapacity` at launch time, fall back to `us-east-1b`, then `c`, then `d`, in that order.
**Also in this change:** this is the first commit in the project's history where AWS was actually reachable. The IAM user started with zero permissions attached (couldn't even call `sts:GetCallerIdentity` successfully at first, then couldn't call any `ec2:Describe*` after that was fixed) — a scoped inline policy (EC2 lifecycle, S3 under a `diloco-measured-*` prefix, `servicequotas`/`pricing` read) was attached before any of this was possible, matching §23's least-privilege principle rather than a wildcard admin grant.

---
**ADR-025 — Phase A also on-demand, not spot (supersedes ADR-022)**
**Status:** Accepted · **Date:** 2026-08-10 · **Decided by:** project owner, explicit choice, after being shown the real numbers
**Context:** ADR-022 (decided minutes earlier, same session) put Phase A on spot based on the general tradeoff (short, re-runnable points; cheaper). Once AWS access existed, the actual account data was checked before building anything on top of that assumption.
**Finding:** the account's **Spot "G and VT" vCPU quota is 8** — enough for one `g6e.2xlarge` node (8 vCPU), not the 4 nodes (32 vCPU) Phase A runs simultaneously. Separately, **live spot pricing for `g6e.2xlarge` in `us-east-1` is ~$2.15–2.24/hr**, essentially identical to the ~$2.24/hr on-demand price already documented in §5.2 — negligible discount at current market conditions.
**Decision:** Phase A runs on-demand, same as Phase B and the Day 7 headline re-runs. §29.4 no longer has a spot tier at all.
**Reason:** ADR-022's premise (spot = cheaper, and short/re-runnable points absorb reclaim risk well) is sound in general, but doesn't survive contact with this account's actual numbers: under-quota'd for the fleet size needed, and not meaningfully cheaper today. Chasing a spot-quota increase for a ~0-4% discount wasn't judged worth the delay.
**Trade-offs:** loses the theoretical spot-reclaim-is-cheap-to-absorb benefit entirely; gains operational simplicity (one instance-market policy for the whole project, no interruption handling to build anywhere) and removes a dependency on an AWS quota-increase request (not always fast or guaranteed) from the critical path.
**Revisit if:** spot pricing later shows a real discount (re-check before Phase A actually runs, since pricing is a live market) AND the spot quota is separately increased to ≥32.

---
**ADR-026 — Real model dims for 130m/500m/1b/debugmodel, computed from torchtitan's actual formulas**
**Status:** Accepted (dims) · Feeds into ADR-009 (still `[PROPOSED]` overall — no GPU test has run) · **Date:** 2026-08-10
**Context:** `configs/models/{130m,500m,1b}.toml` had `[UNKNOWN]` `dim`/`n_layers`/`n_heads`/`vocab_size` since the scaffold session — placeholders nobody had checked against torchtitan's real `TransformerModelArgs`/`FeedForward` code.
**What was done:** installed `torchtitan==0.2.2` for real (pure-Python wheel; hit a `triton` import wall partway through its package `__init__` chain — `triton` has no Windows wheel at all, confirmed by trying — so parameter counts were computed with the exact `dim`/`n_layers`/`n_heads`/`n_kv_heads`/`hidden_dim`/`multiple_of` formulas read directly from `torchtitan/models/llama3/model/{args,model}.py`, in pure Python, rather than instantiating the real model class). Iterated candidate `dim`/`n_layers`/`n_heads` combinations to real parameter counts: 130m -> 134,105,856 (~134.1M); 500m -> 530,248,960 (~530.2M, required switching to GQA with `n_kv_heads=8` — the original flat-MHA placeholder undercounted badly); 1b -> 973,559,552 (~973.6M).
**Also found:** `torchtitan` ships its own tiny **`debugmodel`** preset (`dim=256, n_layers=6, n_heads=16, vocab_size=2048`, ~6.16M params) for exactly the E2E smoke test's use case (CLAUDE.md §30.4) — added as `configs/models/debugmodel.toml`, using torchtitan's own values verbatim rather than hand-rolling a different tiny config nobody but this project has ever run.
**Not resolved by this ADR:** `vocab_size=32000` for the three real-size configs is still `[PROPOSED]`, pending the actual tokenizer choice (Day 0); a `vocab_size` change means re-running the same parameter-count computation, not re-guessing dims from scratch. No GPU has run any of these configs — that's `make smoke` (Phase 1), not this.

---
**ADR-027 — `infra/launch_cluster.sh`, `teardown.sh`, `cost_report.sh` implemented and verified against the live AWS account; `setup_node.sh` implemented but untestable here**
**Status:** Accepted (3 of 4 scripts, dry-run-verified for real) · `setup_node.sh` `[PROPOSED — UNTESTED]` · **Date:** 2026-08-10
**Context:** all four `infra/*.sh` scripts were scaffold stubs that printed a plan and refused to run. AWS access existed this session (ADR-024/025) — real implementations were written and, for the three that only need AWS API calls, actually exercised against the account for real.
**AMI selection:** looked up live (not assumed) — Ubuntu 24.04 Deep Learning AMIs (GPU nodes: `ami-0e2e1c9b9d71cc77f`, PyTorch 2.12, 2026-07-25 build; control node: plain Ubuntu 24.04 `ami-052355af2a014bd2c`, no GPU/CUDA bloat needed for a CPU-only orchestrator) — Ubuntu chosen over the also-available, fresher Amazon Linux 2023 DLAMIs specifically so `setup_node.sh` can assume `apt`, not `dnf`. AMI IDs get retired over time; both scripts note the exact `describe-images` query to re-run before trusting them past a couple of weeks old.
**Two real bugs found and fixed by actually running the dry-run path against the account**, not merely by review:
  1. The shared `log()` helper wrote to stdout. Every function whose return value is captured via `$(...)` (`ensure_security_group`, etc.) was silently swallowing its own diagnostic output into that captured value instead of printing it — messages never appeared, and in the security-group case corrupted the captured group ID. Fixed: `log()` now writes to stderr.
  2. `aws ... --dry-run | grep -q DryRunOperation` looked reasonable but is wrong under `set -o pipefail`: the AWS CLI exits non-zero on a *successful* dry run (that's how it signals "would have succeeded"), and `pipefail` propagates that non-zero exit through the pipeline regardless of whether `grep` matched — every single dry-run permission check would have reported PERMISSION CHECK FAILED even when permissions were fine. Fixed with a shared `check_dry_run()` helper that captures output first (discarding the deliberately-nonzero exit code) and greps the captured string outside any pipe `pipefail` can see.
**Verification:** `launch_cluster.sh`'s full dry-run path (security group, key pair, placement group, both `run-instances` calls) ran clean end-to-end against the real account after both fixes — confirming the IAM policy from ADR-024 is sufficient for the actual launch sequence, not just the read-only checks used to research Q1. `teardown.sh` was run in both dry-run and `--terminate-for-real` mode (a safe no-op — nothing was running). `cost_report.sh`'s no-instances path was run for real; its per-instance arithmetic was checked against a synthetic 2-hours-ago row (confirmed `$4.48` for `g6e.2xlarge` at the documented $2.24/hr rate). **Nothing was created. Nothing was billed.**
**`setup_node.sh` — explicitly NOT verified**: it must run on a real Ubuntu EC2 node with an NVIDIA GPU, which doesn't exist yet (no cluster has been launched). Written against the documented plan and syntax-checked (`bash -n`) only. Its own header says so; treat every step as a first draft to debug live on Day 1 (`make smoke`), not a validated procedure — most likely failure point: the guessed instance-store NVMe device path (`/dev/nvme1n1`), which is not guaranteed stable and is called out explicitly as unverified in the script itself.
**Makefile updated to match:** `cluster-up`/`cluster-down` now default to dry-run (matching the scripts) and require `LAUNCH=1`/`TEARDOWN=1` respectively for the real action — `make cluster-up` alone can never accidentally spend money.

---
**ADR-028 — `measurement/netshape.py`, `probe.py`, `train.py` implemented for real; `run()` stops at a documented, not-guessed boundary**
**Status:** Accepted (netshape.py, probe.py, the sequencing logic in train.py) · `train.py`'s step 6 (torchrun launch) explicitly **not** built · **Date:** 2026-08-10
**Context:** these three modules were the last pieces of the FR-01/FR-02/FR-03 measurement path still stubbed with `NotImplementedError`. `wire.py::snapshot()` was the only planned dependency of `netshape.py`/`probe.py` still missing an SSH mechanism — rather than block all three on it, `netshape.py` grew its own `ssh_run()` (public, since `probe.py` needs it too) and `wire.py::snapshot()` remains separately unimplemented (it accounts bytes-on-wire, a different concern from shaping or bandwidth probing).
**What was done:**
  - `netshape.py`: real `ssh_run()` (subprocess + `shlex.join`, never string concatenation — CLAUDE.md §33), `apply()`/`verify()`/`restore()` driving `tc qdisc` (`tbf`) over SSH and asserting the achieved rate with `iperf3`, exactly matching the existing pure functions (`compute_error_pct`, `build_tbf_add_args`/`build_tbf_del_args`) that were already implemented and tested. `verify()`'s contract (CONTRACT comment at §17.2, line ~1487) — "NEVER returns a passing result it did not measure" — holds: there is no code path that fabricates a `ShapingVerification`.
  - `probe.py`: `log_spaced_message_sizes()` (pure), `sweep_all_reduce_bandwidth()` (drives `torch.distributed.all_reduce` over an already-initialized process group — backend-agnostic by construction, so it's testable on CPU/gloo even with no GPU in this dev environment; real NCCL numbers are Phase 1), `burst_decay_probe()` (SSH-driven `iperf3` over a sustained window).
  - `train.py`: `run()` now performs, for real, every step of CLAUDE.md §10.1 that CAN be done without a training loop: schema + cross-field spec validation, `check_preconditions()`, shaping `apply`+`verify` with exactly one retry (FR-02), environment fingerprinting (`fingerprint.py`, already real), and unconditional `restore()` in a `finally` on every exit path. It then raises `NotImplementedError` naming the exact blocker (§40 Q2, ADR-009) rather than continuing into step 6 (torchrun launch across real ranks running torchtitan/torchft).
**Why `run()` stops there, deliberately:** `torchft` is pinned to a research candidate (`torchft-nightly`, not a confirmed dated SHA — §40 Q2 is still PENDING by design, the pin is a Day-0 action) and ADR-009 (torchtitan as substrate) is still `[PROPOSED]`, explicitly "must be validated on Day 0." Writing training-loop wiring against either API surface now would be exactly the invented-and-presented-as-real code CLAUDE.md §33.2.6 forbids — there is nothing to validate it against without a real GPU node. Stopping at a named, tested boundary (rather than a silent stub) means every *other* correctness guarantee in the run lifecycle — spec validity, shaping verified, network restored — already holds today, and only the training loop itself remains.
**Testing:** `netshape.py`'s pure helpers were already covered (`test_netshape_pure.py`, unchanged). New: `test_probe_pure.py` (7 tests, `log_spaced_message_sizes`), `tests/integration_cpu/test_probe.py` (2 tests, real gloo process group, 2 ranks), `tests/unit/test_train_orchestration.py` (12 tests) — the latter uses fake `apply_fn`/`verify_fn`/`restore_fn` injected through `run()`'s own parameters to check *sequencing* (retry-exactly-once, precondition-gates-before-shaping, restore-unconditional-including-on-the-NotImplementedError-path) without mocking `netshape.py` itself (forbidden, §30.6) — these tests say nothing about whether a real `iperf3` measurement is trustworthy, only that `train.py` calls its dependencies correctly.
**Not resolved by this ADR:** dataset tokenization pipeline (does not exist yet), a cluster-inventory mechanism to supply real `Node` objects from the CLI (does not exist yet — `cli.py`'s `network characterize`/`run`/`grid`/`converge` still raise `NotImplementedError` for this reason), and the actual torchft/torchtitan validation on real Linux/GPU hardware (blocked on cluster time).

---
**ADR-029 — `cli.py analyze`/`figures` wired to the real, already-complete `analysis/` pipeline**
**Status:** Accepted · **Date:** 2026-08-10
**Context:** `analysis/` (`load.py`, `filter.py`, `aggregate.py`, `cu.py`, `figures/fig1_cu_surface.py`, `figures/fig5_bytes_on_wire.py`) has been fully implemented and fixture-tested since earlier this session, but `cli.py` still dispatched every command — including these two, which need no cluster, no GPU, and no credential — to `NotImplementedError`.
**Decision:** wire `analyze` and `figures` for real now, while leaving every cluster-dependent command (`network characterize`, `run`, `grid`, `converge`, `plan`) raising `NotImplementedError`, because those genuinely need infrastructure this repo doesn't have yet (§40 Q2's pin, a cluster-inventory mechanism, dataset tokenization) — half-wiring them now would mean guessing at that infrastructure's shape.
**What was done:** new `analysis/report.py::generate_all_figures()` — orchestrates load → filter → per-algorithm `fig1_cu_surface`/`fig5_bytes_on_wire` generation, saving PNGs to `results/figures/`. It lives in `analysis/`, not `cli.py`: deciding *which algorithms have data* is itself decision logic CLAUDE.md §19.1 reserves for the module being dispatched to, not the dispatcher. `cli.py analyze` now loads, filters, and prints FR-13's exclusion counts (crashed/diverged/other-status/version-mismatch/loader-bound/reconciliation-failed) plus the kept run list; `cli.py figures` calls `generate_all_figures()` and echoes each written path.
**Verification:** both commands were run directly (not just via pytest) against `tests/fixtures/run_results/` — `analyze` printed correct counts (25 loaded, 18 kept, 7 excluded across all 5 categories represented in the fixture corpus), `figures` wrote 8 real PNGs (4 algorithms × 2 figure modules) to a scratch directory. 5 new `integration_cpu` tests (`test_report.py`) cover per-algorithm output, files-on-disk, output-dir creation, the empty-corpus no-op, and the version-mismatch exclusion path.
**Not resolved by this ADR:** the module's own docstring previously claimed "every command raises `NotImplementedError`" — corrected to state the real split honestly.

---
**ADR-030 — `infra/prepare_dataset.py` implemented and tested; tokenizer pin deliberately left open**
**Status:** Accepted (pipeline) · Tokenizer pin still `[PROPOSED]`, same as `vocab_size=32000` (ADR-026) · **Date:** 2026-08-10
**Context:** `methods`/§30.6 always assumed "a small tokenized shard committed for CPU integration tests" would exist, and the Day-0 plan (§31.2) names "pre-tokenize FineWeb-Edu on the control node → S3" as the single most expensive scheduling mistake to defer — but no code existed for either.
**Decision:** build the tokenize/pack/checksum/manifest pipeline for real now; leave the actual tokenizer choice for Day 0, the same way ADR-009/§40 Q2 leaves the torchtitan/torchft pin for Day 0. `--tokenizer` accepts any HuggingFace `AutoTokenizer` name/path — nothing in the packing, checksumming, or manifest logic depends on which one is eventually chosen, so this doesn't block on that decision and doesn't guess at it either.
**What was done:**
  - `infra/prepare_dataset.py`: `pack_tokens()` (pure — flat token stream → fixed-`seq_len` blocks, final incomplete block dropped, not padded, the nanoGPT/torchtitan-style convention), `dtype_for_vocab_size()` (pure — `uint16` vs `uint32`, derived from the tokenizer's actual vocab size, never assumed), `sha256_file()`, `write_shard_npy()`, `build_manifest()`. The one network-touching function, `iter_dataset_documents()` (HuggingFace `datasets` streaming — required, since FineWeb-Edu/C4 are far too large to download in whole), is injected into the orchestrating `prepare_shards()` via a `documents_per_shard: list[DocumentFactory]` parameter — same dependency-injection pattern as `train.py`'s `apply_fn`/`verify_fn`/`restore_fn` (ADR-028) — so the orchestration logic is unit-testable with in-memory strings and zero network access. S3 upload shells out to the `aws` CLI (`aws s3 sync`), matching every other `infra/` script rather than adding a `boto3` dependency.
  - New `dataprep` optional dependency group (`transformers`, `datasets`) — deliberately kept OUT of the base `dependencies` list, since `measurement/` and `analysis/` never import this and `analysis/` must stay GPU-free/network-free/minimal (CLAUDE.md §11.2).
  - A real, small, committed test fixture: `tests/fixtures/dataset/fixture_shard_0000.npy` + `manifest.json`, produced by this actual pipeline using the real, small, public, ungated `gpt2` tokenizer against four sentences written for this repository (not sampled from any real corpus — see `tests/fixtures/dataset/GENERATED.md` for the explicit disambiguation, since the manifest's `license_note` field is the pipeline's standard FineWeb-Edu/C4 boilerplate and does not apply to this fixture).
**Testing:** `tests/unit/test_prepare_dataset.py` (16 tests — packing, dtype selection, checksumming, shard round-trip, manifest assembly, all offline). `tests/integration_cpu/test_prepare_dataset_fixture.py` (4 tests — loads the committed fixture, zero network, zero `dataprep` dependency, so the base test suite never needs network to pass). `tests/integration_cpu/test_prepare_dataset_tokenizer.py` (3 tests — real end-to-end `prepare_shards()` run against a live-fetched `gpt2` tokenizer; skips cleanly via `pytest.importorskip`/a network try-except if `transformers` isn't installed or HF Hub is unreachable, rather than failing the suite in an offline environment). 194 passed, 4 skipped total; ruff and mypy clean including `infra/`.
**Not resolved by this ADR:** which tokenizer/vocab_size actually gets used to train (Day 0, per above); the real FineWeb-Edu/C4 streaming path (`iter_dataset_documents`, the CLI's `main()`) has never been run against the real corpus — only against the tiny in-memory fixture documents; no dataset has been uploaded to S3 by this session.

---
**ADR-031 — R6 materialized for real: `g6e`/`g6` capacity exhausted across every `us-east-1` AZ; `launch_cluster.sh` hardened in response**
**Status:** Accepted (script changes) · Underlying capacity shortage: **unresolved, outside this repo's control** · **Date:** 2026-08-11
**Context:** attempted the real 4-node cluster launch this session. Every attempt's control node (`c7i.2xlarge`) and every dry-run permission check succeeded without exception — the failure was isolated cleanly to GPU instance capacity, every single time, ruling out a quota, IAM, AMI, subnet, or security-group problem (all of which were already separately verified in ADR-024/027).
**What was tried, for real, in order:**
  1. `g6e.2xlarge` (L40S, the primary topology), `--count 4` in one batch call: `InsufficientInstanceCapacity` in `us-east-1a`.
  2. Same, retried in `us-east-1b`, `us-east-1c`, `us-east-1d` (ADR-024's documented fallback order): failed identically in all three.
  3. Rewrote `launch_gpu_nodes()` (see below) to issue 4 independent `--count 1` requests instead of one `--count 4` batch, on the theory that a batch needs 4 simultaneous free slots while 4 single requests each only need 1: retried across all 4 AZs again — **failed identically, including single-instance requests**, meaning this was never a batch-sizing artifact; capacity was genuinely at zero.
  4. `g6e.4xlarge` (still L40S, CLAUDE.md R1's documented 2-node/32-vCPU fallback), 2 nodes, across `us-east-1a`/`us-east-1b`: failed identically — same GPU family, same shortage.
  5. `g6.2xlarge` (L4, CLAUDE.md R6's documented fallback — a genuinely different GPU generation), 4 nodes, across `us-east-1a`/`b`/`c`/`d`, plus `us-east-1f` (a 5th AZ this instance type is offered in, discovered from AWS's own error-message text mid-session): failed identically in all five.
  **Total: ~20 real launch attempts, 3 instance types, 2 GPU generations, every AZ each type is offered in.** Every attempt was cleaned up immediately after (control node terminated, placement group deleted) and the account was verified clean (zero running instances, zero placement groups) after every single attempt and at the end of the session. Total spend: a few cents of per-second billing on briefly-live control nodes.
**Script changes (durable, kept regardless of today's outcome):**
  - `launch_gpu_nodes()` now issues `$remaining` (existing-aware, so idempotent re-runs after a partial failure only request the shortfall) separate single-instance `run-instances` calls instead of one `--count N` batch, each with its own `run_with_retry` backoff.
  - New `DILOCO_GPU_INSTANCE_TYPE` env var (parallel to the existing `DILOCO_GPU_AMI`/`DILOCO_AZ` overrides) so the R1 (`g6e.4xlarge`) and R6 (`g6.2xlarge`) fallbacks are a one-line invocation, not a script edit.
**Reason this is recorded as an ADR, not just a session note:** CLAUDE.md §33.1.11 requires reporting negative results rather than quietly dropping them, and this is a real, load-bearing finding — R6 was a `[PROPOSED]`/`Medium` risk in the original plan; it is now a confirmed, `Critical`, currently-blocking one, and future sessions (or the operator, days from now) need this exact record of what was already tried before spending more time re-discovering it.
**Not resolved by this ADR:** the capacity shortage itself. Remaining options are explicitly outside what a script change can fix: wait for AWS to free up capacity (no timeline given), or request GPU quota in a different region (untested — ADR-024 only confirmed quota in `us-east-1`; `us-east-2`/`us-west-2` are at 0).

---
**ADR-032 — §40 Q2 resolved for real: torchft-nightly + torchtitan validated on live GPU hardware; cluster launch succeeded**
**Status:** Accepted · **Date:** 2026-08-14
**Context:** `us-east-1b` finally succeeded (via `retry_across_azs.sh`, cycle 7 of the automated retry loop from ADR-031) — a real 4x `g6e.2xlarge` + 1x `c7i.2xlarge` cluster came up for the first time. `setup_node.sh` ran for real on all 5 nodes and passed cleanly on the first try (real PyTorch 2.13+cu130, `torch.cuda.is_available()==True`, `NVIDIA L40S` confirmed on all 4 GPU nodes, GPU clocks locked to 2520MHz, NVMe pre-mount correctly recognized). This ADR covers what happened next: actually resolving §40 Q2, which had been "de-risked" by research (2026-08-10) but never installed or run anywhere, since neither package has a Windows wheel.
**A stale SSH rule almost blocked this:** the security group's SSH rule was still scoped to the operator IP set days earlier in the session; the operator's real IP had changed since. All 5 nodes were `running` in AWS but every SSH attempt timed out (not refused) until this was found and fixed — a reminder that "instance state == running" is not the same as "node is usable."
**What was found and fixed, in order:**
  1. **Pin freshness re-verified, not trusted from Aug 10.** `torchft-nightly` latest was `2026.8.14` (published *that morning*) vs. the documented `2026.8.10` candidate. `torchtitan`'s latest `main` was `624c312b67` (also that morning).
  2. **A real bug in the previously-documented candidate pin:** `torchtitan @ git+https://github.com/meta-pytorch/torchtitan@...` 404s — the meta-pytorch org rename (confirmed for `torchft` in §40 Q2's original research) does **not** apply to `torchtitan`, which is still `pytorch/torchtitan`. Nobody had actually tried the URL until now. Fixed in `pyproject.toml`.
  3. **`torchft-nightly==2026.8.14` installs with no Rust/protobuf toolchain** (confirms the prebuilt-manylinux-wheel reasoning from §40 Q2's research) and **its own DiLoCo/LocalSGD test suite (`torchft.local_sgd_test`) passes 8/8** on this real hardware, after installing one missing test-only dependency (`parameterized`) — a real, substantive validation of the library itself, not just an import check.
  4. **`torchtitan @ git+https://github.com/pytorch/torchtitan@624c312b67` installs and imports cleanly** on real Linux (no `triton` wheel wall — that was Windows-only, per ADR-026). Its config API has changed substantially since ADR-026's Aug 10 research: the flat `TransformerModelArgs`/`FeedForward` formula-reading approach is gone, replaced by a `Trainer.Config`/`model_registry()` composition system (`torchtitan.models.llama3.model_registry("debugmodel")` returns a `ModelSpec`). **The debugmodel dimensions themselves still match exactly** — `model_registry("debugmodel").model` reports `dim=256, vocab_size=2048`, and the constructed model has **6,163,712 parameters**, identical to ADR-026's hand-computed ~6.16M.
  5. **A real forward+backward pass on the L40S**, first attempt, hit `AssertionError: attention_masks must be instance of BlockMask, got NoneType` — not a bug, a missing required argument in the ad-hoc test call (the real `Trainer` builds this automatically from its dataloader). Fixed by calling the model's own `get_attention_masks(positions)` helper.
  6. **Second attempt hit a genuine environment gap**: `Python.h: No such file or directory` — FlexAttention's `torch.compile`/Triton JIT path shells out to `gcc` to build a small CUDA-launcher C extension, and the Deep Learning AMI does not ship Python dev headers. Fixed with `apt-get install python3-dev gcc`; added as a new `install_system_packages()` step in `setup_node.sh` (runs before `install_deps`, GPU-role nodes only) so every future node gets this without rediscovering the failure.
  7. **Third attempt succeeded completely**: real Triton kernels autotuned and compiled (13 backward-pass kernel candidates benchmarked, best selected), output shape `[2, 32, 2048]` (correct for batch=2, seq_len=32, vocab_size=2048), `loss.backward()` completed with a real loss value. This is a full, real, end-to-end validation — not an import check, not a simulation.
**Also in this change:** `pyproject.toml`'s commented-out candidate pins replaced with active, verified ones (`torchtitan @ git+https://github.com/pytorch/torchtitan@624c312b67`, `torchft-nightly==2026.8.14`) and its STATUS comment updated from `[PROPOSED]` to `[CONFIRMED]`. ADR-009 (torchtitan as substrate) moved from `Proposed` to `Accepted`.
**Not resolved by this ADR:** this validation was run on one GPU node only, single-process, no distributed/multi-node NCCL collective yet (that's FR-01 network characterization, next). `torchft`'s `DiLoCo`/`Manager` class was inspected (needs a `Manager` wrapping the process group — more involved than plain `gloo`) but the actual reference-vs-torchft cross-implementation equivalence test (US-06, ADR-003/D3) was NOT run this session — real validation of the library itself substituted for it today given time constraints, but the equivalence test remains the real gate before trusting `torchft`'s DiLoCo path over the reference implementation for any real number. Because `torchtitan`'s config surface changed shape since ADR-026, `configs/models/*.toml` (the flat dim/n_layers/n_heads files) do not yet wire directly into a real `Trainer.Config` — that adapter layer does not exist yet and is real remaining work before a `Trainer`-orchestrated run, though the model-construction primitives underneath are now confirmed correct.

---
**ADR-033 — Real primary network interface is `enp39s0`, not `ens5`; `netshape.py`'s default fixed**
**Status:** Accepted · **Date:** 2026-08-14
**Context:** first real multi-node NCCL all-reduce test (4 GPU nodes, `torchrun`, `--rdzv-backend=c10d`) failed with `torch.distributed.DistBackendError: NCCL error ... invalid usage`. Root-caused (not guessed) by checking `ip -brief addr show` on the actual node: the real primary interface is `enp39s0` — `ens5`, the interface name `netshape.py`'s `Node.iface` had defaulted to (per CLAUDE.md §11.1's architecture diagram, explicitly flagged `[PROPOSED]`/unverified in the code's own docstring since it was written), **does not exist on this AMI at all**. `NCCL_SOCKET_IFNAME=ens5` was silently pointing NCCL at a nonexistent device. Verified consistent across all 4 GPU nodes (`enp39s0` on every one), not a fluke of a single instance.
**Why this matters beyond the NCCL test:** every `tc` shaping command in `netshape.py` (`build_tbf_add_args`/`build_tbf_del_args`, the actual FR-02 bandwidth-shaping mechanism this entire project is built around) also targets `node.iface`. Had this shipped unverified into a real experiment, the likely failure mode is not a clean crash — `tc qdisc add dev ens5 root tbf ...` against a nonexistent device fails at the `tc` layer, which `apply()`/`verify()` should catch (the shaping verification gate, FR-02's central integrity mechanism) and abort on — but this was never actually exercised end-to-end against a real device name before now, so the confidence that the gate would have caught it, rather than something more subtle going wrong first, was itself unverified. Caught before a single real shaped run happened.
**Decision:** `Node.iface` default corrected from `"ens5"` to `"enp39s0"`, docstring updated from `[PROPOSED]` to `[CONFIRMED]` with the real finding recorded in place of the old "must confirm on Day 1" note.
**Verification:** local unit tests unaffected (`test_netshape_pure.py`/`test_wire_accounting.py` pass `"ens5"` as an explicit literal argument to pure functions, never relying on the `Node` dataclass default) — 194 passed, 4 skipped, unchanged. The real fix was verified by re-running the NCCL sweep with the corrected interface name (see the network-characterization results this same session).
**Not resolved by this ADR:** the value is still a hardcoded default, not auto-detected — if the pinned AMI or instance family ever changes, this needs re-verification the same way (`ip -brief addr show` on a live node), not assumed. Auto-detecting the primary interface at runtime (e.g. via the default route) would remove this whole class of risk permanently and is a reasonable follow-up, not done here.

---
**ADR-034 — First real training measurement: DiLoCo H-sweep on real GPU hardware, headline hypothesis holds**
**Status:** Accepted (as a first slice, explicitly not the full grid — see "Not resolved") · **Date:** 2026-08-14
**Context:** with ADR-032 (torchft/torchtitan validated), ADR-033 (real interface fixed), and FR-01 network characterization (`results/network/phase1-us-east-1b-20260814.json`) all done, the remaining blocker to a real FR-04 measurement was `measurement/train.py::run()`'s own documented stopping point (ADR-028): it deliberately never built a `Trainer.Config` adapter for torchtitan's `model_registry()` composition system, because that adapter didn't exist and building it against unvalidated APIs would have been exactly the "invented and presented as real" code CLAUDE.md §33.2.6 forbids. That adapter still does not exist. Rather than block the project's headline measurement on it, this session took a narrower, honest path: use the SAME validated primitives (`DiLoCoTrainer`, `StepTimer`, the real `debugmodel` architecture) directly, in a hand-written `torchrun` entrypoint, and be explicit that this bypasses `train.py`'s full orchestration rather than pretending it doesn't.
**What was done:**
  1. **Real dataset**: FineWeb-Edu (`sample-10BT`) streamed and gpt2-tokenized on the control node (`infra/prepare_dataset.py`, ADR-030's real pipeline — not a new one), 4 disjoint shards (one per rank, `.skip(shard_index * DOCS_PER_SHARD)`), `seq_len=512`. Real manifest: 16,375,808 total tokens, 31,984 total sequences, real sha256 checksums. The tokenization process itself crashed with exit 134 (`PyGILState_Release`, a benign HF Hub background-thread shutdown race) **after** the manifest was fully written — verified the 4 `.npy` shards were genuinely intact (`np.load()` shape/dtype/max-token-id checks) before trusting them, rather than assuming a crash meant failure.
  2. **Real model**: `torchtitan.models.llama3.model_registry("debugmodel")` (the exact dims ADR-032 validated: `dim=256, n_layers=6, n_heads=16`) with `vocab_size`/`tok_embeddings.num_embeddings`/`lm_head.out_features` overridden to 50257 (gpt2) — all three must be set together, they are independently settable in torchtitan's config. Real parameter count: **30,846,720** (`torchtitan`'s own `.build()`, not hand-derived). Committed as `configs/models/30m-realvocab.toml`.
  3. **Real training driver** (`experiments/01_cu_grid/train_driver.py`, committed — was a local-only scratchpad script before this ADR): the reference `DiLoCoTrainer` (ADR-012/D3 — NOT `torchft`'s DiLoCo path; US-06's cross-implementation equivalence check is still not run, same gap ADR-032 already flagged) driving real `torch.distributed` NCCL all-reduce across 4 real ranks, `StepTimer` (ADR-014) doing the real phase-marker decomposition (`mark_loader_done`/`mark_compute_done`/`mark_sync_done`/`mark_optimizer_done`), `inner_step()`/`ready_for_outer_step()`/`outer_step()` called separately (not the combined `step()`) specifically so the phase markers land on the right side of the inner/outer boundary.
  4. **Real sweep**: `H ∈ {1, 8, 32, 128}`, 150–200 steps each (10 warmup, discarded), `micro_batch_size=4`, unshaped (no `tc` — this is a baseline slice, not the shaped multi-bandwidth grid `configs/grids/phase_a.yaml` still calls for). Raw per-step telemetry committed as JSON: `experiments/01_cu_grid/raw_step_telemetry/result_h{1,8,32,128}.json`.
  5. **Real aggregation** (`experiments/01_cu_grid/aggregate_results.py`, committed): `analysis/cu.py::measured()` on the real StepRecords for `cu_measured`; `analysis/cu.py::analytic()` fed (a) the real `iperf3` all-pairs mean (`LINK_BANDWIDTH_BPS ≈ 9.530 Gbit/s`) for `cu_analytic_link`, and (b) the real NCCL curve, linearly interpolated at the actual pseudo-gradient message size (`n_params * 4` bytes, fp32), for `cu_analytic_achieved` — one shared code path, per ADR-007/FR-04, not two divergent computations. Fingerprint (`measurement/fingerprint.py::capture()`) captured for real on the rank-0 node and hand-embedded (scrubbed per §23 — no IPs, no account IDs). Written to `results/raw/cu_grid-diloco-30m-h{1,8,32,128}-bwunshaped-r0.json`, all 4 pass schema validation via `analysis/load.py::load_run_results()` and step-time reconciliation (residuals 0.01–0.04%, well under the 5% invariant, CLAUDE.md §15.2).
**The headline finding — the pre-registered hypothesis (CLAUDE.md §2.7) holds on this slice:**

| H | cu_measured | cu_analytic_link | cu_analytic_achieved | discrepancy_link |
| --- | --- | --- | --- | --- |
| 1 | 0.1670 | 0.1989 | 0.2797 | 1.19x |
| 8 | 0.6023 | 0.6560 | 0.7490 | 1.09x |
| 32 | 0.7101 | 0.8827 | 0.9217 | 1.24x |
| 128 | 0.8393 | 0.9682 | 0.9794 | 1.15x |

Measured compute utilization is below **both** analytic predictions at every H tested — real hardware underperforms the simulated literature's model here, same direction §2.7 predicted, roughly 1.1–1.25x discrepancy on link bandwidth. A genuine, unplanned secondary finding surfaced feeding this: the real NCCL all-reduce curve (14.3–15.8 Gbit/s plateau) is **higher**, not lower, than the raw point-to-point `iperf3` rate (~9.53 Gbit/s) — ring-topology parallelism keeping multiple segments of the link busy simultaneously beats a single point-to-point TCP flow. This is why `cu_analytic_achieved` comes out *above* `cu_analytic_link` here, the opposite of §2.7 mechanism 1's original expectation ("NCCL over TCP does not achieve link bandwidth" assumed NCCL would underperform a single-flow measurement, not exceed it) — recorded honestly as a finding that revises, not confirms, that specific pre-registered mechanism, while the headline hypothesis (measured < both analytic variants) still holds. New figure module `analysis/figures/fig4_cu_vs_h.py` (CU vs. H at one fixed bandwidth — the correct grouping for a single-bandwidth-level H-sweep; `fig1_cu_surface` needs ≥2 bandwidth levels and correctly raises `ValueError` here) renders this, wired into `analysis/report.py::generate_all_figures()`. 7 new unit tests (`tests/unit/test_fig4_cu_vs_h.py`) cover the §18 solid/dashed convention and the monotonic-in-H trend.
**Not resolved by this ADR:**
  - **This is a first slice, not the grid.** One bandwidth level (unshaped), one repeat per H, no shaping gate exercised. `configs/grids/phase_a.yaml`'s shaped multi-bandwidth sweep (Phase 2/3, M3/M4, G1/G2) is still real remaining work — nothing here should be read as satisfying those.
  - **This run bypassed `measurement/train.py::run()`'s FR-03 orchestration.** No automated precondition gate, no automated shaping gate (moot for unshaped), no in-process fingerprinting — all done by hand for this slice. Building the real `Trainer.Config` adapter (or an equivalent) so `train.py` can drive this itself, with its gates live, is required before a shaped campaign — CLAUDE.md is explicit that the shaping verification gate is not optional (FR-02/ADR-002), and this ADR's driver has no such gate wired in at all.
  - **Repeats** (§40 Q6, still PENDING) — one repeat per H means no variance estimate; the discrepancy factor's non-monotonicity across H (1.19x → 1.09x → 1.24x → 1.15x) could be real or could be single-run noise, and cannot be distinguished without repeats.
  - **US-06** (reference-vs-torchft cross-implementation equivalence) still not run on GPU — same gap as ADR-032.
  - **Per-step Parquet** (§16.1/ADR-023) not produced — raw telemetry is committed as JSON (`raw_step_telemetry/`) only.

---

# 42. Future Extension Strategy

Documented so the architecture stays open, **not** so it gets built now.

| Extension | Enabled by | Blocked by | Effort |
| --- | --- | --- | --- |
| More replicas (8, 16) | `world_size` in the spec; launcher parameterized | GPU quota | Small |
| `netem` WAN realism (latency/jitter/loss) | `netshape.py` designed to accept additional qdisc parameters | Grid size | Small–medium |
| Streaming DiLoCo / async / decoupled variants | `algorithm` is an enum; the run lifecycle is algorithm-agnostic | torchft support; time | Medium |
| Two-level hierarchy (FSDP inside a replica) | Spec supports `gpus_per_replica` as a future field | Requires multi-GPU nodes → more quota | Medium |
| Larger models (7B+) | Model configs are declarative | VRAM; run duration at low bandwidth | Medium |
| Heterogeneous replicas (mixed GPU types) | Fingerprint is per-node already | Straggler analysis needs extension | Medium |
| Other clouds / bare metal | Only `infra/` is AWS-specific | — | Small |
| A community "measured curves" corpus | Schemas are versioned and records self-contained | Would need a submission and provenance process | Large |

**Extension points that already exist:** the `algorithm` enum, the compression codec interface, the shaping parameter set, versioned schemas, and the analysis layer's independence from how records were produced.

**Explicitly not built now:** a plugin system, a config DSL, a generic experiment framework, an abstract "Trainer" hierarchy. Each would be complexity in service of a hypothetical.

---

# 43. Glossary

| Term | Definition |
| --- | --- |
| **Achieved bandwidth** | Bandwidth actually realized by a collective, as opposed to the nominal link rate. The gap between the two is a central subject of this project |
| **All-reduce** | Collective in which every rank ends with the elementwise sum (or mean) of all ranks' tensors |
| **Bytes on wire** | Bytes actually transmitted on the network interface, measured from `/proc/net/dev`, as distinct from bytes the framework believes it sent |
| **Compute utilization (CU)** | Fraction of wall-clock step time spent computing rather than blocked on communication. The literature's primary metric for semi-sync efficiency |
| **DDP** | Distributed Data Parallel; full gradient all-reduce every step (`H = 1`) |
| **DiLoCo** | Distributed Low-Communication training; inner AdamW for `H` local steps, then an outer Nesterov SGD step on the averaged pseudo-gradient |
| **Discrepancy factor (F)** | Measured required bandwidth ÷ analytically predicted required bandwidth, at a given CU target. The headline number |
| **ENA** | Elastic Network Adapter, AWS's network interface. Burstable on small instance sizes |
| **Error feedback** | Accumulating quantization residual and reinjecting it on the next round, so compression defers error rather than destroying it |
| **FSDP2** | Fully Sharded Data Parallel v2; per-parameter DTensor sharding |
| **H** | Synchronization interval — inner steps between outer synchronizations. The project's central control variable |
| **Harness version** | Version of the measurement code path. Results from different versions are not pooled (ADR-006) |
| **Inner / outer optimizer** | Inner runs locally every step (AdamW); outer runs every `H` steps on the pseudo-gradient (Nesterov SGD) |
| **Lighthouse** | `torchft`'s coordinator process, tracking replica-group membership and enabling recovery |
| **LocalSGD** | Semi-sync training by parameter averaging every `H` steps; no outer optimizer. Serves here as the no-outer-optimizer ablation |
| **MFU** | Model FLOPs Utilization; achieved FLOPs ÷ peak. Recompute-counting convention must be stated |
| **NCCL** | NVIDIA Collective Communications Library. Here running over TCP, not NVLink or InfiniBand |
| **`netem`** | Linux qdisc for emulating latency, jitter, and loss |
| **Placement group (cluster)** | AWS grouping that co-locates instances for low-latency networking |
| **Pseudo-gradient / outer gradient** | `θ_outer − θ_inner` after `H` inner steps; the quantity all-reduced in DiLoCo |
| **Replica** | One DiLoCo worker. Here, exactly one GPU on one host |
| **Shaping gate** | The mandatory verification that a requested bandwidth was achieved before a run proceeds (FR-02) |
| **Straggler** | The slowest rank at a synchronization barrier; determines blocking cost |
| **`tbf`** | Token Bucket Filter, the Linux qdisc used for bandwidth shaping |
| **torchft** | PyTorch fault-tolerance library providing HSDP, LocalSGD, DiLoCo, Streaming DiLoCo |
| **torchtitan** | PyTorch-native LLM pretraining platform used here as the model and training substrate |
| **TTTL** | Time-To-Target-Loss; wall-clock time to reach the single-GPU reference loss. `null` when never reached |

---

# 44. Change Management

Whenever a decision, requirement, or architecture element changes, do all seven steps. No silent changes.

```text
1. IDENTIFY   Which existing decision (§41) or requirement (§6) does this affect?
2. EXPLAIN    Why is it insufficient? What did we learn?
3. PROPOSE    The new decision, with alternatives and trade-offs.
4. IMPACT     Measurement path? → which existing results become non-poolable?
              Schema? → which version? Figures? → which regenerate?
              Cost? → cluster-hours. Claims? → PRIOR_ART / LIMITATIONS.
5. UPDATE     CLAUDE.md §41 (new ADR, supersede the old — never delete it),
              §40 (resolve or add questions), §39 (debt), methods/ docs.
6. RECORD     RESULTS.md if any published number changes.
7. VERSION    Bump harness_version if measurement changed; tag the tree.
```

## 44.1 Superseding a decision

Never delete an ADR. Mark it `Status: Superseded by ADR-0NN` and add the new one. The history of why the design changed is part of the artifact, and in an interview it is often the most interesting part.

## 44.2 Mid-campaign changes

`[CONFIRMED]` **A measurement-path change during a running campaign is forbidden.** Stop the campaign, make the change, bump the version, and either re-run the campaign or record the split explicitly in `RESULTS.md`. There is no third option.

---

# 45. Documentation Maintenance

`CLAUDE.md` is the source of truth for intent. It must not drift from the code.

## 45.1 Sync checklist

After any significant change, check whether these need updating:

```text
[ ] §6  Requirements          — new/changed capability?
[ ] §11 Architecture          — new component or boundary?
[ ] §13 Technology Stack      — new dependency? (justify or revert)
[ ] §14 Repository Structure  — new directory or module?
[ ] §15 Domain Model          — new entity, field, or invariant?
[ ] §16 Data Architecture     — schema version change?
[ ] §17 Interfaces            — CLI or module contract change?
[ ] §27 Performance           — target measured and now known?
[ ] §30 Testing               — new test category?
[ ] §35/§36 Phases/Milestones — plan changed?
[ ] §38 Risks                 — risk materialized, retired, or new?
[ ] §39 Technical Debt        — new debt accepted?
[ ] §40 Open Questions        — resolved or newly discovered?
[ ] §41 Decision Log          — new or superseded ADR?
[ ] §43 Glossary              — new term?
[ ] PRIOR_ART.md              — does this change what we claim is novel?
[ ] LIMITATIONS.md            — new limitation?
```

## 45.2 Where truth lives

| Concern | Source of truth |
| --- | --- |
| Why the project exists, what the gap is | `CLAUDE.md` §2, `PRIOR_ART.md` |
| Requirements and acceptance criteria | `CLAUDE.md` §6, §8 |
| Architecture and its rationale | `CLAUDE.md` §11–§13, §41 |
| **How a number is computed** | `methods/*.md` — **the specification; the code implements it, not the reverse** |
| Record shape | `src/diloco_measured/schemas/*.json` |
| CLI and module contracts | `CLAUDE.md` §17 + docstrings |
| Expected behaviour | `tests/` |
| **What was actually measured** | `results/raw/` — **nothing else** |
| Network conditions of a measurement | `results/network/` |
| Environment of a measurement | `results/environment/` + the embedded fingerprint |
| What happened during a campaign | `experiments/*/NOTES.md`, `RESULTS.md` |
| Infrastructure | `infra/*.sh` |
| Known confounds | `LIMITATIONS.md` |
| Open decisions | `CLAUDE.md` §40 |

## 45.3 Staleness markers

If a section cannot be verified against the code, mark it `[STALE — verify]` rather than leaving it silently wrong. A known-stale section is recoverable; a confidently wrong one is not.

---

# 46. Final Engineering Checklist

Run before declaring the project complete.

## Scientific integrity
```text
[ ] Every bandwidth in every record was independently verified (iperf3), not requested
[ ] Every result carries a complete environment fingerprint
[ ] No results pooled across harness_version without a documented equivalence argument
[ ] Step-time components reconcile within tolerance for every included run
[ ] Warmup discarded and the count recorded everywhere
[ ] Median + IQR reported; never mean-only
[ ] Crashed/diverged/loader-bound runs excluded, counted, and reported
[ ] tttl == null never rendered as a finite number
[ ] Instrumentation overhead measured and reported
[ ] Analytic CU model form documented with every assumption (methods/cu_model.md)
[ ] Sensitivity analysis over alternative analytic model forms
```

## Reproducibility
```text
[ ] `make figures` works on a laptop: no GPU, no network, no credentials
[ ] Every report figure regenerates from results/raw/
[ ] All dependencies pinned in a committed lockfile
[ ] launch_cluster.sh + setup_node.sh reproduce the environment from scratch
[ ] Dataset acquisition and tokenization scripted and checksummed
[ ] Every raw iperf3 and NCCL probe output committed under results/network/
```

## Honesty
```text
[ ] PRIOR_ART.md complete and linked in the first screenful of README
[ ] The novelty claim is scoped to the controlled measurement only
[ ] LIMITATIONS.md covers: no WAN latency/jitter/loss · 4 replicas · one GPU
      generation · ≤1B params · 400M-token budgets · single-GPU replicas ·
      one cloud · one AZ · few seeds
[ ] Negative and null results published in RESULTS.md
[ ] experiments/*/NOTES.md record what went wrong, not a sanitized narrative
[ ] Resume bullets contain measured numbers — NO placeholders left
```

## Engineering
```text
[ ] Measurement and analysis remain strictly separated (no forbidden imports)
[ ] All records schema-valid; loader rejects invalid ones
[ ] CI green: lint, type-check, unit, CPU integration
[ ] DiLoCo cross-implementation equivalence test passes
[ ] No credentials, account IDs, ARNs, bucket names, or private IPs anywhere
[ ] results/raw/ never edited (verify via git history)
```

## Operations
```text
[ ] Cluster terminated; teardown verified idempotent
[ ] No orphaned volumes, IPs, or placement groups
[ ] Final cost recorded and compared against budget
```

## Communication
```text
[ ] README leads with the claim and the headline figure
[ ] PLAYBOOK.md is usable by a practitioner who reads nothing else
[ ] Demo recorded (headline figure walkthrough + `plan --probe` + fault injection)
[ ] Every figure survives the question "how do you know?"
[ ] CLAUDE.md §40 Open Questions all resolved or explicitly deferred with reasons
```

---

## Appendix A — Consistency audit of this document

Performed after drafting, per the authoring brief.

| Check | Result |
| --- | --- |
| Contradictions | None found. Note the intentional tension between "no database" (§16) and the derived SQLite index — resolved explicitly: the index is gitignored and never authoritative |
| Every technology justified | Yes — §13 gives purpose, rationale, alternatives, trade-offs, status for each; §13.5 lists rejections with reasons |
| Requirements → architecture | FR-01→probe/netshape; FR-02→netshape gate; FR-03→train/telemetry; FR-04→analysis/cu; FR-05→wire; FR-06→train+analysis; FR-07→predictor; FR-08→fingerprint; FR-09→faults; FR-10→compress; FR-11/12/13→analysis + schemas |
| Architecture → phases | M1 offline modules → M2 rig → M3 validation → M4 grid → M5/M6 → M7 |
| Journeys → requirements | Journey A→FR-01…FR-06; B→FR-11/12; C→FR-07; D→§33 |
| Requirements → domain/interfaces | Every FR names the entities it touches (§15) and its CLI surface (§17) |
| Security reflected in architecture | §23 controls map to §11 (private subnet, control node), §14 (no secrets in configs), §26 (log exclusions), §16 (fingerprint scrubbing) |
| Testing covers key workflows | §30 covers the shaping gate, DiLoCo correctness, aggregation, and the E2E smoke gate |
| Unknowns marked | §40 holds ten open questions; `[UNKNOWN]` appears in §5.2, §13.4, §27 |
| No invented facts as confirmed | Throughput, bandwidth, run-duration, and cost figures are `[PROPOSED]` or `[UNKNOWN]` pending Day 1 measurement |
| Roadmap logically ordered | §37 dependency graph; hard gate at M2; headline lands at M4 on Day 3 so truncation degrades gracefully |
| Operationally usable by future Claude | §33 rules, §32 protocol, §34 DoD, §44 change management, §45 truth map |

**Known intentional asymmetry:** several template sections (frontend, database, HTTP API, authentication, background queues, scalability) are marked *not applicable* with reasons rather than filled in. That is deliberate — inventing a permission matrix or a REST contract for a single-operator measurement rig would be exactly the over-engineering §12.8 and §43's anti-complexity posture forbid.

---

*End of `CLAUDE.md` v0.1. Update per §44 and §45.*
