# diloco-measured

**Measured, Not Simulated — A Bandwidth-Controlled Evaluation of Semi-Synchronous LLM Training
on Commodity Ethernet.**

> Every published bandwidth-vs-synchronization curve in this literature is simulated. This
> repository contains measured ones, the code that produced them, and the discrepancy between
> the two.

**Document status:** `[PROPOSED]` — repository scaffold only. No experiments have been run.
Headline figure and numbers below are placeholders until Phase 3 (`M4`, see `CLAUDE.md` §36).

👉 **Read [`PRIOR_ART.md`](PRIOR_ART.md) first.** It states exactly what is and is not novel here.

---

## The headline claim (pending measurement)

`[UNKNOWN]` — populated after the Phase A grid (`CLAUDE.md` §35 Phase 3). This section will
report the measured vs. analytic compute-utilization discrepancy factor `F` with uncertainty.
Nothing is written here until it is measured.

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

`make figures` runs with no GPU, no network access, and no AWS credentials (FR-11). If it ever
requires any of those three things, that is a bug — file it.

## Repository map

| Path | What it is |
| --- | --- |
| `CLAUDE.md` | The master engineering specification. Read this first. |
| `PRIOR_ART.md` | What is and isn't novel here. |
| `LIMITATIONS.md` | Every known confound, stated by us. |
| `PLAYBOOK.md` | Practitioner-facing lookup: "N GPUs at X Mbit/s → use H=…" |
| `RESULTS.md` | Every run, including nulls, crashes, and abandoned lines. |
| `methods/` | How every number is computed, with every assumption listed. |
| `src/diloco_measured/measurement/` | Needs GPUs + AWS. Never imported by analysis. |
| `src/diloco_measured/analysis/` | Pure. No GPU, no network, no credentials. |
| `results/` | Committed, append-only measurement corpus. |

## Status

Pre-implementation scaffold (`CLAUDE.md` v0.1, Phase 0 not yet started). See `CLAUDE.md` §40 for
the ten open questions that block Day 1, and §35 for the phase plan.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
