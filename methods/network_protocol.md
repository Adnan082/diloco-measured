# Network Characterization and Shaping Protocol

**Status:** `[CONFIRMED]` — this is the project's central integrity mechanism (FR-02).
Specifies `measurement/netshape.py` and `measurement/probe.py` (FR-01, FR-02).

---

## 1. Characterization (FR-01) — run once per cluster session, before any experiment

1. `iperf3` between all ordered node pairs, both directions, 60s each.
2. NCCL all-reduce probe across all 4 ranks, message size log-spaced 1 MiB → 4 GiB.
3. For each configured shaping rate: requested rate, measured `iperf3` rate, NCCL
   achieved-bandwidth curve — all recorded.
4. 10-minute sustained transfer at the unshaped rate, to detect ENA burst-credit decay
   (`g6e.2xlarge` is rated "up to 20 Gigabit" — burst, not sustained; §2.7 mechanism 3).
5. Write a `NetworkProfile` record to `results/network/`.

**Abort condition:** any node unreachable → abort, report which node, write no partial profile.
**Finding, not error:** sustained throughput decaying > 20% over 10 minutes → record
`burst_decay_detected: true` with the decay curve.

## 2. Shaping + verification gate (FR-02) — before every shaped run

```text
apply tc qdisc ... tbf rate <R> burst <B> latency <L> on ens5 egress, all 4 nodes
        │
        ▼
run iperf3 ≥15s between two nodes
        │
        ▼
assert |measured − requested| / requested ≤ tolerance   (tolerance = [PROPOSED] 10%)
        │
   pass │ fail
        │    └─► retry ONCE → still fail → ABORT run, write ShapingFailure record
        ▼
record the MEASURED rate (never the requested rate) into the run record
        │
        ▼
... run proceeds ...
        │
        ▼
restore original qdisc on every exit path (including SIGINT)
   restore fails → mark node dirty; subsequent runs on that node abort until fixed
```

**The invariant this protocol exists to guarantee:** `ShapingVerification.passed == false` ⇒
no `RunResult` may exist for that run (`CLAUDE.md` §15.2). There is no code path that writes a
requested-but-unverified rate into an analysis-eligible record.

## 3. Why `tbf`, not `netem rate`

`tbf` (Token Bucket Filter) models a bandwidth ceiling only. `netem rate` has a different
queueing model. `tbf` is primary; `netem` is the documented fallback if `tbf` proves unstable
(R3), and is also the mechanism reserved for the optional WAN-realism spot-check (§40 Q8).

## 4. Known confound

`tbf` adds no latency, jitter, or loss (ADR-002 trade-off, TD-3). The measured discrepancy this
project reports is therefore a lower bound on what a real WAN link would show — stated in
`LIMITATIONS.md`.

## 5. Timeouts `[PROPOSED — tune Day 1]`

`iperf3`: 60s. NCCL init: 300s. NCCL collective: 600s (must exceed the worst-case sync at the
lowest bandwidth level under test — compute this from the model size and 50 Mbit/s, don't guess;
miscalibrating this causes spurious failures precisely at the low-bandwidth end that matters
most). S3 sync: 1800s.
