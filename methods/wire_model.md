# Bytes-on-Wire Model

**Status:** `[CONFIRMED]` derivation form; `[PROPOSED]` exact per-algorithm constants pending
Day 1 measurement. Specifies `measurement/wire.py::predict()` and `::account()` (FR-05).

---

## 1. Ground truth vs. prediction

- **Measured:** `/proc/net/dev` snapshot before and after the measurement window, differenced,
  per node. This is the ground truth — independent of what the framework believes it sent
  (FR-05 design note).
- **Predicted:** computed from first principles, below.
- **Recorded:** `wire_bytes_predicted`, `wire_bytes_measured`, `wire_overhead_ratio`, plus
  per-token normalizations (`WireAccount` entity, `CLAUDE.md` §15.2).

## 2. Analytic prediction — ring all-reduce

For a ring all-reduce of a tensor of `N` bytes across `P` ranks, each rank sends and receives
approximately:

```
bytes_per_rank_per_sync = 2 · N · (P − 1) / P
```

Synchronization frequency is `1 / H` (once every `H` inner steps). So per training step:

```
bytes_per_rank_per_step = bytes_per_rank_per_sync / H
```

**Consequence to test (§6.10.3 invariant):** communication volume per *round* is `O(N)`,
independent of `H`; per *step* it is `O(N / H)`. DDP (`H = 1`) sends `H`× more per-step bytes
than DiLoCo at a given `H` — the DDP/DiLoCo per-step byte ratio should equal `H`. This is a unit
test target (`CLAUDE.md` §30.2: "DDP vs DiLoCo ratio equals H").

## 3. What `N` is, per algorithm

| Algorithm | What's synchronized | Size |
| --- | --- | --- |
| DDP | Full gradient, every step | `num_params × dtype_bytes` |
| FSDP2 | Sharded gradients/params per step | depends on sharding — `[UNKNOWN]`, derive in Phase 0 |
| LocalSGD | Full parameter tensor, every `H` steps | `num_params × dtype_bytes` |
| DiLoCo | Pseudo-gradient `Δ = θ_outer − θ_inner`, every `H` steps | `num_params × dtype_bytes` |
| DiLoCo + compression | Compressed pseudo-gradient + any error-feedback overhead | codec-dependent, see `compress.py` (FR-10) |

## 4. Idle baseline

Other traffic on the interface pollutes the counter (FR-05 failure condition). Mitigation:
nothing else runs on the nodes during a measurement window, and the idle-baseline drift is
measured and recorded as `idle_baseline_bytes` so it can be subtracted or reported alongside.

## 5. What a divergence between predicted and measured means

Per FR-05's design note: agreement demonstrates correct understanding of the collectives.
Divergence is attributable to TCP/IP header overhead, NCCL protocol overhead (ring vs. tree
switching), and retransmits — quantifying the gap is itself a reported result, not an error to
explain away.

## 6. Open items

`[UNKNOWN]` — FSDP2's exact per-step communication volume at a given sharding configuration must
be derived empirically on Day 0/1 before this document's FSDP2 row can move to `[CONFIRMED]`.
