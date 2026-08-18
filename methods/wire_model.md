# Bytes-on-Wire Model

**Status:** `[CONFIRMED]` derivation form for every algorithm this project measures, including
FSDP2 (§3a, resolved 2026-08-17); `[PROPOSED]` exact per-algorithm constants pending a real
`/proc/net/dev` cross-check, which no driver has implemented yet (§6). Specifies
`measurement/wire.py::predict()` and `::account()` (FR-05).

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
| DDP | Full gradient, every step (1 all-reduce) | `num_params × dtype_bytes` |
| FSDP2 | Sharded params/grads, every step (2 all-gathers + 1 reduce-scatter) | `num_params × dtype_bytes`, see §3a — **3×** the per-collective volume of DDP |
| LocalSGD | Full parameter tensor, every `H` steps | `num_params × dtype_bytes` |
| DiLoCo | Pseudo-gradient `Δ = θ_outer − θ_inner`, every `H` steps | `num_params × dtype_bytes` |
| DiLoCo + compression | Compressed pseudo-gradient + any error-feedback overhead | codec-dependent, see `compress.py` (FR-10) |

### 3a. FSDP2's per-step communication pattern — `[CONFIRMED]`, derived 2026-08-17

Resolves this document's former `[UNKNOWN]` FSDP2 row (Phase 0's placeholder). Derived
analytically from `torch.distributed.fsdp.fully_shard`'s own documented behavior and
`torchtitan/models/llama3/infra/parallelize.py::apply_fsdp()`'s real wrapping code (read
directly, not guessed) — not yet cross-checked against a real `/proc/net/dev` measurement,
which is the same project-wide gap `fig5_bytes_on_wire` already documents (ADR-038/039) as
unresolved for every algorithm, FSDP2 included.

With `reshard_after_forward=True` (torchtitan's own default for every real parameter-holding
FSDP group when pipeline parallelism is disabled — the case for every configuration this
project runs), one training step does three real ring collectives, each moving the standard
ring-collective volume `N(P−1)/P` bytes per rank:

1. **Forward all-gather** — materialize full (unsharded) parameters before computing.
2. **Backward all-gather** — parameters were freed after forward, so they must be re-gathered
   before gradient computation.
3. **Backward reduce-scatter** — sum gradients across ranks, re-shard the result.

```
bytes_per_rank_per_step (FSDP2) = 3 · N · (P − 1) / P    = 1.5 × DDP's 2·N·(P−1)/P
```

This 1.5× figure is a well-established, independently-documented property of FSDP's
`FULL_SHARD`-equivalent strategy (memory savings traded for extra communication versus DDP) —
this derivation reproduces it from this project's own actual wrapping code rather than citing
it uncritically. `H` has no meaning for FSDP2 in this project's framework (not a
semi-synchronous method) — every FSDP2 grid point fixes `H=1`, same as DDP.

**A separate, simpler convention for the `cu_analytic_*` model** (`methods/cu_model.md` §3):
that model does not account for ring-collective efficiency at all — every real transfer
counts as one full `N`-byte `bytes/B` cost, deliberately simpler than the ring-collective math
above. Under that convention, FSDP2's three separate collective calls (forward all-gather,
backward all-gather, backward reduce-scatter) give `bytes_synced = 3·N`, not `1.5·N` — the
starker ratio is a consequence of the CU model's own simplification, not a different measured
reality. Both numbers are real and both are used, for different purposes; conflating them
would be the mistake, not either individually.

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

`[CONFIRMED — 2026-08-17]` FSDP2's per-step communication volume is now derived, §3a. What
remains open (same as every other algorithm in this project, not FSDP2-specific): no driver
yet captures `/proc/net/dev` before/after a measurement window, so `wire_bytes_measured` has
never been populated for any real record, and this document's ring-collective formulas remain
analytically derived rather than empirically cross-checked (`fig5_bytes_on_wire` is empty for
every algorithm as a direct consequence — ADR-038/039).
