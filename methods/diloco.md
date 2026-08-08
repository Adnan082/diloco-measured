# DiLoCo — Inner/Outer Derivation

**Status:** `[CONFIRMED]` algorithm form (this is DeepMind's published algorithm, not ours —
see `PRIOR_ART.md`). `[PROPOSED]` hyperparameters, pending Day 0/1 validation.

Specifies `measurement/diloco.py` (the in-repo reference implementation, ADR-003) and the
equivalence contract it must satisfy against the `torchft` path.

---

## 1. The loop

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

(Reproduced from `CLAUDE.md` §10.3 — this file is the canonical home for it; §10.3 references it.)

## 2. Why Nesterov for the outer step

`[PROPOSED — cite DiLoCo paper's ablation once reviewed in Phase 0]`. The outer optimizer treats
the pseudo-gradient as if it were a normal gradient signal on a much coarser timescale; momentum
smooths the noise introduced by averaging across only 4 replicas with heterogeneous local data.

## 3. Invariants that MUST hold and MUST be tested

These map directly to `tests/integration_cpu/` (§30.3):

1. **Inner optimizer state persists across rounds.** This is what distinguishes DiLoCo from
   naive FedOpt — resetting AdamW state every round is a correctness bug, not a style choice.
2. **Bit-identical `θ_outer` across replicas after every outer step**, within all-reduce
   determinism tolerance.
3. **Communication volume is `O(N)` per round, `O(N/H)` per step** — see `methods/wire_model.md`.
4. **With compression enabled, the error-feedback residual accumulator persists across outer
   steps and is included in checkpoints** (FR-10 invariant). Dropping it silently is the
   single most likely compression bug (§30.2).

## 4. Hyperparameters

`[PROPOSED — pending Day 0/1]`

| Param | Value | Status |
| --- | --- | --- |
| Inner optimizer | AdamW | `[CONFIRMED — algorithm form]` |
| Outer optimizer | Nesterov SGD | `[CONFIRMED — algorithm form]` |
| Inner LR | `[UNKNOWN]` | pending Day 0 sweep on the tiny/smoke config |
| Outer LR / momentum | `[UNKNOWN]` | pending Day 0 sweep |
| `H` values under test | `{1, 8, 32, 128, 512}` | `[CONFIRMED — CLAUDE.md §1.2]` |

## 5. Cross-implementation equivalence contract (US-06)

Given the same seed, model, data order, and `H`: the reference `diloco.py` (gloo, CPU) and the
`torchft` path must produce loss curves agreeing within a documented tolerance over 200 steps.
Tolerance value: `[PROPOSED]` — to be set empirically on Day 0 from observed float
non-determinism, then frozen. A divergence beyond tolerance fails CI (`CLAUDE.md` §30.3).
