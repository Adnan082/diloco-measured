# Results Log

**Status:** `[CONFIRMED]` — first real campaigns have run. Network characterization (Phase 1)
and a first unshaped DiLoCo `H`-sweep (early Phase 2) are complete. The shaped multi-bandwidth
grid (Phase 3, `M4` — the project's actual headline deliverable) has not run yet.

This file is the human-readable ledger of every campaign: what ran, what didn't, and why.
Per `CLAUDE.md` §16.3 and §25, nothing in `results/raw/` is ever edited or deleted — this file
is where the *narrative* of failures, crashes, and abandoned lines is published rather than
hidden. It is updated after every campaign (`CLAUDE.md` §34.2 Definition of Done).

Do not write a number here that is not backed by a record in `results/raw/`.

---

## Campaign log

| Date | Campaign | `harness_version` | Points attempted | Completed | Crashed | Aborted (shaping) | Diverged | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-14 | Network characterization (FR-01, unshaped baseline) | 0.1.0 | 1 profile (12 `iperf3` pairs, 6-point NCCL curve, 1 burst-decay probe) | 1 | 0 | n/a | n/a | Not run via `cli.py network characterize` (still `NotImplementedError` — see ADR-028) — ad-hoc scripts driving the same real `netshape.py`/`probe.py` primitives. See `experiments/00_network_characterization/NOTES.md` for the mistakes made along the way (a shell-regex parsing bug, the `ens5`→`enp39s0` interface bug, ADR-033). |
| 2026-08-14 | First unshaped DiLoCo `H`-sweep (`H∈{1,8,32,128}`, `01_cu_grid`) | 0.1.0 | 4 | 4 | 0 | n/a (unshaped) | 0 | ADR-034. One repeat per point, no variance estimate yet (§40 Q6 still open). Driven by `experiments/01_cu_grid/train_driver.py` directly, not through `measurement/train.py::run()`'s full FR-03 orchestration — no automated precondition/shaping gate for this run. |

**Not yet run:** the shaped, multi-bandwidth grid (`configs/grids/phase_a.yaml`, Phase 2/3,
`M3`/`M4`, goals `G1`/`G2`) — this is the comparison that actually varies bandwidth and answers
"how much bandwidth is enough." Convergence/TTTL runs (`G3`). Fault injection (`G7`). Predictor
validation (`G4`).

## Cumulative cost

| Date | Cluster-hours | Estimated spend | Running total |
| --- | --- | --- | --- |
| 2026-08-14 | ~2.9 hrs × 5 instances (4× `g6e.2xlarge` + 1× `c7i.2xlarge`), `us-east-1b`, at a combined $9.32/hr burn rate | ~$27 (derived from `infra/cost_report.sh`'s burn rate × observed runtime — the exact AWS invoice may differ slightly due to per-second billing rounding and reporting lag) | ~$27 |

Cluster terminated 2026-08-14 at the end of this session (`infra/teardown.sh`) — all 5
instances confirmed `terminated` and the placement group deleted, verified via
`aws ec2 describe-instances`/`describe-placement-groups` immediately after. Budget ceiling per
`CLAUDE.md` §5.1 is ~$650–800 for the whole project — this session used a small fraction of it.

## Known-bad / superseded records

`[CONFIRMED]` None yet. When a `harness_version` bump supersedes prior records, list the
superseded run IDs here per §16.3, rather than deleting them.

## Null / negative results

`[CONFIRMED]` Reported here as a first-class outcome, per §2.7 and §25.3 — a silent skip is a
bug. None recorded yet — all 4 training runs and the network characterization completed
successfully. The headline finding itself (measured CU below both analytic predictions,
ADR-034) is not a null result, but it is the falsification-committed-in-advance direction: see
`CLAUDE.md` §2.7 for why either direction was pre-declared as a reportable outcome.
