# Results Log

**Status:** `[CONFIRMED]` — the real shaped, multi-bandwidth DiLoCo grid now has 3 repeats/point
(G1/G2 satisfied) and a real convergence campaign has run (G3 satisfied) — DiLoCo only; the
DDP/FSDP2/LocalSGD legs of the full `phase_a.yaml`/`phase_b.yaml` comparisons are still not
built, and G4 (predictor) is still not fitted.

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
| 2026-08-14/15 | Shaped, multi-bandwidth DiLoCo grid, repeat 0 (`H∈{1,8,32,128} × bw∈{50m,200m,1g,5g}`, `01_cu_grid`) | 0.1.0 | 16 | 16 | 0 | 0 | 0 | ADR-035. Real `tc` shaping + real FR-02 `iperf3` verification gate on every point, 16/16 passed on the first attempt (no retries needed, 0.7–2% error, well under 10% tolerance). Cluster relaunched for this campaign (previous one torn down) — hit and fixed 3 real bugs getting it back up: a stale SG SSH rule, a Git-Bash/MSYS path-mangling bug in the AWS CLI invocation, and an undersized control-node root volume (8GB default, no `--block-device-mappings` — fixed live via `ec2:ModifyVolume`, and fixed in `infra/launch_cluster.sh` for future launches). DiLoCo only, 30.8M-param model (not the 1B `phase_a.yaml` calls for). |
| 2026-08-15 | Shaped, multi-bandwidth DiLoCo grid, repeats 1 and 2 (same 16 points × 2, `01_cu_grid`) | 0.1.0 | 32 | 32 | 0 | 0 | 0 | ADR-037. Third cluster relaunch this project. Repeat variance tight (< 1.5% spread on typical points). Combined with repeat 0: 48 real runs, satisfying G1's "3 repeats each" for the first time. Also fixed a real biconditional-validator bug found while preparing this campaign (`H==1 iff algorithm=="ddp"` rejected real, already-committed DiLoCo `H=1` data — relaxed to one-directional, ADR-036) and a real accidental-commit of a scratch working directory (`experiments/*/shaped_grid_run_logs/`, fixed same session). |
| 2026-08-15 | Convergence campaign: single-GPU reference + DiLoCo grid (`H∈{1,8,32,128} × bw∈{unshaped,1g,200m}`, `02_convergence`) | 0.1.0 | 13 (1 reference + 12) | 13 | 0 | 0 | 0 | ADR-037. Single-GPU reference reuses `train_driver.py` unchanged (H set unreachably large so the outer step never fires — verified directly with a smoke test before trusting it). L\*=7.352. **TTTL is `null` for all 12 DiLoCo points — none reached L\* within the 400,000-token budget** (final losses 8.11–8.65 vs. the reference's 7.35). Reported as a real null result, not hidden — see "Null / negative results" below. |

**Not yet run:** DDP/FSDP2/LocalSGD legs of `configs/grids/phase_a.yaml`/`phase_b.yaml`'s
4-algorithm comparisons (no training driver exists for them yet) — both campaigns above are
DiLoCo only. The 1B model `phase_a.yaml` specifies. More than 1 seed per convergence
configuration. Fault injection (`G7`). Compression ablation (`G6`). Predictor validation
(`G4`) — the required-bandwidth table this needs as an input now exists
(`experiments/01_cu_grid/required_bandwidth_table.json`), but the model has not been fitted.

## Cumulative cost

| Date | Cluster-hours | Estimated spend | Running total |
| --- | --- | --- | --- |
| 2026-08-14 | ~2.9 hrs × 5 instances (4× `g6e.2xlarge` + 1× `c7i.2xlarge`), `us-east-1b`, at a combined $9.32/hr burn rate — unshaped `H`-sweep session | ~$27 | ~$27 |
| 2026-08-14/15 | ~1.9 hrs × 5 instances, cluster relaunched (placement group `pg-0ee059f5ef7da671b`), same `us-east-1b`, same $9.32/hr — shaped multi-bandwidth grid session | ~$18 | ~$45 |
| 2026-08-15 | ~2.3 hrs × 5 instances, cluster relaunched again (placement group `pg-08f2faef5740fa3c4`), same `us-east-1b`, same $9.32/hr — repeats 1/2 + convergence campaign session (13:58:23–~16:14 UTC) | ~$21 | ~$66 |

All three clusters terminated at the end of their respective sessions (`infra/teardown.sh`) —
every instance confirmed `terminated` and each placement group deleted, verified via
`aws ec2 describe-instances`/`describe-placement-groups` immediately after. Budget ceiling per
`CLAUDE.md` §5.1 is ~$650–800 for the whole project — cumulative spend so far is well under 10%
of that.

## Known-bad / superseded records

`[CONFIRMED]` None yet. When a `harness_version` bump supersedes prior records, list the
superseded run IDs here per §16.3, rather than deleting them.

## Null / negative results

`[CONFIRMED]` Reported here as a first-class outcome, per §2.7 and §25.3 — a silent skip is a
bug.

**TTTL never crossed, convergence campaign (2026-08-15, ADR-037).** None of the 12 DiLoCo
configurations (`H∈{1,8,32,128} × bandwidth∈{unshaped,1g,200m}`) reached the single-GPU
reference's target loss `L*=7.352` within the 400,000-token budget. `tttl_s` is `null` in all
12 `results/raw/convergence-diloco-*.json` records — never rendered as a finite number, per
the `ConvergenceCurve` invariant. Final losses (8.11–8.65) are all above `L*`, and — a real,
if unwelcome, finding — the gap does not close monotonically with `H`: `H=8` (final loss 8.11)
outperformed `H=128` (8.18) on this single-seed run, the opposite of the CU-grid's H-monotone
trend. With only 1 seed per configuration (§40 Q6/TD-7), whether that ordering is a real effect
or seed noise is not distinguishable from this campaign alone — stated honestly, not resolved
into a cleaner story than the data supports.

All other runs across every campaign so far (65 training runs total: 4 unshaped + 48 shaped
CU-grid + 13 convergence, the last of which includes the single-GPU reference) and the network
characterization completed successfully, with zero shaping-gate aborts and zero crashes. The
CU-grid headline finding itself (measured CU below the naive analytic prediction at every
shaped point, ADR-035/037) is not a null result, but it is the falsification-committed-in-
advance direction: see `CLAUDE.md` §2.7 for why either direction was pre-declared as a
reportable outcome.
