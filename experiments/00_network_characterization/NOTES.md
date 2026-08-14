# NOTES — 00_network_characterization

**Rule (CLAUDE.md §14.1):** this file records what actually happened, including mistakes — a
sanitized narrative does not belong here.

**Run 2026-08-14, real 4x g6e.2xlarge + 1x c7i.2xlarge cluster, us-east-1b (placement group
`pg-04ac04963de1615d8`).** Committed as `results/network/phase1-us-east-1b-20260814.json`.

**Not run via `run.sh`/the CLI.** `diloco-measured network characterize` (`cli.py`) still
raises `NotImplementedError` for the same reason `measurement/train.py::run()` stops short of
a real torchrun launch (ADR-028/ADR-034: no cluster-inventory mechanism to supply real `Node`
objects from the CLI existed yet). The actual measurements were taken with ad-hoc scripts
driving the SAME real primitives `netshape.py`/`probe.py` implement (`iperf3` over SSH, a real
`torch.distributed.all_reduce` NCCL sweep across all 4 ranks, a 600s sustained-transfer burst
probe) and hand-assembled into a schema-valid `NetworkProfile` afterward. This is a real gap
versus FR-01's fully automated path — closing it (wiring `cli.py network characterize` to a
real cluster inventory) is real remaining work, not done here.

**What was found (all real, all in the committed profile):**
- 12 ordered-pair `iperf3` measurements (60s each), all ~9.530 Gbit/s — consistent, no
  meaningful asymmetry between any pair.
- A 6-point NCCL all-reduce bandwidth curve, 1 MiB → 1 GiB message sizes, plateauing around
  **14.3–15.8 Gbit/s** — i.e. NCCL's ring all-reduce achieves noticeably **higher** throughput
  than raw point-to-point `iperf3` TCP at the same physical link, because the ring topology
  keeps multiple segments of the interconnect busy in parallel rather than saturating one
  point-to-point flow. This was not anticipated going in (§2.7's mechanism list only expected
  NCCL to achieve *less* than link rate) and is exactly the kind of "measured, not simulated"
  finding this project exists to surface — see ADR-034 for how this fed into
  `cu_analytic_achieved` coming out *higher*, not lower, than `cu_analytic_link`.
- `burst_decay_detected: false` — 600s sustained transfer, 20 sample points, only 0.012% decay.
  No evidence of ENA burst-credit exhaustion at this instance size over this window (the §2.7
  mechanism-3 hypothesis was not observed here; a longer/heavier sustained-load probe could
  still surface it and hasn't been ruled out).
- No shaping was applied for this profile (`shaping_fidelity: []`) — this is the **unshaped
  baseline** characterization only. FR-02's shaping-with-verification-gate path (5g/1g/200m/50m
  levels per `spec.yaml`) has not been exercised on real hardware yet.

**Mistakes along the way (kept, not sanitized):** a first attempt at the `iperf3` all-pairs
sweep used a shell `grep`/regex pass over the raw JSON output that silently matched nothing
(tab-separated JSON broke the `[0-9.]*` pattern) — caught by a quick isolated check before
trusting a full 12-pair run, fixed by parsing with `python3 -c "import json..."` on the remote
node instead of shell regex. Separately, the very first multi-node NCCL attempt failed
outright (`NCCL error ... invalid usage`) because `netshape.py`'s `Node.iface` default
(`"ens5"`) does not exist on this AMI — the real interface is `enp39s0` (ADR-033).
