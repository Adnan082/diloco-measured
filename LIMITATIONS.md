# Limitations

**Status:** `[CONFIRMED — set of known confounds]`, `[content pending measurement]`.

Per `CLAUDE.md` Architecture Principle #10 ("The limitation section is part of the deliverable"),
every known confound is stated here *before* a reviewer finds it. This list is seeded from the
non-goals (§4.5), technical debt (§39), and risks (§38) already accepted in `CLAUDE.md` — it must
be kept in sync with that document (§45.1 sync checklist) and rewritten in reader-facing language
before publication.

---

## Scope limitations

- **DiLoCo only, so far.** All real measurements to date (`CLAUDE.md` ADR-034/035/037) are
  DiLoCo; DDP, FSDP2, and LocalSGD have no training driver yet. The headline CU-vs-bandwidth
  and convergence results are therefore not yet a cross-algorithm comparison — that is real
  remaining work, not a claim already made and hedged.
- **A 30.8M-parameter model, not the 1B `phase_a.yaml`/`phase_b.yaml` specify.** Chosen for
  validated-working continuity across sessions (`configs/models/30m-realvocab.toml`); the 1B
  config's `torchtitan` `Trainer.Config` adapter does not exist yet.
- **No real WAN emulation.** `tc`/`tbf` shapes bandwidth only — no added latency, jitter, or
  packet loss (TD-3). The measured discrepancy this project reports is therefore a **lower
  bound** on the discrepancy a real, lossy, high-latency WAN would show.
- **Four replicas only**, fixed by the 32-vCPU AWS quota (TD-4). Conclusions may not generalize
  to 8, 16, or more replicas.
- **Single-GPU replicas.** No multi-GPU-per-replica (FSDP-inside-a-DiLoCo-worker) hierarchy is
  tested (TD-5) — untestable on this hardware.
- **Models ≤ ~1B parameters**, short token budgets (~400M tokens) (TD-6). Results may not hold
  at frontier scale.
- **One GPU generation** (NVIDIA L40S, Ada sm89), one cloud (AWS), one region/AZ.
- **One seed per convergence configuration** (TD-7) — convergence conclusions have not been
  checked for seed-sensitivity. This is not hypothetical: the real convergence campaign
  (`CLAUDE.md` ADR-037) found `H=8`'s final loss beat `H=128`'s on the single seed run
  (8.11 vs. 8.18), the opposite ordering of the CU-grid's H-monotone trend — whether that is a
  real effect of `H` or seed noise cannot be told apart with 1 seed, and the result is reported
  as-is rather than resolved into a cleaner story than the data supports.
- **The convergence campaign's target loss was never reached.** None of the 12 real DiLoCo
  configurations tested reached the single-GPU reference's loss within the 400,000-token
  budget (`tttl_s: null` throughout, `CLAUDE.md` ADR-037) — every reported TTTL-adjacent
  finding from this project so far is therefore about *how far short* configurations fell, not
  about comparing crossing times. A larger token budget is needed to know whether DiLoCo
  eventually catches up, and has not been run.

## Methodological limitations

- The analytic CU model attributed to "the literature" is one specific functional form,
  `CU = H·t_compute / (H·t_compute + bytes_synced·8/B)` (`CLAUDE.md` ADR-015 — confirmed by the
  project owner, not derived from a single source paper). A sensitivity analysis against the two
  rejected alternatives (partial overlap; per-paper reproduction) is required before publication
  (`methods/cu_model.md` §6) and has not been run yet — it needs real Phase 3 grid data.
- `torchft`'s LocalSGD/DiLoCo paths are explicitly experimental upstream (R2); mitigated by a
  cross-validated in-repo reference implementation (ADR-003), not eliminated.
- Straggler heterogeneity across nominally-identical EC2 instances is measured but not fully
  separated from bandwidth effects in every figure (R10).

## Dataset licensing (`CLAUDE.md` §40 Q7)

Both candidate pretraining corpora — [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
and [C4](https://huggingface.co/datasets/allenai/c4) — are released under the **Open Data Commons
Attribution License (ODC-BY) v1.0**, which permits use and redistribution with attribution. Both
are also explicitly subject to [Common Crawl's Terms of
Use](https://commoncrawl.org/terms-of-use), since both are derived from the Common Crawl corpus.

**One clause is worth stating plainly rather than discovering later:** Common Crawl's ToU
(Section 9) requires users to indemnify Common Crawl for claims arising from "use of Crawled
Content in connection with artificial intelligence, machine learning, or other similar
technologies, including, without limitation, large language models and neural networks" — which
is exactly what this project does with the data. This is a standard indemnification clause (not a
usage prohibition), the Crawled Content itself is provided "AS IS" with a liability cap, and
using Common-Crawl-derived corpora for LLM pretraining is standard, widespread practice across
the field (both FineWeb-Edu and C4 are published, citable, widely-used-for-exactly-this-purpose
datasets). This is not legal advice and this project carries no commercial stakes, but the clause
exists and a careful reviewer may ask about it, so it's recorded here rather than left for them
to find.

**Mitigation already baked into the project's design (not new because of this research):** this
repository does not redistribute raw or tokenized dataset content — only tokenized-shard
checksums and the download/tokenization scripts, so a reproducer re-fetches the corpus directly
from its original Hugging Face source under the same license terms, rather than from this repo
(`CLAUDE.md` §40 Q7 recommendation, §16.4 retention policy).

## What is explicitly NOT claimed

- This project does not claim DiLoCo is good or bad as an algorithm.
- This project does not claim to have invented DiLoCo, LocalSGD, or the analytic scaling-law
  model — see `PRIOR_ART.md`.
- This project does not claim results generalize beyond the scope stated above (R11 — accepted,
  not mitigated).

---

`[UNKNOWN]` The remaining content of this file is written incrementally as findings land, and
finalized during Phase 6 close-out (`CLAUDE.md` §35, §46 "Honesty" checklist).
