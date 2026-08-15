# NOTES — 02_convergence

**Rule (CLAUDE.md §14.1):** this file records what actually happened, including mistakes — a
sanitized narrative does not belong here.

**First real convergence campaign run 2026-08-15** (see CLAUDE.md ADR-037, G3). Not driven by
`configs/grids/phase_b.yaml` (still a scaffold) — a purpose-built pair of scripts,
`run_convergence_campaign.py` + `aggregate_convergence.py` (this directory), on the same
relaunched cluster the repeats campaign used (`experiments/01_cu_grid/NOTES.md`).

**Design:** single-GPU reference (defines `L*` per ADR-021/§40 Q5) + DiLoCo grid,
`H ∈ {1, 8, 32, 128} × bandwidth ∈ {unshaped, 1g, 200m}` = 12 points, all to a fixed
**400,000-token** budget. `model_config=30m-realvocab`, same model as the CU grid.

**The single-GPU reference reuses `train_driver.py` unchanged** — no new training driver was
written. Setting `--H 999999` (far larger than the run's ~195 steps) means
`DiLoCoTrainer.ready_for_outer_step()` (`_h_count >= H`) never fires, so `inner_step()` alone —
plain per-step AdamW on the live model, zero cross-replica communication — is exactly a
standard single-GPU training loop. This was **verified directly**, not assumed: a 10-step
smoke run on one node showed `outer=False` and `sync_ms≈0.03` throughout before the full
campaign was trusted to run unattended.

`L* = 7.352` (the reference's own final loss, down from 11.00 over 195 steps / 400,000 tokens).

**All 12 DiLoCo points completed, zero shaping-gate failures, zero crashes.** Combined with
the reference: 13 real runs, clearing G3's "≥10 completed convergence runs."

**Headline finding: TTTL is `null` for every one of the 12 DiLoCo points.** None reached `L*`
within the 400,000-token budget:

| H | final_loss (all 3 bandwidth levels — see below) |
| --- | --- |
| 1 | 8.6487 |
| 8 | 8.1075 |
| 32 | 8.5510 |
| 128 | 8.1823 |

All above the reference's 7.352. Recorded as a real, honest null result (CLAUDE.md §2.7/§25.3
— a silent skip is a bug, and so is quietly not mentioning a result that didn't go the
expected way) — `tttl_s: null` is the documented, correct representation for "target never
reached," never rendered as a finite number. Whether a larger token budget would let DiLoCo
catch up is a real open question this campaign doesn't answer.

**A sanity check that became a finding:** `final_loss` is bit-identical across all 3 bandwidth
levels for a given `H` (e.g. `H=1`: `8.64871883392334` at unshaped, 1g, and 200m alike), while
wall-clock time varies enormously for that same sweep — 9.1s / 76.1s / 398.7s. Checked directly
against the raw per-run `started_at` timestamps and `total_wall_s` before trusting this as
correct rather than a bug: same seed, same `H`, same step count produce the exact same sequence
of optimizer updates regardless of how long each sync physically takes — bandwidth is a pure
wall-clock cost multiplier here, with zero effect on the training trajectory (nothing
bandwidth-dependent, like gradient compression, exists yet to change that). This is why
`fig3_convergence_curves.py` plots one bandwidth level's loss curve as representative of all
three for a given `H`.

Figure: `results/figures/fig3_convergence_curves_diloco_bwunshaped.png` — reference (dashed
black) vs. each `H` (solid), with `L*` as a horizontal dotted line. First real rendering of
this figure module.

**Not yet done:** DDP/FSDP2/LocalSGD convergence behavior (DiLoCo-only, same scope gap as the
CU grid). The 1B model. A held-out validation set (`val_loss` is `null` throughout — `train_loss`
serves as both signals, a real simplification, not hidden). More than 1 seed per configuration
(§40 Q6/TD-7). Diagnosing *why* DiLoCo trained slower per-token than the reference here
(candidates: the Nesterov-SGD outer step vs. AdamW's adaptive rates; effective-batch-size
differences between 1 and 4 replicas; too few outer syncs within the budget for H≥32 to help
much) — not distinguished from each other by this campaign.
