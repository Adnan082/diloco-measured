# Synthetic RunResult corpus

**Generated, not hand-edited.** Regenerate with:

```bash
python tests/fixtures/generate_run_result_corpus.py
```

25 schema-valid `RunResult` records (CLAUDE.md §30.6: "a fixture corpus of ~20 synthetic
RunResult records covering every status, used by all analysis tests"), built from
`tests/fixtures/factories.py`. Used by `tests/integration_cpu/test_aggregation_pipeline.py`
and available to any other test via `analysis.load.load_run_results(CORPUS_DIR)`.

## What's covered

| Category | Records |
| --- | --- |
| DDP baseline, H=1, every bandwidth level | 5 (unshaped/5g/1g/200m/50m) |
| DiLoCo across the H sweep at 1g | 4 (H=8/32/128/512) |
| DiLoCo repeat + a second bandwidth point | 2 (a 2nd repeat at H=32/1g; H=32/200m) |
| LocalSGD, FSDP2 ablations | 2 |
| Non-`completed` statuses | 4 (`crashed`, `diverged`, `aborted_shaping`, `oom`) |
| `completed` but must still be excluded by `analysis/filter.py` | 3 (`loader_bound_warning`, an old `harness_version`, a reconciliation failure) |
| Convergence runs (FR-06) | 2 (target reached; target NOT reached — `tttl_s: null`) |
| Compression ablation (FR-10) | 1 (`int8_ef`) |
| Fault injection (FR-09) | 2 (DiLoCo recovers; DDP hangs — the *expected* outcome) |

**Not represented, deliberately:** `invalid_spec` and `aborted_preconditions` statuses — per
CLAUDE.md §15.2's `Run`/`RunResult` state machine, neither ever produces a `RunResult` record
at all (the run aborts before one is written), so no fixture should model them as if a record
existed.

## Why generated rather than hand-written

25 individually hand-typed JSON files, each ~30 fields deep, would drift from the schema the
moment either changes. `factories.py` centralizes the "valid skeleton," and this directory is
regenerated and re-committed deliberately (not at test time — matches the project's "results
are committed, static, human-readable" philosophy, ADR-004) whenever the schema or the
scenarios change.
