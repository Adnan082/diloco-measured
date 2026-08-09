"""Aggregation pipeline test against the real fixture corpus — CLAUDE.md §30.3:
"Aggregation pipeline | Fixture records | Figures generate from fixtures with no GPU, no
network." No GPU/network/distributed backend is used here (hence living alongside the other
CPU integration tests, even though nothing here spawns a process) — it's grouped with the
pipeline-level tests rather than tests/unit because it exercises load -> filter -> aggregate
together, not one function in isolation.

Corpus: tests/fixtures/run_results/*.json (25 records), built by
tests/fixtures/generate_run_result_corpus.py from tests/fixtures/factories.py. See
tests/fixtures/run_results/README.md for what each record represents.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from diloco_measured.analysis.aggregate import aggregate_repeats
from diloco_measured.analysis.filter import apply
from diloco_measured.analysis.load import load_run_results

CORPUS_DIR = Path(__file__).parents[1] / "fixtures" / "run_results"


@pytest.mark.integration_cpu
def test_entire_corpus_loads_and_validates():
    records = load_run_results(CORPUS_DIR)
    assert len(records) == 25


@pytest.mark.integration_cpu
def test_corpus_covers_every_reachable_run_result_status():
    """invalid_spec and aborted_preconditions are deliberately absent: per CLAUDE.md §15.2's
    Run/RunResult state machine, neither ever produces a RunResult record at all (the run
    aborts before one is written) — so no fixture should represent them.
    """
    records = load_run_results(CORPUS_DIR)
    statuses = Counter(r["status"] for r in records)
    assert statuses["completed"] > 0
    assert statuses["crashed"] > 0
    assert statuses["diverged"] > 0
    assert statuses["oom"] > 0
    assert statuses["aborted_shaping"] > 0
    assert "invalid_spec" not in statuses
    assert "aborted_preconditions" not in statuses


@pytest.mark.integration_cpu
def test_filter_excludes_every_documented_category_with_correct_counts():
    records = load_run_results(CORPUS_DIR)
    kept, report = apply(records, harness_version="v1")

    assert report.total == 25
    assert report.excluded_crashed == 1
    assert report.excluded_diverged == 1
    assert report.excluded_other_status == 2  # aborted_shaping + oom
    assert report.excluded_version_mismatch == 1  # the "oldversion" fixture (harness_version=v0)
    assert report.excluded_loader_bound == 1
    assert report.excluded_reconciliation_failed == 1
    assert report.kept == 25 - (1 + 1 + 2 + 1 + 1 + 1)
    assert len(kept) == report.kept
    assert all(r["status"] == "completed" for r in kept)


@pytest.mark.integration_cpu
def test_filter_with_allow_version_mix_recovers_the_old_version_record():
    records = load_run_results(CORPUS_DIR)
    kept_strict, _ = apply(records, harness_version="v1")
    kept_mixed, report_mixed = apply(records, harness_version="v1", allow_version_mix=True)

    assert len(kept_mixed) == len(kept_strict) + 1
    assert report_mixed.excluded_version_mismatch == 0


@pytest.mark.integration_cpu
def test_aggregate_repeats_over_the_two_diloco_h32_bw1g_records():
    """Two repeats exist at cu_grid-diloco-1b-h32-bw1g (r0, r1) with different cu_measured
    values specifically so this aggregation has something real to compute over.
    """
    records = load_run_results(CORPUS_DIR)
    kept, _ = apply(records, harness_version="v1")

    matching = [
        r for r in kept
        if r["spec"]["algorithm"] == "diloco"
        and r["spec"]["H"] == 32
        and r["spec"]["bandwidth_requested_bps"] == 1_000_000_000
    ]
    assert len(matching) == 2

    cu_values = [r["cu"]["cu_measured"] for r in matching]
    result = aggregate_repeats(cu_values)
    assert result.n == 2
    # median of exactly 2 values is their mean
    assert result.median == pytest.approx(sum(cu_values) / 2)


@pytest.mark.integration_cpu
def test_not_reached_convergence_record_has_null_tttl_not_a_large_number():
    records = load_run_results(CORPUS_DIR)
    not_reached = next(
        r for r in records if r["run_id"] == "convergence-diloco-130m-h32-bw200m-notreached-r0"
    )
    assert not_reached["convergence"]["tttl_s"] is None
    assert not_reached["convergence"]["reached_target"] is False


@pytest.mark.integration_cpu
def test_ddp_fault_record_shows_hung_not_recovered():
    """FR-09: for DDP, a hang instead of recovery is the EXPECTED outcome, not a failure."""
    records = load_run_results(CORPUS_DIR)
    ddp_fault = next(r for r in records if r["run_id"] == "faults-ddp-130m-h1-bw1g-r0")
    assert ddp_fault["faults"][0]["outcome"] == "hung"


@pytest.mark.integration_cpu
def test_crashed_record_has_no_telemetry_fields():
    """A crashed run preserves partial telemetry per §15.2, but this fixture models the case
    with none captured — cu/wire/throughput must be ABSENT, not null (schemas/run_result.v1.json
    doesn't accept null for these; see factories.py's docstring on this).
    """
    records = load_run_results(CORPUS_DIR)
    crashed = next(r for r in records if r["status"] == "crashed")
    assert "cu" not in crashed
    assert "wire" not in crashed
    assert "throughput" not in crashed
