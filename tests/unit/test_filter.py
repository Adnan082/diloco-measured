"""Unit tests for analysis/filter.py — exclusion rules, per methods/measurement_windows.md §3.

Every test checks BOTH the partition (kept vs. excluded) and the report counts, since
CLAUDE.md §25.3 requires exclusions to be counted, never just silently dropped.
"""

from __future__ import annotations

import pytest

from diloco_measured.analysis.filter import apply


def _record(**overrides) -> dict:
    base = {
        "run_id": "r0",
        "status": "completed",
        "harness_version": "v1",
        "loader_bound_warning": False,
        "cu": {
            "total_s": 100.0,
            "compute_s": 80.0,
            "sync_blocked_s": 15.0,
            "optimizer_s": 3.0,
            "loader_stall_s": 2.0,
        },
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_all_completed_records_are_kept():
    records = [_record(run_id=f"r{i}") for i in range(5)]
    kept, report = apply(records)
    assert len(kept) == 5
    assert report.total == 5
    assert report.kept == 5
    assert report.excluded_total == 0


@pytest.mark.unit
def test_crashed_and_diverged_are_counted_in_separate_buckets():
    records = [
        _record(run_id="a", status="crashed"),
        _record(run_id="b", status="diverged"),
        _record(run_id="c", status="oom"),
        _record(run_id="d", status="completed"),
    ]
    kept, report = apply(records)
    assert [r["run_id"] for r in kept] == ["d"]
    assert report.excluded_crashed == 1
    assert report.excluded_diverged == 1
    assert report.excluded_other_status == 1  # oom
    assert report.excluded_total == 3


@pytest.mark.unit
def test_version_mismatch_excluded_unless_allowed():
    records = [_record(run_id="a", harness_version="v1"), _record(run_id="b", harness_version="v2")]

    kept, report = apply(records, harness_version="v1")
    assert [r["run_id"] for r in kept] == ["a"]
    assert report.excluded_version_mismatch == 1

    kept, report = apply(records, harness_version="v1", allow_version_mix=True)
    assert len(kept) == 2
    assert report.excluded_version_mismatch == 0


@pytest.mark.unit
def test_version_check_disabled_when_harness_version_is_none():
    records = [_record(run_id="a", harness_version="v1"), _record(run_id="b", harness_version="v2")]
    kept, report = apply(records, harness_version=None)
    assert len(kept) == 2
    assert report.excluded_version_mismatch == 0


@pytest.mark.unit
def test_loader_bound_records_excluded():
    records = [_record(run_id="a", loader_bound_warning=True), _record(run_id="b")]
    kept, report = apply(records)
    assert [r["run_id"] for r in kept] == ["b"]
    assert report.excluded_loader_bound == 1


@pytest.mark.unit
def test_reconciliation_failure_excluded():
    bad_cu = {
        "total_s": 100.0,
        "compute_s": 10.0,  # way under-accounted -> large residual
        "sync_blocked_s": 0.0,
        "optimizer_s": 0.0,
        "loader_stall_s": 0.0,
    }
    records = [_record(run_id="a", cu=bad_cu), _record(run_id="b")]
    kept, report = apply(records)
    assert [r["run_id"] for r in kept] == ["b"]
    assert report.excluded_reconciliation_failed == 1


@pytest.mark.unit
def test_reconciliation_within_tolerance_is_kept():
    # 100 total vs 99 accounted-for => 1% residual, well under the 5% default tolerance.
    ok_cu = {
        "total_s": 100.0,
        "compute_s": 80.0,
        "sync_blocked_s": 15.0,
        "optimizer_s": 3.0,
        "loader_stall_s": 1.0,
    }
    kept, report = apply([_record(cu=ok_cu)])
    assert len(kept) == 1
    assert report.excluded_reconciliation_failed == 0


@pytest.mark.unit
def test_missing_cu_field_does_not_crash_and_is_kept():
    record = _record()
    del record["cu"]
    kept, report = apply([record])
    assert len(kept) == 1


@pytest.mark.unit
def test_exclusion_order_first_match_wins():
    """A crashed + loader-bound + version-mismatched record should only ever be counted once
    (as crashed, the first check), not double-counted across buckets.
    """
    record = _record(
        status="crashed", loader_bound_warning=True, harness_version="v_old"
    )
    kept, report = apply([record], harness_version="v1")
    assert kept == []
    assert report.excluded_crashed == 1
    assert report.excluded_loader_bound == 0
    assert report.excluded_version_mismatch == 0
    assert report.excluded_total == 1
