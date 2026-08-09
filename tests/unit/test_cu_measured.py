"""Unit tests for analysis/cu.py::measured(). analytic() is intentionally NOT tested here —
it stays unimplemented pending CLAUDE.md §40 Q3, see cu.py's module docstring.
"""

from __future__ import annotations

from collections import namedtuple

import pytest

from diloco_measured.analysis.cu import measured

StepRow = namedtuple("StepRow", ["compute_time_ms", "wall_time_ms"])


def _step(compute_time_ms: float, wall_time_ms: float, **extra) -> dict:
    return {"compute_time_ms": compute_time_ms, "wall_time_ms": wall_time_ms, **extra}


@pytest.mark.unit
def test_measured_known_value():
    steps = [_step(80, 100), _step(90, 100)]
    # sum(compute)=170, sum(wall)=200 -> CU = 0.85
    assert measured(steps, warmup=0) == pytest.approx(0.85)


@pytest.mark.unit
def test_measured_discards_warmup_steps():
    # Warmup steps have terrible CU (0.1); if not discarded they'd drag the result down.
    steps = [_step(10, 100), _step(10, 100), _step(90, 100), _step(90, 100)]
    result = measured(steps, warmup=2)
    assert result == pytest.approx(0.9)


@pytest.mark.unit
def test_measured_works_with_attribute_style_records():
    """Not every future StepRecords container will be a plain dict (CLAUDE.md §13.3) —
    confirm namedtuple/attribute-style rows work identically.
    """
    steps = [
        StepRow(compute_time_ms=80, wall_time_ms=100),
        StepRow(compute_time_ms=90, wall_time_ms=100),
    ]
    assert measured(steps, warmup=0) == pytest.approx(0.85)


@pytest.mark.unit
def test_measured_ignores_extra_fields_on_the_record():
    steps = [_step(80, 100, loss=1.23, tokens_processed=4096, is_sync_step=True)]
    assert measured(steps, warmup=0) == pytest.approx(0.8)


@pytest.mark.unit
def test_measured_raises_when_nothing_survives_warmup():
    steps = [_step(80, 100), _step(90, 100)]
    with pytest.raises(ValueError, match="no StepRecords remain"):
        measured(steps, warmup=5)


@pytest.mark.unit
def test_measured_raises_on_negative_warmup():
    with pytest.raises(ValueError, match="warmup"):
        measured([_step(80, 100)], warmup=-1)


@pytest.mark.unit
def test_measured_raises_on_zero_total_wall_time():
    steps = [_step(0, 0), _step(0, 0)]
    with pytest.raises(ValueError, match="wall_time_ms"):
        measured(steps, warmup=0)


@pytest.mark.unit
def test_measured_perfect_utilization_is_one():
    steps = [_step(100, 100), _step(100, 100)]
    assert measured(steps, warmup=0) == pytest.approx(1.0)
