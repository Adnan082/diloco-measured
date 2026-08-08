"""Shared pytest fixtures.

STATUS: [PROPOSED] scaffold.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SCHEMAS_DIR = Path(__file__).parents[1] / "src" / "diloco_measured" / "schemas"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def schemas_dir() -> Path:
    return SCHEMAS_DIR
