"""Unsupported orchestration must reject before importing measurement code."""

import pytest
from typer.testing import CliRunner

from diloco_measured.cli import app

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--spec", "missing.yaml"],
        ["grid", "--config", "missing.yaml"],
        ["converge", "--spec", "missing.yaml"],
        ["plan", "--model", "30m"],
        ["network", "characterize"],
    ],
)
def test_unsupported_commands_are_explicit(arguments):
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 2
    assert "Unsupported unified command" in result.output
    assert "No infrastructure was changed" in result.output
