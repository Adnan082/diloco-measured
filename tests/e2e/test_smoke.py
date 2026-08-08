"""`make smoke` — the gate before every campaign (CLAUDE.md §30.4).

4 real nodes, ~1M-parameter model, 20 steps, one shaped bandwidth level. Asserts: cluster
reachable, shaping verified, run completes, a schema-valid RunResult is emitted, network
restored. If this fails, nothing else runs.

STATUS: [PROPOSED] scaffold. Requires a live 4-node cluster — never runs in CI (pyproject.toml
marks `e2e` accordingly). Skipped unconditionally until Phase 1.
"""

from __future__ import annotations

import pytest


@pytest.mark.e2e
@pytest.mark.skip(reason="Requires a live 4-node cluster — not runnable until Phase 1 (CLAUDE.md §35)")
def test_smoke_run_completes_and_emits_valid_run_result():
    raise NotImplementedError
