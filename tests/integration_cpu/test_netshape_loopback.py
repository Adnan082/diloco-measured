"""netshape gate on loopback / two cheap instances (CLAUDE.md §30.3).

Requesting a rate and measuring a different one must produce passed=False and no RunResult.
No mocks belong in this path (§30.6) — a mocked iperf3 would defeat the purpose of the gate.

STATUS: [PROPOSED] scaffold — skipped until measurement/netshape.py (Phase 0) exists.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration_cpu
@pytest.mark.skip(reason="Blocked on measurement/netshape.py — Phase 0")
def test_verification_gate_rejects_out_of_tolerance_rate():
    raise NotImplementedError


@pytest.mark.integration_cpu
@pytest.mark.skip(reason="Blocked on measurement/netshape.py — Phase 0")
def test_restore_is_idempotent():
    raise NotImplementedError
