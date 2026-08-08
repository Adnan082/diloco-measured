"""DiLoCo cross-implementation equivalence — the correctness backbone of ADR-003 (US-06).

Method: gloo, 4 processes, tiny model, fixed seed, H=4. Reference `diloco.py` and the
`torchft` path must produce loss curves agreeing within a documented tolerance over 200 steps.
A divergence beyond tolerance FAILS CI (CLAUDE.md §30.3).

STATUS: [PROPOSED] scaffold — skipped until measurement/diloco.py (Phase 0) and the torchft
pin (§40 Q2, PENDING) exist. This test is listed here, not merely promised in prose, so CI
surfaces its absence rather than silently having no coverage for the single riskiest
correctness claim in the project (R2).
"""

from __future__ import annotations

import pytest


@pytest.mark.integration_cpu
@pytest.mark.skip(reason="Blocked on measurement/diloco.py + CLAUDE.md §40 Q2 (torchft SHA pin) — Phase 0")
def test_reference_and_torchft_agree_within_tolerance():
    raise NotImplementedError


@pytest.mark.integration_cpu
@pytest.mark.skip(reason="Blocked on measurement/diloco.py — Phase 0")
def test_inner_optimizer_state_persists_across_rounds():
    """methods/diloco.md §3 invariant 1."""
    raise NotImplementedError


@pytest.mark.integration_cpu
@pytest.mark.skip(reason="Blocked on measurement/diloco.py — Phase 0")
def test_replicas_hold_bit_identical_theta_outer_after_outer_step():
    """methods/diloco.md §3 invariant 2."""
    raise NotImplementedError
