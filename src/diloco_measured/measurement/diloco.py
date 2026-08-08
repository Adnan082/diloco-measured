"""Reference DiLoCo implementation — inner AdamW / outer Nesterov SGD.

This is OUR implementation (ADR-003), cross-validated against the `torchft` path because
torchft's semi-sync APIs are marked experimental (R2). See methods/diloco.md for the full
derivation and the invariants this module MUST satisfy — they are restated here as the
contract this file is responsible for upholding, not just documenting.

STATUS: [PROPOSED] scaffold.

INVARIANTS (methods/diloco.md §3 — each MUST have a corresponding test in tests/integration_cpu/):
  1. Inner optimizer state persists across outer rounds.
  2. All replicas hold bit-identical θ_outer after every outer step.
  3. Communication volume is O(N) per round, O(N/H) per step.
  4. With compression enabled, the error-feedback residual persists across rounds and is
     included in checkpoints.
"""

from __future__ import annotations


class DiLoCoTrainer:
    """Reference inner/outer training loop. See methods/diloco.md §1 for the pseudocode.

    STATUS: [PROPOSED] — constructor signature and step API are provisional pending Day 0
    implementation against a real torchtitan model.
    """

    def __init__(self, model, inner_optimizer_cfg: dict, outer_optimizer_cfg: dict, H: int):
        raise NotImplementedError("Phase 0 — see methods/diloco.md")

    def inner_step(self, local_batch) -> float:
        """One local AdamW step. Zero cross-replica communication. Returns the step loss."""
        raise NotImplementedError("Phase 0")

    def outer_step(self) -> None:
        """Compute the pseudo-gradient, all-reduce it, and apply the outer Nesterov SGD step.

        This is the ONLY cross-replica traffic in the training loop (methods/diloco.md §1).
        """
        raise NotImplementedError("Phase 0")
