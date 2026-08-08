"""Environment fingerprinting (FR-08).

A result cannot be written without a complete fingerprint (CLAUDE.md §33.1.4). Every field
captured here is enumerated in schemas/run_result.v1.json#/$defs/EnvironmentFingerprint.

STATUS: [PROPOSED] scaffold.
"""

from __future__ import annotations


def capture() -> dict:
    """Capture the full environment fingerprint for the current run.

    Captures: harness git SHA + dirty flag, harness_version, PyTorch/torchtitan/torchft/
    NCCL/CUDA/driver versions, instance types, AZ, placement group ID, `nvidia-smi topo -m`,
    locked clock settings, NCCL_* environment variables, kernel version, `tc` qdisc dump,
    dataset shard checksum, random seeds.

    CONTRACT: must scrub account IDs, private IPs, ARNs, key names, and bucket names before
    returning (CLAUDE.md §23 — fingerprints are committed publicly).
    """
    raise NotImplementedError("Phase 0")


def scrub(fingerprint: dict) -> dict:
    """Remove AWS account IDs, private IPs, ARNs, key names, bucket names from a fingerprint.

    Called unconditionally by capture() before a fingerprint is ever written or returned to a
    caller that might log it. Easy to forget, embarrassing to fix later (CLAUDE.md §23).
    """
    raise NotImplementedError("Phase 0")
