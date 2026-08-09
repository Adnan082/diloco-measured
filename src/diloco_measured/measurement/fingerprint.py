"""Environment fingerprinting (FR-08).

A result cannot be written without a complete fingerprint (CLAUDE.md §33.1.4). Every field
captured here is enumerated in `schemas/run_result.v1.json#/$defs/EnvironmentFingerprint`.

This module does best-effort LOCAL capture: git state, installed package versions, and
whatever GPU/driver/cloud info is reachable from wherever it runs. Fields that genuinely
require the real cluster (instance type, AZ, driver version, the NCCL runtime version, `tc`
qdisc state) fall back to an explicit `"unknown"` sentinel (or an empty list, for
`instance_types`) when unreachable — never a silent `None`, since the schema requires these
as non-nullable strings, and never a guess. Running this on the actual GPU nodes during
Phase 1 is what makes those fields real; running it anywhere else (a laptop, CI) is what
makes the function itself testable without a cluster.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

UNKNOWN = "unknown"
NCCL_ENV_PREFIX = "NCCL_"
IMDS_BASE = "http://169.254.169.254/latest/meta-data"
# Local network hop only when actually on EC2; must never hang a run on a laptop where the
# link-local address simply doesn't respond.
IMDS_TIMEOUT_S = 0.5


def _run(argv: list[str], timeout_s: float = 5.0) -> str | None:
    """Run `argv`, return stripped stdout, or None on any failure (missing binary, nonzero
    exit, timeout). Never raises — every caller here is a best-effort probe, not a
    precondition check (those live in `netshape.py`/`spec.py`, which DO raise).
    """
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def _git_sha_and_dirty(repo_root: Path) -> tuple[str, bool]:
    sha = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(repo_root), "status", "--porcelain"])
    return (sha or UNKNOWN), bool(status)


def _package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def _pytorch_cuda_nccl_versions() -> tuple[str, str, str]:
    try:
        import torch
    except ImportError:
        return UNKNOWN, UNKNOWN, UNKNOWN

    pytorch_version = torch.__version__ or UNKNOWN
    cuda_version = torch.version.cuda or "cpu"
    try:
        nccl = torch.cuda.nccl.version()
        nccl_version = ".".join(str(part) for part in nccl) if nccl else UNKNOWN
    except Exception:
        nccl_version = UNKNOWN
    return pytorch_version, cuda_version, nccl_version


def _nvidia_driver_version() -> str:
    out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if not out:
        return UNKNOWN
    return out.splitlines()[0].strip()


def _nvidia_smi_topo() -> str | None:
    return _run(["nvidia-smi", "topo", "-m"])


def _imds_get(path: str) -> str | None:
    """Query the EC2 Instance Metadata Service. Returns None off-EC2 or on any error —
    IMDSv1 GET is used here for simplicity; a real deployment should prefer IMDSv2 (token-
    based) — [PROPOSED], revisit before Phase 1 if the launch AMI enforces IMDSv2-only.
    """
    try:
        with urllib.request.urlopen(f"{IMDS_BASE}/{path}", timeout=IMDS_TIMEOUT_S) as resp:
            return resp.read().decode().strip()
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _nccl_env() -> dict:
    return {k: v for k, v in os.environ.items() if k.startswith(NCCL_ENV_PREFIX)}


def capture(
    *,
    seed: int,
    dataset_shard_checksum: str,
    gpu_clocks_locked: bool,
    harness_version: str | None = None,
    repo_root: Path | str | None = None,
) -> dict:
    """Capture the full environment fingerprint for the current run.

    `seed`, `dataset_shard_checksum`, and `gpu_clocks_locked` describe THIS run's
    configuration rather than the machine, so they are not locally detectable — the caller
    must supply them. `harness_version` defaults to the installed package version;
    `repo_root` defaults to the repository containing this file.

    CONTRACT: always returns a schema-shape-complete dict (every required field present with
    the correct type). Undetectable string fields become the `"unknown"` sentinel rather than
    `None` (the schema requires them as non-nullable strings); undetectable `instance_types`
    becomes `[]` rather than a guess. The result is always passed through `scrub()` before
    being returned — callers must not skip that step by reaching into internals.
    """
    resolved_repo_root = Path(repo_root) if repo_root is not None else _default_repo_root()

    harness_git_sha, harness_dirty = _git_sha_and_dirty(resolved_repo_root)
    pytorch_version, cuda_version, nccl_version = _pytorch_cuda_nccl_versions()
    instance_type = _imds_get("instance-type")

    fingerprint: dict = {
        "harness_git_sha": harness_git_sha,
        "harness_dirty": harness_dirty,
        "harness_version": harness_version or _package_version("diloco_measured") or UNKNOWN,
        "pytorch_version": pytorch_version,
        "nccl_version": nccl_version,
        "cuda_version": cuda_version,
        "driver_version": _nvidia_driver_version(),
        "instance_types": [instance_type] if instance_type else [],
        "az": _imds_get("placement/availability-zone") or UNKNOWN,
        "gpu_clocks_locked": gpu_clocks_locked,
        "nccl_env": _nccl_env(),
        "kernel_version": platform.release() or UNKNOWN,
        "dataset_shard_checksum": dataset_shard_checksum,
        "seed": seed,
    }

    torchtitan_version = _package_version("torchtitan")
    if torchtitan_version:
        fingerprint["torchtitan_version"] = torchtitan_version
    torchft_version = _package_version("torchft")
    if torchft_version:
        fingerprint["torchft_version"] = torchft_version
    placement_group_id = _imds_get("placement/group-name")
    if placement_group_id:
        fingerprint["placement_group_id"] = placement_group_id
    topo = _nvidia_smi_topo()
    if topo:
        fingerprint["nvidia_smi_topo"] = topo

    return scrub(fingerprint)


def _default_repo_root() -> Path:
    # src/diloco_measured/measurement/fingerprint.py -> repo root is 3 parents up.
    return Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------------------
# Scrubbing (CLAUDE.md §23) — fingerprints are committed publicly.
# ---------------------------------------------------------------------------------------

_ACCOUNT_ID_RE = re.compile(r"\b\d{12}\b")
_ARN_RE = re.compile(r"arn:aws:[A-Za-z0-9\-:/_.*]+")
_PRIVATE_IP_RE = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
    r"|192\.168\.\d{1,3}\.\d{1,3})\b"
)


def scrub(fingerprint: dict) -> dict:
    """Remove AWS account IDs, ARNs, and private IPv4 addresses from a fingerprint's string
    values, recursively.

    Called unconditionally by `capture()` before a fingerprint is ever returned. Easy to
    forget, embarrassing to fix later (CLAUDE.md §23).

    Scope: pattern-based scrubbing of string VALUES only. Key names and S3 bucket names are
    free-form and not reliably regex-detectable — this is a safety net for the fields that
    ARE detectable (account IDs, ARNs, RFC1918 private IPs showing up in e.g.
    `nvidia_smi_topo` or `nccl_env` values), not a substitute for not putting secrets in
    those fields in the first place.
    """

    def _scrub_value(value):
        if isinstance(value, str):
            value = _ARN_RE.sub("<redacted-arn>", value)
            value = _ACCOUNT_ID_RE.sub("<redacted-account-id>", value)
            value = _PRIVATE_IP_RE.sub("<redacted-ip>", value)
            return value
        if isinstance(value, list):
            return [_scrub_value(v) for v in value]
        if isinstance(value, dict):
            return {k: _scrub_value(v) for k, v in value.items()}
        return value

    return {key: _scrub_value(value) for key, value in fingerprint.items()}
