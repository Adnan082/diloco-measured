#!/usr/bin/env bash
# setup_node.sh — install pinned deps, lock GPU clocks, mount NVMe, sync dataset shards.
#
# STATUS: [PROPOSED] — written against the documented plan (CLAUDE.md §9.1 Journey A, ADR-020
# bare metal / ADR-024 AMI choice) but UNTESTED: it must run on a real Ubuntu 24.04 EC2 node
# with an NVIDIA GPU, which does not exist yet (no cluster has been launched). Unlike
# launch_cluster.sh/teardown.sh/cost_report.sh, which were run for real against this account
# this session, this script's correctness will only be confirmed on Day 1 (`make smoke`,
# CLAUDE.md §30.4) — treat every step here as a first draft to debug live, not a validated
# procedure.
#
# Runs ON each node (invoked via SSH from the operator, or eventually from launch_cluster.sh's
# bootstrap step — that wiring doesn't exist yet either). Assumes Ubuntu 24.04 (`apt`, not
# `dnf`) per ADR-024's AMI choice, and CUDA/NVIDIA drivers already present (baked into the
# Deep Learning AMI, not installed here).
#
# Idempotent where practical: safe to re-run, each step checks before acting.

set -euo pipefail

ROLE="${1:-gpu}"   # "gpu" or "control" — the control node skips GPU-specific steps
case "$ROLE" in
  gpu|control) ;;
  *) echo "[setup_node] usage: setup_node.sh [gpu|control]" >&2; exit 2 ;;
esac

log() { echo "[setup_node] $*" >&2; }

log "role=$ROLE"

# ---- 1. Pinned dependencies -------------------------------------------------------------
# uv.lock does not exist yet (Phase 0 work, CLAUDE.md pyproject.toml's own STATUS comment).
# Until it does, this installs from pyproject.toml directly — NOT a substitute for a real
# lockfile (version drift between nodes is exactly what NFR/R15 warns against); this is a
# placeholder so the rest of bootstrap is exercisable before the lockfile exists.
install_deps() {
  if ! command -v uv >/dev/null 2>&1; then
    log "installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  cd "$(dirname "$0")/.."   # repo root
  if [ -f uv.lock ]; then
    log "uv.lock found — installing from lockfile (reproducible)"
    uv sync --frozen
  else
    log "WARNING: no uv.lock yet — installing from pyproject.toml directly (NOT reproducible, R15 risk)"
    uv sync
  fi
}

# ---- 2. GPU clock lock (NFR-08) ---------------------------------------------------------
lock_gpu_clocks() {
  if [ "$ROLE" != "gpu" ]; then
    log "role=$ROLE — skipping GPU clock lock"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found on a GPU-role node. AMI is wrong or drivers didn't install."
    exit 1
  fi

  # Query the GPU's max supported graphics clock and lock to it — removes clock-speed
  # variance as a confound in throughput measurements (CLAUDE.md NFR-08).
  local max_clock
  max_clock=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits)
  log "locking GPU clock to ${max_clock}MHz (requires sudo)"
  sudo nvidia-smi -lgc "$max_clock,$max_clock"
  log "GPU clocks locked. Verify with: nvidia-smi -q -d CLOCK | grep -A3 'Clocks$'"
}

# ---- 3. Local NVMe ------------------------------------------------------------------------
mount_nvme() {
  local mount_point="/mnt/nvme"
  if mountpoint -q "$mount_point" 2>/dev/null; then
    log "NVMe already mounted at $mount_point"
    return
  fi

  # g6e.2xlarge's instance-store NVMe device name — VERIFY on Day 1: instance-store device
  # naming (/dev/nvme1n1 vs /dev/nvme2n1 etc.) depends on how many EBS volumes are also
  # attached and is not guaranteed stable across instance types; this is a best guess, not
  # a confirmed device path.
  local device="${DILOCO_NVME_DEVICE:-/dev/nvme1n1}"
  if [ ! -b "$device" ]; then
    log "ERROR: expected NVMe device $device not found. List actual block devices with"
    log "'lsblk' and re-run with DILOCO_NVME_DEVICE=<real device>."
    exit 1
  fi

  log "formatting and mounting $device at $mount_point (requires sudo)"
  sudo mkfs.ext4 -F "$device"
  sudo mkdir -p "$mount_point"
  sudo mount "$device" "$mount_point"
  sudo chown "$(whoami):$(whoami)" "$mount_point"
  log "NVMe mounted at $mount_point"
}

# ---- 4. Dataset sync (S3 -> local NVMe) --------------------------------------------------
sync_dataset() {
  local bucket="${DILOCO_S3_BUCKET:-}"
  if [ -z "$bucket" ]; then
    log "DILOCO_S3_BUCKET not set — skipping dataset sync (set it once the bucket exists;"
    log "dataset pre-tokenization is separate Phase 0 work, CLAUDE.md §35)"
    return
  fi
  local dest="/mnt/nvme/dataset"
  mkdir -p "$dest"
  log "syncing s3://$bucket/tokenized/ -> $dest"
  aws s3 sync "s3://$bucket/tokenized/" "$dest" --only-show-errors
}

# ---- 5. Checksum verification (FR-03 precondition) ---------------------------------------
verify_checksums() {
  local dest="/mnt/nvme/dataset"
  local manifest="$dest/CHECKSUMS.sha256"
  if [ ! -f "$manifest" ]; then
    log "no checksum manifest at $manifest — skipping verification (nothing synced, or"
    log "the tokenization step that produces this manifest hasn't run yet)"
    return
  fi
  log "verifying checksums against $manifest"
  (cd "$dest" && sha256sum -c "$manifest") \
    || { log "ERROR: checksum mismatch — do NOT proceed with a run against this shard set."; exit 1; }
  log "checksums verified OK"
}

install_deps
lock_gpu_clocks
mount_nvme
sync_dataset
verify_checksums

log "bootstrap complete for role=$ROLE"
