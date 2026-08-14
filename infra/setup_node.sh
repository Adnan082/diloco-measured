#!/usr/bin/env bash
# setup_node.sh — install pinned deps, lock GPU clocks, mount NVMe, sync dataset shards.
#
# STATUS: [CONFIRMED] — run for real on a live 4x g6e.2xlarge + 1x c7i.2xlarge cluster,
# 2026-08-14 (ADR-032). Steps 1-4 (system packages, deps, GPU clock lock, NVMe mount) all
# verified clean on real hardware; step 1 (install_system_packages) was ADDED as a direct
# result of that run finding a real gap (see its own comment). Steps 5-6 (dataset sync,
# checksum verification) remain unexercised — no S3 bucket/tokenized corpus exists yet.
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

# ---- 1. System packages ------------------------------------------------------------------
# Found for real, 2026-08-14 (ADR-032): torchtitan's default attention backend (FlexAttention)
# JIT-compiles Triton kernels via torch.compile/Inductor, which shells out to `gcc` to build a
# small CUDA-launcher C extension that #includes Python.h. Neither the Deep Learning AMI's
# system Python nor its base package set includes the dev headers, so the FIRST real forward
# pass fails with "Python.h: No such file or directory" — not a torchtitan or torch bug, a
# missing system package. Installing it here, once, up front, rather than letting every run
# hit this the same way.
install_system_packages() {
  if [ "$ROLE" != "gpu" ]; then
    return   # control node never builds/runs a model, and isn't part of FR-01's iperf3 mesh
  fi

  local need_apt_update=false
  local to_install=""

  if ! python3 -c "import sysconfig,os; assert os.path.exists(os.path.join(sysconfig.get_path('include'),'Python.h'))" 2>/dev/null; then
    to_install="$to_install python3-dev gcc"
    need_apt_update=true
  fi
  # FR-01 precondition (CLAUDE.md): "iperf3 installed on all nodes" — needed on every GPU
  # node for the all-pairs bandwidth characterization; the control node isn't part of that
  # mesh (its own network isn't part of the experimental measurement) so this stays gated on
  # role=gpu, same as the compiler toolchain above.
  if ! command -v iperf3 >/dev/null 2>&1; then
    to_install="$to_install iperf3"
    need_apt_update=true
  fi

  if [ -z "$to_install" ]; then
    log "Python.h and iperf3 both already present — skipping system package install"
    return
  fi
  log "installing$to_install (requires sudo)"
  if [ "$need_apt_update" = true ]; then
    sudo apt-get update -qq
  fi
  # shellcheck disable=SC2086
  sudo apt-get install -y -qq $to_install
}

# ---- 2. Pinned dependencies -------------------------------------------------------------
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

# ---- 3. GPU clock lock (NFR-08) ---------------------------------------------------------
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

# ---- 4. Local NVMe ------------------------------------------------------------------------
# STATUS: [CONFIRMED] on the AMI pinned in launch_cluster.sh (Ubuntu 24.04 Deep Learning AMI)
# — verified for real on a live g6e.2xlarge, 2026-08-10: that AMI's boot process ALREADY sets
# up the instance-store NVMe as an LVM volume mounted at /opt/dlami/nvme, writable, ~391GB
# free on a 419GB device. The original version of this function assumed nothing was mounted
# and unconditionally ran `mkfs.ext4` on a guessed raw device (/dev/nvme1n1) — on THIS AMI
# that device is real but is the LVM physical volume backing /opt/dlami/nvme already, so that
# mkfs would have silently destroyed the AMI's own working setup instead of doing anything
# useful. Caught by checking `lsblk` on a real running instance before ever executing this
# function for real, not by reasoning about it in the abstract — reformat-on-guess is exactly
# the kind of hard-to-reverse action worth verifying against reality first.
mount_nvme() {
  local dlami_path="/opt/dlami/nvme"
  mount_point="${DILOCO_NVME_MOUNT:-$dlami_path}"   # not `local` — sync_dataset/verify_checksums read it too

  if mountpoint -q "$mount_point" 2>/dev/null; then
    log "NVMe already mounted at $mount_point (this AMI sets it up at boot — nothing to do)"
    if [ ! -w "$mount_point" ]; then
      log "WARNING: $mount_point is not writable by $(whoami) — dataset sync will fail. Check ownership."
    fi
    return
  fi

  # The control node (c7i.2xlarge) has NO instance storage at all — verified for real,
  # 2026-08-10: just an 8GB EBS root volume. That's expected, not an error (its job is
  # orchestration/tokenization, not the measurement runs that need fast local dataset
  # access) — skip gracefully rather than aborting the whole bootstrap over it.
  local device="${DILOCO_NVME_DEVICE:-/dev/nvme1n1}"
  if [ ! -b "$device" ]; then
    if [ "$ROLE" = "control" ]; then
      log "no instance storage on this control node (expected on c7i.2xlarge) — using \$HOME instead"
      mount_point="$HOME/diloco-data"
      mkdir -p "$mount_point"
      return
    fi
    log "ERROR: expected NVMe device $device not found on a GPU-role node. List actual block"
    log "devices with 'lsblk' and re-run with DILOCO_NVME_DEVICE=<real device>."
    exit 1
  fi

  # Fallback path for a GPU node whose AMI does NOT pre-mount instance storage (i.e. not the
  # AMI launch_cluster.sh pins). Manual mkfs+mount is only safe here because we've already
  # established the default mount point ($dlami_path) is NOT in use AND the device exists —
  # if DILOCO_NVME_DEVICE points at a device that turns out to already hold data on some
  # AMI/instance type nobody has checked yet, this will destroy it. VERIFY with `lsblk` by
  # hand before trusting this path on any AMI other than the one launch_cluster.sh pins.
  log "no pre-mounted instance storage found at $dlami_path — falling back to manual mkfs+mount"
  log "WARNING: about to mkfs.ext4 -F $device — this destroys anything currently on it."
  log "formatting and mounting $device at $mount_point (requires sudo)"
  sudo mkfs.ext4 -F "$device"
  sudo mkdir -p "$mount_point"
  sudo mount "$device" "$mount_point"
  sudo chown "$(whoami):$(whoami)" "$mount_point"
  log "NVMe mounted at $mount_point"
}

# ---- 5. Dataset sync (S3 -> local NVMe) --------------------------------------------------
sync_dataset() {
  local bucket="${DILOCO_S3_BUCKET:-}"
  if [ -z "$bucket" ]; then
    log "DILOCO_S3_BUCKET not set — skipping dataset sync (set it once the bucket exists;"
    log "dataset pre-tokenization is separate Phase 0 work, CLAUDE.md §35)"
    return
  fi
  local dest="$mount_point/dataset"
  mkdir -p "$dest"
  log "syncing s3://$bucket/tokenized/ -> $dest"
  aws s3 sync "s3://$bucket/tokenized/" "$dest" --only-show-errors
}

# ---- 6. Checksum verification (FR-03 precondition) ---------------------------------------
verify_checksums() {
  local dest="$mount_point/dataset"
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

install_system_packages
install_deps
lock_gpu_clocks
mount_nvme
sync_dataset
verify_checksums

log "bootstrap complete for role=$ROLE"
