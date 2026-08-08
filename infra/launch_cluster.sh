#!/usr/bin/env bash
# launch_cluster.sh — provision 4x g6e.2xlarge + 1x c7i.2xlarge in one placement group.
#
# STATUS: [PROPOSED] scaffold, NOT YET IMPLEMENTED. This script currently only validates
# its environment and prints the plan; it takes no destructive action until AWS calls are
# filled in during Phase 0 (CLAUDE.md §35). Supports --dry-run per CLAUDE.md §21.
#
# CONTRACT (CLAUDE.md §29.2): idempotent — re-running against an already-launched cluster
# must not create duplicates. A full rebuild should take under 20 minutes.
# CONTRACT (§23): no credentials, account IDs, or ARNs may be echoed, logged, or committed.

set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
  esac
done

echo "[launch_cluster] STATUS: scaffold only — no AWS calls implemented yet."
echo "[launch_cluster] Plan (CLAUDE.md §5.2):"
echo "  - 1x cluster placement group, single AZ (CLAUDE.md §40 Q1 — PENDING, resolve first)"
echo "  - 4x g6e.2xlarge (1x L40S each, 8 vCPU, 64GiB RAM)"
echo "  - 1x c7i.2xlarge control node (0 GPU quota consumed)"
echo "  - security group: ALL traffic within group (CLAUDE.md §5.2 recommendation — the"
echo "    single most common multi-node NCCL failure is a security group that only opens 22)"

if [ "$DRY_RUN" = true ]; then
  echo "[launch_cluster] --dry-run: exiting without calling AWS."
  exit 0
fi

echo "[launch_cluster] ERROR: real launch path not implemented. Refusing to proceed." >&2
echo "[launch_cluster] Implement this in Phase 0 per CLAUDE.md §35 Phase 0 exit criteria" >&2
echo "[launch_cluster] ('dry-run the launcher') before it may run for real." >&2
exit 1
