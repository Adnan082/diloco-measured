#!/usr/bin/env bash
# setup_node.sh — install pinned deps, lock GPU clocks, mount NVMe, sync dataset shards.
#
# STATUS: [PROPOSED] scaffold, NOT YET IMPLEMENTED.
# CLAUDE.md §40 Q4 is resolved (ADR-020): bare metal + uv lockfile, no Docker. This script
# is bare-metal by design — no container runtime step here.

set -euo pipefail

echo "[setup_node] STATUS: scaffold only. Planned steps (CLAUDE.md §9.1 Journey A):"
echo "  1. Install pinned deps from uv.lock (NOT YET GENERATED — Phase 0 work)"
echo "  2. Lock GPU clocks: nvidia-smi -lgc <min> <max>  (NFR-08 — recorded in fingerprint)"
echo "  3. Mount / verify 450GB local NVMe"
echo "  4. Sync tokenized shards S3 -> local NVMe"
echo "  5. Verify checksums (FR-03 precondition — abort bootstrap on mismatch, never partial)"

echo "[setup_node] ERROR: not implemented yet — this is still a Phase 0 scaffold." >&2
exit 1
