#!/usr/bin/env bash
# teardown.sh — idempotent cluster teardown. MUST leave nothing billable (CLAUDE.md §14).
#
# STATUS: [PROPOSED] scaffold, NOT YET IMPLEMENTED.
#
# This script intentionally FAILS LOUD rather than exiting 0 while doing nothing — an
# orphaned 4-GPU cluster is called out as "the most likely real incident in this project"
# (CLAUDE.md §23, R13). A teardown script that silently no-ops is more dangerous than one
# that refuses to run at all.
#
# CONTRACT: idempotent — safe to run against an already-torn-down cluster, and against a
# partially-launched one.

set -euo pipefail

echo "[teardown] STATUS: scaffold only. Planned steps:"
echo "  1. Terminate 4x g6e.2xlarge + 1x c7i.2xlarge"
echo "  2. Release the placement group"
echo "  3. Verify: no orphaned volumes, IPs, or placement groups (§46 Operations checklist)"
echo "  4. Write final cost to results/environment/ (via cost_report.sh) before exiting"

echo "[teardown] ERROR: not implemented — DO NOT ASSUME THE CLUSTER IS DOWN." >&2
echo "[teardown] Verify manually in the AWS console until this script is real (Phase 0)." >&2
exit 1
