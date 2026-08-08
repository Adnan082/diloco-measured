#!/usr/bin/env bash
# cost_report.sh — cumulative cluster-hours and estimated spend (NFR-05, §26.4).
#
# STATUS: [PROPOSED] scaffold, NOT YET IMPLEMENTED.
# Should be run hourly during cluster mode (§26.4: warn the operator at 80% of the ~$800
# budget ceiling, §5.1) and idle-warn if no run has executed in 30 minutes.

set -euo pipefail

echo "[cost_report] STATUS: scaffold only." >&2
echo "[cost_report] ERROR: not implemented. See CLAUDE.md §26.4, NFR-05." >&2
exit 1
