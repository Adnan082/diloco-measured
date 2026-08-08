#!/usr/bin/env bash
# torchrun_multinode.sh — launch a training job across the 4 GPU nodes via the control
# node's rendezvous endpoint (CLAUDE.md §19.2).
#
# STATUS: [PROPOSED] scaffold, NOT YET IMPLEMENTED.
#
# Usage (planned): torchrun_multinode.sh <run_id> <spec_path>
# CONTRACT: this script is invoked by `measurement/train.py::run()`, never by hand during a
# real campaign (CLAUDE.md §19.3 — runs are strictly serial; never two experiments on the
# cluster at once).

set -euo pipefail

RUN_ID="${1:-}"
SPEC_PATH="${2:-}"

if [ -z "$RUN_ID" ] || [ -z "$SPEC_PATH" ]; then
  echo "usage: torchrun_multinode.sh <run_id> <spec_path>" >&2
  exit 2
fi

echo "[torchrun_multinode] STATUS: scaffold only, run_id=$RUN_ID spec=$SPEC_PATH" >&2
echo "[torchrun_multinode] ERROR: not implemented. See CLAUDE.md §19.2, §40 Q2/Q4." >&2
exit 1
