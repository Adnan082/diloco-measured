#!/usr/bin/env bash
# cost_report.sh — cumulative cluster-hours and estimated spend for CURRENTLY RUNNING
# resources (NFR-05, §26.4).
#
# Read-only. Reports on instances tagged Project=diloco-measured that are running/pending
# right now: how long each has been up and an estimated on-demand cost so far. Hourly rates
# below are the figures already in CLAUDE.md §5.2 — re-verify against the AWS Pricing API
# (see §40 Q2's research method for the pattern) if this script is more than a few weeks old,
# since on-demand pricing does change.
#
# STATUS: [PROPOSED] scope — this reports the CURRENT session's running cost, not a
# cross-session historical ledger (that needs a persistent log written at launch/teardown
# time, not yet built). This answers "is anything costing me money right now" every time
# it's run; it does not answer "what has this project cost in total since Day 0."
#
# Assumes GNU `date` (Git Bash on Windows and the target Ubuntu EC2 nodes both have it) —
# no BSD/macOS `date` fallback, since neither environment this actually runs in needs one.

set -euo pipefail

REGION="${DILOCO_REGION:-us-east-1}"
PROJECT_TAG="diloco-measured"

# CLAUDE.md §5.2 on-demand rates (us-east-1). [PROPOSED] — re-verify if stale.
GPU_RATE_PER_HR="2.24"
CONTROL_RATE_PER_HR="0.36"

AWS="aws --region $REGION"

log() { echo "[cost_report] $*" >&2; }

# Tab-separated: InstanceId, InstanceType, LaunchTime (ISO8601).
ROWS=$($AWS ec2 describe-instances \
  --filters "Name=tag:Project,Values=$PROJECT_TAG" "Name=instance-state-name,Values=pending,running" \
  --query "Reservations[].Instances[].[InstanceId,InstanceType,LaunchTime]" --output text)

if [ -z "$ROWS" ]; then
  log "no instances tagged Project=$PROJECT_TAG currently running/pending in $REGION."
  log "current burn rate: \$0.00/hr. (This does not mean the project has cost nothing"
  log "historically — only that nothing is billing RIGHT NOW.)"
  exit 0
fi

log "region=$REGION"
echo ""
printf "%-20s %-16s %-26s %8s %10s\n" "InstanceId" "Type" "LaunchTime(UTC)" "Hours" "Est.Cost"
printf "%-20s %-16s %-26s %8s %10s\n" "----------" "----" "---------------" "-----" "--------"

NOW_EPOCH=$(date -u +%s)
TOTAL_COST="0.00"
TOTAL_RATE="0.00"

while IFS=$'\t' read -r id itype launch; do
  [ -z "$id" ] && continue
  launch_epoch=$(date -u -d "$launch" +%s)
  hours=$(awk -v now="$NOW_EPOCH" -v then="$launch_epoch" 'BEGIN { printf "%.2f", (now - then) / 3600.0 }')

  case "$itype" in
    g6e.2xlarge) rate="$GPU_RATE_PER_HR" ;;
    c7i.2xlarge) rate="$CONTROL_RATE_PER_HR" ;;
    *)
      rate="0.00"
      log "WARNING: no known rate for instance type '$itype' — costed at \$0.00, report is INCOMPLETE"
      ;;
  esac

  cost=$(awk -v h="$hours" -v r="$rate" 'BEGIN { printf "%.2f", h * r }')
  printf "%-20s %-16s %-26s %8s %10s\n" "$id" "$itype" "$launch" "$hours" "\$$cost"

  TOTAL_COST=$(awk -v a="$TOTAL_COST" -v b="$cost" 'BEGIN { printf "%.2f", a + b }')
  TOTAL_RATE=$(awk -v a="$TOTAL_RATE" -v b="$rate" 'BEGIN { printf "%.2f", a + b }')
done <<< "$ROWS"

echo ""
log "current burn rate: \$$TOTAL_RATE/hr"
log "estimated cost of the current session so far: \$$TOTAL_COST"
log "budget ceiling (CLAUDE.md §5.1): ~\$650-800 for the WHOLE project — the number above is only the current session"
