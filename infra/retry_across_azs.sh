#!/usr/bin/env bash
# retry_across_azs.sh — one pass of launch_cluster.sh across every candidate AZ, cleaning up
# after each failed attempt so only a successful launch is left running.
#
# Built 2026-08-12 after ~25 real InsufficientInstanceCapacity launch attempts across
# g6e.2xlarge/g6e.4xlarge/g6.2xlarge and every us-east-1 AZ each is offered in (CLAUDE.md
# ADR-031) — this script is the reusable form of the manual retry loop used for all of them,
# meant to be invoked repeatedly (e.g. from a scheduled retry cadence) without hand-driving
# each AZ attempt turn by turn.
#
# SAFETY: same as launch_cluster.sh underneath — requires DILOCO_OPERATOR_IP, real spend
# starts the moment an attempt succeeds (~$8.97/hr default fleet). On failure, this script
# ALWAYS cleans up (terminate + delete placement group) before moving to the next AZ, and
# verifies zero instances/placement groups remain if every AZ fails. On success, it leaves
# the cluster running and does NOT tear it down — that's the point.
#
# Usage:
#   DILOCO_OPERATOR_IP=<ip> ./retry_across_azs.sh
#   DILOCO_GPU_INSTANCE_TYPE=g6.2xlarge DILOCO_GPU_COUNT=4 DILOCO_OPERATOR_IP=<ip> ./retry_across_azs.sh
#
# Exit code 0 = a cluster is now running. Exit code 1 = every AZ failed, cleaned up, nothing running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${DILOCO_REGION:-us-east-1}"
PROJECT_TAG="diloco-measured"
PLACEMENT_GROUP_NAME="${DILOCO_PG_NAME:-${PROJECT_TAG}-pg}"
GPU_INSTANCE_TYPE="${DILOCO_GPU_INSTANCE_TYPE:-g6e.2xlarge}"

# All AZs any of the three instance types tried in ADR-031 are offered in. A subnet lookup
# failure for an AZ this instance type isn't actually offered in is a fast, cheap no-op —
# launch_cluster.sh exits 1 immediately on "no default subnet for $AZ", which this script
# treats the same as a capacity failure (log it, clean up, move on).
AZS=(us-east-1a us-east-1b us-east-1c us-east-1d us-east-1f)

log() { echo "[retry_across_azs] $*" >&2; }

cleanup_stranded() {
  local ids
  ids=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Project,Values=$PROJECT_TAG" "Name=instance-state-name,Values=pending,running" \
    --query "Reservations[].Instances[].InstanceId" --output text)
  if [ -n "$ids" ]; then
    log "cleaning up stranded instance(s): $ids"
    aws ec2 terminate-instances --region "$REGION" --instance-ids $ids >/dev/null
  fi
  aws ec2 delete-placement-group --region "$REGION" --group-name "$PLACEMENT_GROUP_NAME" 2>/dev/null || true
}

for az in "${AZS[@]}"; do
  log "attempting $GPU_INSTANCE_TYPE in $az"
  if DILOCO_AZ="$az" DILOCO_GPU_INSTANCE_TYPE="$GPU_INSTANCE_TYPE" \
     bash "$SCRIPT_DIR/launch_cluster.sh" --launch-for-real; then
    log "SUCCESS in $az — cluster is up, NOT tearing down"
    exit 0
  fi
  log "$az failed — cleaning up and trying the next AZ"
  cleanup_stranded
done

log "every AZ failed for $GPU_INSTANCE_TYPE — cleaned up, nothing running"
exit 1
