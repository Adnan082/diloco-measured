#!/usr/bin/env bash
# teardown.sh — idempotent cluster teardown. MUST leave nothing billable (CLAUDE.md §14).
#
# Terminates every EC2 instance tagged Project=diloco-measured (running or pending) and
# deletes the placement group once they're gone. Security group and key pair are left in
# place deliberately — neither is billable, and keeping them means the next launch_cluster.sh
# run doesn't need to re-authorize ingress rules or lose the private key.
#
# Safety pattern matches launch_cluster.sh: dry-run by default (lists what WOULD be
# terminated, terminates nothing), real termination requires --terminate-for-real.
# Unlike launch_cluster.sh, this script is safe to run often — an orphaned running cluster
# is explicitly called out as "the most likely real incident in this project" (CLAUDE.md
# §23, R13), so err on the side of finding this and running it for real.

set -euo pipefail

REGION="${DILOCO_REGION:-us-east-1}"
PROJECT_TAG="diloco-measured"
PLACEMENT_GROUP_NAME="${DILOCO_PG_NAME:-${PROJECT_TAG}-pg}"

DRY_RUN=true
for arg in "$@"; do
  case "$arg" in
    --terminate-for-real) DRY_RUN=false ;;
    --help|-h)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "[teardown] Unknown argument: $arg (see --help)" >&2
      exit 2
      ;;
  esac
done

log() { echo "[teardown] $*" >&2; }

AWS="aws --region $REGION"

log "region=$REGION dry_run=$DRY_RUN"

INSTANCE_IDS=$($AWS ec2 describe-instances \
  --filters "Name=tag:Project,Values=$PROJECT_TAG" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query "Reservations[].Instances[].InstanceId" --output text)

if [ -z "$INSTANCE_IDS" ]; then
  log "no instances tagged Project=$PROJECT_TAG in $REGION — nothing to terminate"
else
  # shellcheck disable=SC2086
  log "instances to terminate: $INSTANCE_IDS"
  if [ "$DRY_RUN" = true ]; then
    log "[dry-run] would terminate the instances listed above. Re-run with --terminate-for-real."
  else
    # shellcheck disable=SC2086
    $AWS ec2 terminate-instances --instance-ids $INSTANCE_IDS >/dev/null
    log "termination requested. Waiting for all instances to reach 'terminated'..."
    # shellcheck disable=SC2086
    $AWS ec2 wait instance-terminated --instance-ids $INSTANCE_IDS
    log "all instances terminated."
  fi
fi

PG_EXISTS=$($AWS ec2 describe-placement-groups --group-names "$PLACEMENT_GROUP_NAME" \
  --query "PlacementGroups[0].GroupName" --output text 2>/dev/null || true)

if [ -z "$PG_EXISTS" ] || [ "$PG_EXISTS" = "None" ]; then
  log "placement group '$PLACEMENT_GROUP_NAME' does not exist — nothing to delete"
elif [ "$DRY_RUN" = true ]; then
  log "[dry-run] would delete placement group '$PLACEMENT_GROUP_NAME' (only possible once all instances in it are terminated)"
else
  $AWS ec2 delete-placement-group --group-name "$PLACEMENT_GROUP_NAME"
  log "deleted placement group: $PLACEMENT_GROUP_NAME"
fi

if [ "$DRY_RUN" = true ]; then
  log "DRY RUN COMPLETE — nothing was terminated. Re-run with --terminate-for-real to actually tear down."
else
  log "Teardown complete. Verify nothing billable remains:"
  log "  aws ec2 describe-instances --region $REGION --filters Name=tag:Project,Values=$PROJECT_TAG --query 'Reservations[].Instances[].{Id:InstanceId,State:State.Name}' --output table"
fi
