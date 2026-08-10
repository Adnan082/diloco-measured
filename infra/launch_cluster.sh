#!/usr/bin/env bash
# launch_cluster.sh — provision 4x g6e.2xlarge + 1x c7i.2xlarge in one placement group.
#
# SAFETY: dry-run by default (validates AWS calls with EC2's own --dry-run flag — checks
# permissions and parameters, creates NOTHING, costs NOTHING). Real launch requires the
# explicit --launch-for-real flag AND DILOCO_OPERATOR_IP set. Once launched for real, the
# fleet bills ~$8.97/hr (CLAUDE.md §5.2) until `teardown.sh` runs — never invoke
# --launch-for-real without deliberately meaning to spend money right now.
#
# Idempotent: safe to re-run. Existing placement group / security group / key pair /
# tagged-and-running instances are reused, not duplicated.
#
# AMI IDs below were looked up live against this account on 2026-08-10 (Ubuntu 24.04 Deep
# Learning AMIs — chosen over Amazon Linux 2023 so setup_node.sh can assume `apt`, not
# `dnf`). AWS retires AMI IDs over time — RE-VERIFY before a real launch if this script is
# more than a couple of weeks old:
#   aws ec2 describe-images --owners amazon --region us-east-1 \
#     --filters "Name=name,Values=Deep Learning* PyTorch*Ubuntu*" "Name=state,Values=available" \
#     --query "sort_by(Images,&CreationDate)[-1].ImageId"

set -euo pipefail

# ---- Configuration (env-overridable) ---------------------------------------------------
REGION="${DILOCO_REGION:-us-east-1}"                      # CLAUDE.md ADR-024 — only region with quota
AZ="${DILOCO_AZ:-us-east-1a}"                              # ADR-024 primary; try 1b/1c/1d if this fails
GPU_AMI_ID="${DILOCO_GPU_AMI:-ami-0e2e1c9b9d71cc77f}"       # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.12 (Ubuntu 24.04) 20260725
CONTROL_AMI_ID="${DILOCO_CONTROL_AMI:-ami-052355af2a014bd2c}"  # ubuntu-noble-24.04-amd64-server-20260714 (plain, no GPU bloat)
GPU_INSTANCE_TYPE="g6e.2xlarge"
CONTROL_INSTANCE_TYPE="c7i.2xlarge"
# Default 4 (the real experimental topology, CLAUDE.md §5.2). Override for a cheaper
# connectivity/setup_node.sh test before committing to the full fleet, e.g.
# DILOCO_GPU_COUNT=1 (~$2.60/hr: 1x g6e.2xlarge + 1x c7i.2xlarge, vs ~$9.33/hr for all 4).
GPU_COUNT="${DILOCO_GPU_COUNT:-4}"
PROJECT_TAG="diloco-measured"
PLACEMENT_GROUP_NAME="${DILOCO_PG_NAME:-${PROJECT_TAG}-pg}"
SECURITY_GROUP_NAME="${DILOCO_SG_NAME:-${PROJECT_TAG}-sg}"
KEY_NAME="${DILOCO_KEY_NAME:-${PROJECT_TAG}-key}"
KEY_FILE="${DILOCO_KEY_FILE:-$HOME/.ssh/${PROJECT_TAG}-key.pem}"
# REQUIRED for --launch-for-real. Intentionally NOT auto-detected: this script may run from
# an environment (CI, a cloud sandbox, a jump host) whose outbound IP is not the operator's
# real IP, and guessing wrong here means either locking yourself out or opening SSH to the
# wrong address. Find yours with: curl -s https://checkip.amazonaws.com
OPERATOR_IP="${DILOCO_OPERATOR_IP:-}"

DRY_RUN=true
for arg in "$@"; do
  case "$arg" in
    --launch-for-real) DRY_RUN=false ;;
    --help|-h)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "[launch_cluster] Unknown argument: $arg (see --help)" >&2
      exit 2
      ;;
  esac
done

log() { echo "[launch_cluster] $*" >&2; }

# Run an AWS CLI command with --dry-run and report whether the permission check passed.
# NOTE ON A BUG THIS REPLACED: `aws ... --dry-run | grep -q DryRunOperation` looks
# reasonable but is wrong under `set -o pipefail` — the AWS CLI exits non-zero (e.g. 254)
# on a *successful* dry-run (that's how it signals "would have succeeded"), and pipefail
# propagates that non-zero exit through the pipeline regardless of whether grep matched,
# so every dry-run check would report PERMISSION CHECK FAILED even when permissions were
# fine. Fixed by capturing output first (with its exit code deliberately discarded) and
# grep-ing the captured string outside any pipe `set -o pipefail` can see.
check_dry_run() {
  local description="$1"
  shift
  local output
  output=$("$@" --dry-run 2>&1 || true)
  if echo "$output" | grep -q "DryRunOperation"; then
    log "[dry-run] $description: permission check OK"
    return 0
  fi
  log "[dry-run] $description: PERMISSION CHECK FAILED"
  log "$output"
  exit 1
}

# ---- Preconditions ----------------------------------------------------------------------
if ! command -v aws >/dev/null 2>&1; then
  log "ERROR: aws CLI not found on PATH."
  exit 1
fi

if [ "$DRY_RUN" = false ] && [ -z "$OPERATOR_IP" ]; then
  log "ERROR: --launch-for-real requires DILOCO_OPERATOR_IP (your public IP, for SSH scoping)."
  log "Find it with: curl -s https://checkip.amazonaws.com"
  log "Then:         DILOCO_OPERATOR_IP=<ip> $0 --launch-for-real"
  exit 1
fi

log "region=$REGION az=$AZ dry_run=$DRY_RUN gpu_count=$GPU_COUNT"
log "gpu_ami=$GPU_AMI_ID control_ami=$CONTROL_AMI_ID"

AWS="aws --region $REGION"

# ---- VPC / subnet (use the account's default VPC — CLAUDE.md doesn't call for a custom one) --
VPC_ID=$($AWS ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" --output text)
if [ "$VPC_ID" = "None" ] || [ -z "$VPC_ID" ]; then
  log "ERROR: no default VPC in $REGION. This script assumes one exists; create/specify a VPC manually if not."
  exit 1
fi
SUBNET_ID=$($AWS ec2 describe-subnets \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=availability-zone,Values=$AZ" "Name=default-for-az,Values=true" \
  --query "Subnets[0].SubnetId" --output text)
if [ "$SUBNET_ID" = "None" ] || [ -z "$SUBNET_ID" ]; then
  log "ERROR: no default subnet for $AZ in $VPC_ID."
  exit 1
fi
log "vpc=$VPC_ID subnet=$SUBNET_ID"

# ---- Security group: SSH from operator only, ALL traffic within the group ---------------
# CLAUDE.md §5.2: "The most common multi-node NCCL failure is a security group that only
# opens port 22." The self-referencing all-traffic rule below is what avoids that.
ensure_security_group() {
  local sg_id
  sg_id=$($AWS ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" "Name=vpc-id,Values=$VPC_ID" \
    --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)
  if [ -n "$sg_id" ] && [ "$sg_id" != "None" ]; then
    log "security group exists: $sg_id"
    echo "$sg_id"
    return
  fi

  if [ "$DRY_RUN" = true ]; then
    log "[dry-run] would create security group '$SECURITY_GROUP_NAME' in $VPC_ID"
    check_dry_run "create-security-group" $AWS ec2 create-security-group \
      --group-name "$SECURITY_GROUP_NAME" --description "$PROJECT_TAG cluster SG" \
      --vpc-id "$VPC_ID"
    echo "sg-DRYRUN"
    return
  fi

  sg_id=$($AWS ec2 create-security-group \
    --group-name "$SECURITY_GROUP_NAME" --description "$PROJECT_TAG cluster SG" \
    --vpc-id "$VPC_ID" --query "GroupId" --output text)
  $AWS ec2 create-tags --resources "$sg_id" --tags "Key=Project,Value=$PROJECT_TAG"

  # SSH from the operator's IP only.
  $AWS ec2 authorize-security-group-ingress --group-id "$sg_id" \
    --protocol tcp --port 22 --cidr "${OPERATOR_IP}/32"

  # ALL traffic (any protocol, any port) between members of this SG — required for NCCL
  # rendezvous, the lighthouse, and torchrun; self-referencing so it only ever applies
  # within our own cluster, never to the wider internet.
  $AWS ec2 authorize-security-group-ingress --group-id "$sg_id" \
    --protocol all --source-group "$sg_id"

  log "created security group: $sg_id"
  echo "$sg_id"
}

# ---- Key pair -----------------------------------------------------------------------------
ensure_key_pair() {
  local existing
  existing=$($AWS ec2 describe-key-pairs --key-names "$KEY_NAME" \
    --query "KeyPairs[0].KeyName" --output text 2>/dev/null || true)
  if [ -n "$existing" ] && [ "$existing" != "None" ]; then
    log "key pair exists: $KEY_NAME (local file expected at $KEY_FILE — this script cannot recover a lost private key; AWS only stores the public half)"
    return
  fi

  if [ "$DRY_RUN" = true ]; then
    log "[dry-run] would create key pair '$KEY_NAME'"
    return
  fi

  mkdir -p "$(dirname "$KEY_FILE")"
  $AWS ec2 create-key-pair --key-name "$KEY_NAME" --query "KeyMaterial" --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  log "created key pair: $KEY_NAME -> $KEY_FILE (back this up — AWS cannot reissue it)"
}

# ---- Placement group ------------------------------------------------------------------
ensure_placement_group() {
  local existing
  existing=$($AWS ec2 describe-placement-groups --group-names "$PLACEMENT_GROUP_NAME" \
    --query "PlacementGroups[0].GroupName" --output text 2>/dev/null || true)
  if [ -n "$existing" ] && [ "$existing" != "None" ]; then
    log "placement group exists: $PLACEMENT_GROUP_NAME"
    return
  fi

  if [ "$DRY_RUN" = true ]; then
    log "[dry-run] would create cluster placement group '$PLACEMENT_GROUP_NAME'"
    check_dry_run "create-placement-group" $AWS ec2 create-placement-group \
      --group-name "$PLACEMENT_GROUP_NAME" --strategy cluster
    return
  fi

  $AWS ec2 create-placement-group --group-name "$PLACEMENT_GROUP_NAME" --strategy cluster
  log "created placement group: $PLACEMENT_GROUP_NAME"
}

# ---- Instances -------------------------------------------------------------------------
# CLAUDE.md §29.4/ADR-025: everything on-demand, no spot, for now.
count_running() {
  local instance_type="$1"
  $AWS ec2 describe-instances \
    --filters "Name=tag:Project,Values=$PROJECT_TAG" "Name=instance-type,Values=$instance_type" \
              "Name=instance-state-name,Values=pending,running" \
    --query "length(Reservations[].Instances[])" --output text
}

launch_control_node() {
  local existing
  existing=$(count_running "$CONTROL_INSTANCE_TYPE")
  if [ "$existing" -ge 1 ]; then
    log "control node already running/pending ($existing) — skipping"
    return
  fi

  if [ "$DRY_RUN" = true ]; then
    # --key-name deliberately omitted here: in dry-run mode ensure_key_pair() doesn't
    # actually create the key pair (nothing should be created on a dry run), so a real key
    # name would make RunInstances fail on InvalidKeyPair.NotFound before it ever reaches
    # the permission check this is meant to validate. This checks "can this identity call
    # RunInstances with these parameters," not "does every referenced resource exist yet."
    check_dry_run "run-instances (control)" $AWS ec2 run-instances \
      --image-id "$CONTROL_AMI_ID" --instance-type "$CONTROL_INSTANCE_TYPE" --count 1 \
      --subnet-id "$SUBNET_ID"
    return
  fi

  $AWS ec2 run-instances \
    --image-id "$CONTROL_AMI_ID" --instance-type "$CONTROL_INSTANCE_TYPE" --count 1 \
    --subnet-id "$SUBNET_ID" --security-group-ids "$SG_ID" --key-name "$KEY_NAME" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$PROJECT_TAG},{Key=Role,Value=control}]" \
    --query "Instances[0].InstanceId" --output text
}

launch_gpu_nodes() {
  local existing
  existing=$(count_running "$GPU_INSTANCE_TYPE")
  if [ "$existing" -ge "$GPU_COUNT" ]; then
    log "GPU nodes already running/pending ($existing/$GPU_COUNT) — skipping"
    return
  fi

  if [ "$DRY_RUN" = true ]; then
    # --key-name and --placement's GroupName omitted for dry-run — same reasoning as
    # launch_control_node: neither the key pair nor the placement group necessarily exists
    # yet in dry-run mode, and this check validates RunInstances permission, not whether
    # every dependent resource has been created.
    check_dry_run "run-instances (gpu x$GPU_COUNT)" $AWS ec2 run-instances \
      --image-id "$GPU_AMI_ID" --instance-type "$GPU_INSTANCE_TYPE" --count "$GPU_COUNT" \
      --subnet-id "$SUBNET_ID" --placement "AvailabilityZone=$AZ"
    return
  fi

  $AWS ec2 run-instances \
    --image-id "$GPU_AMI_ID" --instance-type "$GPU_INSTANCE_TYPE" --count "$GPU_COUNT" \
    --subnet-id "$SUBNET_ID" --security-group-ids "$SG_ID" --key-name "$KEY_NAME" \
    --placement "GroupName=$PLACEMENT_GROUP_NAME,AvailabilityZone=$AZ" \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$PROJECT_TAG},{Key=Role,Value=gpu}]" \
    --query "Instances[].InstanceId" --output text
}

# ---- Main --------------------------------------------------------------------------------
SG_ID=$(ensure_security_group)
ensure_key_pair
ensure_placement_group
launch_control_node
launch_gpu_nodes

if [ "$DRY_RUN" = true ]; then
  log "DRY RUN COMPLETE — nothing was created, nothing was billed."
  log "Run with --launch-for-real (and DILOCO_OPERATOR_IP set) to actually launch."
else
  log "Launch requested. Poll status with:"
  log "  aws ec2 describe-instances --region $REGION --filters Name=tag:Project,Values=$PROJECT_TAG --query 'Reservations[].Instances[].{Id:InstanceId,Type:InstanceType,State:State.Name,IP:PublicIpAddress}' --output table"
  log "Remember: this fleet bills ~\$8.97/hr until infra/teardown.sh runs successfully."
fi
