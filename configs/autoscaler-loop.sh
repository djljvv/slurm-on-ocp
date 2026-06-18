#!/bin/sh
set -eu

# --- Configuration (override via env vars in the Deployment) ---
NAMESPACE="${NAMESPACE:-slurm}"
NODESET="${NODESET:-slurm-worker-slinky}"
CONTROLLER_POD="${CONTROLLER_POD:-slurm-controller-0}"
MIN_REPLICAS="${MIN_REPLICAS:-2}"
MAX_REPLICAS="${MAX_REPLICAS:-8}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
SCALE_DOWN_DELAY="${SCALE_DOWN_DELAY:-300}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

LAST_PENDING_TIME=""
PROVISIONED_FILE="/tmp/provisioned-workers"
touch "$PROVISIONED_FILE"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

get_current_replicas() {
  kubectl get nodeset "$NODESET" -n "$NAMESPACE" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null
}

scale_nodeset() {
  local desired="$1"
  log "SCALING: ${NODESET} -> ${desired} replicas"
  kubectl scale nodeset "$NODESET" -n "$NAMESPACE" --replicas="$desired"
}

get_pending_info() {
  kubectl exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- \
    sh -c 'squeue -h -t PENDING -o "%i %D" 2>/dev/null || echo ""'
}

is_provisioned() {
  grep -qx "$1" "$PROVISIONED_FILE" 2>/dev/null || return 1
  # Verify the pod actually has PyTorch (handles pod restarts with same name)
  kubectl exec -n "$NAMESPACE" "$1" -c slurmd -- \
    python3 -c "import torch" >/dev/null 2>&1
}

mark_provisioned() {
  echo "$1" >> "$PROVISIONED_FILE"
}

provision_worker() {
  local pod="$1"
  log "PROVISION: installing dependencies on ${pod}..."

  if ! kubectl exec -n "$NAMESPACE" "$pod" -c slurmd -- \
    bash -c "apt-get update -qq && apt-get install -y -qq python3-pip" >/dev/null 2>&1; then
    log "PROVISION: WARNING: pip install failed on ${pod}"
    return 1
  fi

  log "PROVISION: installing PyTorch on ${pod} (this takes a few minutes)..."
  if ! kubectl exec -n "$NAMESPACE" "$pod" -c slurmd -- \
    pip3 install --break-system-packages torch --index-url "$PYTORCH_INDEX" >/dev/null 2>&1; then
    log "PROVISION: WARNING: PyTorch install failed on ${pod}"
    return 1
  fi

  # Use cat|exec instead of kubectl cp to follow configmap symlinks
  for script in ddp_test.py submit_job.sh submit_job_autoscale.sh; do
    if [ -f "/scripts/${script}" ]; then
      cat "/scripts/${script}" | kubectl exec -n "$NAMESPACE" "$pod" -c slurmd -i -- sh -c "cat > /tmp/${script}" 2>/dev/null || true
    fi
  done

  log "PROVISION: ${pod} ready"
  mark_provisioned "$pod"
  return 0
}

provision_new_workers() {
  local ready_pods
  ready_pods=$(kubectl get pods -n "$NAMESPACE" \
    -l "nodeset.slinky.slurm.net/name=${NODESET}" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

  for pod in $ready_pods; do
    # Skip pods with a deletionTimestamp (being terminated during scale-down)
    local deleting
    deleting=$(kubectl get pod "$pod" -n "$NAMESPACE" \
      -o jsonpath='{.metadata.deletionTimestamp}' 2>/dev/null)
    if [ -n "$deleting" ]; then
      continue
    fi

    if ! is_provisioned "$pod"; then
      local ready_count
      ready_count=$(kubectl get pod "$pod" -n "$NAMESPACE" \
        -o jsonpath='{.status.containerStatuses[?(@.ready==true)].name} {.status.initContainerStatuses[?(@.ready==true)].name}' 2>/dev/null \
        | tr ' ' '\n' | grep -c . 2>/dev/null)
      ready_count="${ready_count:-0}"
      if [ "$ready_count" -lt 2 ]; then
        continue
      fi
      provision_worker "$pod" &
    fi
  done
  wait
}

# Also copy submit scripts to the controller
provision_controller() {
  log "PROVISION: copying submit scripts to controller..."
  # Use cat|exec instead of kubectl cp to follow configmap symlinks
  for script in submit_job.sh submit_job_autoscale.sh; do
    if [ -f "/scripts/${script}" ]; then
      cat "/scripts/${script}" | kubectl exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -i -- sh -c "cat > /tmp/${script}" 2>/dev/null || true
    fi
  done
  log "PROVISION: controller ready"
}

# ---- Main loop ----
log "Slurm NodeSet Autoscaler started (with auto-provisioning)"
log "  NodeSet:       $NODESET"
log "  Min replicas:  $MIN_REPLICAS"
log "  Max replicas:  $MAX_REPLICAS"
log "  Poll every:    ${POLL_INTERVAL}s"
log "  Scale-down:    after ${SCALE_DOWN_DELAY}s idle"
log "  PyTorch index: $PYTORCH_INDEX"
log ""

log "Provisioning controller and existing workers..."
provision_controller
provision_new_workers

while true; do
  CURRENT=$(get_current_replicas 2>/dev/null || echo "")
  if [ -z "$CURRENT" ]; then
    log "WARNING: could not read NodeSet replicas, retrying..."
    sleep "$POLL_INTERVAL"
    continue
  fi

  PENDING_OUTPUT=$(get_pending_info 2>/dev/null || echo "")

  PENDING_COUNT=0
  MAX_NODES_NEEDED=0
  if [ -n "$PENDING_OUTPUT" ]; then
    PENDING_COUNT=$(echo "$PENDING_OUTPUT" | wc -l | tr -d ' ')
    MAX_NODES_NEEDED=$(echo "$PENDING_OUTPUT" | awk '{if($2>m)m=$2} END{print m+0}')
  fi

  NOW=$(date +%s)

  if [ "$PENDING_COUNT" -gt 0 ]; then
    LAST_PENDING_TIME="$NOW"
    DESIRED="$MAX_NODES_NEEDED"
    [ "$DESIRED" -lt "$MIN_REPLICAS" ] && DESIRED="$MIN_REPLICAS"
    [ "$DESIRED" -gt "$MAX_REPLICAS" ] && DESIRED="$MAX_REPLICAS"

    if [ "$DESIRED" -gt "$CURRENT" ]; then
      log "DEMAND: ${PENDING_COUNT} pending job(s), largest needs ${MAX_NODES_NEEDED} node(s), have ${CURRENT}"
      scale_nodeset "$DESIRED"
    else
      log "OK: ${PENDING_COUNT} pending, ${CURRENT} replicas sufficient"
    fi

    provision_new_workers
  else
    if [ -n "$LAST_PENDING_TIME" ]; then
      IDLE_FOR=$((NOW - LAST_PENDING_TIME))
      if [ "$IDLE_FOR" -ge "$SCALE_DOWN_DELAY" ] && [ "$CURRENT" -gt "$MIN_REPLICAS" ]; then
        log "IDLE: no pending jobs for ${IDLE_FOR}s, scaling down"
        scale_nodeset "$MIN_REPLICAS"
        LAST_PENDING_TIME=""
      elif [ "$CURRENT" -gt "$MIN_REPLICAS" ]; then
        REMAINING=$((SCALE_DOWN_DELAY - IDLE_FOR))
        log "COOLDOWN: no pending jobs, scale-down in ${REMAINING}s (${CURRENT} replicas)"
      else
        log "IDLE: no pending jobs, already at min (${CURRENT} replicas)"
      fi
    else
      if [ "$CURRENT" -lt "$MIN_REPLICAS" ]; then
        log "BELOW MIN: have ${CURRENT}, scaling to ${MIN_REPLICAS}"
        scale_nodeset "$MIN_REPLICAS"
      else
        log "IDLE: no pending jobs (${CURRENT} replicas)"
      fi
    fi

    provision_new_workers
  fi

  sleep "$POLL_INTERVAL"
done
