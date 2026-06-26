#!/bin/sh
set -u

# Slurm NodeSet Autoscaler — Scale-Down Watchdog
#
# This loop runs in-cluster and handles:
#   - Scaling DOWN after an idle period (no pending jobs)
#   - Provisioning any newly-scaled workers with PyTorch
#
# Scale-UP is handled proactively by `python ddp_test.py --launch`
# which sizes the cluster BEFORE submitting jobs.

NAMESPACE="${NAMESPACE:-slurm}"
NODESET="${NODESET:-slurm-worker-slinky}"
CONTROLLER_POD="${CONTROLLER_POD:-}"
MIN_REPLICAS="${MIN_REPLICAS:-2}"
MAX_REPLICAS="${MAX_REPLICAS:-8}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
SCALE_DOWN_DELAY="${SCALE_DOWN_DELAY:-300}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

LAST_PENDING_TIME=""

log() { echo "[$(date '+%H:%M:%S')] $*"; }

discover_controller() {
  local pod
  pod=$(kubectl get pods -n "$NAMESPACE" \
    -l app.kubernetes.io/name=slurmctld \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [ -n "$pod" ]; then
    CONTROLLER_POD="$pod"
  elif [ -z "$CONTROLLER_POD" ]; then
    CONTROLLER_POD="slurm-controller-0"
  fi
}

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
  # Live check only — no file cache (survives pod restarts)
  kubectl exec -n "$NAMESPACE" "$1" -c slurmd -- \
    python3 -c "import torch" >/dev/null 2>&1
}

provision_worker() {
  local pod="$1"
  log "PROVISION: installing dependencies on ${pod}..."

  if ! kubectl exec -n "$NAMESPACE" "$pod" -c slurmd -- \
    bash -c "apt-get update -qq && apt-get install -y -qq python3-pip" >/dev/null 2>&1; then
    log "PROVISION: WARNING: pip install failed on ${pod}"
    return 1
  fi

  log "PROVISION: installing PyTorch on ${pod}..."
  if ! kubectl exec -n "$NAMESPACE" "$pod" -c slurmd -- \
    pip3 install --break-system-packages torch --index-url "$PYTORCH_INDEX" >/dev/null 2>&1; then
    log "PROVISION: WARNING: PyTorch install failed on ${pod}"
    return 1
  fi

  for script in ddp_test.py; do
    if [ -f "/scripts/${script}" ]; then
      kubectl cp "/scripts/${script}" "${NAMESPACE}/${pod}:/tmp/${script}" -c slurmd || {
        log "PROVISION: WARNING: failed to copy ${script} to ${pod}"
        return 1
      }
    fi
  done

  log "PROVISION: ${pod} ready"
  return 0
}

provision_new_workers() {
  local ready_pods
  ready_pods=$(kubectl get pods -n "$NAMESPACE" \
    -l "nodeset.slinky.slurm.net/name=${NODESET}" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

  local pids=""
  for pod in $ready_pods; do
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
      pids="$pids $!"
    fi
  done

  # Wait for all provisioners, tolerating individual failures
  local failed=0
  for pid in $pids; do
    wait "$pid" || failed=$((failed + 1))
  done
  [ "$failed" -eq 0 ] || log "WARNING: $failed worker(s) failed provisioning"
}

# ---- Main loop ----
log "Slurm NodeSet Autoscaler (scale-down watchdog)"
log "  NodeSet:       $NODESET"
log "  Min replicas:  $MIN_REPLICAS"
log "  Max replicas:  $MAX_REPLICAS"
log "  Poll every:    ${POLL_INTERVAL}s"
log "  Scale-down:    after ${SCALE_DOWN_DELAY}s idle"
log ""

discover_controller
log "  Controller:    $CONTROLLER_POD"
log ""

log "Provisioning any existing workers..."
provision_new_workers

PREV_REPLICAS=""

while true; do
  # Re-discover controller each iteration in case it restarted
  discover_controller

  CURRENT=$(get_current_replicas 2>/dev/null || echo "")
  if [ -z "$CURRENT" ]; then
    log "WARNING: could not read NodeSet replicas, retrying..."
    sleep "$POLL_INTERVAL"
    continue
  fi

  # Detect external scale-up (e.g. from --launch) and reset cooldown
  if [ -n "$PREV_REPLICAS" ] && [ "$CURRENT" -gt "$PREV_REPLICAS" ]; then
    log "EXTERNAL SCALE-UP detected: ${PREV_REPLICAS} -> ${CURRENT} (resetting cooldown)"
    LAST_PENDING_TIME=""
  fi
  PREV_REPLICAS="$CURRENT"

  PENDING_OUTPUT=$(get_pending_info 2>/dev/null || echo "")

  PENDING_COUNT=0
  RUNNING_COUNT=0
  if [ -n "$PENDING_OUTPUT" ]; then
    PENDING_COUNT=$(printf '%s' "$PENDING_OUTPUT" | grep -c '^' 2>/dev/null || echo 0)
  fi

  # Also check for running jobs — don't scale down while jobs are active
  RUNNING_OUTPUT=$(kubectl exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- \
    sh -c 'squeue -h -t RUNNING -o "%i" 2>/dev/null || echo ""' 2>/dev/null || echo "")
  if [ -n "$RUNNING_OUTPUT" ]; then
    RUNNING_COUNT=$(printf '%s' "$RUNNING_OUTPUT" | grep -c '^' 2>/dev/null || echo 0)
  fi

  NOW=$(date +%s)

  if [ "$PENDING_COUNT" -gt 0 ] || [ "$RUNNING_COUNT" -gt 0 ]; then
    LAST_PENDING_TIME="$NOW"
    if [ "$PENDING_COUNT" -gt 0 ]; then
      log "ACTIVE: ${PENDING_COUNT} pending, ${RUNNING_COUNT} running (${CURRENT} replicas)"
    else
      log "BUSY: ${RUNNING_COUNT} running job(s) (${CURRENT} replicas)"
    fi
    provision_new_workers
  else
    if [ -n "$LAST_PENDING_TIME" ]; then
      IDLE_FOR=$((NOW - LAST_PENDING_TIME))
      if [ "$IDLE_FOR" -ge "$SCALE_DOWN_DELAY" ] && [ "$CURRENT" -gt "$MIN_REPLICAS" ]; then
        log "IDLE: no jobs for ${IDLE_FOR}s, scaling down"
        scale_nodeset "$MIN_REPLICAS"
        LAST_PENDING_TIME=""
      elif [ "$CURRENT" -gt "$MIN_REPLICAS" ]; then
        REMAINING=$((SCALE_DOWN_DELAY - IDLE_FOR))
        log "COOLDOWN: no jobs, scale-down in ${REMAINING}s (${CURRENT} replicas)"
      else
        log "IDLE: no jobs, already at min (${CURRENT} replicas)"
      fi
    else
      if [ "$CURRENT" -lt "$MIN_REPLICAS" ]; then
        log "BELOW MIN: have ${CURRENT}, scaling to ${MIN_REPLICAS}"
        scale_nodeset "$MIN_REPLICAS"
      elif [ "$CURRENT" -gt "$MIN_REPLICAS" ]; then
        LAST_PENDING_TIME="$NOW"
        log "OVER MIN: ${CURRENT} replicas with no demand, starting cooldown"
      else
        log "IDLE: no jobs (${CURRENT} replicas)"
      fi
    fi

    provision_new_workers
  fi

  sleep "$POLL_INTERVAL"
done
