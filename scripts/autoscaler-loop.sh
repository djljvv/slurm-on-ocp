#!/bin/sh
set -u

# Slurm NodeSet Autoscaler — queue-driven scale-up + idle scale-down
#
# Scales the NodeSet to match Slurm queue demand. Works for any Slurm job —
# scaling is driven purely by squeue node demand (%D), not by a specific
# workload launcher.
#
# Provisioning (PyTorch + training scripts) is NOT handled here anymore.
# It's handled at the pod level: the NodeSet's init container (see
# configs/slurm-cluster.yaml) installs everything a worker needs BEFORE
# slurmd starts, so a node never registers with the Slurm controller — and
# therefore can never be scheduled onto — until it's actually ready. That
# removes the race this loop used to work around, and lets Kubernetes
# provision multiple new nodes in parallel instead of one at a time.

NAMESPACE="${NAMESPACE:-slurm}"
NODESET="${NODESET:-slurm-worker-slinky}"
CONTROLLER_POD="${CONTROLLER_POD:-}"
MIN_REPLICAS="${MIN_REPLICAS:-2}"
MAX_REPLICAS="${MAX_REPLICAS:-8}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
SCALE_DOWN_DELAY="${SCALE_DOWN_DELAY:-300}"

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
    squeue -h -t PENDING -o "%i %D"
}

get_running_info() {
  kubectl exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- \
    squeue -h -t RUNNING -o "%i %D"
}

max_job_nodes() {
  printf '%s\n%s\n' "$1" "$2" | awk 'NF >= 2 { n = $2 + 0; if (n > m) m = n } END { print m + 0 }'
}

# ---- Main loop ----
log "Slurm NodeSet Autoscaler (queue-driven scale-up + idle scale-down)"
log "  NodeSet:       $NODESET"
log "  Min replicas:  $MIN_REPLICAS"
log "  Max replicas:  $MAX_REPLICAS"
log "  Poll every:    ${POLL_INTERVAL}s"
log "  Scale-down:    after ${SCALE_DOWN_DELAY}s idle"
log ""

discover_controller
log "  Controller:    $CONTROLLER_POD"
log ""

while true; do
  discover_controller

  CURRENT=$(get_current_replicas 2>/dev/null || echo "")
  if [ -z "$CURRENT" ]; then
    log "WARNING: could not read NodeSet replicas, retrying..."
    sleep "$POLL_INTERVAL"
    continue
  fi

  NOW=$(date +%s)

  if ! PENDING_OUTPUT=$(get_pending_info 2>/dev/null); then
    log "WARNING: failed to query pending jobs (controller unreachable?), skipping cycle"
    sleep "$POLL_INTERVAL"
    continue
  fi

  PENDING_COUNT=0
  if [ -n "$PENDING_OUTPUT" ]; then
    PENDING_COUNT=$(printf '%s' "$PENDING_OUTPUT" | grep -c '^' 2>/dev/null || echo 0)
  fi

  if ! RUNNING_OUTPUT=$(get_running_info 2>/dev/null); then
    log "WARNING: failed to query running jobs (controller unreachable?), skipping cycle"
    sleep "$POLL_INTERVAL"
    continue
  fi

  RUNNING_COUNT=0
  if [ -n "$RUNNING_OUTPUT" ]; then
    RUNNING_COUNT=$(printf '%s' "$RUNNING_OUTPUT" | grep -c '^' 2>/dev/null || echo 0)
  fi

  MAX_DEMAND=$(max_job_nodes "$PENDING_OUTPUT" "$RUNNING_OUTPUT")
  if [ "$MAX_DEMAND" -gt 0 ]; then
    NEEDED=$MAX_DEMAND
    if [ "$NEEDED" -lt "$MIN_REPLICAS" ]; then
      NEEDED=$MIN_REPLICAS
    fi
    if [ "$NEEDED" -gt "$MAX_REPLICAS" ]; then
      log "WARNING: job demand ${NEEDED} nodes exceeds MAX_REPLICAS ${MAX_REPLICAS}, capping"
      NEEDED=$MAX_REPLICAS
    fi
    if [ "$CURRENT" -lt "$NEEDED" ]; then
      log "SCALE-UP: ${PENDING_COUNT} pending + ${RUNNING_COUNT} running need ${NEEDED} nodes (have ${CURRENT})"
      scale_nodeset "$NEEDED"
      CURRENT=$NEEDED
    fi
  fi

  if [ "$PENDING_COUNT" -gt 0 ] || [ "$RUNNING_COUNT" -gt 0 ]; then
    LAST_PENDING_TIME="$NOW"
    if [ "$PENDING_COUNT" -gt 0 ]; then
      log "ACTIVE: ${PENDING_COUNT} pending, ${RUNNING_COUNT} running (${CURRENT} replicas)"
    else
      log "BUSY: ${RUNNING_COUNT} running job(s) (${CURRENT} replicas)"
    fi
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
  fi

  sleep "$POLL_INTERVAL"
done
