#!/bin/bash
set -euo pipefail

###############################################################################
# Run Autoscale End-to-End Test
#
# Single script that copies files, submits the DDP autoscale job, waits for
# completion, and retrieves results locally.
#
# Usage: ./scripts/run-autoscale-test.sh [options]
#
# Options:
#   --nodes N             Number of nodes to request (default: 2)
#   --namespace NS        Kubernetes namespace (default: slurm)
#   --results-dir DIR     Local directory for results (default: results/)
#   --timeout SECONDS     Max wait time for job completion (default: 600)
#   --help                Show this help message
###############################################################################

NAMESPACE="${NAMESPACE:-slurm}"
NODES=2
RESULTS_DIR="results"
TIMEOUT=600
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

while [[ $# -gt 0 ]]; do
  case $1 in
    --nodes)      NODES="$2"; shift 2 ;;
    --namespace)  NAMESPACE="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --timeout)    TIMEOUT="$2"; shift 2 ;;
    --help)
      sed -n '3,15p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $1"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)]${NC} $1"; }
section() { echo ""; echo -e "${BLUE}━━━ $1 ━━━${NC}"; echo ""; }

CONTROLLER="slurm-controller-0"

section "Autoscale End-to-End Test (${NODES} nodes)"

# ─── Step 1: Discover running workers ────────────────────────────────────────
log "Discovering running workers..."
WORKERS=$(oc get pods -n "$NAMESPACE" -l nodeset.slinky.slurm.net/name=slurm-worker-slinky \
  --field-selector=status.phase=Running -o jsonpath='{.items[*].metadata.name}')

if [ -z "$WORKERS" ]; then
  err "No running worker pods found in namespace $NAMESPACE"
  exit 1
fi
log "Found workers: $WORKERS"

# ─── Step 2: Copy files to cluster ───────────────────────────────────────────
section "Copying files to cluster"

log "Copying submit script to controller..."
oc cp "$REPO_ROOT/demos/submit_job_autoscale.sh" \
  "$NAMESPACE/$CONTROLLER:/tmp/submit_job_autoscale.sh" -c slurmctld

log "Copying training script to workers..."
for worker in $WORKERS; do
  oc cp "$REPO_ROOT/demos/ddp_test.py" "$NAMESPACE/$worker:/tmp/ddp_test.py" -c slurmd
  log "  → $worker"
done

# ─── Step 3: Submit the job ──────────────────────────────────────────────────
section "Submitting job"

SBATCH_OUTPUT=$(oc exec -n "$NAMESPACE" "$CONTROLLER" -c slurmctld -- \
  sbatch -N "$NODES" --ntasks="$NODES" /tmp/submit_job_autoscale.sh 2>&1)

JOB_ID=$(echo "$SBATCH_OUTPUT" | grep -oP 'Submitted batch job \K[0-9]+')

if [ -z "$JOB_ID" ]; then
  err "Failed to submit job: $SBATCH_OUTPUT"
  exit 1
fi
log "Submitted batch job $JOB_ID (requesting $NODES nodes)"

# ─── Step 4: Wait for job completion ─────────────────────────────────────────
section "Waiting for job $JOB_ID to complete (timeout: ${TIMEOUT}s)"

ELAPSED=0
POLL=10
LAST_STATE=""

while [ $ELAPSED -lt $TIMEOUT ]; do
  STATE=$(oc exec -n "$NAMESPACE" "$CONTROLLER" -c slurmctld -- \
    squeue -j "$JOB_ID" -h -o "%T" 2>/dev/null || echo "COMPLETED")

  if [ -z "$STATE" ] || [ "$STATE" = "COMPLETED" ]; then
    log "Job $JOB_ID finished."
    break
  fi

  if [ "$STATE" != "$LAST_STATE" ]; then
    log "Job $JOB_ID: $STATE (${ELAPSED}s elapsed)"
    LAST_STATE="$STATE"
  fi

  sleep $POLL
  ELAPSED=$((ELAPSED + POLL))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
  err "Timed out after ${TIMEOUT}s. Job $JOB_ID may still be running."
  err "Check manually: oc exec -n $NAMESPACE $CONTROLLER -c slurmctld -- squeue -j $JOB_ID"
  exit 1
fi

# ─── Step 5: Identify batch host ─────────────────────────────────────────────
section "Retrieving results"

BATCH_HOST=$(oc exec -n "$NAMESPACE" "$CONTROLLER" -c slurmctld -- \
  scontrol show job "$JOB_ID" 2>/dev/null | grep -oP 'BatchHost=\K\S+' || echo "")

if [ -z "$BATCH_HOST" ]; then
  warn "Could not determine BatchHost (job may have been purged). Trying slinky-0..."
  BATCH_HOST="slinky-0"
fi

# Map Slurm hostname to pod name (slinky-N → slurm-worker-slinky-N)
BATCH_POD="slurm-worker-${BATCH_HOST}"
log "Batch host: $BATCH_HOST (pod: $BATCH_POD)"

# ─── Step 6: Retrieve output files ───────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

log "Fetching job output..."
oc exec -n "$NAMESPACE" "$BATCH_POD" -c slurmd -- \
  cat "/tmp/ddp-elastic-${JOB_ID}.out" > "$RESULTS_DIR/job-${JOB_ID}.out" 2>/dev/null && \
  log "  → $RESULTS_DIR/job-${JOB_ID}.out" || \
  warn "  Output file not found on $BATCH_POD"

log "Fetching job error log..."
oc exec -n "$NAMESPACE" "$BATCH_POD" -c slurmd -- \
  cat "/tmp/ddp-elastic-${JOB_ID}.err" > "$RESULTS_DIR/job-${JOB_ID}.err" 2>/dev/null && \
  log "  → $RESULTS_DIR/job-${JOB_ID}.err" || \
  warn "  Error file not found on $BATCH_POD"

log "Fetching training artifacts..."
oc cp "$NAMESPACE/$BATCH_POD:/tmp/ddp-results" "$RESULTS_DIR/ddp-results" -c slurmd 2>/dev/null && \
  log "  → $RESULTS_DIR/ddp-results/" || \
  warn "  Training artifacts not found (job may have failed before saving)"

# ─── Step 7: Print summary ───────────────────────────────────────────────────
section "Summary"

if [ -f "$RESULTS_DIR/job-${JOB_ID}.out" ]; then
  # Extract key lines from output
  if grep -q "TEST PASSED" "$RESULTS_DIR/job-${JOB_ID}.out"; then
    echo -e "${GREEN}✓ JOB $JOB_ID PASSED${NC}"
  elif grep -q "RESULT: FAILED" "$RESULTS_DIR/job-${JOB_ID}.out"; then
    echo -e "${RED}✗ JOB $JOB_ID FAILED${NC}"
  else
    echo -e "${YELLOW}? JOB $JOB_ID - check output for details${NC}"
  fi
  echo ""
  grep -E "(Training Complete|Total time|Avg throughput|Scaling efficiency|Recommendation)" \
    "$RESULTS_DIR/job-${JOB_ID}.out" 2>/dev/null | sed 's/^/  /'
fi

echo ""
log "Results saved to: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR"/job-${JOB_ID}.* 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
[ -d "$RESULTS_DIR/ddp-results" ] && \
  find "$RESULTS_DIR/ddp-results" -type f | awk '{print "  " $0}' 2>/dev/null
