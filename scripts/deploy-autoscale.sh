#!/bin/bash
set -euo pipefail

###############################################################################
# Slurm Autoscaler — Setup & Run
#
# Single entry point that:
#   1. Ensures the autoscaler infrastructure is deployed (ConfigMap, RBAC, SCC)
#   2. Submits a self-sizing DDP job that calculates minimum nodes from dataset
#      size and pod memory limits
#
# The autoscaler loop (autoscaler-loop.sh) detects pending jobs and scales
# the NodeSet up/down. This script just sets up that machinery and submits work.
#
# Usage:
#   ./scripts/deploy-autoscale.sh                          # Deploy + submit default job
#   ./scripts/deploy-autoscale.sh --setup-only             # Deploy autoscaler, don't submit
#   ./scripts/deploy-autoscale.sh --submit-only            # Submit job (assumes deployed)
#   NUM_SAMPLES=16384 ./scripts/deploy-autoscale.sh        # Larger dataset → more nodes
#   INTENSITY=light ./scripts/deploy-autoscale.sh          # Light mode (1 node sufficient)
#
# Environment variables:
#   NAMESPACE         Kubernetes namespace (default: slurm)
#   NUM_SAMPLES       Dataset size (default: 8192)
#   INTENSITY         light or medium (default: medium)
#   POD_MEM_LIMIT_MB  Pod memory limit in MB (default: 4096)
#   OVERHEAD_MB       PyTorch/model overhead in MB (default: 1200)
#   MAX_NODES         Maximum nodes to request (default: 8)
#   EPOCHS            Training epochs (default: 5)
#   BATCH_SIZE        Per-rank batch size (default: 64)
###############################################################################

NAMESPACE="${NAMESPACE:-slurm}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SETUP_ONLY=false
SUBMIT_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --setup-only)  SETUP_ONLY=true; shift ;;
    --submit-only) SUBMIT_ONLY=true; shift ;;
    --namespace)   NAMESPACE="$2"; shift 2 ;;
    --help|-h)
      sed -n '3,30p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

CONTROLLER_POD="slurm-controller-0"

# ─── Colors ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $1"; }
section() { echo ""; echo -e "${BLUE}━━━ $1 ━━━${NC}"; echo ""; }

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Setup — Ensure autoscaler infrastructure is deployed
# ═══════════════════════════════════════════════════════════════════════════════

setup_autoscaler() {
  section "Setting up autoscaler"

  # Check if already running
  if oc get deployment slurm-autoscaler -n "$NAMESPACE" &>/dev/null; then
    READY=$(oc get deployment slurm-autoscaler -n "$NAMESPACE" \
      -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
    if [ "${READY:-0}" -ge 1 ]; then
      log "Autoscaler already running — updating ConfigMap..."
    fi
  fi

  # Create/update ConfigMap from source files
  log "Creating ConfigMap 'slurm-autoscaler-script'..."
  oc create configmap slurm-autoscaler-script -n "$NAMESPACE" \
    --from-file=autoscaler.sh="${SCRIPT_DIR}/autoscaler-loop.sh" \
    --from-file=ddp_test.py="${REPO_ROOT}/demos/ddp_test.py" \
    --from-file=submit_job_autoscale.sh="${SCRIPT_DIR}/submit_job_autoscale.sh" \
    --dry-run=client -o yaml | oc apply -f -

  # Apply RBAC + Deployment
  log "Applying autoscaler RBAC and Deployment..."
  oc apply -f "${REPO_ROOT}/configs/slurm-autoscaler.yaml"

  # Grant privileged SCC (required to exec into worker pods)
  log "Granting privileged SCC to autoscaler service account..."
  oc adm policy add-scc-to-user privileged -z slurm-autoscaler -n "$NAMESPACE" 2>/dev/null || true

  # Wait for pod to be ready
  log "Waiting for autoscaler pod..."
  oc wait --for=condition=available deployment/slurm-autoscaler \
    -n "$NAMESPACE" --timeout=120s 2>/dev/null || {
    warn "Autoscaler may still be starting"
  }

  log "Autoscaler is deployed and running"
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Submit — Calculate nodes needed and submit self-sizing job
# ═══════════════════════════════════════════════════════════════════════════════

submit_job() {
  section "Submitting self-sizing DDP job"

  # --- Job parameters (override via env vars) ---
  NUM_SAMPLES="${NUM_SAMPLES:-8192}"
  INTENSITY="${INTENSITY:-medium}"
  POD_MEM_LIMIT_MB="${POD_MEM_LIMIT_MB:-4096}"
  OVERHEAD_MB="${OVERHEAD_MB:-1200}"
  MAX_NODES="${MAX_NODES:-8}"
  EPOCHS="${EPOCHS:-5}"
  BATCH_SIZE="${BATCH_SIZE:-64}"

  # --- Calculate minimum nodes ---
  if [ "$INTENSITY" = "light" ]; then
    MIN_NODES=1
    DATASET_MB=0
    BUDGET_MB=0
  else
    BYTES_PER_SAMPLE=$((224 * 224 * 3 * 4))
    DATASET_MB=$(( (NUM_SAMPLES * BYTES_PER_SAMPLE) / (1024 * 1024) ))
    BUDGET_MB=$((POD_MEM_LIMIT_MB - OVERHEAD_MB))
    if [ "$BUDGET_MB" -le 0 ]; then BUDGET_MB=1024; fi
    MIN_NODES=$(( (DATASET_MB + BUDGET_MB - 1) / BUDGET_MB ))
    if [ "$MIN_NODES" -lt 1 ]; then MIN_NODES=1; fi
    if [ "$MIN_NODES" -gt "$MAX_NODES" ]; then MIN_NODES="$MAX_NODES"; fi
  fi

  log "Configuration:"
  echo "  Intensity:        $INTENSITY"
  echo "  Num samples:      $NUM_SAMPLES"
  echo "  Dataset memory:   ${DATASET_MB} MB total"
  echo "  Per-node budget:  ${BUDGET_MB} MB"
  echo "  Minimum nodes:    ${MIN_NODES}"
  echo "  Node range:       ${MIN_NODES}-${MAX_NODES}"
  echo ""

  # --- Submit via controller ---
  log "Submitting to Slurm (--nodes=${MIN_NODES}-${MAX_NODES})..."

  SBATCH_OUTPUT=$(oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- \
    sbatch --nodes=${MIN_NODES}-${MAX_NODES} --ntasks-per-node=1 --job-name=ddp-elastic \
    --cpus-per-task=2 --time=00:30:00 --time-min=00:10:00 \
    --output=/tmp/ddp-elastic-%j.out --error=/tmp/ddp-elastic-%j.err \
    --export=ALL,NUM_SAMPLES=${NUM_SAMPLES},INTENSITY=${INTENSITY},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE} \
    --requeue --wrap="
MAX_WAIT=600; ELAPSED=0
echo \"[\$(hostname)] Waiting for PyTorch and training script...\"
while ! python3 -c 'import torch' 2>/dev/null || [ ! -f /tmp/ddp_test.py ]; do
  if [ \$ELAPSED -ge \$MAX_WAIT ]; then echo \"[\$(hostname)] TIMEOUT\"; exit 1; fi
  echo \"[\$(hostname)] Not ready yet (\${ELAPSED}/\${MAX_WAIT}s)\"; sleep 15; ELAPSED=\$((ELAPSED+15))
done
echo \"[\$(hostname)] Ready (torch \$(python3 -c 'import torch; print(torch.__version__)'))\"
python3 /tmp/ddp_test.py --intensity $INTENSITY --epochs $EPOCHS --batch-size $BATCH_SIZE --num-samples $NUM_SAMPLES --autoscale --output-dir /tmp/ddp-results
" 2>&1)

  JOB_ID=$(echo "$SBATCH_OUTPUT" | awk '/Submitted batch job/{print $4}')

  if [ -z "$JOB_ID" ]; then
    warn "Failed to submit job: $SBATCH_OUTPUT"
    exit 1
  fi

  log "Submitted batch job $JOB_ID (nodes: ${MIN_NODES}-${MAX_NODES})"
  echo ""
  echo "  Monitor autoscaler:  oc logs -n $NAMESPACE -l app.kubernetes.io/name=slurm-autoscaler -f --tail=20"
  echo "  Monitor job:         oc exec -n $NAMESPACE $CONTROLLER_POD -c slurmctld -- squeue -l"
  echo "  View output:         oc exec -n $NAMESPACE slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-elastic-${JOB_ID}.out"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if [ "$SUBMIT_ONLY" = true ]; then
  submit_job
elif [ "$SETUP_ONLY" = true ]; then
  setup_autoscaler
else
  setup_autoscaler
  submit_job
fi
