#!/bin/bash
set -euo pipefail

###############################################################################
# Slurm Autoscaler — Deploy the scale-down watchdog
#
# Deploys the background autoscaler loop that handles scale-DOWN after idle.
# Scale-UP is now handled by `python ddp_test.py --launch` which proactively
# sizes the cluster before job submission.
#
# Usage:
#   ./scripts/deploy-autoscale.sh              # Deploy the autoscaler watchdog
#   ./scripts/deploy-autoscale.sh --teardown   # Remove it
#
# Environment variables:
#   NAMESPACE         Kubernetes namespace (default: slurm)
###############################################################################

NAMESPACE="${NAMESPACE:-slurm}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEARDOWN=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --teardown)  TEARDOWN=true; shift ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --help|-h)
      sed -n '3,14p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $1"; }
section() { echo ""; echo -e "${BLUE}━━━ $1 ━━━${NC}"; echo ""; }

if [ "$TEARDOWN" = true ]; then
  section "Removing autoscaler"
  oc delete -f "${REPO_ROOT}/configs/slurm-autoscaler.yaml" --ignore-not-found
  oc delete configmap slurm-autoscaler-script -n "$NAMESPACE" --ignore-not-found
  log "Autoscaler removed"
  exit 0
fi

section "Deploying autoscaler (scale-down watchdog)"

log "Creating ConfigMap 'slurm-autoscaler-script'..."
oc create configmap slurm-autoscaler-script -n "$NAMESPACE" \
  --from-file=autoscaler.sh="${SCRIPT_DIR}/autoscaler-loop.sh" \
  --from-file=ddp_test.py="${REPO_ROOT}/demos/ddp_test.py" \
  --dry-run=client -o yaml | oc apply -f -

log "Applying autoscaler RBAC and Deployment..."
oc apply -f "${REPO_ROOT}/configs/slurm-autoscaler.yaml"

log "Granting privileged SCC to autoscaler service account..."
oc adm policy add-scc-to-user privileged -z slurm-autoscaler -n "$NAMESPACE" 2>/dev/null || true

log "Waiting for autoscaler pod..."
waited=0
while [ $waited -lt 30 ]; do
  if oc get pods -n "$NAMESPACE" -l app.kubernetes.io/name=slurm-autoscaler --no-headers 2>/dev/null | grep -qE "Running|ContainerCreating|Pulling"; then
    log "Autoscaler pod is starting"
    break
  fi
  sleep 3
  waited=$((waited + 3))
done

log "Autoscaler watchdog deployed"
echo ""
echo "  The autoscaler now only handles scale-DOWN after idle periods."
echo "  Scale-UP is handled automatically by:"
echo "    python demos/ddp_test.py --launch"
echo ""
