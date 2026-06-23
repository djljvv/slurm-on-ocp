#!/bin/bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-slurm}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Deploying Slurm Autoscaler to namespace: ${NAMESPACE} ==="

# Step 1: Create/update the ConfigMap from source files
echo "[1/2] Creating ConfigMap 'slurm-autoscaler-script'..."
oc create configmap slurm-autoscaler-script -n "$NAMESPACE" \
  --from-file=autoscaler.sh="${SCRIPT_DIR}/autoscaler-loop.sh" \
  --from-file=ddp_test.py="${REPO_ROOT}/demos/ddp_test.py" \
  --from-file=submit_job.sh="${REPO_ROOT}/demos/submit_job.sh" \
  --from-file=submit_job_autoscale.sh="${REPO_ROOT}/demos/submit_job_autoscale.sh" \
  --dry-run=client -o yaml | oc apply -f -

# Step 2: Apply RBAC + Deployment
echo "[2/3] Applying autoscaler RBAC and Deployment..."
oc apply -f "${SCRIPT_DIR}/slurm-autoscaler.yaml"

# Step 3: Grant privileged SCC so autoscaler can exec into worker pods
echo "[3/3] Granting privileged SCC to autoscaler service account..."
oc adm policy add-scc-to-user privileged -z slurm-autoscaler -n "$NAMESPACE" 2>/dev/null || true

echo ""
echo "=== Autoscaler deployed. Verify with: ==="
echo "  oc get pods -n ${NAMESPACE} -l app.kubernetes.io/name=slurm-autoscaler"
echo "  oc logs -n ${NAMESPACE} -l app.kubernetes.io/name=slurm-autoscaler -f"
