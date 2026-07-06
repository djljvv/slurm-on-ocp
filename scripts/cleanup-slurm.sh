#!/bin/bash

###############################################################################
# Cleanup Script - Remove Slurm Cluster and optionally Operator
#
# By default: removes only the Slurm CLUSTER (workload) in the given namespace.
# With --remove-operator: removes cluster + operator (OperatorHub and/or Helm)
#   + slinky namespace + CRDs = full uninstall.
#
# Usage: ./cleanup-slurm.sh [OPTIONS] [NAMESPACE]
#
#   NAMESPACE           Cluster namespace to clean (default: slurm)
#   --remove-operator   Also remove operator, slinky namespace, and CRDs (full uninstall)
#   --help              Show this help
#
# Examples:
#   ./cleanup-slurm.sh                    # Remove cluster in slurm namespace only
#   ./cleanup-slurm.sh my-slurm          # Remove cluster in my-slurm namespace
#   ./cleanup-slurm.sh --remove-operator  # Remove cluster + operator + slinky + CRDs
#   ./cleanup-slurm.sh slurm --remove-operator
###############################################################################

set -euo pipefail

NAMESPACE="slurm"
REMOVE_OPERATOR=false

# Parse arguments (flags first, then optional namespace)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --remove-operator)
      REMOVE_OPERATOR=true
      shift
      ;;
    --help|-h)
      head -30 "$0" | tail -25
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      NAMESPACE="$1"
      shift
      ;;
  esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Run a command; log and continue on failure (for optional cleanup steps)
run_ignore() {
  if "$@" 2>/dev/null; then
    return 0
  fi
  log_warn "Optional step failed or nothing to delete: $*"
  return 0
}

echo "🧹 Slurm cleanup: namespace=$NAMESPACE, remove_operator=$REMOVE_OPERATOR"
echo ""

# ---------------------------------------------------------------------------
# 1. Cluster namespace cleanup (delete in order so operator doesn't recreate)
# ---------------------------------------------------------------------------
if oc get namespace "$NAMESPACE" &>/dev/null; then
  log_info "Cleaning cluster namespace: $NAMESPACE"

  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  log_info "Deleting autoscaler (if deployed)..."
  run_ignore oc delete -f "$REPO_ROOT/configs/slurm-autoscaler.yaml" --ignore-not-found --timeout=30s
  run_ignore oc delete configmap slurm-autoscaler-script -n "$NAMESPACE" --ignore-not-found --timeout=30s

  # Discover controller SA before deleting pods (needed for SCC cleanup later)
  _CTRL_SA=$(oc get pod slurm-controller-0 -n "$NAMESPACE" \
    -o jsonpath='{.spec.serviceAccountName}' 2>/dev/null || echo "")

  log_info "Deleting Controller and NodeSet (custom resources)..."
  run_ignore oc delete controller --all -n "$NAMESPACE" --ignore-not-found --timeout=60s
  run_ignore oc delete nodeset --all -n "$NAMESPACE" --ignore-not-found --timeout=60s
  sleep 5
  log_info "Deleting workloads..."
  run_ignore oc delete statefulset --all -n "$NAMESPACE" --ignore-not-found --timeout=60s
  run_ignore oc delete deployment --all -n "$NAMESPACE" --ignore-not-found --timeout=60s
  run_ignore oc delete replicaset --all -n "$NAMESPACE" --ignore-not-found --timeout=30s
  run_ignore oc delete pods --all -n "$NAMESPACE" --ignore-not-found --timeout=30s
  run_ignore oc delete job --all -n "$NAMESPACE" --ignore-not-found --timeout=30s

  log_info "Deleting Services, ConfigMaps, Secrets, PVCs..."
  run_ignore oc delete svc --all -n "$NAMESPACE" --ignore-not-found --timeout=30s
  run_ignore oc delete configmap --all -n "$NAMESPACE" --ignore-not-found --timeout=30s
  run_ignore oc delete secret --all -n "$NAMESPACE" --ignore-not-found --timeout=30s
  run_ignore oc delete pvc --all -n "$NAMESPACE" --ignore-not-found --timeout=60s

  log_info "Removing anyuid/privileged SCC grants..."
  run_ignore oc adm policy remove-scc-from-user anyuid -z slurm-workload -n "$NAMESPACE"
  run_ignore oc adm policy remove-scc-from-user privileged -z slurm-workload -n "$NAMESPACE"
  run_ignore oc adm policy remove-scc-from-user anyuid -z default -n "$NAMESPACE"
  if [ -n "${_CTRL_SA:-}" ] && [ "$_CTRL_SA" != "slurm-workload" ] && [ "$_CTRL_SA" != "default" ]; then
    run_ignore oc adm policy remove-scc-from-user anyuid -z "$_CTRL_SA" -n "$NAMESPACE"
  fi

  log_info "Deleting namespace $NAMESPACE..."
  if oc delete namespace "$NAMESPACE" --ignore-not-found --timeout=120s 2>/dev/null; then
    log_info "Namespace $NAMESPACE deletion requested"
  else
    log_warn "Namespace $NAMESPACE may still be terminating (check: oc get namespace $NAMESPACE)"
  fi
else
  log_warn "Namespace $NAMESPACE does not exist; skipping cluster cleanup"
fi

# ---------------------------------------------------------------------------
# 2. Operator removal (--remove-operator): OperatorHub + Helm (slinky) + CRDs
# ---------------------------------------------------------------------------
if [ "$REMOVE_OPERATOR" = true ]; then
  log_info "Removing Slurm operator (OperatorHub and/or Helm)..."

  # 2a. OperatorHub (openshift-operators)
  # Note: `grep` exits 1 when it finds no matches (e.g. operator installed via Helm,
  # not OperatorHub), which under `set -o pipefail` would otherwise abort the whole
  # script here due to `set -e`. The trailing `|| true` on each pipeline makes "no
  # matching subscription/CSV" a non-fatal, expected outcome instead of a hard stop.
  log_info "Removing OperatorHub subscription and CSV..."
  oc get subscription -n openshift-operators -o name 2>/dev/null | grep -i slurm | while read -r sub; do
    oc delete -n openshift-operators "$sub" --ignore-not-found --timeout=30s 2>/dev/null || true
  done || true
  oc get csv -n openshift-operators -o name 2>/dev/null | grep -i slurm | while read -r csv; do
    oc delete -n openshift-operators "$csv" --ignore-not-found --timeout=60s 2>/dev/null || true
  done || true
  run_ignore oc delete pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator --ignore-not-found --timeout=30s

  # 2b. Helm (slinky namespace)
  OPERATOR_NS="slinky"
  if oc get namespace "$OPERATOR_NS" &>/dev/null; then
    log_info "Removing Helm releases in $OPERATOR_NS (if any)..."
    if command -v helm &>/dev/null; then
      for release in slurm-operator slurm-operator-crds slurm; do
        if helm uninstall "$release" -n "$OPERATOR_NS" 2>/dev/null; then
          log_info "Uninstalled Helm release: $release"
        else
          log_warn "Helm uninstall $release failed or not found (may be installed via OpenShift Console)"
        fi
      done
    else
      log_warn "helm CLI not found; will delete namespace resources directly"
    fi
    # Remove any remaining workloads so namespace can terminate (works even if Helm CLI didn't remove them)
    log_info "Removing remaining resources in $OPERATOR_NS..."
    run_ignore oc delete deployment,statefulset,replicaset,pod,service,configmap,secret,job --all -n "$OPERATOR_NS" --ignore-not-found --timeout=60s
    run_ignore oc delete subscription,csv --all -n "$OPERATOR_NS" --ignore-not-found --timeout=30s 2>/dev/null || true
    sleep 3
    log_info "Deleting namespace $OPERATOR_NS..."
    oc delete namespace "$OPERATOR_NS" --ignore-not-found --timeout=120s 2>/dev/null || true
  fi

  # 2c. CRDs (delete after operator/instances are gone)
  log_info "Deleting Slurm/Slinky CRDs..."
  for crd in \
    controllers.slinky.slurm.net \
    nodesets.slinky.slurm.net \
    loginsets.slinky.slurm.net \
    accountings.slinky.slurm.net \
    restapis.slinky.slurm.net \
    tokens.slinky.slurm.net \
    slurmclusters.slurm.schedmd.com \
    slurmjobs.slurm.schedmd.com \
    ; do
    oc delete crd "$crd" --ignore-not-found --timeout=60s 2>/dev/null && log_info "Deleted CRD: $crd" || true
  done

  log_info "✅ Operator removal completed (OperatorHub + slinky + CRDs)"
  log_warn "Reinstall operator from OperatorHub or: helm install slurm-operator-crds ... && helm install slurm-operator ..."
else
  echo ""
  log_info "Operator kept. To remove everything (cluster + operator + slinky + CRDs), run:"
  log_info "   ./scripts/cleanup-slurm.sh $NAMESPACE --remove-operator"
fi

# ---------------------------------------------------------------------------
# 3. Wait and report
# ---------------------------------------------------------------------------
echo ""
log_info "Waiting for namespace deletion..."
sleep 5

if oc get namespace "$NAMESPACE" &>/dev/null; then
  log_warn "Namespace $NAMESPACE still exists (may be terminating). Check: oc get namespace $NAMESPACE"
else
  log_info "✅ Namespace $NAMESPACE deleted"
fi

# If --remove-operator was used but slinky is still there, show manual steps
if [ "$REMOVE_OPERATOR" = true ] && oc get namespace slinky &>/dev/null; then
  echo ""
  log_warn "slinky namespace is still present (Helm/Console may manage it). Remove manually:"
  log_info "  helm uninstall slurm-operator -n slinky"
  log_info "  helm uninstall slurm-operator-crds -n slinky"
  log_info "  oc delete namespace slinky"
  log_info "If namespace is stuck in Terminating: oc get namespace slinky -o yaml  # check metadata.finalizers"
fi

echo ""
log_info "📋 Summary:"
log_info "   ✅ Cluster namespace $NAMESPACE cleaned"
if [ "$REMOVE_OPERATOR" = true ]; then
  log_info "   ✅ Operator and CRDs removed (full uninstall)"
else
  log_info "   ✅ Operator kept (in openshift-operators and/or slinky)"
fi
echo ""
log_info "Redeploy cluster: ./scripts/deploy-slurm.sh"
echo ""
