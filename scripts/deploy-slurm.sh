#!/bin/bash

###############################################################################
# Slurm Deployment Script for OpenShift
# 
# This script automates the complete deployment of Slurm on OpenShift:
# 1. Installs cert-manager (if needed)
# 2. Installs Slinky operator (if not already installed)
# 3. Deploys Slurm cluster (Controller and NodeSet)
# 
# Works with:
# - Operator installed via OperatorHub (most common)
# - Operator installed via Helm
# 
# Naming Convention: All scripts use "slurm" (the workload/cluster being managed)
#                    not "slinky" (the operator that manages it)
# 
# Usage: ./deploy-slurm.sh [options]
# 
# Options:
#   --namespace NAMESPACE    Target namespace for Slurm cluster (default: slurm)
#   --operator-ns NS         Namespace for operator if installing via Helm (default: slinky)
#   --cert-manager-ns NS     Namespace for cert-manager (default: cert-manager)
#   --skip-cert-manager      Skip cert-manager installation
#   --skip-operator          Skip operator installation (assumes already installed)
#   --skip-cluster           Skip cluster deployment (only install operator)
#   --dry-run                Show what would be done without executing
#
# Cluster is always deployed via direct YAML (configs/slurm-cluster.yaml).
# Edit that file to customize resources, GPU settings, replicas, etc.
# 
# Note: If CRDs are already installed via OperatorHub, the script will automatically
#       detect and skip CRD installation. You don't need to do anything manually.
# 
# Note: If operator is already installed (via OperatorHub or Helm), it will be
#       detected and the script will proceed directly to cluster deployment.
###############################################################################

set -euo pipefail

# Default values
NAMESPACE="slurm"
OPERATOR_NS="slinky"
CERT_MANAGER_NS="cert-manager"
SKIP_CERT_MANAGER=false
SKIP_OPERATOR=false
SKIP_CLUSTER=false
DRY_RUN=false

# Pinned to the last chart release on Slurm app version 25.11, matching the
# slurmctld/slurmd image tags hardcoded in configs/slurm-cluster.yaml. Chart
# 1.2.0 bumped the operator to app version 26.05, whose NodeSet reconciler
# builds worker pods requesting `privileged: true` plus BPF/NET_ADMIN/SYS_ADMIN
# capabilities — no OpenShift SCC (including anyuid) allows that, so pods are
# rejected at admission and no workers ever get created. Pinning avoids
# silently picking up a future breaking chart release the same way.
SLURM_OPERATOR_CHART_VERSION="1.1.1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --operator-ns)
      OPERATOR_NS="$2"
      shift 2
      ;;
    --cert-manager-ns)
      CERT_MANAGER_NS="$2"
      shift 2
      ;;
    --skip-cert-manager)
      SKIP_CERT_MANAGER=true
      shift
      ;;
    --skip-operator)
      SKIP_OPERATOR=true
      shift
      ;;
    --skip-cluster)
      SKIP_CLUSTER=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Functions
log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
  log_info "Checking prerequisites..."
  
  # Check if oc is installed
  if ! command -v oc &> /dev/null; then
    log_error "oc (OpenShift CLI) is not installed"
    exit 1
  fi
  
  # Check if helm is installed (only needed for operator installation)
  if [ "$SKIP_OPERATOR" = false ] && ! command -v helm &> /dev/null; then
    log_error "helm is not installed (required for operator install; use --skip-operator if already installed)"
    exit 1
  fi
  
  # Check if logged in to OpenShift
  if ! oc whoami &> /dev/null; then
    log_error "Not logged in to OpenShift. Please run 'oc login'"
    exit 1
  fi
  
  log_info "Prerequisites check passed"
}

install_cert_manager() {
  if [ "$SKIP_CERT_MANAGER" = true ]; then
    log_info "Skipping cert-manager installation"
    return
  fi
  
  log_info "Installing cert-manager..."
  
  # Check if cert-manager is already installed
  if oc get namespace "$CERT_MANAGER_NS" &> /dev/null; then
    log_warn "Namespace $CERT_MANAGER_NS already exists"
    if oc get deployment -n "$CERT_MANAGER_NS" cert-manager &> /dev/null; then
      log_info "cert-manager appears to be already installed"
      return
    fi
  fi
  
  if [ "$DRY_RUN" = false ]; then
    # Add Jetstack Helm repository
    helm repo add jetstack https://charts.jetstack.io || true
    helm repo update
    
    # Install cert-manager
    helm install cert-manager jetstack/cert-manager \
      --namespace "$CERT_MANAGER_NS" \
      --create-namespace \
      --set installCRDs=true \
      --set 'crds.enabled=true' \
      --version v1.13.0 \
      --wait --timeout 5m
    
    log_info "Waiting for cert-manager to be ready..."
    oc wait --for=condition=ready pod \
      -l app.kubernetes.io/name=cert-manager \
      -n "$CERT_MANAGER_NS" \
      --timeout=300s || true
  else
    log_info "[DRY RUN] Would install cert-manager"
  fi
  
  log_info "cert-manager installation completed"
}

install_slurm_operator_crds() {
  log_info "Checking Slurm Operator CRDs..."
  
  # Check if CRDs already exist
  if oc get crd controllers.slinky.slurm.net &> /dev/null; then
    log_info "Slurm CRDs already exist"
    
    # Check if CRDs are Helm-managed
    MANAGED_BY=$(oc get crd controllers.slinky.slurm.net -o jsonpath='{.metadata.labels.app\.kubernetes\.io/managed-by}' 2>/dev/null || echo "")
    
    if [ "$MANAGED_BY" = "Helm" ]; then
      log_info "CRDs are already managed by Helm, skipping installation"
      return
    else
      log_warn "CRDs exist but are not Helm-managed (likely installed via OperatorHub/OLM)"
      log_info "Skipping CRD installation - existing CRDs will be used"
      return
    fi
  fi
  
  # CRDs don't exist, install them via Helm
  log_info "Installing Slurm Operator CRDs via Helm..."
  
  if [ "$DRY_RUN" = false ]; then
    local err
    if ! err=$(helm upgrade --install slurm-operator-crds \
      oci://ghcr.io/slinkyproject/charts/slurm-operator-crds \
      --version "$SLURM_OPERATOR_CHART_VERSION" \
      --namespace "$OPERATOR_NS" \
      --create-namespace \
      --server-side=false 2>&1); then
      log_warn "--server-side=false failed: $err — retrying without flag"
      helm upgrade --install slurm-operator-crds \
        oci://ghcr.io/slinkyproject/charts/slurm-operator-crds \
        --version "$SLURM_OPERATOR_CHART_VERSION" \
        --namespace "$OPERATOR_NS" \
        --create-namespace || {
          log_error "Failed to install CRDs via Helm"
          log_info "CRDs may already exist. Check with: oc get crd | grep slinky"
          exit 1
        }
    fi
    
    # Wait a moment for CRDs to be registered
    sleep 5
    
    # Verify CRDs (check for the correct CRD names)
    if oc get crd controllers.slinky.slurm.net &> /dev/null; then
      log_info "Slurm CRDs installed and verified successfully"
    else
      log_error "Failed to verify Slurm CRDs after installation"
      exit 1
    fi
  else
    log_info "[DRY RUN] Would install Slurm Operator CRDs"
  fi
}

install_slurm_operator() {
  log_info "Checking if Slurm Operator is already installed..."
  
  # Check if operator is installed via Helm
  if helm list -n "$OPERATOR_NS" | grep -q "slurm-operator"; then
    log_info "Slurm Operator is already installed via Helm (release: slurm-operator)"
    log_info "Skipping operator installation"
    
    # Verify operator pods are running
    if oc get pods -n "$OPERATOR_NS" -l app.kubernetes.io/name=slurm-operator --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -q .; then
      log_info "✓ Slurm Operator pods are running"
      return
    else
      log_warn "Operator Helm release exists but pods may not be ready"
      log_info "Checking operator status..."
    fi
  fi
  
  # Check if operator is installed via OperatorHub (in openshift-operators)
  if oc get pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -q .; then
    log_info "Slurm Operator is already installed via OperatorHub (in openshift-operators namespace)"
    log_info "Skipping operator installation"
    return
  fi
  
  # Operator not found, proceed with installation
  log_info "Installing Slurm Operator via Helm..."
  
  if [ "$DRY_RUN" = false ]; then
    local err
    if ! err=$(helm upgrade --install slurm-operator \
      oci://ghcr.io/slinkyproject/charts/slurm-operator \
      --version "$SLURM_OPERATOR_CHART_VERSION" \
      --namespace "$OPERATOR_NS" \
      --create-namespace \
      --server-side=false \
      --wait --timeout 5m 2>&1); then
      log_warn "--server-side=false failed: $err — retrying without flag"
      helm upgrade --install slurm-operator \
        oci://ghcr.io/slinkyproject/charts/slurm-operator \
        --version "$SLURM_OPERATOR_CHART_VERSION" \
        --namespace "$OPERATOR_NS" \
        --create-namespace \
        --wait --timeout 5m || {
          log_error "Failed to install Slurm Operator"
          log_info "If operator is already installed, you may need to:"
          log_info "  1. Check: helm list -n $OPERATOR_NS"
          log_info "  2. Or use: --skip-operator flag"
          exit 1
        }
    fi
    
    log_info "Waiting for Slurm Operator to be ready..."
    oc wait --for=condition=ready pod \
      -l app.kubernetes.io/name=slurm-operator \
      -n "$OPERATOR_NS" \
      --timeout=300s || {
        log_error "Slurm Operator did not become ready"
        exit 1
      }
    
    log_info "Slurm Operator is ready"
  else
    log_info "[DRY RUN] Would install Slurm Operator"
  fi
}

deploy_slurm_cluster() {
  if [ "$SKIP_CLUSTER" = true ]; then
    log_info "Skipping cluster deployment (--skip-cluster flag set)"
    return
  fi
  
  log_info "Deploying Slurm cluster..."
  
  # Detect operator (either namespace) — informational only, YAML deploy works with both
  if oc get pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -q .; then
    log_info "Operator detected in openshift-operators (OperatorHub)"
  elif oc get pods -n "$OPERATOR_NS" -l app.kubernetes.io/name=slurm-operator --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -q .; then
    log_info "Operator detected in $OPERATOR_NS (Helm)"
  else
    log_warn "Operator not detected in openshift-operators or $OPERATOR_NS"
    log_warn "Proceeding with YAML deployment anyway (operator may be in another namespace)"
  fi
  deploy_cluster_via_yaml
}

deploy_cluster_via_yaml() {
  log_info "Deploying cluster via direct YAML (Controller and NodeSet)..."
  
  if [ "$DRY_RUN" = true ]; then
    log_info "[DRY RUN] Would deploy Controller and NodeSet via oc apply"
    return
  fi
  
  # Resolve path to config (repo root = parent of scripts/)
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  CLUSTER_YAML="$REPO_ROOT/configs/slurm-cluster.yaml"
  
  if [ ! -f "$CLUSTER_YAML" ]; then
    log_error "Config file not found: $CLUSTER_YAML"
    log_info "Run this script from the repository root or ensure configs/slurm-cluster.yaml exists"
    exit 1
  fi
  
  # Create namespace if it doesn't exist
  oc create namespace "$NAMESPACE" 2>/dev/null || true

  # Create service account if it doesn't exist
  oc create serviceaccount slurm-workload -n "$NAMESPACE" 2>/dev/null || true
  
  # Grant anyuid SCC to slurm-workload (for NodeSet workers) and default
  # (bootstrap for controller — the operator may use default or its own SA;
  #  we narrow this after discovering the actual controller SA below)
  log_info "Granting anyuid SCC to slurm-workload and default (bootstrap)..."
  oc adm policy add-scc-to-user anyuid -z slurm-workload -n "$NAMESPACE" 2>/dev/null || true
  oc adm policy add-scc-to-user anyuid -z default -n "$NAMESPACE" 2>/dev/null || true

  # Also grant privileged SCC to slurm-workload. This is a genuine upstream
  # requirement, not an OpenShift-specific workaround: Slurm's cgroup/v2 plugin
  # manages per-job GPU/device access via an eBPF program that slurmstepd loads
  # into the kernel at runtime (see https://slurm.schedmd.com/cgroup_v2.html),
  # which needs the BPF/NET_ADMIN/SYS_ADMIN capabilities and cgroup write access
  # that only the privileged SCC provides — anyuid alone explicitly forbids
  # both. Without this, the operator's NodeSet reconciler fails admission on
  # every worker pod ("unable to validate against any security context
  # constraint") and no worker pods are ever created.
  log_info "Granting privileged SCC to slurm-workload (required for Slurm's eBPF-based cgroup/v2 device control)..."
  oc adm policy add-scc-to-user privileged -z slurm-workload -n "$NAMESPACE" 2>/dev/null || true
  
  # Create/update the ConfigMap holding worker training script(s). The NodeSet
  # mounts this directly onto every worker pod (see configs/slurm-cluster.yaml),
  # so newly autoscaled nodes always have the script — no runtime copy needed.
  # Re-run this script (or just this step) after editing demos/ddp_test.py so
  # future scale-ups pick up the change; already-running pods keep whatever
  # was mounted at pod start until they're recreated.
  log_info "Creating/updating ConfigMap 'slurm-worker-scripts'..."
  oc create configmap slurm-worker-scripts -n "$NAMESPACE" \
    --from-file=ddp_test.py="$REPO_ROOT/demos/ddp_test.py" \
    --dry-run=client -o yaml | oc apply -f -

  # Create/update gres.conf so slurmd can report its GPU at dynamic (-Z)
  # registration time. Uses a static File= mapping rather than
  # AutoDetect=nvml: the slurmd image (ghcr.io/slinkyproject/slurmd) isn't
  # built with the NVML plugin, so AutoDetect=nvml always reports 0 GPUs on
  # this image regardless of the driver being present. See the comment above
  # `configFileRefs` in configs/slurm-cluster.yaml for why the old static
  # NodeName=... Gres=gpu:1 line was removed instead of kept alongside this.
  log_info "Creating/updating ConfigMap 'slurm-gres-conf'..."
  oc create configmap slurm-gres-conf -n "$NAMESPACE" \
    --from-literal=gres.conf="NodeName=slinky-[0-7] Name=gpu File=/dev/nvidia0" \
    --dry-run=client -o yaml | oc apply -f -

  # Create secrets if they don't exist (operator default names/keys)
  if ! oc get secret slurm-auth-jwths256 -n "$NAMESPACE" &>/dev/null; then
    log_info "Creating slurm-auth-jwths256 and slurm-auth-slurm secrets..."
    JWT_KEY=$(openssl rand -base64 32)
    SLURM_KEY=$(openssl rand -base64 32)
    oc create secret generic slurm-auth-jwths256 -n "$NAMESPACE" \
      --from-literal=jwt_hs256.key="$JWT_KEY"
    oc create secret generic slurm-auth-slurm -n "$NAMESPACE" \
      --from-literal=slurm.key="$SLURM_KEY"
  else
    log_info "Secrets slurm-auth-jwths256/slurm-auth-slurm already exist, skipping creation"
  fi
  
  # Deploy Controller and NodeSet from config file (substitute namespace if different from slurm)
  log_info "Applying configs/slurm-cluster.yaml..."
  if [ "$NAMESPACE" = "slurm" ]; then
    oc apply -f "$CLUSTER_YAML"
  else
    sed "s/namespace: slurm/namespace: $NAMESPACE/g" "$CLUSTER_YAML" | oc apply -f -
  fi
  sleep 5
  
  log_info "Checking Controller and NodeSet status..."
  oc get controller,nodeset -n "$NAMESPACE" || {
    log_error "Failed to create Controller or NodeSet"
    exit 1
  }
  
  # Wait for pods to be Running instead of CR conditions (operator may not set them)
  log_info "Waiting for controller pod to be Running..."
  local waited=0
  while [ $waited -lt 120 ]; do
    if oc get pod slurm-controller-0 -n "$NAMESPACE" --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -q .; then
      log_info "Controller pod is Running"
      break
    fi
    sleep 5
    waited=$((waited + 5))
  done
  if [ $waited -ge 120 ]; then
    log_warn "Controller pod not Running after 120s (may still be pulling image)"
  fi

  # Discover which SA the operator assigned to the controller pod.
  # The pod may exist even if it failed admission (SCC rejection), so we can
  # still read spec.serviceAccountName from a non-Running pod.
  CTRL_SA=$(oc get pod slurm-controller-0 -n "$NAMESPACE" \
    -o jsonpath='{.spec.serviceAccountName}' 2>/dev/null || echo "")

  if [ -z "$CTRL_SA" ]; then
    log_warn "Could not determine controller SA — keeping anyuid on 'default'"
  elif [ "$CTRL_SA" = "default" ]; then
    log_info "Controller uses 'default' SA — anyuid already granted (bootstrap)"
  elif [ "$CTRL_SA" = "slurm-workload" ]; then
    log_info "Controller uses slurm-workload SA — anyuid already granted, revoking from 'default'"
    oc adm policy remove-scc-from-user anyuid -z default -n "$NAMESPACE" 2>/dev/null || true
  else
    log_info "Controller uses SA '$CTRL_SA' — granting anyuid, revoking from 'default'"
    oc adm policy add-scc-to-user anyuid -z "$CTRL_SA" -n "$NAMESPACE" 2>/dev/null || true
    oc adm policy remove-scc-from-user anyuid -z default -n "$NAMESPACE" 2>/dev/null || true
  fi

  log_info "Waiting for worker pods to be Running..."
  waited=0
  while [ $waited -lt 120 ]; do
    local running_count
    running_count=$(oc get pods -n "$NAMESPACE" -l nodeset.slinky.slurm.net/name=slurm-worker-slinky --field-selector=status.phase=Running --no-headers 2>/dev/null | wc -l)
    local expected
    expected=$(oc get nodeset slurm-worker-slinky -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo 2)
    if [ "$running_count" -ge "$expected" ] 2>/dev/null; then
      log_info "All $running_count worker pods are Running"
      break
    fi
    sleep 5
    waited=$((waited + 5))
  done
  if [ $waited -ge 120 ]; then
    log_warn "Not all worker pods Running after 120s — check: oc get pods -n $NAMESPACE"
  fi
  
  log_info "Cluster deployment initiated"
  log_info "Pods: oc get pods -n $NAMESPACE"
}

verify_deployment() {
  log_info "Verifying deployment..."
  
  if [ "$DRY_RUN" = true ]; then
    log_info "[DRY RUN] Would verify deployment"
    return
  fi
  
  # Check operator (in operator namespace)
  log_info "Checking Slurm Operator (namespace: $OPERATOR_NS)..."
  if oc get pods -n "$OPERATOR_NS" -l app.kubernetes.io/name=slurm-operator --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -q .; then
    log_info "✓ Slurm Operator is running in namespace: $OPERATOR_NS"
    oc get pods -n "$OPERATOR_NS" -l app.kubernetes.io/name=slurm-operator
  else
    log_warn "⚠ Slurm Operator not found in namespace: $OPERATOR_NS"
    log_info "Checking if operator is installed via OperatorHub..."
    if oc get pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -q .; then
      log_info "✓ Slurm Operator is running in namespace: openshift-operators (OperatorHub installation)"
      oc get pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator
    else
      log_error "✗ Slurm Operator is not running"
      return 1
    fi
  fi
  
  echo ""
  
  # Check cluster (in target namespace)
  log_info "Checking Slurm Cluster (namespace: $NAMESPACE)..."
  if oc get pods -n "$NAMESPACE" --field-selector=status.phase=Running 2>/dev/null | grep -q controller; then
    log_info "✓ Slurm Controller is running in namespace: $NAMESPACE"
    oc get pods -n "$NAMESPACE" | grep controller || true
  else
    log_error "✗ Slurm Controller not found running in namespace: $NAMESPACE"
    oc get pods -n "$NAMESPACE" 2>/dev/null || log_error "Namespace $NAMESPACE may not exist"
    return 1
  fi
  
  if oc get pods -n "$NAMESPACE" -l app.kubernetes.io/name=slurmd --field-selector=status.phase=Running --no-headers 2>/dev/null | grep -q .; then
    log_info "✓ Slurm compute nodes are running in namespace: $NAMESPACE"
    oc get pods -n "$NAMESPACE" -l app.kubernetes.io/name=slurmd 2>/dev/null || true
  else
    log_error "✗ No running compute pods found in namespace: $NAMESPACE"
    return 1
  fi
  
  echo ""
  log_info "Deployment verification completed"
  log_info ""
  log_info "Summary:"
  log_info "  - Operator namespace: $OPERATOR_NS (or openshift-operators if installed via OperatorHub)"
  log_info "  - Cluster namespace: $NAMESPACE"
}


# Main execution
main() {
  log_info "Starting Slinky Operator deployment..."
  log_info "Target namespace: $NAMESPACE"
  log_info "Operator namespace: $OPERATOR_NS"
  
  check_prerequisites
  
  if [ "$SKIP_CERT_MANAGER" = false ]; then
    install_cert_manager
  fi
  
  if [ "$SKIP_OPERATOR" = false ]; then
    install_slurm_operator_crds
    install_slurm_operator
  else
    log_info "Skipping operator installation (--skip-operator flag set)"
    log_info "Assuming operator is already installed"
  fi
  
  deploy_slurm_cluster
  verify_deployment

  # Deploy autoscaler watchdog (handles scale-down after idle)
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ "$DRY_RUN" = true ]; then
    log_info "[DRY RUN] Would deploy autoscaler via ./scripts/deploy-autoscale.sh"
  elif [ -f "$SCRIPT_DIR/deploy-autoscale.sh" ]; then
    log_info "Deploying autoscaler..."
    NAMESPACE="$NAMESPACE" "$SCRIPT_DIR/deploy-autoscale.sh"
    log_info "✓ Autoscaler deployed"
  else
    log_warn "Autoscaler script not found at $SCRIPT_DIR/deploy-autoscale.sh, skipping"
  fi
  
  log_info "Deployment completed successfully!"
  log_info ""
  log_info "Next steps:"
  log_info "1. Run DDP training (auto-detects cluster, scales, submits, monitors):"
  log_info "   python demos/ddp_test.py --launch"
  log_info "2. Or check autoscaler watchdog logs:"
  log_info "   oc logs -n $NAMESPACE -l app.kubernetes.io/name=slurm-autoscaler -f --tail=15"
}

# Run main
main

