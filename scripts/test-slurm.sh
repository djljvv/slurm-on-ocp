#!/bin/bash

###############################################################################
# Slurm Cluster Test Script
# 
# Comprehensive test script for Slurm cluster on OpenShift
# 
# Usage: ./test-slurm.sh [options]
# 
# Options:
#   --namespace NAMESPACE    Cluster namespace (default: slurm)
#   --controller-pod POD     Controller pod name (auto-detected if not provided)
#   --quick                  Quick end-to-end test (single job, verify output)
#   --comprehensive          Comprehensive test suite (multiple job types)
#   --help                   Show this help message
# 
# Default: Quick test (--quick)
###############################################################################

set -euo pipefail

# Default values
NAMESPACE="slurm"
CONTROLLER_POD=""
QUICK_MODE=true
COMPREHENSIVE_MODE=false

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --controller-pod)
      CONTROLLER_POD="$2"
      shift 2
      ;;
    --quick)
      QUICK_MODE=true
      COMPREHENSIVE_MODE=false
      shift
      ;;
    --comprehensive)
      QUICK_MODE=false
      COMPREHENSIVE_MODE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --namespace NAMESPACE    Cluster namespace (default: slurm)"
      echo "  --controller-pod POD     Controller pod name (auto-detected if not provided)"
      echo "  --quick                  Quick end-to-end test (default)"
      echo "  --comprehensive          Comprehensive test suite"
      echo "  --help                   Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
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

log_section() {
  echo ""
  echo -e "${BLUE}=== $1 ===${NC}"
  echo ""
}

# Get controller pod if not provided
if [ -z "$CONTROLLER_POD" ]; then
  log_info "Auto-detecting controller pod..."
  CONTROLLER_POD=$(oc get pods -n "$NAMESPACE" -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
  
  if [ -z "$CONTROLLER_POD" ]; then
    log_error "Controller pod not found in namespace $NAMESPACE"
    log_info "Please provide pod name: $0 --controller-pod <pod-name> [--namespace <ns>]"
    exit 1
  fi
fi

log_info "Using controller pod: $CONTROLLER_POD"
log_info "Namespace: $NAMESPACE"
echo ""

# Quick test mode (default)
if [ "$QUICK_MODE" = true ]; then
  log_section "Quick End-to-End Test"
  
  log_info "1. Checking cluster status..."
  oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- sinfo
  echo ""
  
  log_info "2. Checking available nodes..."
  oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- scontrol show nodes | head -20
  echo ""
  
  log_info "3. Submitting test job..."
  JOB_OUTPUT=$(oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- sbatch --output=/tmp/quick-test.out --wrap="echo 'Quick test job' && hostname && date && echo 'Job completed successfully'")
  echo "$JOB_OUTPUT"
  JOB_ID=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/{print $4}')
  log_info "Job ID: $JOB_ID"
  echo ""
  
  log_info "4. Waiting for job to complete (5 seconds)..."
  sleep 5
  
  log_info "5. Checking job queue..."
  oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- squeue
  echo ""
  
  log_info "6. Checking detailed job status..."
  oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- scontrol show job "$JOB_ID" 2>/dev/null | grep -E "(JobId|JobState|ExitCode|NodeList|StdOut|SubmitTime|StartTime|EndTime)" || log_warn "Job may have been purged from memory (check output file instead)"
  echo ""
  
  log_info "7. Finding which compute node ran the job..."
  NODE_LIST=$(oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- scontrol show job "$JOB_ID" 2>/dev/null | grep "NodeList=" | awk '{print $1}' | cut -d= -f2)
  if [ -n "$NODE_LIST" ]; then
    log_info "Job ran on: $NODE_LIST"
    # Derive pod name from Slurm node name (slinky-N → slurm-worker-slinky-N)
    FIRST_NODE=$(echo "$NODE_LIST" | sed 's/\[.*//;s/,.*//')
    COMPUTE_NODE="slurm-worker-${FIRST_NODE}"
  else
    log_warn "Could not determine node, will try first compute node"
    COMPUTE_NODE="slurm-worker-slinky-0"
  fi
  echo ""
  
  log_info "8. Viewing job output..."
  if [ -n "$COMPUTE_NODE" ]; then
    log_info "Checking $COMPUTE_NODE (from NodeList=$NODE_LIST)..."
    oc exec -n "$NAMESPACE" "$COMPUTE_NODE" -c slurmd -- cat /tmp/quick-test.out 2>/dev/null || log_warn "File not found on $COMPUTE_NODE"
  else
    log_info "Trying both compute nodes..."
    if oc exec -n "$NAMESPACE" slurm-worker-slinky-0 -c slurmd -- cat /tmp/quick-test.out 2>/dev/null; then
      echo ""
    elif oc exec -n "$NAMESPACE" slurm-worker-slinky-1 -c slurmd -- cat /tmp/quick-test.out 2>/dev/null; then
      echo ""
    else
      log_warn "Output file not found. Checking for other output files..."
      oc exec -n "$NAMESPACE" slurm-worker-slinky-0 -c slurmd -- ls -la /tmp/*.out 2>/dev/null || log_warn "No output files found on compute-0"
    fi
  fi
  echo ""
  
  log_section "Quick Test Complete"
  
  echo "To submit more jobs, use:"
  echo "  oc exec -n $NAMESPACE $CONTROLLER_POD -c slurmctld -- sbatch --output=/tmp/myjob.out --wrap=\"echo 'My job'\""
  echo ""
  echo "To view job status:"
  echo "  oc exec -n $NAMESPACE $CONTROLLER_POD -c slurmctld -- scontrol show job <JOB_ID>"
  echo ""
  echo "To view job output:"
  echo "  # First, find which node ran the job:"
  echo "  oc exec -n $NAMESPACE $CONTROLLER_POD -c slurmctld -- scontrol show job <JOB_ID> | grep NodeList"
  echo "  # Then view output:"
  echo "  # If NodeList=slinky-0: oc exec -n $NAMESPACE slurm-worker-slinky-0 -c slurmd -- cat /tmp/myjob.out"
  echo "  # If NodeList=slinky-1: oc exec -n $NAMESPACE slurm-worker-slinky-1 -c slurmd -- cat /tmp/myjob.out"
  echo "  # Or try both: oc exec -n $NAMESPACE slurm-worker-slinky-0 -c slurmd -- cat /tmp/myjob.out 2>/dev/null || oc exec -n $NAMESPACE slurm-worker-slinky-1 -c slurmd -- cat /tmp/myjob.out"
fi

# Comprehensive test mode
if [ "$COMPREHENSIVE_MODE" = true ]; then
  log_section "Comprehensive Test Suite"
  
  log_info "Test 1: Simple Hello World job"
  JOB_OUTPUT=$(oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- sbatch --wrap="echo 'Hello from Slurm' && hostname && date" 2>&1)
  JOB_ID=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/{print $4}')
  if [ -n "$JOB_ID" ]; then
    log_info "Job submitted: $JOB_ID"
    sleep 5
    oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- squeue -j "$JOB_ID" || true
  else
    log_warn "Failed to submit job"
  fi
  echo ""
  
  log_info "Test 2: CPU-intensive job"
  JOB_OUTPUT=$(oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- sbatch \
    --cpus-per-task=2 \
    --wrap="echo 'CPU test' && nproc && sleep 30" 2>&1)
  JOB_ID=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/{print $4}')
  if [ -n "$JOB_ID" ]; then
    log_info "Job submitted: $JOB_ID"
    sleep 5
    oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- squeue -j "$JOB_ID" || true
  else
    log_warn "Failed to submit job"
  fi
  echo ""
  
  log_info "Test 3: Memory-intensive job"
  JOB_OUTPUT=$(oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- sbatch \
    --mem=2G \
    --wrap="echo 'Memory test' && free -h && sleep 30" 2>&1)
  JOB_ID=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/{print $4}')
  if [ -n "$JOB_ID" ]; then
    log_info "Job submitted: $JOB_ID"
    sleep 5
    oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- squeue -j "$JOB_ID" || true
  else
    log_warn "Failed to submit job"
  fi
  echo ""
  
  log_info "Test 4: Submitting 5 jobs to test scheduling"
  for i in {1..5}; do
    JOB_OUTPUT=$(oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- sbatch \
      --wrap="echo 'Job $i' && sleep 60" 2>&1)
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/{print $4}')
    if [ -n "$JOB_ID" ]; then
      log_info "Job $i submitted: $JOB_ID"
    fi
  done
  
  sleep 5
  log_info "Current queue status:"
  oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- squeue
  echo ""
  
  log_info "Test 5: Job with dependencies"
  JOB_OUTPUT=$(oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- sbatch --wrap="echo 'Job 1' && sleep 10" 2>&1)
  JOB1=$(echo "$JOB_OUTPUT" | awk '/Submitted batch job/{print $4}')
  if [ -n "$JOB1" ]; then
    log_info "First job submitted: $JOB1"
    JOB2_OUTPUT=$(oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- sbatch --dependency=afterok:"$JOB1" --wrap="echo 'Job 2 (depends on $JOB1)'" 2>&1)
    JOB2=$(echo "$JOB2_OUTPUT" | awk '/Submitted batch job/{print $4}')
    if [ -n "$JOB2" ]; then
      log_info "Dependent job submitted: $JOB2"
      sleep 5
      oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- squeue -j "$JOB1,$JOB2" || true
    fi
  fi
  echo ""
  
  log_info "Cluster status:"
  oc exec -n "$NAMESPACE" "$CONTROLLER_POD" -c slurmctld -- sinfo
  echo ""
  
  log_section "Comprehensive Test Complete"
  
  log_info "All test jobs submitted. Monitor with:"
  echo "  oc exec -n $NAMESPACE $CONTROLLER_POD -c slurmctld -- squeue"
  echo "  oc exec -n $NAMESPACE $CONTROLLER_POD -c slurmctld -- sacct"
fi
