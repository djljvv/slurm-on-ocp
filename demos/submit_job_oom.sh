#!/bin/bash
#SBATCH --job-name=ddp-oom-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=/tmp/ddp-oom-%j.out
#SBATCH --error=/tmp/ddp-oom-%j.err
#SBATCH --export=ALL

# ============================================================
#  Autoscaling Demo - Step 1: Resource Exhaustion
# ============================================================
#
# This job deliberately exceeds the constrained pod memory limits
# to demonstrate what happens when resources are insufficient.
#
# Pair with configs/slurm-cluster-constrained.yaml (1Gi memory limit).
# The medium-intensity workload pre-allocates ~1.1 GB of training data
# plus ~400 MB for PyTorch/model, triggering an OOMKill.
#
# Usage:
#   sbatch submit_job_oom.sh                    # Medium intensity (~1.5 GB)
#   sbatch submit_job_oom.sh --intensity heavy  # Heavy intensity (~4 GB)
#
# Expected outcome: OOMKilled / non-zero exit code
# ============================================================

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "============================================================"
echo "  AUTOSCALING DEMO - Resource Exhaustion Test"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Nodes:        $SLURM_NNODES"
echo "  Tasks:        $SLURM_NTASKS"
echo "  Node list:    $SLURM_NODELIST"
echo "  Master:       $MASTER_ADDR:$MASTER_PORT"
echo "  World Size:   $WORLD_SIZE"
echo "============================================================"
echo ""
echo "  This job is expected to FAIL with OOMKill."
echo "  Pod memory limit: 1Gi"
echo "  Workload memory:  ~1.5 GB (medium intensity)"
echo ""
echo "============================================================"

srun python3 /tmp/ddp_test.py \
    --intensity medium \
    --epochs 3 \
    --batch-size 128 \
    --num-samples 2000 \
    --output-dir /tmp/ddp-results

JOB_EXIT=$?
echo ""
echo "============================================================"
echo "  Job exit code: $JOB_EXIT"
if [ $JOB_EXIT -ne 0 ]; then
    echo "  EXPECTED: Job failed due to resource constraints."
    echo "  This proves the cluster cannot handle this workload"
    echo "  without more resources or autoscaling."
else
    echo "  UNEXPECTED: Job succeeded. Increase --num-samples or"
    echo "  use --intensity heavy to exceed the memory limit."
fi
echo "============================================================"
