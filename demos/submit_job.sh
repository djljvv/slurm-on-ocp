#!/bin/bash
#SBATCH --job-name=ddp-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=/tmp/ddp-test-%j.out
#SBATCH --error=/tmp/ddp-test-%j.err
#SBATCH --export=ALL

# ============================================================
#  Slurm on OCP - DDP Training Test Submission Script
# ============================================================
#
# Usage:
#   sbatch submit_job.sh              # Default: 2 nodes
#   sbatch -N 4 submit_job.sh         # Override: 4 nodes
#   sbatch --gres=gpu:1 submit_job.sh # Request 1 GPU per node
#
# ============================================================

# Derive master address from Slurm node list
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Nodes:        $SLURM_NNODES"
echo "  Tasks:        $SLURM_NTASKS"
echo "  Node list:    $SLURM_NODELIST"
echo "  Master:       $MASTER_ADDR:$MASTER_PORT"
echo "  World Size:   $WORLD_SIZE"
echo "============================================================"

# Results are written by rank 0 to this directory inside the container.
# Use `oc cp` after the job completes to retrieve them locally.
export DDP_OUTPUT_DIR="/tmp/ddp-results"

# Launch distributed training via srun
# Each srun task becomes one rank in the distributed group
srun python3 /tmp/ddp_test.py \
    --epochs 5 \
    --batch-size 64 \
    --num-samples 8192 \
    --output-dir "$DDP_OUTPUT_DIR"
