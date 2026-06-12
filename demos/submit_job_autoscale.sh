#!/bin/bash
#SBATCH --job-name=ddp-elastic
#SBATCH --nodes=1-4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --time-min=00:10:00
#SBATCH --output=/tmp/ddp-elastic-%j.out
#SBATCH --error=/tmp/ddp-elastic-%j.err
#SBATCH --export=ALL
#SBATCH --requeue

# ============================================================
#  Slurm on OCP - Elastic DDP Training Submission Script
# ============================================================
#
# Autoscaling-aware job submission that adapts to available resources.
# Uses --nodes=min-max so Slurm can start the job as soon as the
# minimum number of nodes is available, then grow the allocation
# if more nodes become free.
#
# Key differences from submit_job.sh:
#   --nodes=1-4       Elastic: runs with 1 to 4 nodes
#   --time-min        Backfill-friendly minimum runtime
#   --requeue         Re-enqueue if preempted by higher-priority work
#   --autoscale       Enables resource monitoring in ddp_test.py
#
# Usage:
#   sbatch submit_job_autoscale.sh                     # Default: 1-4 nodes
#   sbatch --nodes=2-8 submit_job_autoscale.sh         # Override: 2-8 nodes
#   sbatch --gres=gpu:1 submit_job_autoscale.sh        # Request GPUs
#   sbatch --partition=gpu submit_job_autoscale.sh      # Target GPU partition
#
# With autoscaling enabled (configs/slurm-autoscaler.yaml deployed):
#   - Submitting this job triggers KEDA to scale up the NodeSet
#   - After the job completes, idle nodes scale back down
#   - Multiple jobs can queue and the cluster expands to meet demand
#
# ============================================================

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Nodes:        $SLURM_NNODES (requested: $SLURM_JOB_NUM_NODES)"
echo "  Tasks:        $SLURM_NTASKS"
echo "  Node list:    $SLURM_NODELIST"
echo "  Master:       $MASTER_ADDR:$MASTER_PORT"
echo "  World Size:   $WORLD_SIZE"
echo "  Requeue:      enabled"
echo "============================================================"

srun python3 /tmp/ddp_test.py \
    --epochs 5 \
    --batch-size 64 \
    --num-samples 8192 \
    --autoscale
