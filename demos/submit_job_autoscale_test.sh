#!/bin/bash
#SBATCH --job-name=ddp-autoscale-demo
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/tmp/ddp-autoscale-%j.out
#SBATCH --error=/tmp/ddp-autoscale-%j.err
#SBATCH --export=ALL

# ============================================================
#  Autoscaler End-to-End Test
# ============================================================
#
# Requests 4 nodes when only 2 exist. The autoscaler should:
#   1. Detect the pending job needing 4 nodes
#   2. Scale the NodeSet from 2 -> 4 replicas
#   3. Provision new workers with PyTorch
#   4. Job runs DDP training across all 4 nodes
#
# Uses light intensity for quick execution (~30s training).
# Includes a readiness gate that waits for PyTorch to be
# installed by the autoscaler before starting training.
# ============================================================

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

echo "============================================================"
echo "  AUTOSCALER END-TO-END TEST"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Nodes:        $SLURM_NNODES"
echo "  Tasks:        $SLURM_NTASKS"
echo "  Node list:    $SLURM_NODELIST"
echo "  Master:       $MASTER_ADDR:$MASTER_PORT"
echo "  World Size:   $WORLD_SIZE"
echo "  Hostname:     $(hostname)"
echo "  Start time:   $(date)"
echo "============================================================"
echo ""

srun bash -c '
MAX_WAIT=600
ELAPSED=0
echo "[$(hostname)] Waiting for PyTorch and training script..."
while ! python3 -c "import torch" 2>/dev/null || [ ! -f /tmp/ddp_test.py ]; do
  if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "[$(hostname)] TIMEOUT after ${MAX_WAIT}s"
    exit 1
  fi
  echo "[$(hostname)] Not ready yet, retrying in 15s... (${ELAPSED}/${MAX_WAIT}s)"
  sleep 15
  ELAPSED=$((ELAPSED + 15))
done
echo "[$(hostname)] Ready after ${ELAPSED}s (torch $(python3 -c "import torch; print(torch.__version__)"))"

python3 /tmp/ddp_test.py \
    --intensity light \
    --epochs 5 \
    --batch-size 64 \
    --num-samples 8192 \
    --autoscale \
    --output-dir /tmp/ddp-results
'

JOB_EXIT=$?
echo ""
echo "============================================================"
echo "  Job completed at: $(date)"
echo "  Exit code:        $JOB_EXIT"
if [ $JOB_EXIT -eq 0 ]; then
    echo "  RESULT: SUCCESS — DDP training ran across $SLURM_NNODES nodes"
    echo "  The autoscaler successfully:"
    echo "    1. Detected demand for $SLURM_NNODES nodes"
    echo "    2. Scaled up the NodeSet"
    echo "    3. Provisioned workers with PyTorch"
    echo "    4. Job completed distributed training"
    echo ""
    echo "  Results saved on batch host: $(hostname)"
    echo "  Retrieve with:"
    echo "    oc cp slurm/$(hostname):/tmp/ddp-results results/ -c slurmd"
else
    echo "  RESULT: FAILED (exit code $JOB_EXIT)"
fi
echo "============================================================"
