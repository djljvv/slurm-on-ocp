#!/bin/bash
# ============================================================
#  Slurm on OCP - Self-Sizing Elastic DDP Training
# ============================================================
#
# Self-sizing job: calculates minimum nodes needed based on dataset
# memory requirements and pod memory limits, then submits with the
# correct --nodes=MIN-MAX. The autoscaler detects pending demand
# and provisions new workers automatically.
#
# Node calculation:
#   dataset_bytes = NUM_SAMPLES * 224 * 224 * 3 * 4
#   per_node_budget = POD_MEM_LIMIT - OVERHEAD (PyTorch, model, etc.)
#   min_nodes = ceil(dataset_bytes / per_node_budget)
#
# Usage (from controller):
#   bash /tmp/submit_job_autoscale.sh                          # Default: 8192 samples
#   NUM_SAMPLES=16384 bash /tmp/submit_job_autoscale.sh        # Larger → more nodes
#   NUM_SAMPLES=32768 bash /tmp/submit_job_autoscale.sh        # Forces 4+ nodes
#   INTENSITY=light bash /tmp/submit_job_autoscale.sh          # Light mode (1 node ok)
#
# ============================================================

# --- Configurable parameters (override via env vars) ---
NUM_SAMPLES="${NUM_SAMPLES:-8192}"
INTENSITY="${INTENSITY:-medium}"
POD_MEM_LIMIT_MB="${POD_MEM_LIMIT_MB:-4096}"
OVERHEAD_MB="${OVERHEAD_MB:-1200}"
MAX_NODES="${MAX_NODES:-8}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# --- Calculate minimum nodes needed ---
if [ "$INTENSITY" = "light" ]; then
    MIN_NODES=1
else
    BYTES_PER_SAMPLE=$((224 * 224 * 3 * 4))
    DATASET_MB=$(( (NUM_SAMPLES * BYTES_PER_SAMPLE) / (1024 * 1024) ))
    BUDGET_MB=$((POD_MEM_LIMIT_MB - OVERHEAD_MB))
    if [ "$BUDGET_MB" -le 0 ]; then BUDGET_MB=1024; fi
    MIN_NODES=$(( (DATASET_MB + BUDGET_MB - 1) / BUDGET_MB ))
    if [ "$MIN_NODES" -lt 1 ]; then MIN_NODES=1; fi
    if [ "$MIN_NODES" -gt "$MAX_NODES" ]; then MIN_NODES="$MAX_NODES"; fi
fi

echo "============================================================"
echo "  AUTO-SIZING: ${NUM_SAMPLES} samples (${INTENSITY})"
echo "  Dataset memory:   ${DATASET_MB:-0} MB total"
echo "  Per-node budget:  ${BUDGET_MB:-N/A} MB"
echo "  Minimum nodes:    ${MIN_NODES}"
echo "  Requesting:       --nodes=${MIN_NODES}-${MAX_NODES}"
echo "============================================================"

# --- Generate batch script with #SBATCH directives (avoids env retrieval bug) ---
BATCH_FILE="/tmp/ddp-elastic-batch-$$.sh"
cat > "$BATCH_FILE" << EOF
#!/bin/bash
#SBATCH --job-name=ddp-elastic
#SBATCH --nodes=${MIN_NODES}-${MAX_NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --time-min=00:10:00
#SBATCH --output=/tmp/ddp-elastic-%j.out
#SBATCH --error=/tmp/ddp-elastic-%j.err
#SBATCH --export=ALL

export MASTER_ADDR=\$(scontrol show hostname \$SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=\$SLURM_NTASKS
export INTENSITY="${INTENSITY}"
export NUM_SAMPLES="${NUM_SAMPLES}"
export EPOCHS="${EPOCHS}"
export BATCH_SIZE="${BATCH_SIZE}"

echo "============================================================"
echo "  ELASTIC DDP TRAINING"
echo "============================================================"
echo "  Job ID:       \$SLURM_JOB_ID"
echo "  Nodes:        \$SLURM_NNODES (requested: \$SLURM_JOB_NUM_NODES)"
echo "  Tasks:        \$SLURM_NTASKS"
echo "  Node list:    \$SLURM_NODELIST"
echo "  Master:       \$MASTER_ADDR:\$MASTER_PORT"
echo "  World Size:   \$WORLD_SIZE"
echo "  Hostname:     \$(hostname)"
echo "  Start time:   \$(date)"
echo "  Intensity:    \$INTENSITY"
echo "  Num samples:  \$NUM_SAMPLES"
echo "============================================================"
echo ""

srun bash -c '
MAX_WAIT=600
ELAPSED=0
echo "[\$(hostname)] Waiting for PyTorch and training script..."
while ! python3 -c "import torch" 2>/dev/null || [ ! -f /tmp/ddp_test.py ]; do
  if [ \$ELAPSED -ge \$MAX_WAIT ]; then
    echo "[\$(hostname)] TIMEOUT after \${MAX_WAIT}s"
    exit 1
  fi
  echo "[\$(hostname)] Not ready yet, retrying in 15s... (\${ELAPSED}/\${MAX_WAIT}s)"
  sleep 15
  ELAPSED=\$((ELAPSED + 15))
done
echo "[\$(hostname)] Ready after \${ELAPSED}s (torch \$(python3 -c "import torch; print(torch.__version__)"))"

python3 /tmp/ddp_test.py \
    --intensity \$INTENSITY \
    --epochs \$EPOCHS \
    --batch-size \$BATCH_SIZE \
    --num-samples \$NUM_SAMPLES \
    --autoscale \
    --output-dir /tmp/ddp-results
'

JOB_EXIT=\$?
echo ""
echo "============================================================"
echo "  Job completed at: \$(date)"
echo "  Exit code:        \$JOB_EXIT"
if [ \$JOB_EXIT -eq 0 ]; then
    echo "  RESULT: SUCCESS — DDP training ran across \$SLURM_NNODES node(s)"
    echo ""
    echo "  Results saved on batch host: \$(hostname)"
    echo "  Retrieve with:"
    echo "    oc cp slurm/\$(hostname):/tmp/ddp-results results/ -c slurmd"
else
    echo "  RESULT: FAILED (exit code \$JOB_EXIT)"
fi
echo "============================================================"
EOF

sbatch "$BATCH_FILE"
rm -f "$BATCH_FILE"
