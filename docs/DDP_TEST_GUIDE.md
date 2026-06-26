# PyTorch DDP Training Guide — Slurm on OpenShift

## Overview

This guide covers running **distributed PyTorch training** on Slurm-managed OpenShift pods with automatic scaling. The `ddp_test.py --launch` command handles everything: discovering the cluster, calculating resource needs, scaling nodes, provisioning workers, submitting the job, and retrieving results.

### What This Proves

| Capability | What's Validated |
|------------|-----------------|
| **Gang Scheduling** | Slurm allocates all N nodes simultaneously before starting the job |
| **Inter-Pod Communication** | `all_reduce` works across pods over the K8s SDN |
| **Distributed Training** | Gradient synchronization is correct (loss converges consistently) |
| **Autoscaling** | Cluster expands/contracts based on workload requirements |
| **Zero-Config Launch** | Python auto-detects pod memory limits and calculates optimal node count |

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  User's workstation                                            │
│                                                                │
│  python ddp_test.py --launch                                   │
│    1. Discovers cluster (pod mem limits, NodeSet capacity)     │
│    2. Calculates resource plan (intensity, nodes, samples)     │
│    3. Scales NodeSet up if needed                              │
│    4. Provisions workers (pip, PyTorch, scripts)               │
│    5. Submits sbatch job                                       │
│    6. Monitors until completion                                │
│    7. Retrieves results locally                                │
└────────────────────────────┬───────────────────────────────────┘
                             │ oc/kubectl
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  OpenShift Cluster                                             │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Slurm Controller (slurmctld)                            │  │
│  │  - Receives sbatch, schedules across nodes               │  │
│  │  - Elastic: --nodes=MIN-MAX allows flexible allocation   │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                      │
│            ┌────────────┼──────────────┐                       │
│            ▼            ▼              ▼                        │
│     ┌──────────┐ ┌──────────┐  ┌──────────┐                   │
│     │ Worker 0 │ │ Worker 1 │  │ Worker N │                   │
│     │ (slurmd) │ │ (slurmd) │  │ (scaled) │                   │
│     └──────────┘ └──────────┘  └──────────┘                   │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Scale-Down Watchdog (optional)                          │  │
│  │  - Scales NodeSet back to MIN after 5 min idle           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Files

| File | Purpose |
|------|---------|
| `demos/ddp_test.py` | Training script + `--launch` orchestrator |
| `scripts/deploy-slurm.sh` | Deploy the Slurm cluster |
| `scripts/deploy-autoscale.sh` | Deploy the scale-down watchdog (optional) |
| `scripts/cleanup-slurm.sh` | Tear down cluster |
| `configs/slurm-cluster.yaml` | Cluster config (GPU requests, tolerations, memory limits) |

---

## Quick Start

```bash
# Deploy cluster (if not already running)
./scripts/deploy-slurm.sh

# Run DDP training — auto-detects everything:
python demos/ddp_test.py --launch
```

That's it. The `--launch` flag discovers the cluster, calculates nodes needed, scales up, provisions PyTorch, submits the job, monitors it, and retrieves results locally.

**Scale to more nodes** by increasing dataset size:

```bash
python demos/ddp_test.py --launch                      # Default (~2 nodes)
python demos/ddp_test.py --launch --num-samples 10000  # 3 nodes
python demos/ddp_test.py --launch --num-samples 15000  # 4 nodes
python demos/ddp_test.py --launch --num-samples 25000  # 6 nodes
```

Node scaling reference (medium intensity, 4Gi pods):

| `--num-samples` | Dataset memory | Nodes needed |
|-----------------|---------------|--------------|
| 5,000 (default) | 2.9 GB | 2 |
| 10,000 | 5.7 GB | 3 |
| 15,000 | 8.6 GB | 4 |
| 25,000 | 14.4 GB | 6 |
| 40,000 | 22.9 GB | 8 (max) |

**Optional overrides:**

```bash
python demos/ddp_test.py --launch --intensity heavy    # Force intensity
python demos/ddp_test.py --launch --max-nodes 4        # Cap node count
python demos/ddp_test.py --launch --no-monitor         # Fire-and-forget
```

**Optional scale-down watchdog** (reclaims nodes after idle):

```bash
./scripts/deploy-autoscale.sh
```

---

## Prerequisites

- Slurm on OCP deployed and healthy (see [Deployment Guide](DEPLOYMENT_GUIDE.md))
- `oc` CLI logged in with cluster access
- At least 2 Slurm worker pods running
- Python 3.x on your workstation

```bash
# Verify cluster is ready
oc get pods -n slurm
# Expected: slurm-controller-0 (3/3 Running), slurm-worker-slinky-{0,1} (2/2 Running)

oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo
# Expected: slinky-[0-1] in "idle" state
```

---

## Usage

### Automated (Recommended)

```bash
python demos/ddp_test.py --launch
```

This runs these phases:

| Phase | What happens |
|-------|-------------|
| **Discover** | Queries NodeSet for pod memory limits, current replicas, max capacity |
| **Plan** | Selects intensity level, calculates dataset size, determines minimum nodes |
| **Scale** | Scales NodeSet up if more nodes are needed (waits for Ready) |
| **Provision** | Installs pip + PyTorch on workers, copies training script |
| **Submit** | Generates sbatch script with `--nodes=MIN-MAX`, submits to slurmctld |
| **Monitor** | Polls `squeue` until job completes (or timeout) |
| **Retrieve** | Pulls stdout, stderr, and training artifacts to local `results/` |

### Manual Submission (for debugging)

If you need direct control (assumes workers already have PyTorch installed):

```bash
# Light intensity — SmallCNN, 32x32 images, ~1.4 GB RAM, fast (~15s on GPU)
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  sbatch --export=ALL --nodes=2 --ntasks-per-node=1 \
  --output=/tmp/ddp-test-%j.out --error=/tmp/ddp-test-%j.err \
  --wrap="srun python3 /tmp/ddp_test.py --intensity light --autoscale"

# Medium intensity — ResNet-18, 224x224 images, ~2.4 GB RAM/node (with sharding)
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  sbatch --export=ALL --nodes=2 --ntasks-per-node=1 \
  --output=/tmp/ddp-test-%j.out --error=/tmp/ddp-test-%j.err \
  --wrap="srun python3 /tmp/ddp_test.py --intensity medium --num-samples 8192 --autoscale"

# Heavy intensity — ResNet-18, 224x224, larger dataset (~2.9 GB RAM/node, needs 2+ nodes)
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  sbatch --export=ALL --nodes=2 --ntasks-per-node=1 \
  --output=/tmp/ddp-test-%j.out --error=/tmp/ddp-test-%j.err \
  --wrap="srun python3 /tmp/ddp_test.py --intensity heavy --num-samples 10000 --autoscale"
```

| Intensity | Model | Image size | RAM per node (2 nodes) | GPU time |
|-----------|-------|-----------|----------------------|----------|
| `light` | SmallCNN (9.3M params) | 32x32 | ~1.4 GB | ~15s |
| `medium` | ResNet-18 (11.7M params) | 224x224 | ~2.4 GB | ~45s |
| `heavy` | ResNet-18 (11.7M params) | 224x224 | ~2.9 GB | ~90s |

### Monitoring

Open in separate terminals:

```bash
# Autoscaler/watchdog logs
oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler -f --tail=20

# Watch pods
watch -n 5 'oc get pods -n slurm -o wide'

# Slurm node registration (see new nodes appear as idle)
watch -n 10 'oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo -N -l 2>/dev/null'

# Job queue
watch -n 5 'oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue -l 2>/dev/null'

# NodeSet replicas
watch -n 5 'oc get nodeset -n slurm'
```

**macOS:** `watch` is not installed by default. Install via `brew install watch`, or use:
```bash
while true; do clear; oc get pods -n slurm -o wide; sleep 5; done
```

### View Results

```bash
# Find batch host
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job JOB_ID | grep BatchHost

# Read output (on batch host, usually worker-0)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-autoscale-JOB_ID.out

# Copy training artifacts locally
oc cp slurm/slurm-worker-slinky-0:/tmp/ddp-results results/ -c slurmd
```

Results saved per run:

| File | Contents |
|------|----------|
| `model_checkpoint.pt` | Trained model state dict (~36 MB) |
| `metrics.json` | Training config, timing, throughput, Slurm job info |
| `predictions.pt` | Model inference on synthetic batch |

> **Note:** Output is on the **batch host** worker pod, not the controller. The `unexpected EOF` warning from `oc cp` is harmless.

---

## Expected Output

### `--launch` Output (on your workstation)

```
============================================================
  Slurm on OCP — Auto-Launch DDP Training
============================================================

[14:30:01] Discovering cluster...
[14:30:01]   Pod mem limit:  4096 MB
[14:30:01]   Replicas:       2 / max 8
[14:30:01]   Workers online: 2

[14:30:02] Resource plan:
[14:30:02]   Intensity:       medium
[14:30:02]   Num samples:     5000
[14:30:02]   Dataset memory:  2867 MB
[14:30:02]   Nodes needed:    2-8

[14:30:02] Cluster has 2 replicas, need 2 — no scaling required
[14:30:02] Provisioning workers...
[14:30:03] Provisioning complete

[14:30:04] Submitted batch job 42 (nodes: 2-8)
[14:30:14] Job 42: RUNNING (10s)
[14:31:44] Job 42 finished

[14:31:45] Retrieving results...
[14:31:45] RESULT: PASSED
============================================================
```

### Job Output (inside the cluster, 4-node GPU)

```
[Comm Test] all_reduce: got 6.0, expected 6.0 - PASSED
[Bandwidth] all_reduce 64MB: 127.0ms (0.49 GB/s)

============================================================
  Training Configuration
============================================================
  Intensity:       light
  World size:      4
  Device:          cuda:0
  Backend:         nccl
  Model:           SmallCNN
  Model params:    9,356,554
  Dataset size:    8,192
  Batch size/rank: 64
  Global batch:    256
  Num workers:     2
  Epochs:          5
============================================================

  Epoch   1/5 | Loss: 2.3335 | Throughput: 562 samples/s | Time: 3.65s | RSS: 1384MB
  Epoch   2/5 | Loss: 2.3031 | Throughput: 771 samples/s | Time: 2.66s | RSS: 1385MB
  Epoch   3/5 | Loss: 2.3042 | Throughput: 785 samples/s | Time: 2.61s | RSS: 1385MB
  Epoch   4/5 | Loss: 2.3027 | Throughput: 768 samples/s | Time: 2.67s | RSS: 1385MB
  Epoch   5/5 | Loss: 2.3026 | Throughput: 773 samples/s | Time: 2.65s | RSS: 1385MB

============================================================
  Training Complete
============================================================
  Total time:         14.23s
  Avg throughput:     2879 samples/s (global)
  Samples processed:  40,960 (across all ranks)
============================================================

============================================================
  Autoscaling Report
============================================================
  Current world size:     4
  Per-rank throughput:    732 samples/s
  Global throughput:      2927 samples/s
  Communication overhead: -30.3% (first epoch vs steady state)
  Slurm job:              33
  Allocated nodes:        4
  Recommendation:         SCALE UP  - Minimal communication overhead, more nodes would help
  GPU memory utilization: 0.3%
  Process RSS:            1383 MB
============================================================

  TEST PASSED - All ranks completed successfully
```

### Autoscaler Watchdog Log (scale-down after idle)

The watchdog only handles scale-down (scale-up is done by `ddp_test.py --launch`):

```
[19:38:05] POLL: 0 pending, 0 running — idle timer started (4 replicas)
[19:39:05] POLL: 0 pending, 0 running — idle 60s / 300s
[19:42:05] POLL: 0 pending, 0 running — idle 240s / 300s
[19:43:05] IDLE: no jobs for 300s, scaling down
[19:43:05] SCALING: slurm-worker-slinky -> 2 replicas
```

### Key Metrics

- **Throughput**: GPU ~400-1000 samples/s per rank, CPU ~40 samples/s
- **Bandwidth**: 0.5-1 GB/s over K8s SDN (InfiniBand would be 10-50 GB/s)
- **Loss**: Should decrease across epochs (proves gradient sync is correct)
- **Communication overhead**: Low % means DDP comms are cheap relative to compute; negative values mean steady-state is faster than first epoch (DDP bucket rebuilding overhead amortizes)

The `.err` file should only contain `UserWarning: Failed to initialize NumPy` — harmless.

---

## Autoscaling Flow

When you request more nodes than exist (e.g., `--num-samples 15000` needs 4 nodes but only 2 are running):

1. `--launch` scales the NodeSet from 2 to 4 replicas
2. Slinky operator creates worker pods 2 and 3
3. `--launch` waits for pods to reach Ready state
4. `--launch` provisions new workers (pip + PyTorch + script)
5. Submits job with `--nodes=4-8`
6. Slurm schedules across all 4 nodes
7. DDP training runs, results retrieved
8. Watchdog (if deployed) scales back to 2 after 5 min idle

If new pods can't schedule (no free GPUs), `--launch` will timeout and report which pods are stuck. See Troubleshooting for fixes.

---

## Configuration

### Scale-Down Watchdog

Env vars in `configs/slurm-autoscaler.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_REPLICAS` | 2 | Floor for scale-down |
| `MAX_REPLICAS` | 8 | Ceiling for scale-up |
| `POLL_INTERVAL` | 30s | Seconds between `squeue` checks |
| `SCALE_DOWN_DELAY` | 300s | Idle seconds before scaling down |
| `PYTORCH_INDEX` | `https://download.pytorch.org/whl/cu124` | PyTorch package index |

After editing:

```bash
oc apply -f configs/slurm-autoscaler.yaml
oc rollout restart deployment slurm-autoscaler -n slurm
```

### Using CPU-Only PyTorch

Change `PYTORCH_INDEX` to `https://download.pytorch.org/whl/cpu`.

### Updating Scripts

```bash
./scripts/deploy-autoscale.sh
oc rollout restart deployment slurm-autoscaler -n slurm
```

### Disabling Autoscaling

```bash
oc delete -f configs/slurm-autoscaler.yaml
oc delete configmap slurm-autoscaler-script -n slurm
oc scale nodeset slurm-worker-slinky --replicas=2 -n slurm
```

---

## Troubleshooting

### Clear stuck jobs

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue
oc exec -n slurm slurm-controller-0 -c slurmctld -- scancel --state=PENDING -u slurm
```

**Root cause:** Slurm tries `su -l` for environment retrieval which fails in containers. Fix: `#SBATCH --export=ALL` (already included in all submission paths).

### Scaled-up workers stuck in Pending (no GPUs)

```bash
oc describe pod slurm-worker-slinky-2 -n slurm | grep -A 3 "FailedScheduling"

# Fix: scale back and cancel pending jobs to break the autoscaler loop
oc scale nodeset slurm-worker-slinky -n slurm --replicas=2
oc exec -n slurm slurm-controller-0 -c slurmctld -- scancel --state=PENDING -u slurm
```

### Job PENDING after scale-up (nodes drained)

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo -N -l | grep drain
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  scontrol update nodename=ALL state=resume reason="cleared"
```

### Workers don't have PyTorch

Check logs: `oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler --tail=50 | grep PROVISION`

Manual fix:
```bash
oc exec -n slurm slurm-worker-slinky-2 -c slurmd -- \
  bash -c "apt-get update -qq && apt-get install -y -qq python3-pip"
oc exec -n slurm slurm-worker-slinky-2 -c slurmd -- \
  pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu124
```

### Connection timeout during distributed init

```bash
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- getent hosts slinky-1
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- bash -c "echo | nc -w 3 slinky-1 29500"
```

### Slurm auth errors after scaling

Wait 30s for pods to re-register, or:
```bash
oc delete pod slurm-controller-0 -n slurm
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  scontrol update nodename=ALL state=resume reason="cleared"
```

### Output file not found

Output is on the **batch host** (first node in allocation), not the controller:
```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job JOB_ID | grep BatchHost
```

---

## How It Works

### `ddp_test.py --launch` (orchestrator)

Runs on your workstation. Discovers cluster state via `oc`, calculates the highest intensity that fits in pod memory, scales the NodeSet if needed, provisions workers (pip + PyTorch + script copy), generates an sbatch file with `--nodes=MIN-MAX --export=ALL`, submits to slurmctld, polls `squeue` until completion, then retrieves results locally.

### `ddp_test.py` (training, inside cluster)

Auto-detects Slurm (`SLURM_PROCID`) or torchrun (`RANK`) launch mode. Picks NCCL for GPU / Gloo for CPU. Runs: communication test (`all_reduce` correctness) → bandwidth test (32MB tensor timing) → DDP training (CNN on synthetic data, gradient sync across ranks) → saves checkpoint + metrics + predictions.

### Scale-down watchdog (`autoscaler-loop.sh`)

Runs inside a `bitnami/kubectl` pod. Every 30s, execs into slurmctld and checks `squeue` for pending jobs. If no pending jobs for 5 minutes, scales the NodeSet back to `MIN_REPLICAS`. Does NOT handle scale-up (that's done by `--launch` before submission).

---

## Manual Setup (Without `--launch`)

For debugging or environments where you can't run Python locally:

### Install PyTorch on Workers

```bash
for i in 0 1; do
  oc exec -n slurm slurm-worker-slinky-$i -c slurmd -- \
    bash -c "apt-get update -qq && apt-get install -y -qq python3-pip"
  oc exec -n slurm slurm-worker-slinky-$i -c slurmd -- \
    pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu124
done
```

For CPU-only: use `https://download.pytorch.org/whl/cpu`. If switching from CPU to CUDA, add `--force-reinstall`.

### Copy Script and Submit

```bash
oc cp demos/ddp_test.py slurm/slurm-worker-slinky-0:/tmp/ddp_test.py -c slurmd
oc cp demos/ddp_test.py slurm/slurm-worker-slinky-1:/tmp/ddp_test.py -c slurmd

oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  sbatch --export=ALL --nodes=2 --ntasks-per-node=1 \
  --output=/tmp/ddp-test-%j.out --error=/tmp/ddp-test-%j.err \
  --wrap="srun python3 /tmp/ddp_test.py --intensity light --autoscale --output-dir /tmp/ddp-results"
```

### View Output

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-test-JOB_ID.out
```

---

## Next Steps

1. **Larger model** — Swap SmallCNN for ResNet-50 or use FSDP for model-parallel training
2. **Real dataset** — Replace synthetic data with domain-specific data
3. **Custom container image** — Pre-install PyTorch to eliminate runtime provisioning
4. **Multi-tenancy** — Configure Slurm accounts, QoS policies, and fairshare scheduling
5. **Higher scale** — Increase `MAX_REPLICAS` and test with 8+ node jobs
