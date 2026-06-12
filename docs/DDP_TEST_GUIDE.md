# PyTorch DDP Test Guide - Slurm on OpenShift

## Overview

This guide walks through running a **distributed PyTorch training test** on Slurm-managed OpenShift pods. The test validates that Slurm's gang scheduling and inter-pod collective communication (via Gloo/NCCL) work correctly over the Kubernetes network — the foundation for scaling to larger GPU workloads.

### What This Test Proves

| Capability | What's Validated |
|------------|-----------------|
| **Gang Scheduling** | Slurm allocates all N nodes simultaneously before starting the job |
| **Inter-Pod Communication** | `all_reduce` works across pods over the K8s SDN |
| **Distributed Training** | Gradient synchronization is correct (loss converges consistently) |
| **Slurm Job Lifecycle** | Submit → queue → allocate → execute → complete works end-to-end |

### Files

| File | Purpose |
|------|---------|
| `demos/ddp_test.py` | PyTorch DDP training script (torchrun + Slurm compatible) |
| `demos/submit_job.sh` | Slurm batch submission script (fixed node count) |
| `demos/submit_job_autoscale.sh` | Elastic submission script (variable node count, `--nodes=min-max`) |
| `configs/slurm-autoscaler.yaml` | KEDA ScaledObject + Slurm REST API for NodeSet autoscaling |

---

## Prerequisites

- Slurm on OCP deployed and healthy (see [Deployment Guide](DEPLOYMENT_GUIDE.md))
- `oc` CLI logged in with cluster access
- At least 2 Slurm worker pods running

```bash
# Verify cluster is ready
oc get pods -n slurm
# Expected: slurm-controller-0 (3/3 Running), slurm-worker-slinky-{0,1} (2/2 Running)

# Verify Slurm nodes are idle
oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo
# Expected: slinky-[0-1] in "idle" state
```

---

## Step 1: Clear Any Stuck Jobs

Previous jobs may be stuck with "user env retrieval failed requeued held" — a common issue in containerized Slurm where login environment resolution fails.

```bash
# Check for stuck jobs
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue

# Cancel all stuck pending jobs (if any)
oc exec -n slurm slurm-controller-0 -c slurmctld -- scancel --state=PENDING -u slurm

# Verify queue is empty
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue
```

**Root Cause:** Slurm tries to retrieve the user's login environment via `su -l` which fails in containers. The fix is `--export=ALL` in the sbatch script (already included in `submit_job.sh`).

---

## Step 2: Configure GPU Access (Required for GPU Testing)

By default, Slurm worker pods don't request GPU resources and may land on CPU-only nodes. To run the DDP test on GPUs, you need to patch the NodeSet to:
1. Request `nvidia.com/gpu` resources (so the pod gets a GPU allocated)
2. Add a `nodeSelector` targeting GPU-labeled nodes
3. Add a `toleration` for any GPU node taints

**Check if your cluster has GPUs:**

```bash
# List nodes with GPU allocatable resources
oc get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'

# Check for GPU node labels
oc get nodes -l nvidia.com/gpu.present=true -o name

# Check GPU node taints (note the taint key for the toleration below)
oc get nodes -l nvidia.com/gpu.present=true -o custom-columns='NAME:.metadata.name,TAINTS:.spec.taints[*].key'
```

**Patch the NodeSet to request GPUs:**

```bash
# Add GPU resource requests/limits and target GPU nodes
oc patch nodeset slurm-worker-slinky -n slurm --type='merge' -p '{
  "spec": {
    "slurmd": {
      "resources": {
        "limits": {
          "cpu": "2",
          "memory": "4Gi",
          "nvidia.com/gpu": "1"
        },
        "requests": {
          "cpu": "1",
          "memory": "2Gi",
          "nvidia.com/gpu": "1"
        }
      }
    },
    "template": {
      "spec": {
        "nodeSelector": {
          "kubernetes.io/os": "linux",
          "nvidia.com/gpu.present": "true"
        },
        "tolerations": [
          {
            "key": "g5-gpu",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule"
          }
        ]
      }
    }
  }
}'
```

**Note:** The `tolerations` key (`g5-gpu` above) must match your cluster's GPU node taint. Check with:
```bash
oc get nodes <gpu-node-name> -o jsonpath='{.spec.taints}'
```

**Wait for pods to restart on GPU nodes:**

```bash
# Watch pods roll out (old pods terminate, new ones start on GPU nodes)
oc get pods -n slurm -o wide -w

# Verify workers are on GPU nodes
oc get pods -n slurm -o wide | grep worker
# NODE column should show GPU-labeled nodes
```

**Revert to CPU-only (if needed):**

```bash
# Remove GPU requests and node targeting
oc patch nodeset slurm-worker-slinky -n slurm --type='merge' -p '{
  "spec": {
    "slurmd": {
      "resources": {
        "limits": {
          "cpu": "2",
          "memory": "4Gi"
        },
        "requests": {
          "cpu": "1",
          "memory": "2Gi"
        }
      }
    },
    "template": {
      "spec": {
        "nodeSelector": {
          "kubernetes.io/os": "linux"
        },
        "tolerations": []
      }
    }
  }
}'
```

---

## Step 3: Install PyTorch on Worker Pods

The default Slurm container images don't include PyTorch. Install the CPU-only version on each worker (this is ephemeral — lost on pod restart).

```bash
# Install pip on both workers
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- \
  bash -c "apt-get update -qq && apt-get install -y -qq python3-pip"

oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- \
  bash -c "apt-get update -qq && apt-get install -y -qq python3-pip"

# Install PyTorch (CPU-only, ~200MB)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- \
  pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu

oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- \
  pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
```

**Verify installation:**

```bash
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- \
  python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'Gloo: {torch.distributed.is_gloo_available()}')"
# Expected: PyTorch 2.x, Gloo: True
```

**Note:** For GPU pods, install the CUDA version instead:

**Important:** If you previously installed the CPU-only version (from `https://download.pytorch.org/whl/cpu`), you must **force-reinstall** — otherwise pip will see the existing CPU version as "already satisfied" and skip the install. Additionally, the worker pods themselves must have GPUs exposed via the NVIDIA GPU Operator and device plugin on OpenShift. Without actual GPU hardware allocated to the pods, `torch.cuda.is_available()` will return `False` regardless of which PyTorch version is installed.

```bash
# Install pip on both GPU workers
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- \
  bash -c "apt-get update -qq && apt-get install -y -qq python3-pip"

oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- \
  bash -c "apt-get update -qq && apt-get install -y -qq python3-pip"

# Install PyTorch with CUDA 12.4 support (~2.5GB per node)
# Use --force-reinstall if a CPU-only version was previously installed
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- \
  pip3 install --break-system-packages --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124

oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- \
  pip3 install --break-system-packages --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124

# Verify GPU installation
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- \
  python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'NCCL: {torch.distributed.is_nccl_available()}')"
# Expected: PyTorch 2.x+cu124, CUDA available: True, NCCL: True
```

**If CUDA shows False after installing the CUDA version:**
- The pods don't have GPU hardware allocated. You need:
  1. GPU-equipped nodes in your OpenShift cluster
  2. NVIDIA GPU Operator installed (exposes GPUs to pods via device plugin)
  3. Slurm GRES configured for GPUs (`--gres=gpu:1` in sbatch)
- The `ddp_test.py` script auto-detects the environment — it uses NCCL when GPUs are present and falls back to Gloo on CPU-only pods. No code changes needed.

**Uninstall PyTorch (CPU or GPU):**

If you need to remove PyTorch entirely (e.g. to switch between CPU and CUDA versions, or to clean up):

```bash
# Uninstall PyTorch and its dependencies from both workers
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- \
  pip3 uninstall --break-system-packages -y torch torchvision torchaudio

oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- \
  pip3 uninstall --break-system-packages -y torch torchvision torchaudio

# Verify removal
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- \
  python3 -c "import torch" 2>&1
# Expected: ModuleNotFoundError: No module named 'torch'
```

---

## Step 4: Copy Scripts to the Cluster

The training script goes to both workers (where `srun` executes). The submit script goes to the controller (where `sbatch` runs).

```bash
# Copy training script to both worker nodes
oc cp demos/ddp_test.py slurm/slurm-worker-slinky-0:/tmp/ddp_test.py -c slurmd
oc cp demos/ddp_test.py slurm/slurm-worker-slinky-1:/tmp/ddp_test.py -c slurmd

# Copy submission script to controller
oc cp demos/submit_job.sh slurm/slurm-controller-0:/tmp/submit_job.sh -c slurmctld
```

---

## Step 5: Submit the Job

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch /tmp/submit_job.sh
# Expected: "Submitted batch job <JOB_ID>"
```

---

## Step 6: Monitor the Job

```bash
# Check queue (should show RUNNING on 2 nodes)
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue

# Check job details (replace 24 with your actual job ID from sbatch output)
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job 24

# Watch live output (on the batch host node)
# Find batch host from scontrol output (BatchHost field), then:
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- tail -f /tmp/ddp-test-24.out
```

**Note:** Replace `24` in the commands above with your actual job ID (printed by `sbatch`).

**Expected queue output:**
```
JOBID PARTITION  NAME     USER ST  TIME  NODES NODELIST(REASON)
   21       all  ddp-test slurm  R  0:12     2  slinky-[0-1]
```

---

## Step 7: View Results

Once the job completes (5 epochs on GPU takes ~25 seconds, on CPU takes ~9 minutes):

```bash
# Find which node was batch host
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job 24 | grep BatchHost

# Read output (substitute the batch host node — replace 24 with your job ID)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-test-24.out

# Read errors (should only show numpy warning — harmless)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-test-24.err
```

**Note:** Replace `24` with your actual job ID. Output is written to the **batch host** node — check `BatchHost` in the `scontrol` output to know which worker pod to read from.

---

## Expected Output

### CPU Mode (Gloo backend)

A successful CPU run looks like this:

```
============================================================
  Job ID:       21
  Nodes:        2
  Tasks:        2
  Node list:    slinky-[1,0]
  Master:       slinky-1:29500
  World Size:   2
============================================================

############################################################
  Slurm on OCP - Distributed Training Test
############################################################
  Rank 0/2 on cpu
  Slurm Job ID: 21
  Slurm Nodes:  slinky-[1,0]
  CUDA available: False

[Timer] Communication Test...
[Comm Test] all_reduce: got 1.0, expected 1.0 - PASSED
[Timer] Communication Test: 0.00s
[Timer] Bandwidth Test...
[Bandwidth] all_reduce 32MB: 55.4ms (0.56 GB/s)
[Timer] Bandwidth Test: 0.09s
[Timer] Training...

============================================================
  Training Configuration
============================================================
  World size:      2
  Device:          cpu
  Backend:         gloo
  Model params:    9,356,554
  Dataset size:    8,192
  Batch size/rank: 64
  Global batch:    128
  Epochs:          5
============================================================

  Epoch   1/5 | Loss: 2.3479 | Throughput: 42 samples/s | Time: 97.94s
  Epoch   2/5 | Loss: 2.3028 | Throughput: 39 samples/s | Time: 105.53s
  Epoch   3/5 | Loss: 2.3036 | Throughput: 30 samples/s | Time: 137.59s
  Epoch   4/5 | Loss: 2.3030 | Throughput: 38 samples/s | Time: 107.44s
  Epoch   5/5 | Loss: 2.3028 | Throughput: 38 samples/s | Time: 107.16s

============================================================
  Training Complete
============================================================
  Total time:         555.66s
  Avg throughput:     74 samples/s (global)
  Samples processed:  40,960 (across all ranks)
============================================================

[Timer] Training: 556.58s
############################################################
  TEST PASSED - All ranks completed successfully
############################################################
```

### GPU Mode (NCCL backend)

A successful GPU run looks like this:

```
============================================================
  Job ID:       23
  Nodes:        2
  Tasks:        2
  Node list:    slinky-[0-1]
  Master:       slinky-0:29500
  World Size:   2
============================================================

############################################################
  Slurm on OCP - Distributed Training Test
############################################################
  Rank 0/2 on cuda:0
  Slurm Job ID: 23
  Slurm Nodes:  slinky-[0-1]
  CUDA available: True
  GPU: NVIDIA A10G
  GPU Memory: 23.7 GB

[Timer] Communication Test...
[Comm Test] all_reduce: got 1.0, expected 1.0 - PASSED
[Timer] Communication Test: 0.39s
[Timer] Bandwidth Test...
[Bandwidth] all_reduce 32MB: 58.7ms (0.53 GB/s)
[Timer] Bandwidth Test: 0.07s
[Timer] Training...

============================================================
  Training Configuration
============================================================
  World size:      2
  Device:          cuda:0
  Backend:         nccl
  Model params:    9,356,554
  Dataset size:    8,192
  Batch size/rank: 64
  Global batch:    128
  Epochs:          5
============================================================

  Epoch   1/5 | Loss: 2.3448 | Throughput: 908 samples/s | Time: 4.51s
  Epoch   2/5 | Loss: 2.3028 | Throughput: 1009 samples/s | Time: 4.06s
  Epoch   3/5 | Loss: 2.3029 | Throughput: 996 samples/s | Time: 4.11s
  Epoch   4/5 | Loss: 2.3028 | Throughput: 1006 samples/s | Time: 4.07s
  Epoch   5/5 | Loss: 2.3024 | Throughput: 987 samples/s | Time: 4.15s

============================================================
  Training Complete
============================================================
  Total time:         20.90s
  Avg throughput:     1959 samples/s (global)
  Samples processed:  40,960 (across all ranks)
============================================================

[Timer] Training: 22.43s
############################################################
  TEST PASSED - All ranks completed successfully
############################################################
```

### Performance Comparison

| Metric | CPU (Gloo) | GPU (NCCL) | Speedup |
|--------|-----------|-----------|---------|
| Total time | 555s | 21s | **26x** |
| Throughput | 74 samples/s | 1,959 samples/s | **26x** |
| Per-epoch | ~105s | ~4s | **26x** |
| Backend | Gloo | NCCL | - |
| Device | CPU | NVIDIA A10G (24 GB) | - |

---

## Understanding the Results

### Test Phases

| Phase | What It Does | Success Criteria |
|-------|-------------|------------------|
| **Communication Test** | Each rank sends its ID via `all_reduce`; verifies sum is correct | Sum matches `N*(N-1)/2` |
| **Bandwidth Test** | Times a 32MB `all_reduce` across all ranks | Completes without timeout; reports GB/s |
| **Training** | Runs CNN on synthetic data with DDP gradient sync | Loss decreases; all epochs complete |

### Key Metrics

- **Throughput (samples/s)**: How fast each rank processes data. CPU pods: ~40 samples/s. GPU pods: expect 50-100x more.
- **Bandwidth (GB/s)**: Inter-pod communication speed over K8s network. 0.5-1 GB/s is normal for pod-to-pod. InfiniBand HPC would be 10-50 GB/s.
- **Loss convergence**: Loss should decrease over epochs, proving gradients are synchronized correctly across ranks.

---

## Troubleshooting

### "user env retrieval failed requeued held"

**Cause:** Slurm can't resolve user login environment in the container.
**Fix:** Ensure `#SBATCH --export=ALL` is in the submit script. Cancel stuck jobs with `scancel --state=PENDING -u slurm`.

### Job stays in PD (pending) state

```bash
# Check reason
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue -o "%.10i %.10P %.15j %.8u %.2t %.10M %.6D %R"
```

Common reasons:
- **(Resources)** — Not enough nodes available. Check `sinfo` for node states.
- **(Priority)** — Another job is ahead in queue. Wait or cancel it.

### "No module named 'torch'"

PyTorch not installed on the worker that ran the job. Re-run Step 3 for that worker.

### Connection timeout during distributed init

The ranks can't reach each other over the network. Check:
```bash
# Verify pods can resolve each other's hostnames
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- getent hosts slinky-1
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- getent hosts slinky-0

# Verify port 29500 is reachable between pods
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- bash -c "echo | nc -w 3 slinky-1 29500"
```

### Job completes but output file not found

Output is written to the **batch host** node (the first node in the allocation). Check `BatchHost` in `scontrol show job <ID>` and read from that specific worker pod.

---

## Autoscaling

The DDP test supports autoscaling at two levels: **cluster-level** (Slinky/KEDA scales the NodeSet based on job demand) and **job-level** (Slurm allocates a variable number of nodes per job). Together, these allow the cluster to expand when workloads arrive and contract when idle.

### Architecture

```
                         ┌──────────────┐
                         │  User submits│
                         │  sbatch job  │
                         └──────┬───────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  Slurm Controller                                           │
│  - Sees pending job, needs N nodes                          │
│  - Elastic: --nodes=1-4 allows partial starts               │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼────────────────┐
          ▼            ▼                ▼
   ┌────────────┐ ┌────────────┐  ┌────────────┐
   │ Prometheus │ │   KEDA     │  │  Slinky    │
   │ scrapes    │→│ ScaledObj  │→ │  Operator  │
   │ slurm_*    │ │ threshold  │  │  scales    │
   │ metrics    │ │ triggers   │  │  NodeSet   │
   └────────────┘ └────────────┘  └────────────┘
                                        │
                       ┌────────────────┼──────────────┐
                       ▼                ▼              ▼
                ┌──────────┐     ┌──────────┐   ┌──────────┐
                │ Worker 0 │     │ Worker 1 │   │ Worker N │
                │ (slurmd) │     │ (slurmd) │   │ (scaled) │
                └──────────┘     └──────────┘   └──────────┘
```

### Prerequisites

Autoscaling requires three additional components beyond the base Slurm/Slinky deployment:

1. **Prometheus** — scrapes Slurm REST API metrics (`slurm_partition_jobs_pending`, etc.)
2. **KEDA** — bridges Prometheus metrics to Kubernetes HPA, supports scale-to-zero
3. **Slurm REST API** — exposes Slurm metrics (deployed via `configs/slurm-autoscaler.yaml`)

### Step 1: Install Prometheus and KEDA

```bash
# Prometheus (skip if OpenShift monitoring is already enabled)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace

# KEDA
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda -n keda --create-namespace

# Verify both are running
oc get pods -n monitoring | grep prometheus
oc get pods -n keda | grep keda
```

### Step 2: Deploy the Autoscaler

```bash
# Deploys: Slurm REST API + ServiceMonitor + KEDA ScaledObject
oc apply -f configs/slurm-autoscaler.yaml

# Verify the REST API pod is running
oc get pods -n slurm -l app.kubernetes.io/name=slurmrestd

# Verify KEDA created the HPA
oc get hpa -n slurm
# Expected: keda-hpa-scale-slurm-workers targeting NodeSet
```

The default autoscaler configuration:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `idleReplicaCount` | 1 | Standby nodes when no jobs are pending |
| `minReplicaCount` | 2 | Minimum nodes when jobs are active |
| `maxReplicaCount` | 8 | Maximum nodes under heavy demand |
| `cooldownPeriod` | 300s | Seconds of inactivity before scaling to idle |
| `pollingInterval` | 15s | How often KEDA checks Prometheus |

Edit `configs/slurm-autoscaler.yaml` to adjust these values for your cluster.

### Step 3: Submit an Elastic Job

The elastic submit script uses `--nodes=min-max` so Slurm starts the job as soon as the minimum node count is available, rather than waiting for the maximum:

```bash
# Copy elastic submit script to controller
oc cp demos/submit_job_autoscale.sh slurm/slurm-controller-0:/tmp/submit_job_autoscale.sh -c slurmctld

# Copy training script to ALL workers (including any that may scale up)
# For a static 2-node test:
oc cp demos/ddp_test.py slurm/slurm-worker-slinky-0:/tmp/ddp_test.py -c slurmd
oc cp demos/ddp_test.py slurm/slurm-worker-slinky-1:/tmp/ddp_test.py -c slurmd

# Submit the elastic job
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch /tmp/submit_job_autoscale.sh
```

Key differences from `submit_job.sh`:

| Feature | `submit_job.sh` | `submit_job_autoscale.sh` |
|---------|----------------|--------------------------|
| Node count | `--nodes=2` (fixed) | `--nodes=1-4` (elastic range) |
| Preemption | None | `--requeue` (re-enqueue on preemption) |
| Backfill | No | `--time-min` enables backfill scheduling |
| Metrics | Standard output | `--autoscale` flag adds scaling report |

**Override the node range at submission time:**

```bash
# Run with 2-8 nodes
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  sbatch --nodes=2-8 /tmp/submit_job_autoscale.sh

# Run on GPU partition with autoscaling
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  sbatch --nodes=1-4 --gres=gpu:1 --partition=gpu /tmp/submit_job_autoscale.sh
```

### Step 4: Monitor Autoscaling Behavior

```bash
# Watch NodeSet replica count change in real time
oc get nodeset slurm-worker-slinky -n slurm -w

# Watch pods scale up/down
oc get pods -n slurm -l nodeset.slinky.slurm.net/name=slurm-worker-slinky -w

# Check KEDA ScaledObject status
oc describe scaledobject scale-slurm-workers -n slurm

# Check HPA scaling decisions
oc describe hpa -n slurm

# View Slurm's perspective on node states
oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo -N -l
```

### Step 5: View Autoscaling Report

When `--autoscale` is passed to `ddp_test.py`, the job output includes a scaling efficiency report after training completes:

```
============================================================
  Autoscaling Report
============================================================
  Current world size:     4
  Per-rank throughput:    38 samples/s
  Global throughput:      152 samples/s
  Scaling efficiency:     95.2%
  Slurm job:              42
  Node range:             1-4
  Recommendation:         SCALE UP  - High efficiency, adding nodes would increase throughput
  Host memory usage:      34.2%
============================================================

[Autoscale] Metrics written to /tmp/ddp-autoscale-metrics.json
```

The report also writes a machine-readable JSON metrics file (`/tmp/ddp-autoscale-metrics.json`) that external tools can consume.

### Autoscaling Scenarios

**Scenario 1: Burst of jobs**
1. User submits 5 DDP jobs with `sbatch submit_job_autoscale.sh`
2. Prometheus reports `slurm_partition_jobs_pending = 5`
3. KEDA scales NodeSet from idle (1) to max (8) over ~60s
4. Slurm scheduler assigns nodes to jobs as they become available
5. Jobs complete, pending count drops to 0
6. After 5 minutes of no pending jobs, KEDA scales back to idle (1)

**Scenario 2: Single elastic job**
1. User submits `sbatch --nodes=2-8 submit_job_autoscale.sh`
2. Only 3 nodes are currently available — Slurm starts the job with 3 nodes
3. KEDA sees the pending node demand and scales up the NodeSet
4. Slurm cannot dynamically add nodes mid-job (elastic allocation is at submit time), but the next job benefits from the expanded pool

**Scenario 3: Priority preemption**
1. A low-priority job is running on 4 nodes
2. A high-priority job is submitted needing 4 nodes
3. Slurm preempts the low-priority job (`--requeue` causes it to re-enter the queue)
4. KEDA detects pending jobs and scales up for both to run

### Customizing the Autoscaler

**Change the Prometheus query** to target specific partitions or use different metrics:

```yaml
# Scale based on CPU-seconds of pending work (more nuanced than job count)
triggers:
  - type: prometheus
    metadata:
      query: slurm_partition_cpus_pending{partition="all"}
      threshold: "4"
```

**Add a separate GPU autoscaler** — uncomment the optional ScaledObject in `configs/slurm-autoscaler.yaml` and adjust for your GPU NodeSet name and partition.

**Scale to zero** — set `idleReplicaCount: 0` in the ScaledObject to completely remove worker pods when no jobs are pending. KEDA handles the 0-to-1 transition when a new job arrives.

### Disabling Autoscaling

```bash
# Remove the autoscaler (keeps static NodeSet replicas)
oc delete -f configs/slurm-autoscaler.yaml

# Set a fixed replica count
oc scale nodeset slurm-worker-slinky --replicas=2 -n slurm
```

---

## Next Steps

Once this test passes, you've validated the core infrastructure. Next steps toward production GPU workloads:

1. **GPU version** — Use pods with NVIDIA GPU operator, install full PyTorch (with CUDA), switch to NCCL backend
2. **Larger model** — Swap `SmallCNN` for ResNet-50 or use FSDP for model-parallel training
3. **Real dataset** — Replace synthetic data with Tiny-ImageNet or domain-specific data
4. **Custom container image** — Build a PyTorch image with all dependencies pre-installed (no runtime `pip install`)
5. **Multi-node scaling test** — Add more worker nodes and measure linear scaling efficiency
6. **Autoscaled workloads** — Deploy the KEDA autoscaler (`configs/slurm-autoscaler.yaml`) and use elastic job submission to let the cluster scale with demand

See the **Roadmap After Test Success** section below for next steps.

---

## How the Test Works

### `submit_job.sh` — Launching the Job

The batch script requests 2 nodes with gang scheduling (`--nodes=2`), one task per node, and a 15-minute time limit. Before launching the Python script, it derives the master address from Slurm's node list:

```bash
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
```

`srun` then launches one copy of `ddp_test.py` per node, with Slurm injecting `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_NODELIST`, and `SLURM_LOCALID` into each process.

### `ddp_test.py` — Script Internals

**Setup:** The script auto-detects its launch mode — Slurm (via `SLURM_PROCID`), torchrun (via `RANK`), or single-process fallback. It picks NCCL for GPU or Gloo for CPU, then calls `dist.init_process_group()` to connect all ranks through `MASTER_ADDR:MASTER_PORT`.

**Phase 1 — Communication Test:** Each rank creates a tensor containing its rank value. An `all_reduce(SUM)` should produce `N*(N-1)/2`. If it doesn't match, the script aborts — no point continuing if basic communication is broken.

**Phase 2 — Bandwidth Test:** A 32 MB `all_reduce` is timed to measure inter-pod throughput. Typical result is ~0.5 GB/s over K8s SDN, which is 10–50x slower than InfiniBand but adequate for compute-bound workloads.

**Phase 3 — Training:** A small CNN (~1.2M parameters) trains on synthetic 3x32x32 images (8,192 samples) using DDP. A `DistributedSampler` shards data across ranks. DDP hooks into the backward pass to `all_reduce` gradients before each optimizer step, keeping weights identical across ranks. Decreasing loss across epochs proves gradient sync is correct.

**Phase 4 — Autoscaling Report (optional):** When `--autoscale` is passed, the script collects per-rank resource utilization (GPU memory, host memory, CPU count) and computes scaling efficiency metrics. It compares per-rank throughput across epochs to determine whether adding or removing nodes would improve overall performance. Results are printed and also written to `/tmp/ddp-autoscale-metrics.json` for consumption by external tooling.

CLI args: `--epochs` (default 5), `--batch-size` (default 64), `--num-samples` (default 8192), `--autoscale` (enable scaling report).

### `submit_job_autoscale.sh` — Elastic Job Submission

The elastic variant uses `--nodes=1-4` (min-max syntax) so Slurm can start the job as soon as the minimum node count is available. This pairs with KEDA-driven NodeSet autoscaling: when the job is pending, KEDA detects `slurm_partition_jobs_pending > 0` and scales up the NodeSet. The `--requeue` flag allows Slurm to re-enqueue the job if it's preempted by higher-priority work, and `--time-min` enables backfill scheduling.

---

## Why This Test Matters

The DDP test validates the core value proposition of Slurm on OpenShift: multi-node distributed training that pure Kubernetes cannot do without significant workarounds.

### What Kubernetes Alone Cannot Do

| Capability | Pure K8s | Slurm on OCP |
|------------|----------|--------------|
| Gang scheduling | Pods start independently, risking deadlock | `sbatch -N 2` guarantees all-or-nothing |
| Job queue with fairshare | No built-in queueing | Native `squeue`, priority, backfill |
| Multi-node coordination | Manual StatefulSet + DNS setup | Automatic `$SLURM_*` env vars |
| Batch job accounting | Custom Prometheus exporters | Native `sacct` with runtime, memory, exit codes |
| User experience | Write YAML manifests | `sbatch job.sh` |

### Roadmap After Test Success

1. **Custom container images** — Build a PyTorch image with all dependencies pre-installed to eliminate runtime `pip install` and ensure reproducibility.
2. **Real ML workloads** — Move from the small CNN to ResNet-50 on Tiny-ImageNet with FSDP, or 3D U-Net for medical imaging.
3. **Hybrid cloud** — Add bare-metal GPU nodes to the Slurm cluster so jobs can burst beyond OpenShift using the same `sbatch` scripts (see [Add Nodes Guide](ADD_NODES.md)).
4. **Multi-tenancy** — Configure Slurm accounts, QoS policies, and fairshare scheduling for shared environments with `sacct`-based chargeback.
