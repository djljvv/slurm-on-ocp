# PyTorch DDP Training Guide — Slurm on OpenShift

## Overview

This guide walks through running **distributed PyTorch training** on Slurm-managed OpenShift pods with **automatic scaling**. The autoscaler handles provisioning workers with PyTorch, scaling the cluster to meet demand, and scaling back down when idle.

### What This Proves

| Capability | What's Validated |
|------------|-----------------|
| **Gang Scheduling** | Slurm allocates all N nodes simultaneously before starting the job |
| **Inter-Pod Communication** | `all_reduce` works across pods over the K8s SDN |
| **Distributed Training** | Gradient synchronization is correct (loss converges consistently) |
| **Autoscaling** | Cluster expands/contracts based on job demand |
| **Auto-Provisioning** | New workers get PyTorch + scripts without manual intervention |

### Architecture

```
                         ┌──────────────┐
                         │  User submits│
                         │  sbatch job  │
                         └──────┬───────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│  Slurm Controller (slurmctld)                               │
│  - Queues the job, marks it PENDING if nodes unavailable    │
│  - Elastic: --nodes=1-4 allows partial starts               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Autoscaler Pod │
              │  (polls squeue) │
              │                 │
              │  Pending > 0?   │
              │  → kubectl scale│
              │    NodeSet up   │
              │                 │
              │  Provisions new │
              │  workers with   │
              │  PyTorch + deps │
              │                 │
              │  Idle > 5 min?  │
              │  → kubectl scale│
              │    NodeSet down │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Slinky Operator │
              │ reconciles      │
              │ NodeSet replicas│
              └────────┬────────┘
                       │
          ┌────────────┼──────────────┐
          ▼            ▼              ▼
   ┌──────────┐ ┌──────────┐  ┌──────────┐
   │ Worker 0 │ │ Worker 1 │  │ Worker N │
   │ (slurmd) │ │ (slurmd) │  │ (scaled) │
   └──────────┘ └──────────┘  └──────────┘
```

### Files Reference

| File | Purpose |
|------|---------|
| `scripts/deploy-slurm.sh` | Full deploy (cluster + autoscaler) in one command |
| `scripts/deploy-autoscale.sh` | Setup autoscaler infra + submit self-sizing jobs |
| `scripts/run-autoscale-test.sh` | End-to-end test (submit, wait, retrieve results, pass/fail) |
| `scripts/cleanup-slurm.sh` | Tear down cluster and autoscaler |
| `scripts/autoscaler-loop.sh` | The polling script that runs inside the autoscaler pod |
| `scripts/submit_job_autoscale.sh` | Self-sizing elastic batch script (calculates min nodes from dataset) |
| `configs/slurm-autoscaler.yaml` | ServiceAccount, RBAC, and Deployment for the autoscaler pod |
| `configs/slurm-cluster.yaml` | Standard cluster config (4Gi worker memory) |
| `demos/ddp_test.py` | DDP training script with intensity levels and `--autoscale` flag |

---

## Quick Start (Scripted)

If the cluster is already deployed, you can run the full DDP training test with a single command:

```bash
# Deploy cluster + autoscaler (if not already running)
./scripts/deploy-slurm.sh

# Run end-to-end test (submits light job, waits for completion, retrieves results, prints pass/fail)
./scripts/run-autoscale-test.sh

# Or: setup autoscaler + submit a medium-intensity self-sizing job (fire-and-forget)
./scripts/deploy-autoscale.sh

# Submit only (autoscaler already deployed), with custom dataset size:
NUM_SAMPLES=12000 ./scripts/deploy-autoscale.sh --submit-only

# Tear everything down when done
./scripts/cleanup-slurm.sh
```

| Script | What it does | Blocks? | Retrieves results? |
|--------|-------------|---------|-------------------|
| `deploy-autoscale.sh` | Setup infra + submit job | No | No |
| `run-autoscale-test.sh` | Submit light job + wait + retrieve + verdict | Yes | Yes |

The rest of this guide walks through each step manually for understanding and debugging.

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

## Step 1: Deploy the Autoscaler

The autoscaler handles **everything** after this point — it installs PyTorch on workers, copies training scripts, and scales the cluster up/down based on job demand.

```bash
./scripts/deploy-autoscale.sh --setup-only
```

This creates:

| Resource | Purpose |
|----------|---------|
| `ConfigMap` slurm-autoscaler-script | Scripts mounted into the autoscaler pod (autoscaler loop, ddp_test.py, submit scripts) |
| `ServiceAccount` slurm-autoscaler | Identity for the autoscaler pod |
| `Role` + `RoleBinding` | Permissions to scale NodeSets, list pods, and exec into slurmctld |
| `Deployment` | Runs the polling script in a `bitnami/kubectl` container |

**Verify it's running:**

```bash
oc get pods -n slurm -l app.kubernetes.io/name=slurm-autoscaler
# Expected: 1/1 Running

oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler --tail=15
```

Expected startup log:

```
[HH:MM:SS] Slurm NodeSet Autoscaler started (with auto-provisioning)
[HH:MM:SS]   NodeSet:       slurm-worker-slinky
[HH:MM:SS]   Min replicas:  2
[HH:MM:SS]   Max replicas:  8
[HH:MM:SS]   Poll every:    30s
[HH:MM:SS]   Scale-down:    after 300s idle
[HH:MM:SS]   PyTorch index: https://download.pytorch.org/whl/cu124
[HH:MM:SS] Provisioning controller and existing workers...
[HH:MM:SS] PROVISION: installing PyTorch on slurm-worker-slinky-0 (this takes a few minutes)...
[HH:MM:SS] PROVISION: installing PyTorch on slurm-worker-slinky-1 (this takes a few minutes)...
[HH:MM:SS] PROVISION: slurm-worker-slinky-0 ready
[HH:MM:SS] PROVISION: slurm-worker-slinky-1 ready
[HH:MM:SS] IDLE: no pending jobs (2 replicas)
```

The autoscaler provisions existing workers on startup — no need to manually install PyTorch.

### Default Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_REPLICAS` | 2 | Minimum worker nodes (maintained at idle) |
| `MAX_REPLICAS` | 8 | Maximum worker nodes under heavy demand |
| `POLL_INTERVAL` | 30s | How often the script checks `squeue` |
| `SCALE_DOWN_DELAY` | 300s | Seconds with no pending jobs before scaling down |
| `PYTORCH_INDEX` | `https://download.pytorch.org/whl/cu124` | PyTorch package index (CUDA 12.4 by default) |

These are set as env vars in `configs/slurm-autoscaler.yaml`.

---

## Step 2: Submit the Job

Once the autoscaler logs show workers are provisioned (`PROVISION: ... ready`), submit the elastic DDP job:

```bash
# Default (8192 samples, medium intensity) — auto-calculates 2 nodes minimum
oc exec -n slurm slurm-controller-0 -c slurmctld -- bash /tmp/submit_job_autoscale.sh
# Expected: "Submitted batch job <JOB_ID>"
```

**Force autoscaling by increasing dataset size:**

```bash
# 16384 samples → needs 4 nodes minimum (triggers autoscaler)
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  bash -c 'NUM_SAMPLES=16384 bash /tmp/submit_job_autoscale.sh'

# 32768 samples → needs 7 nodes minimum
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  bash -c 'NUM_SAMPLES=32768 bash /tmp/submit_job_autoscale.sh'

# Light mode (small model, minimal memory) — 1 node is enough
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  bash -c 'INTENSITY=light bash /tmp/submit_job_autoscale.sh'
```

The script calculates the minimum nodes needed based on dataset memory requirements and pod limits. With data sharding, each rank only loads `num_samples / world_size` images — so bigger datasets require more nodes to fit in 4Gi pod memory.

| NUM_SAMPLES | Dataset total | Min nodes (4Gi pods) |
|-------------|--------------|---------------------|
| 8,192 | 4.7 GB | 2 |
| 16,384 | 9.4 GB | 4 |
| 24,576 | 14.1 GB | 5 |
| 32,768 | 18.8 GB | 7 |

---

## Step 3: Monitor the Job

Open these in **separate terminals** before or after submitting:

```bash
# Terminal 1: Autoscaler logs (scaling decisions + provisioning)
oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler -f --tail=20

# Terminal 2: Watch pods appear/disappear
watch -n 5 'oc get pods -n slurm -o wide'

# Terminal 3: Slurm node registration
watch -n 10 'oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo -N -l 2>/dev/null'

# Terminal 4: Job queue (PENDING → RUNNING → gone)
watch -n 5 'oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue -l 2>/dev/null'

# Terminal 5: NodeSet replica count
watch -n 5 'oc get nodeset -n slurm'
```

**macOS:** `watch` is not installed by default. Install via `brew install watch`, or use a loop:
```bash
# macOS alternative for any watch command above:
while true; do clear; oc get pods -n slurm -o wide; sleep 5; done
```

### What You'll See (4-node autoscale flow)

1. `squeue` shows the job as **PD** (pending) with reason `Resources`
2. Autoscaler log: `DEMAND: 1 pending job(s), largest needs 4 node(s), have 2`
3. Autoscaler log: `SCALING: slurm-worker-slinky -> 4 replicas`
4. `oc get pods` shows `slurm-worker-slinky-2` and `slurm-worker-slinky-3` appearing
5. Autoscaler log: `PROVISION: installing PyTorch on slurm-worker-slinky-2...`
6. `sinfo` shows `slinky-[2-3]` registering as `idle`
7. Job's readiness gate prints `Not ready yet, retrying in 15s...` on new workers
8. Autoscaler log: `PROVISION: slurm-worker-slinky-2 ready`
9. `squeue` shows the job flip to **R** (running) across 4 nodes
10. Job completes, prints pass/fail verdict
11. After 5 minutes idle, autoscaler scales back to 2

---

## Step 4: View Results

```bash
# Find which worker was the batch host
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job JOB_ID | grep BatchHost

# Read job output (replace JOB_ID; output is on batch host, usually worker-0)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-elastic-JOB_ID.out

# Read error log (should only show harmless numpy warnings)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-elastic-JOB_ID.err
```

> **Note:** Job output is on the **batch host** worker pod, not the controller. Always check `BatchHost` in `scontrol show job`.

### Retrieve Training Artifacts

Rank 0 saves a model checkpoint, metrics JSON, and predictions to `/tmp/ddp-results/<timestamp>/` on the batch host:

```bash
# Verify results exist
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- ls -lh /tmp/ddp-results/

# Copy to local results/ folder (gitignored)
oc cp slurm/slurm-worker-slinky-0:/tmp/ddp-results results/ -c slurmd
```

| File | Contents |
|------|----------|
| `model_checkpoint.pt` | Trained model state dict + metadata (~36 MB) |
| `metrics.json` | Training config, timing, throughput, Slurm job info |
| `predictions.pt` | Model inference on a synthetic batch (inputs, logits, probabilities) |

> **Tip:** The `unexpected EOF` warning from `oc cp` is harmless — files transfer correctly. Results are saved on slinky-0 (a base replica), so they persist after scale-down.

---

## Expected Output

### Autoscaler Log (successful run)

```
[13:31:11] IDLE: no pending jobs (2 replicas)
[13:34:34] DEMAND: 1 pending job(s), largest needs 4 node(s), have 2
[13:34:34] SCALING: slurm-worker-slinky -> 4 replicas
nodeset.slinky.slurm.net/slurm-worker-slinky scaled
[13:35:12] PROVISION: installing dependencies on slurm-worker-slinky-2...
[13:35:13] PROVISION: installing dependencies on slurm-worker-slinky-3...
[13:35:29] PROVISION: installing PyTorch on slurm-worker-slinky-2 (this takes a few minutes)...
[13:35:29] PROVISION: installing PyTorch on slurm-worker-slinky-3 (this takes a few minutes)...
[13:37:31] PROVISION: slurm-worker-slinky-3 ready
[13:37:35] PROVISION: slurm-worker-slinky-2 ready
[13:38:05] COOLDOWN: no pending jobs, scale-down in 89s (4 replicas)
[13:39:58] IDLE: no pending jobs for 324s, scaling down
[13:39:58] SCALING: slurm-worker-slinky -> 2 replicas
nodeset.slinky.slurm.net/slurm-worker-slinky scaled
[13:40:33] IDLE: no pending jobs (2 replicas)
```

### Job Output — GPU Mode (4-node autoscale)

```
============================================================
  ELASTIC DDP TRAINING
============================================================
  Job ID:       30
  Nodes:        4 (requested: 4)
  Tasks:        4
  Node list:    slinky-[0-3]
  Master:       slinky-0:29500
  World Size:   4
  Hostname:     slinky-0
  Start time:   Thu Jun 18 13:34:42 UTC 2026
  Requeue:      enabled
============================================================

[slinky-0] Waiting for PyTorch and training script...
[slinky-1] Waiting for PyTorch and training script...
[slinky-2] Waiting for PyTorch and training script...
[slinky-3] Waiting for PyTorch and training script...
[slinky-0] Ready after 0s (torch 2.6.0+cu124)
[slinky-1] Ready after 0s (torch 2.6.0+cu124)
[slinky-2] Not ready yet, retrying in 15s... (0/600s)
[slinky-3] Not ready yet, retrying in 15s... (0/600s)
...
[slinky-3] Ready after 165s (torch 2.6.0+cu124)
[slinky-2] Ready after 180s (torch 2.6.0+cu124)

############################################################
  Slurm on OCP - Distributed Training Test
  Intensity: LIGHT
  [Autoscaling mode enabled]
############################################################
  Rank 0/4 on cuda:0
  Slurm Job ID: 30
  Slurm Nodes:  slinky-[0-3]
  Allocated:    4 nodes
  CUDA available: True
  GPU: NVIDIA L40S
  GPU Memory: 47.7 GB
  Host RSS (pre-load): 408 MB

[Comm Test] all_reduce: got 6.0, expected 6.0 - PASSED
[Bandwidth] all_reduce 64MB: 145.3ms (0.43 GB/s)

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

  Epoch   1/5 | Loss: 2.3439 | Throughput: 431 samples/s | Time: 4.75s | RSS: 1383MB
  Epoch   2/5 | Loss: 2.3038 | Throughput: 799 samples/s | Time: 2.56s | RSS: 1384MB
  Epoch   3/5 | Loss: 2.3022 | Throughput: 770 samples/s | Time: 2.66s | RSS: 1384MB
  Epoch   4/5 | Loss: 2.3029 | Throughput: 798 samples/s | Time: 2.57s | RSS: 1384MB
  Epoch   5/5 | Loss: 2.3034 | Throughput: 795 samples/s | Time: 2.58s | RSS: 1384MB

============================================================
  Training Complete
============================================================
  Total time:         15.12s
  Avg throughput:     2709 samples/s (global)
  Samples processed:  40,960 (across all ranks)
============================================================

============================================================
  Autoscaling Report
============================================================
  Current world size:     4
  Per-rank throughput:    719 samples/s
  Global throughput:      2874 samples/s
  Scaling efficiency:     166.7%
  Slurm job:              30
  Allocated nodes:        4
  Recommendation:         SCALE UP  - High efficiency, adding nodes would increase throughput
  GPU memory utilization: 0.0%
  Host memory usage:      8.2%
  Process RSS:            1384 MB
============================================================

############################################################
  TEST PASSED - All ranks completed successfully
############################################################

============================================================
  Job completed at: Thu Jun 18 13:38:09 UTC 2026
  Exit code:        0
  RESULT: SUCCESS — DDP training ran across 4 node(s)

  Results saved on batch host: slinky-0
  Retrieve with:
    oc cp slurm/slinky-0:/tmp/ddp-results results/ -c slurmd
============================================================
```

### Performance Reference (GPU, 4 nodes)

| Metric | Value |
|--------|-------|
| Total time | 15s |
| Throughput | 2,709 samples/s (global) |
| Per-epoch | ~3s |
| Backend | NCCL |
| Device | NVIDIA L40S (48 GB) |

### Job Error Log

The `.err` file should only contain harmless PyTorch warnings about NumPy:

```
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```

This does not affect training. NumPy is optional for PyTorch.

---

## Understanding the Results

### Test Phases

| Phase | What It Does | Success Criteria |
|-------|-------------|------------------|
| **Readiness Gate** | Each rank waits for PyTorch + `ddp_test.py` to be provisioned | All ranks report `Ready after Xs` |
| **Communication Test** | Each rank sends its ID via `all_reduce`; verifies sum is correct | Sum matches `N*(N-1)/2` |
| **Bandwidth Test** | Times a 32MB `all_reduce` across all ranks | Completes without timeout; reports GB/s |
| **Training** | Runs CNN on synthetic data with DDP gradient sync | Loss decreases; all epochs complete |
| **Autoscaling Report** | Per-rank resource utilization and scaling recommendation | Reports efficiency and recommendation |

### Key Metrics

- **Throughput (samples/s)**: How fast each rank processes data. CPU pods: ~40 samples/s. GPU pods: 400-1000+ samples/s per rank.
- **Bandwidth (GB/s)**: Inter-pod communication speed over K8s network. 0.5-1 GB/s is normal for pod-to-pod. InfiniBand HPC would be 10-50 GB/s.
- **Loss convergence**: Loss should decrease over epochs, proving gradients are synchronized correctly across ranks.
- **Scaling efficiency**: > 100% means adding more nodes would still improve throughput.

---

## Customizing the Autoscaler

### Configuration Parameters

All configuration is via environment variables in `configs/slurm-autoscaler.yaml`:

```yaml
env:
  - name: MIN_REPLICAS
    value: "2"        # Minimum workers (floor for scale-down)
  - name: MAX_REPLICAS
    value: "8"        # Maximum workers (ceiling for scale-up)
  - name: POLL_INTERVAL
    value: "30"       # Seconds between squeue checks
  - name: SCALE_DOWN_DELAY
    value: "300"      # Seconds idle before scaling down
  - name: PYTORCH_INDEX
    value: "https://download.pytorch.org/whl/cu124"  # PyTorch package index
```

After editing, apply and restart:

```bash
oc apply -f configs/slurm-autoscaler.yaml
oc rollout restart deployment slurm-autoscaler -n slurm
```

### Updating Scripts

If you modify `autoscaler-loop.sh`, `ddp_test.py`, or submit scripts, re-run the deploy script and restart:

```bash
./scripts/deploy-autoscale.sh --setup-only
oc rollout restart deployment slurm-autoscaler -n slurm
```

### Using CPU-Only PyTorch

If your workers don't have GPUs, change `PYTORCH_INDEX` in `configs/slurm-autoscaler.yaml`:

```yaml
- name: PYTORCH_INDEX
  value: "https://download.pytorch.org/whl/cpu"
```

### Disabling Autoscaling

```bash
# Remove the autoscaler and its ConfigMap
oc delete -f configs/slurm-autoscaler.yaml
oc delete configmap slurm-autoscaler-script -n slurm

# Set a fixed replica count
oc scale nodeset slurm-worker-slinky --replicas=2 -n slurm
```

---

## Autoscaling Scenarios

### Scenario 1: Elastic start (default)

1. User submits `sbatch /tmp/submit_job_autoscale.sh` (requests 1-4 nodes)
2. Job starts immediately on the 2 available workers
3. If more nodes later scale up for other jobs, Slurm can expand the allocation

### Scenario 2: Force multi-node (autoscale test)

1. User submits `sbatch --nodes=4 /tmp/submit_job_autoscale.sh` (hard 4-node requirement)
2. Only 2 nodes exist — job stays PENDING with reason `Resources`
3. Autoscaler detects pending job needing 4 nodes, scales NodeSet from 2 to 4
4. New worker pods start, autoscaler installs PyTorch and copies scripts
5. New nodes register with Slurm, job transitions to RUNNING
6. DDP training runs across all 4 nodes
7. After completion and 5-minute cooldown, autoscaler scales back to 2

### Scenario 3: Burst of jobs

1. User submits 5 DDP jobs
2. Autoscaler sees 5 pending jobs, scales toward max (8)
3. Slurm distributes jobs across available nodes as they come online
4. Jobs complete, pending count drops to 0
5. After 5 minutes idle, autoscaler scales back to min (2)

### Scenario 4: Priority preemption

1. A low-priority job is running
2. A high-priority job is submitted
3. Slurm preempts the low-priority job (`--requeue` causes re-enqueue)
4. Autoscaler detects both jobs pending and scales up for both to run

---

## Troubleshooting

### Clear stuck jobs

Previous jobs may be stuck with "user env retrieval failed requeued held" — a common issue in containerized Slurm where login environment resolution fails.

```bash
# Check for stuck jobs
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue

# Cancel all stuck pending jobs (if any)
oc exec -n slurm slurm-controller-0 -c slurmctld -- scancel --state=PENDING -u slurm

# Verify queue is empty
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue
```

**Root Cause:** Slurm tries to retrieve the user's login environment via `su -l` which fails in containers. The fix is `#SBATCH --export=ALL` in the submit script (already included).

### Scaled-up workers stuck in Pending (no GPUs available)

The autoscaler increased replicas, but new pods can't schedule because the cluster has no free GPUs. You'll see `Insufficient nvidia.com/gpu` in pod events:

```bash
# Check why the pod can't schedule
oc describe pod slurm-worker-slinky-2 -n slurm | grep -A 3 "FailedScheduling"

# Fix: scale back to what's available and cancel pending Slurm jobs
oc scale nodeset slurm-worker-slinky -n slurm --replicas=2
oc exec -n slurm slurm-controller-0 -c slurmctld -- scancel --state=PENDING -u slurm
```

The autoscaler may try to scale up again on its next poll if pending jobs remain. Cancel the Slurm jobs first to break the loop.

### Job stays PENDING after scale-up

The autoscaler scaled up and pods are Running, but the Slurm job is still pending:

```bash
# Are the new nodes registered with Slurm?
oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo -N -l

# Are any nodes drained?
oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo -N -l | grep drain

# Resume drained nodes
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  scontrol update nodename=ALL state=resume reason="cleared"
```

### New workers don't have PyTorch

The autoscaler provisions workers automatically, but provisioning can fail if:
- The pod doesn't have internet access (for `pip install`)
- The pod's memory limit is too low to install CUDA PyTorch

Check autoscaler logs for `PROVISION: WARNING` messages:

```bash
oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler --tail=50 | grep PROVISION
```

To manually provision a worker:

```bash
oc exec -n slurm slurm-worker-slinky-2 -c slurmd -- \
  bash -c "apt-get update -qq && apt-get install -y -qq python3-pip"
oc exec -n slurm slurm-worker-slinky-2 -c slurmd -- \
  pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu124
```

### Autoscaler not detecting pending jobs

```bash
# Verify the autoscaler can exec into the controller
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue -h -t PENDING -o "%i %D"

# Check autoscaler pod logs for errors
oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler --tail=30
```

### "user env retrieval failed requeued held"

**Cause:** Slurm can't resolve user login environment in the container.
**Fix:** Ensure `#SBATCH --export=ALL` is in the submit script. Cancel stuck jobs:

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- scancel --state=PENDING -u slurm
```

### Job stays in PD (pending) state

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue -o "%.10i %.10P %.15j %.8u %.2t %.10M %.6D %R"
```

Common reasons:
- **(Resources)** — Not enough nodes available. Wait for autoscaler to scale up, or check `sinfo` for drained nodes.
- **(Priority)** — Another job is ahead in queue. Wait or cancel it.

### Connection timeout during distributed init

The ranks can't reach each other over the network:

```bash
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- getent hosts slinky-1
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- getent hosts slinky-0
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- bash -c "echo | nc -w 3 slinky-1 29500"
```

### Slurm auth errors after scaling

If you see `failed to connect to any sack sockets` errors after new nodes come up, wait 30 seconds for pods to re-register, or restart the controller:

```bash
oc delete pod slurm-controller-0 -n slurm
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  scontrol update nodename=ALL state=resume reason="cleared"
```

### Job output file not found

Output files are on the **batch host** worker pod, not the controller:

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job JOB_ID | grep BatchHost
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-elastic-JOB_ID.out
```

---

## Advanced: Manual Setup (Without Autoscaler)

If you prefer to run without the autoscaler (fixed 2-node cluster, manually installed PyTorch):

### Install PyTorch Manually

```bash
for i in 0 1; do
  oc exec -n slurm slurm-worker-slinky-$i -c slurmd -- \
    bash -c "apt-get update -qq && apt-get install -y -qq python3-pip"
  oc exec -n slurm slurm-worker-slinky-$i -c slurmd -- \
    pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
done
```

For GPU workers use `https://download.pytorch.org/whl/cu124` instead. If switching from CPU to CUDA, add `--force-reinstall`.

### Copy Scripts and Submit

```bash
oc cp demos/ddp_test.py slurm/slurm-worker-slinky-0:/tmp/ddp_test.py -c slurmd
oc cp demos/ddp_test.py slurm/slurm-worker-slinky-1:/tmp/ddp_test.py -c slurmd

oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  sbatch --export=ALL --nodes=2 --ntasks=2 --ntasks-per-node=1 \
  --output=/tmp/ddp-test-%j.out --error=/tmp/ddp-test-%j.err \
  --wrap="srun python3 /tmp/ddp_test.py --intensity light --num-samples 8192 --epochs 5"
```

### View Output

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-test-JOB_ID.out
```

---

## How It Works (Internals)

### The Autoscaler Loop

The autoscaler is a shell script (`scripts/autoscaler-loop.sh`) running in a `bitnami/kubectl` pod:

1. **Poll:** Every `POLL_INTERVAL` seconds, exec into `slurmctld` and run `squeue -h -t PENDING -o "%i %D"` to get pending job IDs and their requested node counts.

2. **Scale decision:**
   - Pending jobs requesting more nodes than available → scale up (capped at `MAX_REPLICAS`)
   - No pending jobs for `SCALE_DOWN_DELAY` seconds → scale down to `MIN_REPLICAS`

3. **Provision:** After scaling up, installs `python3-pip`, PyTorch, and copies training scripts from the ConfigMap to each new worker.

4. **Readiness gate:** The `submit_job_autoscale.sh` script includes a per-rank loop that waits up to 10 minutes for PyTorch and `ddp_test.py` to appear. This allows jobs to be scheduled on nodes while they're still being provisioned.

### What the autoscaler does NOT do

- It does **not** use Prometheus, KEDA, or any external metrics pipeline
- It does **not** modify Slurm configuration (partitions, nodes, etc.)
- It does **not** drain nodes before scale-down (the Slinky operator handles graceful pod termination)
- It does **not** persist state across restarts (provisioning status is tracked in `/tmp`)

### `submit_job_autoscale.sh` — Elastic Job Submission

The batch script uses `--nodes=1-4` (min-max syntax) so Slurm can start the job as soon as the minimum node count is available. The `--requeue` flag allows re-enqueue on preemption, and `--time-min` enables backfill scheduling.

### `ddp_test.py` — Training Script

**Setup:** Auto-detects launch mode — Slurm (via `SLURM_PROCID`), torchrun (via `RANK`), or single-process fallback. Picks NCCL for GPU or Gloo for CPU, then calls `dist.init_process_group()`.

**Communication Test:** Each rank creates a tensor containing its rank value. `all_reduce(SUM)` should produce `N*(N-1)/2`.

**Bandwidth Test:** A 32 MB `all_reduce` is timed to measure inter-pod throughput.

**Training:** A CNN (~9.3M parameters) trains on synthetic 3x32x32 images using DDP. A `DistributedSampler` shards data across ranks. Decreasing loss proves gradient sync is correct.

**Save Results:** Rank 0 saves model checkpoint, metrics JSON, and synthetic predictions to `--output-dir`.

**Autoscaling Report:** When `--autoscale` is passed, collects per-rank resource utilization and computes scaling efficiency.

---

## Next Steps

Once training succeeds, you've validated the full infrastructure. Next:

1. **Larger model** — Swap `SmallCNN` for ResNet-50 or use FSDP for model-parallel training
2. **Real dataset** — Replace synthetic data with domain-specific data
3. **Custom container image** — Build a PyTorch image with all dependencies pre-installed (eliminates runtime provisioning)
4. **Multi-tenancy** — Configure Slurm accounts, QoS policies, and fairshare scheduling
5. **Higher scale** — Increase `MAX_REPLICAS` and test with 8+ node jobs
