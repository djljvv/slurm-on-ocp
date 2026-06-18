# Autoscaling Guide — Slurm on OpenShift

This guide covers the complete autoscaling story for Slurm on OpenShift: prove the cluster can't handle a workload (Part 1), then show that automatic scaling fixes it (Part 2). The autoscaler is a lightweight polling loop — no Prometheus, no KEDA, no metrics pipeline required.

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Files Reference](#files-reference)
- [Part 1: Resource Exhaustion (the "Before" State)](#part-1-resource-exhaustion-the-before-state)
- [Part 2: Deploy the Autoscaler](#part-2-deploy-the-autoscaler)
- [Part 3: End-to-End Autoscale Test](#part-3-end-to-end-autoscale-test)
- [Monitoring Commands](#monitoring-commands)
- [Expected Output](#expected-output)
- [Customizing the Autoscaler](#customizing-the-autoscaler)
- [Autoscaling Scenarios](#autoscaling-scenarios)
- [Troubleshooting](#troubleshooting)

---

## Architecture

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

The autoscaler runs as a single pod (`bitnami/kubectl` image) that:

1. Every 30 seconds, execs into `slurmctld` and runs `squeue` to check for pending jobs
2. If a pending job needs more nodes than available, scales the NodeSet up via `kubectl scale`
3. Waits for new pods to become Ready, then installs PyTorch and copies training scripts
4. After 5 minutes with no pending jobs, scales back down to the minimum

No additional operators, Prometheus, or monitoring infrastructure required.

---

## Prerequisites

- Slurm on OCP deployed and healthy (see [Deployment Guide](DEPLOYMENT_GUIDE.md))
- `oc` CLI logged in with cluster access
- At least 2 Slurm worker pods running

```bash
oc get pods -n slurm
oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `configs/slurm-autoscaler.yaml` | ServiceAccount, RBAC, and Deployment for the autoscaler pod |
| `configs/autoscaler-loop.sh` | The polling script that runs inside the autoscaler pod |
| `configs/slurm-cluster.yaml` | Standard cluster config (4Gi worker memory) |
| `configs/slurm-cluster-constrained.yaml` | Constrained config (1Gi) for the OOM demo |
| `demos/ddp_test.py` | DDP training script with intensity levels and `--autoscale` flag |
| `demos/submit_job.sh` | Standard fixed-node Slurm batch script |
| `demos/submit_job_autoscale.sh` | Elastic batch script (`--nodes=1-4`, `--requeue`) |
| `demos/submit_job_autoscale_test.sh` | 4-node autoscale end-to-end test (with PyTorch readiness gate) |
| `demos/submit_job_oom.sh` | OOM-triggering batch script (for constrained demo) |

---

## Part 1: Resource Exhaustion (the "Before" State)

This section proves that a Slurm cluster with **insufficient worker pod memory** cannot run a realistic training workload. This establishes the baseline problem that autoscaling solves.

### Step 1: Constrain Worker Memory

Patch the NodeSet to reduce worker memory from 4Gi to 1Gi:

```bash
oc patch nodeset slurm-worker-slinky -n slurm --type='merge' -p '{
  "spec": {
    "slurmd": {
      "resources": {
        "requests": {
          "cpu": "1",
          "memory": "500Mi",
          "nvidia.com/gpu": "1"
        },
        "limits": {
          "cpu": "2",
          "memory": "1Gi",
          "nvidia.com/gpu": "1"
        }
      }
    }
  }
}'
```

> **Note:** Remove the `nvidia.com/gpu` lines if your cluster doesn't have GPUs.

| Setting | Normal | Constrained |
|---------|--------|-------------|
| Worker memory requests | 2Gi | 500Mi |
| Worker memory limits | 4Gi | **1Gi** |

Wait for pods to restart and verify:

```bash
oc get pods -n slurm -w

oc get pod slurm-worker-slinky-0 -n slurm \
  -o jsonpath='{.spec.containers[?(@.name=="slurmd")].resources.limits.memory}'
# Expected: 1Gi
```

### Step 2: Install PyTorch and Copy Scripts

Since the pods restarted, PyTorch needs to be reinstalled. Use the **CPU-only** version — the full CUDA version (~2.5 GB) would OOM during installation within the 1Gi limit.

```bash
for i in 0 1; do
  oc exec -n slurm slurm-worker-slinky-$i -c slurmd -- \
    bash -c "apt-get update -qq && apt-get install -y -qq python3-pip"
  oc exec -n slurm slurm-worker-slinky-$i -c slurmd -- \
    pip3 install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
done

oc cp demos/ddp_test.py slurm/slurm-worker-slinky-0:/tmp/ddp_test.py -c slurmd
oc cp demos/ddp_test.py slurm/slurm-worker-slinky-1:/tmp/ddp_test.py -c slurmd
oc cp demos/submit_job_oom.sh slurm/slurm-controller-0:/tmp/submit_job_oom.sh -c slurmctld
```

### Step 3: Submit the Workload

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch /tmp/submit_job_oom.sh
# Expected: "Submitted batch job <JOB_ID>"
```

The job uses `--intensity medium` which pre-allocates ~1148 MB of training data plus ~400 MB for PyTorch and the model. Total process RSS reaches ~1408 MB — well over the 1Gi pod limit.

### Step 4: Observe the Failure

```bash
# Check job state (replace JOB_ID with the number from sbatch output)
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job JOB_ID

# Read job output (try both workers — output is on the batch host)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-oom-JOB_ID.out 2>/dev/null || \
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- cat /tmp/ddp-oom-JOB_ID.out

# Read error log (shows the crash)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-oom-JOB_ID.err 2>/dev/null || \
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- cat /tmp/ddp-oom-JOB_ID.err
```

**Expected job output** (`ddp-oom-JOB_ID.out`) — the job starts normally but crashes during training:

```
  [Dataset] Pre-allocating 2000 images (3x224x224) in host memory...
  [Dataset] Estimated memory: 1148 MB
  [Dataset] Allocation complete.
  [Memory] Process RSS after dataset+model load: 1407 MB

============================================================
  Job exit code: 1
  EXPECTED: Job failed due to resource constraints.
============================================================
```

**Expected error log** (`ddp-oom-JOB_ID.err`) — shared memory exhaustion:

```
[rank0]: RuntimeError: unable to allocate shared memory(shm) for file
          </torch_4304_2962414746_0>: No space left on device (28)
srun: error: slinky-0: task 0: Exited with exit code 1
srun: error: slinky-1: task 1: Exited with exit code 1
```

**Why it fails:** The dataset pre-allocation puts ~1148 MB into host RAM, PyTorch + ResNet-18 adds ~260 MB, pushing process RSS to ~1407 MB — well over the 1Gi (1024 MB) pod limit. When DataLoader workers try to create shared memory segments, the allocation fails.

> **Tip:** If the job succeeds instead of crashing, increase intensity: add `--intensity heavy --num-samples 5000` (~4 GB, guaranteed to fail at 1Gi).

### Step 5: Restore Normal Resources

```bash
oc patch nodeset slurm-worker-slinky -n slurm --type='merge' -p '{
  "spec": {
    "slurmd": {
      "resources": {
        "requests": {
          "cpu": "1",
          "memory": "2Gi",
          "nvidia.com/gpu": "1"
        },
        "limits": {
          "cpu": "2",
          "memory": "4Gi",
          "nvidia.com/gpu": "1"
        }
      }
    }
  }
}'
```

> **Note:** Remove the `nvidia.com/gpu` lines if your cluster doesn't have GPUs.

Wait for pods to restart. If any Slurm nodes are drained after the failure:

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  scontrol update nodename=ALL state=resume reason="cleared"
```

---

## Part 2: Deploy the Autoscaler

### Step 1: Create the ConfigMap

The autoscaler pod mounts scripts from a ConfigMap. Create it from the repo files:

```bash
oc create configmap slurm-autoscaler-script -n slurm \
  --from-file=autoscaler.sh=configs/autoscaler-loop.sh \
  --from-file=ddp_test.py=demos/ddp_test.py \
  --from-file=submit_job.sh=demos/submit_job.sh \
  --from-file=submit_job_autoscale.sh=demos/submit_job_autoscale.sh \
  --from-file=submit_job_autoscale_test.sh=demos/submit_job_autoscale_test.sh \
  --dry-run=client -o yaml | oc apply -f -
```

### Step 2: Deploy RBAC and the Autoscaler Pod

```bash
oc apply -f configs/slurm-autoscaler.yaml
```

This creates:

| Resource | Purpose |
|----------|---------|
| `ServiceAccount` slurm-autoscaler | Identity for the autoscaler pod |
| `Role` + `RoleBinding` | Permissions to scale NodeSets, list pods, and exec into slurmctld |
| `Deployment` | Runs the polling script in a `bitnami/kubectl` container |

### Step 3: Verify the Autoscaler

```bash
oc get pods -n slurm -l app.kubernetes.io/name=slurm-autoscaler
# Expected: 1/1 Running

oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler --tail=10
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
[HH:MM:SS] PROVISION: copying submit scripts to controller...
[HH:MM:SS] IDLE: no pending jobs (2 replicas)
```

The autoscaler also provisions existing workers with PyTorch on startup, so you don't need to manually install it.

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

## Part 3: End-to-End Autoscale Test

This test submits a job requesting **4 nodes** when only **2 exist**, forcing the autoscaler to scale up, provision new workers, and run the job across all 4 nodes.

### Step 1: Verify Clean State

```bash
# 2 workers, no pending jobs
oc get pods -n slurm
oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue
```

### Step 2: Open Monitoring Windows

Open these in **separate terminals** before submitting the job (see [Monitoring Commands](#monitoring-commands) below for details):

```bash
# Terminal 1: Autoscaler logs (scaling decisions + provisioning)
oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler -f --tail=20

# Terminal 2: Watch pods appear
watch -n 5 'oc get pods -n slurm -o wide'

# Terminal 3: Watch Slurm node registration
watch -n 10 'oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo -N -l 2>/dev/null'

# Terminal 4: Watch job queue
watch -n 5 'oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue -l 2>/dev/null'

# Terminal 5: Watch NodeSet replica count
watch -n 5 'oc get nodeset -n slurm'
```

### Step 3: Submit the 4-Node Job

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch /tmp/submit_job_autoscale_test.sh
# Expected: "Submitted batch job <JOB_ID>"
```

### Step 4: Watch the Autoscale Flow

You will see this sequence across your monitoring terminals:

1. `squeue` shows the job as **PD** (pending) with reason `Resources`
2. Autoscaler log: `DEMAND: 1 pending job(s), largest needs 4 node(s), have 2`
3. Autoscaler log: `SCALING: slurm-worker-slinky -> 4 replicas`
4. `oc get pods` shows `slurm-worker-slinky-2` and `slurm-worker-slinky-3` appearing
5. Autoscaler log: `PROVISION: installing PyTorch on slurm-worker-slinky-2...` (takes ~2 minutes)
6. `sinfo` shows `slinky-[2-3]` registering as `idle`
7. The job's built-in readiness gate waits for PyTorch — you'll see `Not ready yet, retrying in 15s...`
8. Autoscaler log: `PROVISION: slurm-worker-slinky-2 ready`
9. `squeue` shows the job flip to **R** (running) across 4 nodes
10. Job output shows DDP training with autoscale report
11. After 5 minutes idle, autoscaler scales back to 2

### Step 5: View Results

```bash
# Find which worker was the batch host
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job JOB_ID | grep BatchHost

# Read job output (replace JOB_ID; output is on batch host, usually worker-0)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-autoscale-JOB_ID.out

# Read error log (should only show harmless numpy warnings)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-autoscale-JOB_ID.err
```

> **Important:** Job output files are written on the **batch host worker pod**, not the controller. Always check `BatchHost` in `scontrol show job` to know which pod to read from.

### Step 6: Retrieve Training Artifacts

The job saves model checkpoints, metrics, and predictions to `/tmp/ddp-results/` on the batch host. Copy them locally before pods scale down:

```bash
# Verify results exist on batch host
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- ls -lh /tmp/ddp-results/

# Copy to local results/ folder (gitignored)
oc cp slurm/slurm-worker-slinky-0:/tmp/ddp-results results/ -c slurmd
```

> **Tip:** The `unexpected EOF` warning from `oc cp` is harmless — files transfer correctly despite the message. Verify with `ls results/` locally.
>
> **Autoscale note:** Results are saved by rank 0 on the batch host (first node in SLURM_NODELIST, typically slinky-0). Since slinky-0 is a base replica, it persists after scale-down. You don't need to race the scale-down timer.

---

## Monitoring Commands

### Autoscaler Logs

The most important view — shows scaling decisions, provisioning progress, and cooldown status:

```bash
oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler -f --tail=20
```

### Worker Pods

See new pods appear during scale-up and disappear during scale-down:

```bash
watch -n 5 'oc get pods -n slurm -o wide'
```

### Slurm Node Registration

See when Slurm recognizes new nodes as available for scheduling:

```bash
watch -n 10 'oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo -N -l 2>/dev/null'
```

### Job Queue

See the job transition from PENDING to RUNNING:

```bash
watch -n 5 'oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue -l 2>/dev/null'
```

### NodeSet Replica Count

See the autoscaler changing the desired replica count:

```bash
watch -n 5 'oc get nodeset -n slurm'
```

---

## Expected Output

### Autoscaler Log (successful run)

```
[13:31:11] IDLE: no pending jobs (2 replicas)
[13:34:00] IDLE: no pending jobs (2 replicas)
[13:34:34] DEMAND: 1 pending job(s), largest needs 4 node(s), have 2
[13:34:34] SCALING: slurm-worker-slinky -> 4 replicas
nodeset.slinky.slurm.net/slurm-worker-slinky scaled
[13:35:08] COOLDOWN: no pending jobs, scale-down in 266s (4 replicas)
[13:35:12] PROVISION: installing dependencies on slurm-worker-slinky-2...
[13:35:13] PROVISION: installing dependencies on slurm-worker-slinky-3...
[13:35:29] PROVISION: installing PyTorch on slurm-worker-slinky-3 (this takes a few minutes)...
[13:35:29] PROVISION: installing PyTorch on slurm-worker-slinky-2 (this takes a few minutes)...
[13:37:31] PROVISION: slurm-worker-slinky-3 ready
[13:37:35] PROVISION: slurm-worker-slinky-2 ready
[13:38:05] COOLDOWN: no pending jobs, scale-down in 89s (4 replicas)
[13:39:58] IDLE: no pending jobs for 324s, scaling down
[13:39:58] SCALING: slurm-worker-slinky -> 2 replicas
nodeset.slinky.slurm.net/slurm-worker-slinky scaled
[13:40:33] IDLE: no pending jobs (2 replicas)
```

### Job Output (successful 4-node run)

```
============================================================
  AUTOSCALER END-TO-END TEST
============================================================
  Job ID:       30
  Nodes:        4
  Tasks:        4
  Node list:    slinky-[0-3]
  Master:       slinky-0:29500
  World Size:   4
  Hostname:     slinky-0
  Start time:   Thu Jun 18 13:34:42 UTC 2026
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
  RESULT: SUCCESS — DDP training ran across 4 nodes
  The autoscaler successfully:
    1. Detected demand for 4 nodes
    2. Scaled up the NodeSet
    3. Provisioned workers with PyTorch
    4. Job completed distributed training
============================================================
```

### Job Error Log

The `.err` file should only contain harmless PyTorch warnings about NumPy:

```
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```

This does not affect training. NumPy is optional for PyTorch.

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

If you modify `autoscaler-loop.sh`, `ddp_test.py`, or any submit scripts, update the ConfigMap and restart the autoscaler:

```bash
oc create configmap slurm-autoscaler-script -n slurm \
  --from-file=autoscaler.sh=configs/autoscaler-loop.sh \
  --from-file=ddp_test.py=demos/ddp_test.py \
  --from-file=submit_job.sh=demos/submit_job.sh \
  --from-file=submit_job_autoscale.sh=demos/submit_job_autoscale.sh \
  --from-file=submit_job_autoscale_test.sh=demos/submit_job_autoscale_test.sh \
  --dry-run=client -o yaml | oc apply -f -

oc rollout restart deployment slurm-autoscaler -n slurm
```

### Using CPU-Only PyTorch

If your workers don't have GPUs, change the `PYTORCH_INDEX` to the CPU-only index:

```yaml
- name: PYTORCH_INDEX
  value: "https://download.pytorch.org/whl/cpu"
```

### Disabling Autoscaling

```bash
# Remove the autoscaler (keeps static NodeSet replicas)
oc delete -f configs/slurm-autoscaler.yaml
oc delete configmap slurm-autoscaler-script -n slurm

# Set a fixed replica count
oc scale nodeset slurm-worker-slinky --replicas=2 -n slurm
```

---

## Autoscaling Scenarios

### Scenario 1: Large multi-node job (the end-to-end test)

1. User submits `sbatch submit_job_autoscale_test.sh` (requests 4 nodes)
2. Only 2 nodes exist — job stays PENDING with reason `Resources`
3. Autoscaler detects pending job needing 4 nodes, scales NodeSet from 2 to 4
4. New worker pods start, autoscaler installs PyTorch and copies scripts
5. New nodes register with Slurm, job transitions to RUNNING
6. DDP training runs across all 4 nodes
7. After completion and 5-minute cooldown, autoscaler scales back to 2

### Scenario 2: Elastic job (starts with what's available)

1. User submits `sbatch submit_job_autoscale.sh` (requests 1-4 nodes)
2. Job starts immediately on the 2 available nodes
3. If more nodes scale up, Slurm can expand the allocation

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

### Job stays PENDING after scale-up

The autoscaler scaled up but the job is still pending. Check:

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

### Slurm auth errors after scaling

If you see `failed to connect to any sack sockets` errors after new nodes come up, the Slurm inter-daemon auth may be desynced. Wait 30 seconds for pods to re-register, or restart the controller:

```bash
oc delete pod slurm-controller-0 -n slurm
# Wait for it to come back
oc exec -n slurm slurm-controller-0 -c slurmctld -- \
  scontrol update nodename=ALL state=resume reason="cleared"
```

### Job output file not found

Output files are written on the **batch host** worker pod, not the controller. Find the right pod:

```bash
# Check which pod was the batch host
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job JOB_ID | grep BatchHost

# Then read from that pod (e.g., if BatchHost=slinky-0)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/ddp-autoscale-JOB_ID.out
```

### "user env retrieval failed requeued held"

Slurm can't resolve user login environment in the container. Ensure `#SBATCH --export=ALL` is in the submit script. Cancel stuck jobs:

```bash
oc exec -n slurm slurm-controller-0 -c slurmctld -- scancel --state=PENDING -u slurm
```

---

## How the Autoscaler Works (Internals)

The autoscaler is a shell script (`configs/autoscaler-loop.sh`) running in a `bitnami/kubectl` pod. Here's the main loop logic:

1. **Poll:** Every `POLL_INTERVAL` seconds, exec into `slurmctld` and run `squeue -h -t PENDING -o "%i %D"` to get pending job IDs and their requested node counts.

2. **Scale decision:**
   - If there are pending jobs requesting more nodes than currently available → scale up to meet demand (capped at `MAX_REPLICAS`)
   - If there are no pending jobs and it's been idle for `SCALE_DOWN_DELAY` seconds → scale down to `MIN_REPLICAS`

3. **Provision:** After scaling up, the script checks for unprovisioned worker pods and installs:
   - `python3-pip` (via apt-get)
   - PyTorch (from the configured `PYTORCH_INDEX`)
   - Training scripts and submit scripts (from the ConfigMap)

4. **Readiness gate:** The `submit_job_autoscale_test.sh` script includes a built-in readiness loop — each rank waits up to 10 minutes for PyTorch and `ddp_test.py` to appear before starting training. This allows the job to start on the scaled-up nodes while the autoscaler is still provisioning them.

### What the autoscaler does NOT do

- It does **not** use Prometheus, KEDA, or any external metrics pipeline
- It does **not** modify Slurm configuration (partitions, nodes, etc.)
- It does **not** drain nodes before scale-down (the Slinky operator handles graceful pod termination)
- It does **not** persist state across restarts (provisioning status is tracked in `/tmp`)
