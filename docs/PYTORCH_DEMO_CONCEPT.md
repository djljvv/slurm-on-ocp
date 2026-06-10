# PyTorch Application Concept for Slurm on OCP

## Executive Summary

This document outlines a **Distributed Multi-Node ResNet-50 Training** application to demonstrate the unique value proposition of running Slurm workload manager on OpenShift for AI/ML containerized workloads. The demo showcases how Slurm's HPC-grade scheduling capabilities complement Kubernetes' container orchestration, addressing gaps that pure Kubernetes cannot solve for scientific AI workloads.

---

## The Application: Distributed ResNet-50 Training with FSDP

### Overview

Train a ResNet-50 CNN using PyTorch's **Fully Sharded Data Parallel (FSDP)** or **Distributed Data Parallel (DDP)** across multiple compute nodes (OpenShift pods) managed by Slurm's batch scheduler.

### Dataset

- **Primary**: Tiny-ImageNet (200 classes, 100K images) or synthetic ImageNet-like data
- **Rationale**: Avoids I/O bottlenecks while maintaining realistic computational complexity
- **Size**: ~250MB compressed, easily cached in ephemeral storage

### Training Configuration

```
Model: ResNet-50 (25M parameters)
Framework: PyTorch 2.x with torchrun + FSDP/DDP
Nodes: 2-4 Slurm worker pods (configurable via sbatch -N flag)
GPUs: Optional (works with CPU-only or GPU acceleration)
Batch Size: 64 per GPU/CPU core
Epochs: 10-20 (adjustable for demo duration)
```

---

## Why This Demonstrates Slurm on OCP Value

### The Core Value Proposition

**Kubernetes alone cannot handle HPC-grade AI workloads effectively** because:

1. **No Gang Scheduling**: K8s lacks native gang scheduling (all-or-nothing pod allocation). For distributed training requiring N synchronized nodes, K8s may start 2/4 pods and deadlock waiting for resources, wasting GPU time.

2. **No Job Queuing/Prioritization**: K8s has basic priority classes but no job queue management (backfill, fairshare, preemption policies). Multiple teams submitting training jobs leads to resource contention chaos.

3. **No MPI/Parallel Job Semantics**: HPC frameworks (MPI, NCCL) expect synchronized process launch with `$SLURM_PROCID`, `$SLURM_NODELIST` environment variables. K8s requires complex manual coordination.

4. **Poor Multi-Tenancy for Batch Jobs**: K8s is designed for long-running services, not transient batch jobs with strict resource limits, accounting, and time limits.

### How Slurm on OCP Solves This

| Problem | Kubernetes Alone | Slurm on OCP |
|---------|------------------|--------------|
| **Gang Scheduling** | Requires MPIJob operator workarounds | Native via `sbatch -N 4` (allocates all 4 nodes or queues) |
| **Job Queue Management** | No queue; pods compete directly | FIFO/backfill queue with fairshare, QoS limits |
| **Multi-Node Coordination** | Manual `StatefulSet` + `Headless Service` DNS hacks | Automatic via `$SLURM_NODELIST` and `torchrun` integration |
| **Resource Accounting** | Prometheus metrics (post-hoc) | Built-in `sacct` per-job accounting (CPU-hours, memory, exit codes) |
| **Time Limits** | Manual `activeDeadlineSeconds` | `sbatch --time=01:00:00` with automatic SIGTERM/SIGKILL |
| **GPU Sharing** | Requires device plugins + MIG complexity | `--gres=gpu:2` syntax with Slurm GRES management |

---

## Technical Architecture

### Job Submission Flow

```
┌─────────────────────────────────────────────────────────────┐
│  User (Login Pod or oc exec)                                │
│  $ sbatch --nodes=4 --gres=gpu:1 train_resnet.sh            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Slurm Controller (slurmctld pod in slurm namespace)        │
│  - Queues job with JobID=123                                │
│  - Waits for 4 slurmd pods available in partition "all"     │
│  - Allocates nodes: slurm-worker-slinky-{0,1,2,3}           │
└────────────────────┬────────────────────────────────────────┘
                     │
      ┌──────────────┼──────────────┬──────────────┐
      ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ slurmd-0 │  │ slurmd-1 │  │ slurmd-2 │  │ slurmd-3 │
│ (Rank 0) │  │ (Rank 1) │  │ (Rank 2) │  │ (Rank 3) │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                   │
                   ▼
         torchrun --nnodes=4 \
           --nproc_per_node=1 \
           --rdzv_backend=c10d \
           --rdzv_endpoint=$MASTER_ADDR:29500 \
           train.py
```

### Key Integration Points

1. **Slurm Environment Variables → PyTorch**:
   ```bash
   export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
   export MASTER_PORT=29500
   export WORLD_SIZE=$SLURM_NTASKS
   export RANK=$SLURM_PROCID
   ```

2. **FSDP Configuration**:
   ```python
   model = FSDP(
       ResNet50(),
       sharding_strategy=ShardingStrategy.FULL_SHARD,
       cpu_offload=CPUOffload(offload_params=True),  # For CPU-only demo
       device_id=torch.cuda.current_device(),  # Or omit for CPU
   )
   ```

3. **Data Parallel Strategy**:
   - **DDP**: Each node holds full model copy, gradients synchronized
   - **FSDP**: Model sharded across nodes (better for large models >1GB)

---

## Limitations & Feasibility

### Limitations

1. **Network Latency**:
   - **Issue**: OpenShift pod-to-pod networking (SDN/OVN) has higher latency (~50-100μs) vs. InfiniBand HPC (<5μs)
   - **Impact**: 10-20% slower gradient synchronization for small models
   - **Mitigation**: Use gradient compression (PowerSGD) or FSDP to reduce communication volume

2. **Storage I/O**:
   - **Issue**: Persistent volumes (Ceph/NFS) slower than parallel filesystems (Lustre/GPFS)
   - **Impact**: Dataset loading bottleneck if using large datasets
   - **Mitigation**: Use ephemeral `emptyDir` volumes with pre-downloaded datasets, or synthetic data

3. **GPU MIG Complexity**:
   - **Issue**: Multi-Instance GPU (MIG) on A100s requires NVIDIA device plugin + Slurm GRES integration
   - **Impact**: Complex setup; may defer to CPU-only or single-GPU-per-pod
   - **Mitigation**: Start with CPU demo or full GPU pods

4. **Container Overhead**:
   - **Issue**: Container runtime adds 1-3% CPU overhead vs. bare metal
   - **Impact**: Negligible for compute-bound training; noticeable for I/O-bound jobs
   - **Mitigation**: Acceptable tradeoff for cloud-native portability

### Feasibility: High

| Aspect | Feasibility | Notes |
|--------|-------------|-------|
| **PyTorch + torchrun** | ✅ High | Standard distributed training; well-documented |
| **Slurm Integration** | ✅ High | `torchrun` reads `$SLURM_*` env vars natively |
| **Multi-Node OCP** | ✅ High | Slinky operator handles multi-pod NodeSets |
| **Dataset Handling** | ✅ High | Tiny-ImageNet small enough for `emptyDir` |
| **GPU Support (optional)** | ⚠️ Medium | Requires NVIDIA GPU operator + GRES config (doable but adds complexity) |
| **Demo Duration** | ✅ High | 10 epochs on Tiny-ImageNet: ~5-10 min on 4 CPU pods |

---

## Benefits Showcase

### What This Demo Proves

1. **Batch Job Semantics Work**:
   - Submit job → queues → allocates gang → executes → completes → deallocates
   - Show `squeue`, `sinfo`, `sacct` output to demonstrate HPC workflow

2. **Multi-Node Coordination is Automatic**:
   - No manual StatefulSet indexing or headless service DNS lookup
   - `torchrun` finds peers via `$SLURM_NODELIST` automatically

3. **Resource Accounting is Built-In**:
   - `sacct -j 123 --format=JobID,Elapsed,MaxRSS,CPUTime,ExitCode`
   - No need for Prometheus + custom exporters

4. **Job Fails Gracefully**:
   - If 1 of 4 pods OOMKills, Slurm marks job failed (not zombie pods)
   - Re-queue with `sbatch --dependency=afternotok:123`

5. **Hybrid Cloud Ready**:
   - Same sbatch script works on OpenShift + external HPC nodes (see `ADD_NODES.md`)
   - Burst to bare-metal GPU cluster when OCP is full

### Comparison Table

| Capability | Pure K8s (MPIJob Operator) | Slurm on OCP |
|------------|----------------------------|--------------|
| **Setup Complexity** | Medium (Kubeflow/MPIJob CRDs) | Low (sbatch script) |
| **Gang Scheduling** | Yes (via MPIJob controller) | Yes (native) |
| **Job Queue** | No (requires Volcano/Kueue) | Yes (slurmctld built-in) |
| **Fairshare** | No | Yes (via `sacctmgr`) |
| **Time Limits** | Manual (activeDeadlineSeconds) | Automatic (--time flag) |
| **Accounting** | Custom (Prometheus) | Native (sacct, sreport) |
| **Hybrid Cloud** | No (K8s-only) | Yes (same sbatch on OCP + bare-metal) |

---

## Implementation Checklist

### Phase 1: Basic Single-Node Training (Proof of Concept)
- [ ] Create PyTorch training script (`train_resnet.py`) with Tiny-ImageNet
- [ ] Create Dockerfile with PyTorch 2.x, torchvision, FSDP dependencies
- [ ] Build and push container to registry (`quay.io/username/slurm-pytorch:latest`)
- [ ] Write sbatch script (`train_resnet.sh`) requesting 1 node
- [ ] Test on Slurm cluster: `sbatch train_resnet.sh`

### Phase 2: Multi-Node Distributed Training (Core Demo)
- [ ] Update `train_resnet.py` to initialize `torch.distributed` with Slurm env vars
- [ ] Implement FSDP model wrapping
- [ ] Update sbatch script to request 4 nodes: `sbatch -N 4 train_resnet.sh`
- [ ] Test gang scheduling: verify all 4 pods start simultaneously
- [ ] Verify gradient synchronization (loss decreases consistently)

### Phase 3: Validation & Metrics (Show the Benefits)
- [ ] Add logging: training loss, throughput (samples/sec), GPU utilization
- [ ] Compare runtime: 1 node vs. 4 nodes (expect ~3.5x speedup)
- [ ] Show `sacct` output: job accounting (CPU time, memory, exit code)
- [ ] Demonstrate job queueing: submit 2 jobs, show backfill behavior
- [ ] (Optional) Add Prometheus metrics export for hybrid monitoring

### Phase 4: Documentation & Presentation
- [ ] Write `docs/PYTORCH_DEMO.md` with step-by-step instructions
- [ ] Create comparison screenshots: K8s approach vs. Slurm approach
- [ ] Record demo video: job submission → queue → execution → results
- [ ] Prepare architecture diagram showing Slurm integration

---

## Expected Outcomes

### Success Metrics

1. **Functional**: ResNet-50 trains successfully across 4 Slurm worker pods
2. **Performance**: 80-90% linear scaling (4 nodes = 3.2-3.6x speedup vs. 1 node)
3. **Usability**: Job submission via `sbatch` simpler than writing K8s YAML manifests
4. **Reliability**: Job failures (OOM, timeout) handled cleanly by Slurm

### Demo Script (5-minute walkthrough)

```bash
# 1. Show Slurm cluster status
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo

# 2. Submit training job
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch /jobs/train_resnet.sh

# 3. Monitor queue
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- squeue

# 4. Watch logs (job starts across 4 nodes simultaneously)
oc logs -n slurm -f slurm-worker-slinky-0 -c slurmd | grep "Epoch"

# 5. Show results (job completes, accounting data)
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sacct -j 123 --format=JobID,Elapsed,MaxRSS,State
```

---

## Alternative: 3D U-Net Medical Image Segmentation

If a **more visually compelling / scientifically relevant** demo is preferred:

### Concept

Train a 3D U-Net to segment brain tumors from MRI volumes (BraTS dataset subset).

### Why This is Compelling

- **Real-world use case**: Medical imaging is a major AI workload in healthcare
- **Memory pressure**: 3D volumes (128x128x128 voxels) easily exceed single-GPU memory
- **Tensor parallelism**: Demonstrate model sharding across nodes (not just data parallelism)

### Complexity Tradeoff

- **Pros**: More impressive visuals (show segmentation masks), clear scientific value
- **Cons**: Larger dataset (~10GB), requires careful memory management, harder to debug

**Recommendation**: Start with ResNet-50 (simpler), offer 3D U-Net as "advanced demo".

---

## Conclusion

The **Distributed ResNet-50 training** application is the optimal choice for demonstrating Slurm on OCP because it:

1. ✅ **Feasible**: Standard PyTorch + torchrun, no exotic dependencies
2. ✅ **Shows Slurm Value**: Gang scheduling, job queuing, multi-node coordination all visible
3. ✅ **Exposes K8s Gaps**: Native K8s cannot do this without complex operators (MPIJob, Volcano)
4. ✅ **HPC-Authentic**: Same sbatch workflow researchers use on bare-metal clusters
5. ✅ **Hybrid-Ready**: Same job runs on OCP + external HPC nodes (future extensibility)

**Next Steps**: Proceed to Phase 1 implementation (single-node training) to validate container build and Slurm integration.
