"""
Distributed Data Parallel (DDP) training test for Slurm on OpenShift.

Validates that Slurm-managed pods can coordinate multi-node GPU training
over the Kubernetes network using NCCL/Gloo collective communications.

Launch methods:
  1. Via Slurm:   sbatch submit_job.sh
  2. Via torchrun: torchrun --nnodes=N --nproc_per_node=1 ddp_test.py
  3. Single-node:  python ddp_test.py (auto-detects single GPU or CPU)

Autoscaling mode (--autoscale):
  Enables resource utilization monitoring and scaling efficiency metrics.
  Pairs with submit_job_autoscale.sh for elastic node allocation (--nodes=min-max)
  and configs/slurm-autoscaler.yaml for KEDA-driven NodeSet autoscaling.
"""

import os
import time
import json
import argparse
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class SyntheticImageDataset(Dataset):
    """Generates random image-like tensors with class labels on-the-fly."""

    def __init__(self, num_samples: int, image_size: int = 32, num_channels: int = 3, num_classes: int = 10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(self.num_channels, self.image_size, self.image_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label


class SmallCNN(nn.Module):
    """
    Small CNN with enough parameters to generate meaningful GPU work
    without being so large that it requires >1 GPU of memory.
    ~1.2M parameters — enough to stress NCCL gradient sync.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


@contextmanager
def timer(label: str, rank: int):
    """Context manager that prints elapsed time for rank 0."""
    if rank == 0:
        print(f"[Timer] {label}...", flush=True)
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if rank == 0:
        print(f"[Timer] {label}: {elapsed:.2f}s", flush=True)


def setup_distributed():
    """
    Initialize distributed process group.
    Supports three launch modes:
      - Slurm (SLURM_PROCID, SLURM_NODELIST env vars)
      - torchrun (RANK, WORLD_SIZE, MASTER_ADDR env vars)
      - Single-process fallback
    """
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))

        if "MASTER_ADDR" not in os.environ:
            import subprocess
            nodelist = os.environ["SLURM_NODELIST"]
            result = subprocess.run(
                ["scontrol", "show", "hostname", nodelist],
                capture_output=True, text=True
            )
            os.environ["MASTER_ADDR"] = result.stdout.strip().split("\n")[0]
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)

    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    else:
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = torch.device(f"cuda:{local_rank}")
    else:
        backend = "gloo"
        device = torch.device("cpu")

    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    return rank, world_size, local_rank, device


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def run_communication_test(rank, world_size, device):
    """
    Verify that collective communication works across all ranks.
    Each rank contributes its rank value; the sum should equal N*(N-1)/2.
    """
    if world_size <= 1:
        print("[Comm Test] Skipped (single process)", flush=True)
        return True

    tensor = torch.tensor([float(rank)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = world_size * (world_size - 1) / 2.0
    passed = abs(tensor.item() - expected) < 1e-5

    if rank == 0:
        status = "PASSED" if passed else "FAILED"
        print(f"[Comm Test] all_reduce: got {tensor.item()}, expected {expected} - {status}", flush=True)

    return passed


def collect_resource_utilization(rank, world_size, device):
    """Collect per-rank resource utilization for autoscaling decisions."""
    stats = {
        "rank": rank,
        "device": str(device),
        "cpu_count": os.cpu_count(),
    }

    if torch.cuda.is_available() and device.type == "cuda":
        stats["gpu_name"] = torch.cuda.get_device_name(device)
        stats["gpu_memory_total_gb"] = round(torch.cuda.get_device_properties(device).total_memory / 1e9, 2)
        stats["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(device) / 1e9, 2)
        stats["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(device) / 1e9, 2)
        stats["gpu_utilization_pct"] = round(
            torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory * 100, 1
        )

    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                stats["mem_total_gb"] = round(int(line.split()[1]) / 1e6, 2)
            elif line.startswith("MemAvailable:"):
                stats["mem_available_gb"] = round(int(line.split()[1]) / 1e6, 2)
    except (FileNotFoundError, PermissionError):
        pass

    return stats


def print_autoscale_report(rank, world_size, device, epoch_throughputs, train_elapsed):
    """Print scaling efficiency report with recommendations."""
    if rank != 0:
        return

    avg_throughput = sum(epoch_throughputs) / len(epoch_throughputs) if epoch_throughputs else 0
    global_throughput = avg_throughput * world_size

    per_node_efficiency = avg_throughput / max(epoch_throughputs[0], 1) * 100 if epoch_throughputs else 0
    scaling_efficiency = global_throughput / (epoch_throughputs[0] * world_size) * 100 if epoch_throughputs else 0

    print(f"\n{'='*60}", flush=True)
    print(f"  Autoscaling Report", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Current world size:     {world_size}", flush=True)
    print(f"  Per-rank throughput:    {avg_throughput:.0f} samples/s", flush=True)
    print(f"  Global throughput:      {global_throughput:.0f} samples/s", flush=True)
    print(f"  Scaling efficiency:     {scaling_efficiency:.1f}%", flush=True)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    slurm_nnodes = os.environ.get("SLURM_NNODES", "N/A")
    slurm_job_num_nodes = os.environ.get("SLURM_JOB_NUM_NODES", slurm_nnodes)
    print(f"  Slurm job:              {slurm_job_id}", flush=True)
    print(f"  Allocated nodes:        {slurm_job_num_nodes}", flush=True)

    # Scaling recommendation based on throughput efficiency
    if scaling_efficiency > 85:
        recommendation = "SCALE UP  - High efficiency, adding nodes would increase throughput"
    elif scaling_efficiency > 60:
        recommendation = "HOLD      - Moderate efficiency, communication overhead is acceptable"
    else:
        recommendation = "SCALE DOWN - Low efficiency, communication overhead dominates"

    print(f"  Recommendation:         {recommendation}", flush=True)

    resource_stats = collect_resource_utilization(rank, world_size, device)
    if "gpu_utilization_pct" in resource_stats:
        gpu_util = resource_stats["gpu_utilization_pct"]
        print(f"  GPU memory utilization: {gpu_util:.1f}%", flush=True)
        if gpu_util < 30:
            print(f"  GPU hint:               Low GPU memory usage — consider larger batch size", flush=True)
        elif gpu_util > 90:
            print(f"  GPU hint:               High GPU memory — reduce batch size or add nodes", flush=True)

    if "mem_available_gb" in resource_stats and "mem_total_gb" in resource_stats:
        mem_used_pct = (1 - resource_stats["mem_available_gb"] / resource_stats["mem_total_gb"]) * 100
        print(f"  Host memory usage:      {mem_used_pct:.1f}%", flush=True)

    print(f"{'='*60}\n", flush=True)

    # Write machine-readable metrics to a JSON file for external autoscaling tools
    metrics_path = os.environ.get("AUTOSCALE_METRICS_PATH", "/tmp/ddp-autoscale-metrics.json")
    metrics = {
        "job_id": slurm_job_id,
        "world_size": world_size,
        "avg_throughput_per_rank": round(avg_throughput, 2),
        "global_throughput": round(global_throughput, 2),
        "scaling_efficiency_pct": round(scaling_efficiency, 2),
        "train_elapsed_s": round(train_elapsed, 2),
        "recommendation": recommendation.split(" - ")[0].strip(),
        "resource_stats": resource_stats,
    }
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[Autoscale] Metrics written to {metrics_path}", flush=True)
    except (PermissionError, OSError) as e:
        print(f"[Autoscale] Could not write metrics: {e}", flush=True)


def run_bandwidth_test(rank, world_size, device, size_mb=32):
    """Measure point-to-point bandwidth with a large tensor all_reduce."""
    if world_size <= 1:
        return

    numel = (size_mb * 1024 * 1024) // 4  # float32
    tensor = torch.randn(numel, device=device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()

    start = time.perf_counter()
    dist.all_reduce(tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    bandwidth_gbps = (size_mb / 1024) / elapsed
    if rank == 0:
        print(f"[Bandwidth] all_reduce {size_mb}MB: {elapsed*1000:.1f}ms ({bandwidth_gbps:.2f} GB/s)", flush=True)


def train(rank, world_size, device, args):
    """Main training loop with throughput measurement."""
    dataset = SyntheticImageDataset(
        num_samples=args.num_samples,
        image_size=32,
        num_classes=10,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = SmallCNN(num_classes=10).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}", flush=True)
        print(f"  Training Configuration", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  World size:      {world_size}", flush=True)
        print(f"  Device:          {device}", flush=True)
        print(f"  Backend:         {'nccl' if device.type == 'cuda' else 'gloo'}", flush=True)
        print(f"  Model params:    {param_count:,}", flush=True)
        print(f"  Dataset size:    {len(dataset):,}", flush=True)
        print(f"  Batch size/rank: {args.batch_size}", flush=True)
        print(f"  Global batch:    {args.batch_size * world_size}", flush=True)
        print(f"  Epochs:          {args.epochs}", flush=True)
        print(f"{'='*60}\n", flush=True)

    total_samples = 0
    epoch_throughputs = []
    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_samples = 0

        model.train()
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            epoch_samples += images.size(0)

        epoch_elapsed = time.perf_counter() - epoch_start
        throughput = epoch_samples / epoch_elapsed
        avg_loss = epoch_loss / epoch_samples
        epoch_throughputs.append(throughput)

        if rank == 0:
            print(
                f"  Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Throughput: {throughput:.0f} samples/s | "
                f"Time: {epoch_elapsed:.2f}s",
                flush=True,
            )

        total_samples += epoch_samples

    train_elapsed = time.perf_counter() - train_start

    if rank == 0:
        global_throughput = (total_samples * world_size) / train_elapsed
        print(f"\n{'='*60}", flush=True)
        print(f"  Training Complete", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Total time:         {train_elapsed:.2f}s", flush=True)
        print(f"  Avg throughput:     {global_throughput:.0f} samples/s (global)", flush=True)
        print(f"  Samples processed:  {total_samples * world_size:,} (across all ranks)", flush=True)
        print(f"{'='*60}\n", flush=True)

    return train_elapsed, epoch_throughputs


def main():
    parser = argparse.ArgumentParser(description="DDP Training Test for Slurm on OCP")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per rank")
    parser.add_argument("--num-samples", type=int, default=8192, help="Synthetic dataset size")
    parser.add_argument(
        "--autoscale", action="store_true",
        help="Enable resource monitoring and scaling efficiency report",
    )
    args = parser.parse_args()

    rank, world_size, local_rank, device = setup_distributed()

    if rank == 0:
        print(f"\n{'#'*60}", flush=True)
        print(f"  Slurm on OCP - Distributed Training Test", flush=True)
        if args.autoscale:
            print(f"  [Autoscaling mode enabled]", flush=True)
        print(f"{'#'*60}", flush=True)
        print(f"  Rank {rank}/{world_size} on {device}", flush=True)
        if "SLURM_JOB_ID" in os.environ:
            print(f"  Slurm Job ID: {os.environ['SLURM_JOB_ID']}", flush=True)
            print(f"  Slurm Nodes:  {os.environ.get('SLURM_NODELIST', 'N/A')}", flush=True)
            if args.autoscale:
                print(f"  Allocated:    {os.environ.get('SLURM_NNODES', '?')} nodes", flush=True)
        print(f"  CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(device)}", flush=True)
            print(f"  GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB", flush=True)
        print()

    if args.autoscale and rank == 0:
        stats = collect_resource_utilization(rank, world_size, device)
        print(f"[Autoscale] Initial resource snapshot:", flush=True)
        for k, v in stats.items():
            if k != "rank":
                print(f"  {k}: {v}", flush=True)
        print(flush=True)

    # Phase 1: Communication test
    with timer("Communication Test", rank):
        comm_ok = run_communication_test(rank, world_size, device)

    if not comm_ok:
        print(f"[FATAL] Communication test failed on rank {rank}. Aborting.", flush=True)
        cleanup()
        return

    # Phase 2: Bandwidth test
    with timer("Bandwidth Test", rank):
        run_bandwidth_test(rank, world_size, device)

    # Phase 3: Training
    with timer("Training", rank):
        train_elapsed, epoch_throughputs = train(rank, world_size, device, args)

    # Phase 4: Autoscaling report
    if args.autoscale:
        print_autoscale_report(rank, world_size, device, epoch_throughputs, train_elapsed)

    # Final summary
    if rank == 0:
        print(f"{'#'*60}", flush=True)
        print(f"  TEST PASSED - All ranks completed successfully", flush=True)
        print(f"{'#'*60}\n", flush=True)

    cleanup()


if __name__ == "__main__":
    main()
