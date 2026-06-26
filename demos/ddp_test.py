"""
Distributed Data Parallel (DDP) training test for Slurm on OpenShift.

Simulates a realistic GPU training workload (ResNet-18 on ImageNet-scale data)
to validate multi-node coordination and stress-test resource limits.

Launch methods:
  1. Auto-launch:  python ddp_test.py --launch
                   (discovers cluster, scales nodes, submits, monitors — zero config)
  2. Via Slurm:    sbatch ... ddp_test.py --intensity medium
  3. Via torchrun: torchrun --nnodes=N --nproc_per_node=1 ddp_test.py
  4. Single-node:  python ddp_test.py (auto-detects single GPU or CPU)

Intensity levels (--intensity):
  light   - Small CNN, 32x32 images, minimal memory (~300MB host RAM)
  medium  - ResNet-18, 224x224, pre-loaded dataset (~1-2GB host RAM)
  heavy   - ResNet-18, 224x224, large dataset + workers (~3-6GB host RAM)

Autoscaling mode (--autoscale):
  Enables resource utilization monitoring and scaling efficiency metrics.
"""

import os
import sys
import time
import json
import math
import argparse
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

LAUNCH_MODE = "--launch" in sys.argv

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler
except ImportError:
    if not LAUNCH_MODE:
        raise
    # Stubs so class definitions don't crash (never instantiated in launch mode)
    class _Stub:
        Module = object
    nn = _Stub()
    Dataset = object


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Standard residual block with two 3x3 convolutions."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet18(nn.Module):
    """
    ResNet-18 (~11.2M parameters).
    Realistic model size that creates meaningful GPU memory pressure
    and requires substantial host RAM for gradient sync buffers.
    """

    def __init__(self, num_classes=1000):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, stride=2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, stride=2), ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, stride=2), ResBlock(512, 512))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SmallCNN(nn.Module):
    """Lightweight CNN for quick validation (~1.2M parameters)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreallocatedDataset(Dataset):
    """
    Pre-allocates a SHARD of the dataset in host memory (data sharding).
    Each rank only loads its portion (total_samples / world_size), reducing
    per-node memory proportionally with the number of workers.

    Memory usage per rank: (num_samples / world_size) * channels * H * W * 4 bytes
      - 8192 images at 224x224x3, 2 ranks = ~2.35 GB each
      - 8192 images at 224x224x3, 4 ranks = ~1.18 GB each
    """

    def __init__(self, num_samples, image_size=224, num_channels=3, num_classes=1000,
                 rank=0, world_size=1):
        self.num_classes = num_classes
        shard_size = num_samples // max(world_size, 1)
        self.num_samples = shard_size
        print(f"  [Dataset] Shard {rank}/{world_size}: allocating {shard_size}/{num_samples} images "
              f"({num_channels}x{image_size}x{image_size}) in host memory...", flush=True)
        mem_estimate_mb = (shard_size * num_channels * image_size * image_size * 4) / (1024 * 1024)
        print(f"  [Dataset] Estimated memory for this shard: {mem_estimate_mb:.0f} MB", flush=True)

        torch.manual_seed(42 + rank)
        self.images = torch.randn(shard_size, num_channels, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (shard_size,))
        print(f"  [Dataset] Shard allocation complete.", flush=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class OnTheFlyDataset(Dataset):
    """Generates data on-the-fly (minimal memory, for light mode)."""

    def __init__(self, num_samples, image_size=32, num_channels=3, num_classes=10):
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@contextmanager
def timer(label, rank):
    if rank == 0:
        print(f"[Timer] {label}...", flush=True)
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if rank == 0:
        print(f"[Timer] {label}: {elapsed:.2f}s", flush=True)


def get_host_memory_mb():
    """Get current process RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (FileNotFoundError, PermissionError):
        pass
    return 0


def setup_distributed():
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_communication_test(rank, world_size, device):
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


def run_bandwidth_test(rank, world_size, device, size_mb=64):
    """Measure collective bandwidth with a large all_reduce."""
    if world_size <= 1:
        return

    numel = (size_mb * 1024 * 1024) // 4
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(rank, world_size, device, args):
    """Main training loop — model and dataset chosen by intensity level."""

    if args.intensity == "light":
        dataset = OnTheFlyDataset(num_samples=args.num_samples, image_size=32, num_classes=10)
        model = SmallCNN(num_classes=10).to(device)
        default_workers = 2
        use_sampler = True
    else:
        image_size = 224
        num_classes = 1000
        dataset = PreallocatedDataset(
            num_samples=args.num_samples,
            image_size=image_size,
            num_classes=num_classes,
            rank=rank,
            world_size=world_size,
        )
        model = ResNet18(num_classes=num_classes).to(device)
        default_workers = 0
        use_sampler = False

    num_workers = args.num_workers if args.num_workers is not None else default_workers

    if num_workers > 0:
        shm_path = "/dev/shm"
        try:
            shm_stats = os.statvfs(shm_path)
            shm_avail_mb = (shm_stats.f_bavail * shm_stats.f_frsize) / (1024 * 1024)
            if shm_avail_mb < 512:
                if rank == 0:
                    print(f"  [Warning] /dev/shm only {shm_avail_mb:.0f} MB — "
                          f"falling back to num_workers=0 to avoid Bus errors", flush=True)
                num_workers = 0
        except OSError:
            pass

    if rank == 0:
        rss = get_host_memory_mb()
        print(f"  [Memory] Process RSS after dataset+model load: {rss:.0f} MB", flush=True)

    sampler = None
    if use_sampler and world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*60}", flush=True)
        print(f"  Training Configuration", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Intensity:       {args.intensity}", flush=True)
        print(f"  World size:      {world_size}", flush=True)
        print(f"  Device:          {device}", flush=True)
        print(f"  Backend:         {'nccl' if device.type == 'cuda' else 'gloo'}", flush=True)
        print(f"  Model:           {'ResNet-18' if args.intensity != 'light' else 'SmallCNN'}", flush=True)
        print(f"  Model params:    {param_count:,}", flush=True)
        print(f"  Dataset size:    {len(dataset):,}", flush=True)
        print(f"  Batch size/rank: {args.batch_size}", flush=True)
        print(f"  Global batch:    {args.batch_size * world_size}", flush=True)
        print(f"  Num workers:     {num_workers}", flush=True)
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
            rss = get_host_memory_mb()
            print(
                f"  Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Throughput: {throughput:.0f} samples/s | "
                f"Time: {epoch_elapsed:.2f}s | "
                f"RSS: {rss:.0f}MB",
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

    return model, train_elapsed, epoch_throughputs


# ---------------------------------------------------------------------------
# Autoscale report
# ---------------------------------------------------------------------------

def collect_resource_utilization(rank, world_size, device):
    stats = {"rank": rank, "device": str(device), "cpu_count": os.cpu_count()}

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

    stats["process_rss_mb"] = round(get_host_memory_mb(), 1)
    return stats


def print_autoscale_report(rank, world_size, device, epoch_throughputs, train_elapsed):
    if rank != 0:
        return

    avg_throughput = sum(epoch_throughputs) / len(epoch_throughputs) if epoch_throughputs else 0
    global_throughput = avg_throughput * world_size

    # Scaling efficiency: compare later epochs (steady state) to first epoch.
    # First epoch includes DDP bucket rebuilding overhead, so steady-state
    # throughput relative to first epoch shows how well distributed comms amortize.
    # For true multi-node scaling efficiency you'd need a single-node baseline.
    if len(epoch_throughputs) >= 2:
        steady_state = sum(epoch_throughputs[1:]) / len(epoch_throughputs[1:])
        first_epoch = epoch_throughputs[0]
        comm_efficiency = (steady_state / first_epoch * 100) if first_epoch > 0 else 100
    else:
        comm_efficiency = 100.0

    print(f"\n{'='*60}", flush=True)
    print(f"  Autoscaling Report", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Current world size:     {world_size}", flush=True)
    print(f"  Per-rank throughput:    {avg_throughput:.0f} samples/s", flush=True)
    print(f"  Global throughput:      {global_throughput:.0f} samples/s", flush=True)
    print(f"  Communication overhead: {100 - comm_efficiency:.1f}% (first epoch vs steady state)", flush=True)

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    slurm_nnodes = os.environ.get("SLURM_NNODES", "N/A")
    slurm_job_num_nodes = os.environ.get("SLURM_JOB_NUM_NODES", slurm_nnodes)
    print(f"  Slurm job:              {slurm_job_id}", flush=True)
    print(f"  Allocated nodes:        {slurm_job_num_nodes}", flush=True)

    # Recommendations based on throughput trend and world size
    if world_size == 1:
        recommendation = "SCALE UP  - Single node; adding workers will parallelize data loading"
    elif comm_efficiency > 95:
        recommendation = "SCALE UP  - Minimal communication overhead, more nodes would help"
    elif comm_efficiency > 80:
        recommendation = "HOLD      - Moderate overhead, current node count is reasonable"
    else:
        recommendation = "SCALE DOWN - High communication overhead, fewer nodes may be faster"

    print(f"  Recommendation:         {recommendation}", flush=True)

    resource_stats = collect_resource_utilization(rank, world_size, device)
    if "gpu_utilization_pct" in resource_stats:
        gpu_util = resource_stats["gpu_utilization_pct"]
        print(f"  GPU memory utilization: {gpu_util:.1f}%", flush=True)
    if "mem_available_gb" in resource_stats and "mem_total_gb" in resource_stats:
        mem_used_pct = (1 - resource_stats["mem_available_gb"] / resource_stats["mem_total_gb"]) * 100
        print(f"  Host memory usage:      {mem_used_pct:.1f}%", flush=True)

    print(f"  Process RSS:            {resource_stats.get('process_rss_mb', 0):.0f} MB", flush=True)
    print(f"{'='*60}\n", flush=True)

    metrics_path = os.environ.get("AUTOSCALE_METRICS_PATH", "/tmp/ddp-autoscale-metrics.json")
    metrics = {
        "job_id": slurm_job_id,
        "world_size": world_size,
        "avg_throughput_per_rank": round(avg_throughput, 2),
        "global_throughput": round(global_throughput, 2),
        "comm_efficiency_pct": round(comm_efficiency, 2),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

INTENSITY_DEFAULTS = {
    "light":  {"batch_size": 64,  "num_samples": 8192,  "epochs": 5, "num_workers": 2},
    "medium": {"batch_size": 128, "num_samples": 2000,  "epochs": 3, "num_workers": 4},
    "heavy":  {"batch_size": 256, "num_samples": 5000,  "epochs": 5, "num_workers": 4},
}

MEMORY_PER_INTENSITY = {
    "light":  {"image_size": 32,  "channels": 3, "model_mb": 5,   "overhead_mb": 300},
    "medium": {"image_size": 224, "channels": 3, "model_mb": 45,  "overhead_mb": 1200},
    "heavy":  {"image_size": 224, "channels": 3, "model_mb": 45,  "overhead_mb": 1500},
}


# ---------------------------------------------------------------------------
# Launch Mode — Auto-discovery, scaling, submission, and monitoring
# ---------------------------------------------------------------------------

class LaunchError(Exception):
    pass


def _run(cmd, check=True, capture=True):
    """Run a shell command, return stdout. Raises LaunchError on failure."""
    result = subprocess.run(
        cmd, shell=True, capture_output=capture, text=True,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise LaunchError(f"Command failed: {cmd}\n{stderr}")
    return result.stdout.strip() if capture else ""


def _find_oc():
    """Find oc or kubectl binary."""
    for binary in ("oc", "kubectl"):
        if shutil.which(binary):
            return binary
    raise LaunchError("Neither 'oc' nor 'kubectl' found in PATH")


def discover_cluster(namespace="slurm", nodeset="slurm-worker-slinky"):
    """
    Query the cluster and return everything needed for resource planning:
      - Pod memory limits (from NodeSet spec)
      - Current replica count
      - Max replicas (from autoscaler deployment if present, else NodeSet)
      - Running worker pod names
      - Controller pod name
    """
    oc = _find_oc()
    info = {"namespace": namespace, "nodeset": nodeset, "oc": oc}

    # Read NodeSet spec
    mem_limit_raw = _run(
        f"{oc} get nodeset {nodeset} -n {namespace} "
        f"-o jsonpath='{{.spec.slurmd.resources.limits.memory}}'"
    ).strip("'")
    info["pod_mem_limit_mb"] = _parse_memory_to_mb(mem_limit_raw)

    replicas = _run(
        f"{oc} get nodeset {nodeset} -n {namespace} "
        f"-o jsonpath='{{.spec.replicas}}'"
    ).strip("'")
    info["current_replicas"] = int(replicas) if replicas else 2

    # Check autoscaler deployment for MAX_REPLICAS, fall back to 8
    try:
        max_rep = _run(
            f"{oc} get deployment slurm-autoscaler -n {namespace} "
            f"-o jsonpath='{{.spec.template.spec.containers[0].env[?(@.name==\"MAX_REPLICAS\")].value}}'"
        ).strip("'")
        info["max_replicas"] = int(max_rep) if max_rep else 8
    except LaunchError:
        info["max_replicas"] = 8

    # Discover running workers
    workers_raw = _run(
        f"{oc} get pods -n {namespace} "
        f"-l nodeset.slinky.slurm.net/name={nodeset} "
        f"--field-selector=status.phase=Running "
        f"-o jsonpath='{{.items[*].metadata.name}}'"
    ).strip("'")
    info["workers"] = workers_raw.split() if workers_raw else []

    # Controller pod
    info["controller"] = "slurm-controller-0"

    return info


def _parse_memory_to_mb(mem_str):
    """Parse Kubernetes memory strings like '4Gi', '4096Mi', '4000M' to MB."""
    if not mem_str:
        return 4096
    mem_str = mem_str.strip()
    if mem_str.endswith("Gi"):
        return int(float(mem_str[:-2]) * 1024)
    elif mem_str.endswith("Mi"):
        return int(float(mem_str[:-2]))
    elif mem_str.endswith("G"):
        return int(float(mem_str[:-1]) * 1000)
    elif mem_str.endswith("M"):
        return int(float(mem_str[:-1]))
    elif mem_str.endswith("Ki"):
        return int(float(mem_str[:-2]) / 1024)
    else:
        return int(mem_str) // (1024 * 1024)


def calculate_plan(cluster_info, intensity_override=None, num_samples_override=None):
    """
    Given cluster constraints, determine the best intensity, dataset size,
    and number of nodes — all automatically.

    Returns a dict with the full execution plan.
    """
    pod_mem_mb = cluster_info["pod_mem_limit_mb"]
    max_nodes = cluster_info["max_replicas"]

    # Select the highest intensity that fits in the pod memory budget
    if intensity_override:
        selected_intensity = intensity_override
    else:
        selected_intensity = "light"
        for intensity in ("heavy", "medium", "light"):
            mem_profile = MEMORY_PER_INTENSITY[intensity]
            if pod_mem_mb > mem_profile["overhead_mb"] + mem_profile["model_mb"] + 200:
                selected_intensity = intensity
                break

    defaults = INTENSITY_DEFAULTS[selected_intensity]
    mem_profile = MEMORY_PER_INTENSITY[selected_intensity]

    num_samples = num_samples_override if num_samples_override else defaults["num_samples"]
    batch_size = defaults["batch_size"]
    epochs = defaults["epochs"]

    # Calculate dataset memory footprint
    bytes_per_sample = (
        mem_profile["image_size"] ** 2
        * mem_profile["channels"]
        * 4  # float32
    )
    dataset_mb = (num_samples * bytes_per_sample) / (1024 * 1024)

    # Per-node memory budget = pod limit - overhead - model
    per_node_budget_mb = pod_mem_mb - mem_profile["overhead_mb"] - mem_profile["model_mb"]
    if per_node_budget_mb <= 0:
        per_node_budget_mb = 512

    # Minimum nodes to fit the dataset (each rank loads a shard)
    if selected_intensity == "light":
        min_nodes = 1
    else:
        min_nodes = math.ceil(dataset_mb / per_node_budget_mb)
        min_nodes = max(1, min(min_nodes, max_nodes))

    return {
        "intensity": selected_intensity,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "epochs": epochs,
        "dataset_mb": round(dataset_mb, 1),
        "per_node_budget_mb": round(per_node_budget_mb, 1),
        "pod_mem_limit_mb": pod_mem_mb,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
    }


def ensure_capacity(cluster_info, plan):
    """Scale the NodeSet up if the plan requires more nodes than currently available."""
    oc = cluster_info["oc"]
    namespace = cluster_info["namespace"]
    nodeset = cluster_info["nodeset"]
    current = cluster_info["current_replicas"]
    needed = plan["min_nodes"]

    if current >= needed:
        _log(f"Cluster has {current} replicas, need {needed} — no scaling required")
        return

    _log(f"Scaling NodeSet: {current} -> {needed} replicas")
    _run(f"{oc} scale nodeset {nodeset} -n {namespace} --replicas={needed}")

    _log("Waiting for pods to be Ready...")
    deadline = time.time() + 300
    while time.time() < deadline:
        workers_raw = _run(
            f"{oc} get pods -n {namespace} "
            f"-l nodeset.slinky.slurm.net/name={nodeset} "
            f"--field-selector=status.phase=Running "
            f"-o jsonpath='{{.items[*].metadata.name}}'"
        ).strip("'")
        ready_pods = workers_raw.split() if workers_raw else []
        if len(ready_pods) >= needed:
            cluster_info["workers"] = ready_pods
            _log(f"All {needed} pods are Running")
            break
        _log(f"  {len(ready_pods)}/{needed} pods ready, waiting...")
        time.sleep(10)
    else:
        raise LaunchError(f"Timed out waiting for {needed} pods to be Ready")

    # Wait for new nodes to register with Slurm
    _log("Waiting for nodes to register with Slurm...")
    deadline = time.time() + 180
    idle_nodes = []
    while time.time() < deadline:
        try:
            sinfo_out = _run(
                f"{oc} exec -n {namespace} {cluster_info['controller']} -c slurmctld -- "
                f"sinfo -h -N -o '%N %T'",
                check=False,
            )
            idle_nodes = [
                line.split()[0] for line in sinfo_out.splitlines()
                if line.strip() and any(s in line for s in ("idle", "mix"))
            ]
            # Deduplicate (nodes appear once per partition)
            idle_nodes = list(set(idle_nodes))
            if len(idle_nodes) >= needed:
                _log(f"  {len(idle_nodes)} Slurm nodes ready: {', '.join(sorted(idle_nodes))}")
                return
        except LaunchError:
            pass
        _log(f"  {len(idle_nodes)}/{needed} Slurm nodes registered, waiting...")
        time.sleep(10)

    raise LaunchError(f"Timed out waiting for {needed} nodes to register with Slurm")


def _provision_single_worker(pod, oc, namespace, script_path, pytorch_index):
    """Provision a single worker pod with PyTorch and the training script.
    Designed to run in a thread pool for parallel provisioning."""
    # Check if already provisioned and working
    ret = subprocess.run(
        f"{oc} exec -n {namespace} {pod} -c slurmd -- "
        f"python3 -c \"import torch; print(torch.__version__)\"",
        shell=True, capture_output=True, text=True,
    )
    if ret.returncode == 0:
        _log(f"  {pod}: PyTorch verified ({ret.stdout.strip()}), copying script...")
    else:
        _log(f"  {pod}: Installing pip + PyTorch (this takes a few minutes)...")

        # Install pip — retry until it works
        for attempt in range(3):
            pip_ret = subprocess.run(
                f"{oc} exec -n {namespace} {pod} -c slurmd -- "
                f"bash -c 'apt-get update -qq && apt-get install -y -qq python3-pip'",
                shell=True, capture_output=True, text=True,
            )
            if pip_ret.returncode == 0:
                break
            _log(f"  {pod}: pip install attempt {attempt+1} failed, retrying in 15s...")
            time.sleep(15)
        else:
            raise LaunchError(f"Failed to install pip on {pod}: {pip_ret.stderr.strip()}")

        # Install PyTorch — this is the slow step
        _log(f"  {pod}: Installing PyTorch (this is the slow part)...")
        torch_ret = subprocess.run(
            f"{oc} exec -n {namespace} {pod} -c slurmd -- "
            f"pip3 install --break-system-packages torch "
            f"--index-url {pytorch_index}",
            shell=True, capture_output=True, text=True,
        )
        if torch_ret.returncode != 0:
            _log(f"  {pod}: First torch install failed, retrying...")
            torch_ret = subprocess.run(
                f"{oc} exec -n {namespace} {pod} -c slurmd -- "
                f"pip3 install --break-system-packages --force-reinstall torch "
                f"--index-url {pytorch_index}",
                shell=True, capture_output=True, text=True,
            )
            if torch_ret.returncode != 0:
                raise LaunchError(
                    f"PyTorch installation failed on {pod}: {torch_ret.stderr[-500:]}"
                )

        # Verify installation succeeded
        _log(f"  {pod}: Verifying PyTorch import...")
        verify = subprocess.run(
            f"{oc} exec -n {namespace} {pod} -c slurmd -- "
            f"python3 -c \"import torch; print(torch.__version__)\"",
            shell=True, capture_output=True, text=True,
        )
        if verify.returncode != 0:
            _log(f"  {pod}: WARNING — PyTorch import failed, retrying install...")
            _run(
                f"{oc} exec -n {namespace} {pod} -c slurmd -- "
                f"pip3 install --break-system-packages --force-reinstall torch "
                f"--index-url {pytorch_index}",
                check=False,
            )
            verify2 = subprocess.run(
                f"{oc} exec -n {namespace} {pod} -c slurmd -- "
                f"python3 -c \"import torch; print(torch.__version__)\"",
                shell=True, capture_output=True, text=True,
            )
            if verify2.returncode != 0:
                raise LaunchError(
                    f"PyTorch installation failed on {pod}: {verify2.stderr.strip()}"
                )
        _log(f"  {pod}: PyTorch ready ({verify.stdout.strip() if verify.returncode == 0 else 'reinstalled'})")

    # Copy training script
    _run(f"{oc} cp {script_path} {namespace}/{pod}:/tmp/ddp_test.py -c slurmd")
    return pod


def provision_workers(cluster_info, plan, pytorch_index="https://download.pytorch.org/whl/cu124"):
    """Install PyTorch and copy ddp_test.py to all workers + controller.
    Workers are provisioned in parallel to minimize wait time."""
    oc = cluster_info["oc"]
    namespace = cluster_info["namespace"]
    controller = cluster_info["controller"]
    workers = cluster_info["workers"]
    script_path = str(Path(__file__).resolve())

    _log(f"Provisioning {len(workers)} worker(s) in parallel...")

    # Provision all workers concurrently
    errors = []
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = {
            executor.submit(_provision_single_worker, pod, oc, namespace, script_path, pytorch_index): pod
            for pod in workers
        }
        for future in as_completed(futures):
            pod = futures[future]
            try:
                future.result()
            except Exception as e:
                errors.append(f"{pod}: {e}")

    if errors:
        raise LaunchError(f"Provisioning failed:\n  " + "\n  ".join(errors))

    # Final readiness gate: confirm ALL workers can import torch
    _log("Verifying all workers are ready...")
    for pod in workers:
        deadline = time.time() + 120
        while time.time() < deadline:
            ret = subprocess.run(
                f"{oc} exec -n {namespace} {pod} -c slurmd -- "
                f"python3 -c \"import torch\"",
                shell=True, capture_output=True,
            )
            if ret.returncode == 0:
                break
            _log(f"  {pod}: waiting for PyTorch to be ready...")
            time.sleep(10)
        else:
            raise LaunchError(f"{pod} failed readiness check — PyTorch not importable after 120s")

    # Also copy to controller for sbatch access
    _run(f"{oc} cp {script_path} {namespace}/{controller}:/tmp/ddp_test.py -c slurmctld")
    _log("All workers verified and ready")


def submit_job(cluster_info, plan):
    """Submit this script as a Slurm batch job with the calculated node count."""
    oc = cluster_info["oc"]
    namespace = cluster_info["namespace"]
    controller = cluster_info["controller"]

    min_nodes = plan["min_nodes"]
    max_nodes = plan["max_nodes"]
    intensity = plan["intensity"]
    epochs = plan["epochs"]
    batch_size = plan["batch_size"]
    num_samples = plan["num_samples"]

    batch_script = (
        "#!/bin/bash\n"
        f"#SBATCH --job-name=ddp-autoscale\n"
        f"#SBATCH --nodes={min_nodes}-{max_nodes}\n"
        "#SBATCH --ntasks-per-node=1\n"
        "#SBATCH --cpus-per-task=2\n"
        "#SBATCH --time=00:30:00\n"
        "#SBATCH --output=/tmp/ddp-autoscale-%j.out\n"
        "#SBATCH --error=/tmp/ddp-autoscale-%j.err\n"
        "#SBATCH --export=ALL\n"
        "\n"
        "export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)\n"
        "export MASTER_PORT=29500\n"
        "export WORLD_SIZE=$SLURM_NTASKS\n"
        "\n"
        f"srun python3 /tmp/ddp_test.py \\\n"
        f"    --intensity {intensity} \\\n"
        f"    --epochs {epochs} \\\n"
        f"    --batch-size {batch_size} \\\n"
        f"    --num-samples {num_samples} \\\n"
        f"    --autoscale \\\n"
        f"    --output-dir /tmp/ddp-results\n"
    )

    # Write batch script via stdin pipe to avoid shell escaping issues
    write_proc = subprocess.run(
        f"{oc} exec -n {namespace} {controller} -c slurmctld -i -- "
        f"tee /tmp/ddp-autoscale-batch.sh",
        shell=True, input=batch_script, capture_output=True, text=True,
    )
    if write_proc.returncode != 0:
        raise LaunchError(f"Failed to write batch script: {write_proc.stderr}")

    output = _run(
        f"{oc} exec -n {namespace} {controller} -c slurmctld -- "
        f"sbatch /tmp/ddp-autoscale-batch.sh"
    )

    for word in output.split():
        if word.isdigit():
            _log(f"Submitted batch job {word} (nodes: {min_nodes}-{max_nodes})")
            return int(word)

    raise LaunchError(f"Failed to parse job ID from: {output}")


def monitor_job(cluster_info, job_id, timeout=600):
    """Poll Slurm until the job completes or times out."""
    oc = cluster_info["oc"]
    namespace = cluster_info["namespace"]
    controller = cluster_info["controller"]

    _log(f"Monitoring job {job_id} (timeout: {timeout}s)")
    start = time.time()
    last_state = ""

    while (time.time() - start) < timeout:
        try:
            state = _run(
                f"{oc} exec -n {namespace} {controller} -c slurmctld -- "
                f"squeue -j {job_id} -h -o '%T'",
                check=False,
            ).strip().strip("'")
        except LaunchError:
            state = ""

        if not state or state == "":
            _log(f"Job {job_id} finished")
            return True

        elapsed = int(time.time() - start)
        if state != last_state:
            _log(f"Job {job_id}: {state} ({elapsed}s)")
            last_state = state

        time.sleep(10)

    _log(f"WARNING: Job {job_id} timed out after {timeout}s (may still be running)")
    return False


def retrieve_results(cluster_info, job_id):
    """Pull job output and training artifacts back to the local machine."""
    oc = cluster_info["oc"]
    namespace = cluster_info["namespace"]
    controller = cluster_info["controller"]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Find which host ran the job
    try:
        batch_host = _run(
            f"{oc} exec -n {namespace} {controller} -c slurmctld -- "
            f"scontrol show job {job_id}",
            check=False,
        )
        host = ""
        for line in batch_host.splitlines():
            if "BatchHost=" in line:
                host = line.split("BatchHost=")[1].split()[0]
                break
    except LaunchError:
        host = ""

    if not host:
        host = "slinky-0"

    batch_pod = f"slurm-worker-{host}"
    _log(f"Retrieving results from {batch_pod}...")

    # Job stdout
    out_file = results_dir / f"job-{job_id}.out"
    try:
        content = _run(
            f"{oc} exec -n {namespace} {batch_pod} -c slurmd -- "
            f"cat /tmp/ddp-autoscale-{job_id}.out",
            check=False,
        )
        if content:
            out_file.write_text(content)
            _log(f"  -> {out_file}")
    except LaunchError:
        _log(f"  Output file not found on {batch_pod}")

    # Job stderr
    err_file = results_dir / f"job-{job_id}.err"
    try:
        content = _run(
            f"{oc} exec -n {namespace} {batch_pod} -c slurmd -- "
            f"cat /tmp/ddp-autoscale-{job_id}.err",
            check=False,
        )
        if content:
            err_file.write_text(content)
            _log(f"  -> {err_file}")
    except LaunchError:
        pass

    # Training artifacts
    try:
        _run(
            f"{oc} cp {namespace}/{batch_pod}:/tmp/ddp-results "
            f"{results_dir}/ddp-results -c slurmd",
            check=False,
        )
        if (results_dir / "ddp-results").exists():
            _log(f"  -> {results_dir}/ddp-results/")
    except LaunchError:
        pass

    # Print summary from output
    if out_file.exists():
        text = out_file.read_text()
        if "TEST PASSED" in text:
            _log("RESULT: PASSED")
        elif "FATAL" in text or "Traceback" in text:
            _log("RESULT: FAILED")
        else:
            _log("RESULT: check output for details")

        for line in text.splitlines():
            if any(k in line for k in ("Training Complete", "Total time",
                                        "Avg throughput", "Communication overhead",
                                        "Recommendation")):
                print(f"  {line.strip()}")

    _log(f"Results saved to: {results_dir}/")


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def launch_main():
    """
    Orchestration entry point — discovers the cluster, calculates resource
    requirements, scales nodes, provisions workers, submits the training job,
    and retrieves results. No inputs required.
    """
    parser = argparse.ArgumentParser(
        description="Auto-launch DDP training on Slurm/OCP (zero-config)",
    )
    parser.add_argument("--launch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--namespace", default="slurm",
                        help="Kubernetes namespace (default: slurm)")
    parser.add_argument("--nodeset", default="slurm-worker-slinky",
                        help="NodeSet name (default: slurm-worker-slinky)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Job timeout in seconds (default: 600)")
    parser.add_argument("--no-monitor", action="store_true",
                        help="Submit and exit without waiting for completion")
    parser.add_argument("--intensity", choices=["light", "medium", "heavy"], default=None,
                        help="Override auto-detected intensity")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Override auto-detected dataset size")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="Override maximum node count")
    parser.add_argument("--pytorch-index", default=None,
                        help="PyTorch package index URL (default: env PYTORCH_INDEX or cu124)")
    args = parser.parse_args()

    pytorch_index = (
        args.pytorch_index
        or os.environ.get("PYTORCH_INDEX")
        or "https://download.pytorch.org/whl/cu124"
    )

    print(flush=True)
    print("=" * 60, flush=True)
    print("  Slurm on OCP — Auto-Launch DDP Training", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # Phase 1: Discover cluster state
    _log("Discovering cluster...")
    cluster_info = discover_cluster(args.namespace, args.nodeset)
    _log(f"  Namespace:      {cluster_info['namespace']}")
    _log(f"  NodeSet:        {cluster_info['nodeset']}")
    _log(f"  Pod mem limit:  {cluster_info['pod_mem_limit_mb']} MB")
    _log(f"  Replicas:       {cluster_info['current_replicas']}")
    _log(f"  Max replicas:   {cluster_info['max_replicas']}")
    _log(f"  Workers online: {len(cluster_info['workers'])}")
    print(flush=True)

    # Phase 2: Calculate resource plan
    if args.max_nodes:
        cluster_info["max_replicas"] = args.max_nodes

    plan = calculate_plan(cluster_info,
                          intensity_override=args.intensity,
                          num_samples_override=args.num_samples)

    _log("Resource plan:")
    _log(f"  Intensity:       {plan['intensity']}")
    _log(f"  Num samples:     {plan['num_samples']}")
    _log(f"  Dataset memory:  {plan['dataset_mb']} MB")
    _log(f"  Per-node budget: {plan['per_node_budget_mb']} MB")
    _log(f"  Nodes needed:    {plan['min_nodes']}-{plan['max_nodes']}")
    _log(f"  Batch size:      {plan['batch_size']}")
    _log(f"  Epochs:          {plan['epochs']}")
    print(flush=True)

    # Phase 3: Scale cluster
    _log("Ensuring cluster capacity...")
    ensure_capacity(cluster_info, plan)
    print(flush=True)

    # Phase 4: Provision workers
    _log("Provisioning workers...")
    provision_workers(cluster_info, plan, pytorch_index=pytorch_index)
    print(flush=True)

    # Phase 5: Submit job
    _log("Submitting training job...")
    job_id = submit_job(cluster_info, plan)
    print(flush=True)

    if args.no_monitor:
        _log("Job submitted. Use these commands to monitor:")
        _log(f"  oc exec -n {args.namespace} slurm-controller-0 -c slurmctld -- squeue -l")
        _log(f"  oc logs -n {args.namespace} -l app.kubernetes.io/name=slurm-autoscaler -f")
        return

    # Phase 6: Monitor
    success = monitor_job(cluster_info, job_id, timeout=args.timeout)
    print(flush=True)

    # Phase 7: Retrieve results
    if success:
        retrieve_results(cluster_info, job_id)

    print(flush=True)
    print("=" * 60, flush=True)
    if success:
        print("  Launch complete", flush=True)
    else:
        print("  Launch timed out — job may still be running", flush=True)
    print("=" * 60, flush=True)


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def save_results(model, device, args, rank, world_size, train_elapsed, epoch_throughputs, output_dir):
    """
    Save training artifacts to disk (rank 0 only):
      - model checkpoint (.pt)
      - training metrics summary (.json)
      - synthetic predictions dataset (.pt) — model inference on a generated batch
    """
    if rank != 0:
        return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Model checkpoint ---
    checkpoint_path = run_dir / "model_checkpoint.pt"
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "model_state_dict": state_dict,
        "intensity": args.intensity,
        "epochs": args.epochs,
        "world_size": world_size,
    }, checkpoint_path)
    print(f"  [Results] Model checkpoint saved: {checkpoint_path}", flush=True)

    # --- Training metrics ---
    avg_throughput = sum(epoch_throughputs) / len(epoch_throughputs) if epoch_throughputs else 0
    metrics = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intensity": args.intensity,
        "world_size": world_size,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "train_elapsed_s": round(train_elapsed, 3),
        "avg_throughput_samples_per_s": round(avg_throughput, 2),
        "global_throughput_samples_per_s": round(avg_throughput * world_size, 2),
        "epoch_throughputs": [round(t, 2) for t in epoch_throughputs],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodes": os.environ.get("SLURM_NODELIST"),
    }
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [Results] Metrics saved: {metrics_path}", flush=True)

    # --- Synthetic predictions dataset ---
    # Run inference on a small generated batch to produce a reusable predictions file
    model.eval()
    image_size = 32 if args.intensity == "light" else 224
    num_classes = 10 if args.intensity == "light" else 1000
    num_prediction_samples = min(200, args.num_samples)

    with torch.no_grad():
        sample_inputs = torch.randn(num_prediction_samples, 3, image_size, image_size, device=device)
        logits = model(sample_inputs)
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)

    predictions_path = run_dir / "predictions.pt"
    torch.save({
        "inputs": sample_inputs.cpu(),
        "logits": logits.cpu(),
        "probabilities": probabilities.cpu(),
        "predicted_classes": predicted_classes.cpu(),
        "num_classes": num_classes,
        "image_size": image_size,
        "description": "Synthetic predictions from trained model on generated data",
    }, predictions_path)
    print(f"  [Results] Predictions dataset saved: {predictions_path}", flush=True)

    print(f"\n  [Results] All artifacts saved to: {run_dir}/", flush=True)
    print(f"  [Results] Contents:", flush=True)
    for p in sorted(run_dir.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"    - {p.name} ({size_kb:.1f} KB)", flush=True)

    # Print metrics inline so they're captured in the job output file
    print(f"\n  [Results] === METRICS (inline) ===", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"  [Results] === END METRICS ===", flush=True)

    # Print retrieval instructions for cluster runs
    hostname = os.environ.get("HOSTNAME", "unknown")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if slurm_job_id:
        print(f"\n  [Results] To retrieve artifacts from this node:", flush=True)
        print(f"  oc cp slurm/{hostname}:{run_dir} results/ -c slurmd", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="DDP Training Test for Slurm on OCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Intensity levels control memory pressure on the pod:
  light  - ~300MB host RAM  (will NOT OOM with 1Gi limit)
  medium - ~1.5GB host RAM  (WILL OOM with 1Gi limit)
  heavy  - ~4GB+ host RAM   (WILL OOM with anything < 4Gi)
        """,
    )
    parser.add_argument("--intensity", choices=["light", "medium", "heavy"], default="medium",
                        help="Workload intensity level (default: medium)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size per rank")
    parser.add_argument("--num-samples", type=int, default=None, help="Override dataset size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--autoscale", action="store_true",
                        help="Enable resource monitoring and scaling efficiency report")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (model checkpoint, metrics, predictions). "
                             "Defaults to results/ relative to the repo root.")
    args = parser.parse_args()

    defaults = INTENSITY_DEFAULTS[args.intensity]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.num_samples is None:
        args.num_samples = defaults["num_samples"]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.num_workers is None:
        args.num_workers = defaults["num_workers"]

    if args.output_dir is None:
        script_dir = Path(__file__).resolve().parent
        candidate = script_dir.parent / "results"
        if script_dir.parent.name == "slurm-on-ocp" or (script_dir.parent / "demos").is_dir():
            args.output_dir = str(candidate)
        else:
            args.output_dir = "/tmp/ddp-results"

    rank, world_size, local_rank, device = setup_distributed()

    if rank == 0:
        print(f"\n{'#'*60}", flush=True)
        print(f"  Slurm on OCP - Distributed Training Test", flush=True)
        print(f"  Intensity: {args.intensity.upper()}", flush=True)
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
        rss = get_host_memory_mb()
        print(f"  Host RSS (pre-load): {rss:.0f} MB", flush=True)
        print(flush=True)

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
        sys.exit(1)

    # Phase 2: Bandwidth test
    with timer("Bandwidth Test", rank):
        run_bandwidth_test(rank, world_size, device)

    # Phase 3: Training (this is where OOM will likely occur)
    with timer("Training", rank):
        model, train_elapsed, epoch_throughputs = train(rank, world_size, device, args)

    # Phase 4: Save results
    with timer("Saving Results", rank):
        save_results(model, device, args, rank, world_size, train_elapsed, epoch_throughputs, args.output_dir)

    # Phase 5: Autoscaling report
    if args.autoscale:
        print_autoscale_report(rank, world_size, device, epoch_throughputs, train_elapsed)

    # Final summary
    if rank == 0:
        print(f"{'#'*60}", flush=True)
        print(f"  TEST PASSED - All ranks completed successfully", flush=True)
        print(f"{'#'*60}\n", flush=True)

    cleanup()


if __name__ == "__main__":
    if LAUNCH_MODE:
        launch_main()
    else:
        main()
