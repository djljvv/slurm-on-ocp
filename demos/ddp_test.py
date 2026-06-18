"""
Distributed Data Parallel (DDP) training test for Slurm on OpenShift.

Simulates a realistic GPU training workload (ResNet-18 on ImageNet-scale data)
to validate multi-node coordination and stress-test resource limits.

Launch methods:
  1. Via Slurm:   sbatch submit_job.sh
  2. Via torchrun: torchrun --nnodes=N --nproc_per_node=1 ddp_test.py
  3. Single-node:  python ddp_test.py (auto-detects single GPU or CPU)

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
import argparse
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


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
    Pre-allocates entire dataset in host memory.
    This is realistic (mimics loading image files into RAM) and is the primary
    mechanism that stresses pod memory limits.

    Memory usage: num_samples * channels * H * W * 4 bytes
      - 5000 images at 224x224x3  = ~2.8 GB
      - 10000 images at 224x224x3 = ~5.6 GB
      - 2000 images at 224x224x3  = ~1.1 GB
    """

    def __init__(self, num_samples, image_size=224, num_channels=3, num_classes=1000):
        self.num_samples = num_samples
        self.num_classes = num_classes
        print(f"  [Dataset] Pre-allocating {num_samples} images "
              f"({num_channels}x{image_size}x{image_size}) in host memory...", flush=True)
        mem_estimate_mb = (num_samples * num_channels * image_size * image_size * 4) / (1024 * 1024)
        print(f"  [Dataset] Estimated memory: {mem_estimate_mb:.0f} MB", flush=True)

        self.images = torch.randn(num_samples, num_channels, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
        print(f"  [Dataset] Allocation complete.", flush=True)

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
        num_workers = 2
    else:
        image_size = 224
        num_classes = 1000
        dataset = PreallocatedDataset(
            num_samples=args.num_samples,
            image_size=image_size,
            num_classes=num_classes,
        )
        model = ResNet18(num_classes=num_classes).to(device)
        num_workers = args.num_workers

    if rank == 0:
        rss = get_host_memory_mb()
        print(f"  [Memory] Process RSS after dataset+model load: {rss:.0f} MB", flush=True)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

INTENSITY_DEFAULTS = {
    "light":  {"batch_size": 64,  "num_samples": 8192,  "epochs": 5, "num_workers": 2},
    "medium": {"batch_size": 128, "num_samples": 2000,  "epochs": 3, "num_workers": 4},
    "heavy":  {"batch_size": 256, "num_samples": 5000,  "epochs": 5, "num_workers": 4},
}


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
    main()
