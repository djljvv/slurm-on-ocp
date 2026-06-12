# Slurm on OpenShift (OCP)

Complete guide for setting up and using Slurm workload manager on Red Hat OpenShift using the Slinky operator.

## Quick Links

- **[Quick Start Guide](QUICK_START.md)** - Get Slurm on OpenShift running in a few steps
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Full deployment (CLI and UI)
- **[DDP Test Guide](docs/DDP_TEST_GUIDE.md)** - Run distributed PyTorch training on Slurm (CPU & GPU)
- **[Add Nodes](docs/ADD_NODES.md)** - Add physical/virtual/containerized nodes
- **[Architecture Guide](docs/ARCHITECTURE.md)** - Slurm on OCP architecture

## Overview

This repository provides a complete, reproducible setup for running Slurm workload manager on Red Hat OpenShift. Slurm is an open-source job scheduler and workload manager used in HPC environments. The Slinky operator runs Slurm in a cloud-native way on OpenShift.

## Prerequisites

- OpenShift 4.x cluster with admin access
- `oc` CLI installed and configured
- `helm` 3.x installed
- Sufficient cluster resources (or use the deploy script to install cert-manager and operator)

## Quick Start

If you already have an OpenShift cluster, follow the [Quick Start Guide](QUICK_START.md):

1. Ensure cert-manager is installed (or let the script install it)
2. Install Slurm Operator CRDs and operator (Slinky)
3. Create Slurm auth secrets
4. Deploy Slurm cluster (Controller + NodeSet)
5. Verify and run test jobs

## Documentation

### Quick Start Guide

The [Quick Start Guide](QUICK_START.md) provides a streamlined setup to get Slurm running on OpenShift quickly.

### Deployment Guide

The [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) includes:
- CLI and UI deployment methods
- cert-manager and operator installation
- Cluster deployment and verification
- Test scenarios and troubleshooting

### Architecture & Add Nodes

- **[Architecture](docs/ARCHITECTURE.md)** - How Slurm on OCP works (operator, controller, compute nodes)
- **[Add Nodes](docs/ADD_NODES.md)** - Adding OpenShift, physical, or virtual nodes

### DDP Test Guide

The [DDP Test Guide](docs/DDP_TEST_GUIDE.md) covers:
- Configuring GPU access for Slurm worker pods
- Running distributed PyTorch training across Slurm-managed nodes
- Validating NCCL/Gloo communication over the K8s network
- Autoscaling with KEDA — elastic job submission and NodeSet scaling based on demand
- Troubleshooting common issues

## Repository Structure

```
slurm-on-ocp/
├── README.md                    # This file
├── QUICK_START.md               # Quick setup guide
├── configs/
│   ├── slurm-cluster.yaml       # Controller + NodeSet (required): oc apply -f configs/slurm-cluster.yaml
│   ├── slurm-autoscaler.yaml    # KEDA ScaledObject + Slurm REST API for autoscaling
│   └── slurm-values.yaml        # Optional: only for Helm-based cluster deploy (helm install slurm ... -f this)
├── demos/
│   ├── ddp_test.py              # PyTorch DDP distributed training test (supports --autoscale)
│   ├── submit_job.sh            # Slurm batch submission script (fixed nodes)
│   └── submit_job_autoscale.sh  # Elastic submission script (--nodes=min-max, --requeue)
├── docs/
│   ├── DEPLOYMENT_GUIDE.md      # Step-by-step deployment (CLI and UI)
│   ├── DDP_TEST_GUIDE.md        # Distributed training test guide (CPU & GPU)
│   ├── ADD_NODES.md             # Adding physical/virtual/containerized nodes
│   └── ARCHITECTURE.md          # Slurm on OCP architecture
└── scripts/
    ├── deploy-slurm.sh          # Deploy Slurm (operator + cluster)
    ├── cleanup-slurm.sh         # Remove Slurm resources
    └── test-slurm.sh            # Cluster tests
```

## Support

- Check the [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) troubleshooting section
- [Slinky Project](https://slinky.schedmd.com/)
- [Slurm Documentation](https://slurm.schedmd.com/)
- Open an issue in this repository

## License

This repository and its contents are provided as-is for educational and reference purposes.
