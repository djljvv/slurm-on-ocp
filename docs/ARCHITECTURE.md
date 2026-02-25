# Slurm on OpenShift - Architecture Guide

This document explains how Slurm runs on OpenShift with the Slinky operator: namespaces, components, and job flow.

## Table of Contents

1. [Overview](#overview)
2. [Namespaces](#namespaces)
3. [Key Components](#key-components)
4. [Architecture Diagram](#architecture-diagram)
5. [Job Flow on OCP](#job-flow-on-ocp)
6. [Controller vs NodeSet](#controller-vs-nodeset)

## Overview

Slurm on OpenShift uses the **Slinky** operator to run Slurm in a cloud-native way. The operator watches custom resources (Controller, NodeSet) and reconciles them into Slurm controller and compute pods. You get the same Slurm semantics (sbatch, srun, sinfo, etc.) while running on Kubernetes/OpenShift.

## Namespaces

| Namespace | Purpose | What runs there |
|-----------|---------|------------------|
| **slinky** | Operator | Slurm Operator pod, CRDs. The operator watches the cluster and reconciles Controller/NodeSet resources. |
| **slurm** | Workload | Controller CR, NodeSet CR, slurmctld pod, slurmd pods, secrets, services. This is the actual Slurm cluster. |
| **cert-manager** | Prerequisite | cert-manager pods (for operator TLS). |

**In short:** The operator lives in **slinky** and manages resources you create in **slurm**. You need both.

## Key Components

### Slurm Operator (Slinky)

- **Role**: Watches `Controller` and `NodeSet` CRs and creates/updates Slurm pods and config.
- **Location**: `slinky` namespace (or `openshift-operators` if installed via OperatorHub).
- **Responsibilities**:
  - Install and manage Slurm CRDs
  - Reconcile Controller → StatefulSet + Services for slurmctld
  - Reconcile NodeSet → pods/StatefulSet for slurmd
  - Manage certificates (via cert-manager)
  - Apply Slurm config (partitions, etc.) from CRs

### slurmctld (Slurm Controller Daemon)

- **Role**: Central Slurm controller and scheduler (same as on bare metal/RHEL).
- **Location**: Pod(s) in **slurm** namespace, created from the `Controller` CR.
- **Responsibilities**: Job scheduling, queue management, node status, authentication (e.g. JWT/slurm key).

### slurmd (Slurm Daemon)

- **Role**: Compute node daemon; runs jobs assigned by slurmctld.
- **Location**: Pods in **slurm** namespace, created from the `NodeSet` CR (one or more replicas).
- **Responsibilities**: Execute jobs, report node resources, communicate with slurmctld.

### Client Tools

Used inside the controller pod (or via `oc exec`):

- **srun**, **sbatch**: Submit jobs
- **squeue**, **sinfo**, **scontrol**: Inspect queue and nodes
- **scancel**: Cancel jobs

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│              OpenShift Cluster                          │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Slurm Operator (slinky namespace)               │   │
│  │  - Watches Controller / NodeSet CRs              │   │
│  │  - Creates/updates slurmctld and slurmd pods     │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│  ┌───────────────────────┴──────────────────────────┐   │
│  │  slurm namespace                                 │   │
│  │                                                  │   │
│  │  ┌──────────────┐      ┌──────────────────┐      │   │
│  │  │ Controller   │      │ NodeSet          │      │   │
│  │  │ (slurmctld)  │◄────►│ (slurmd pods)    │      │   │
│  │  │              │      │ (compute)        │      │   │
│  │  └──────────────┘      └──────────────────┘      │   │
│  │         │                         │              │   │
│  │         │ sbatch / srun           │ run jobs     │   │
│  │         ▼                         ▼              │   │
│  │  ┌──────────────────────────────────────────┐    │   │
│  │  │  Job submission (oc exec ... srun/sbatch)│    │   │
│  │  └──────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Job Flow on OCP

1. **Submit**: You run `oc exec -n slurm <controller-pod> -c slurmctld -- sbatch job.sh` (or use scripts that do this).
2. **slurmctld**: Receives the job, queues it, and schedules it onto a slurmd pod.
3. **slurmd**: Runs the job on the chosen compute pod and reports completion.
4. **Result**: Output can be read from the controller or compute pod (e.g. shared volume or job output path).

Same Slurm semantics as on RHEL; only the “nodes” are pods and access is via `oc exec` or port-forward.

## Controller vs NodeSet

| Aspect | Controller CR | NodeSet CR |
|--------|----------------|------------|
| **Maps to** | slurmctld (1 per cluster) | slurmd (many pods) |
| **Namespace** | slurm | slurm |
| **References** | — | controllerRef → Controller |
| **Resources** | spec.slurmctld.resources | spec.slurmd.resources |
| **Secrets** | jwtHs256KeyRef, slurmKeyRef | — |
| **Replicas** | 1 | spec.replicas (e.g. 2) |

## Summary

- **slinky**: Operator namespace; runs the Slurm operator that reconciles CRs.
- **slurm**: Workload namespace; Controller (slurmctld) + NodeSet (slurmd pods).
- **cert-manager**: Provides TLS for the operator.
- Jobs are submitted via the controller pod; compute runs on NodeSet pods, same as Slurm on RHEL but containerized.

## Additional Resources

- [Slurm Architecture (generic)](https://slurm.schedmd.com/overview.html)
- [Slinky Project](https://slinky.schedmd.com/)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)
- [Add Nodes](ADD_NODES.md)
