# Adding Physical or Virtual Nodes to Slurm on OpenShift

## Executive Summary

Procedures for adding physical (bare metal) or virtual (VM) nodes to an existing Slurm cluster on OpenShift. Covers four scenarios: OpenShift worker nodes, external physical nodes, external virtual nodes, and scaling containerized compute nodes.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Scenario 1: Adding OpenShift Worker Nodes](#scenario-1-adding-openshift-worker-nodes)
4. [Scenario 2: Adding External Physical Nodes](#scenario-2-adding-external-physical-nodes)
5. [Scenario 3: Adding External Virtual Nodes](#scenario-3-adding-external-virtual-nodes)
6. [Scenario 4: Scaling Containerized Nodes](#scenario-4-scaling-containerized-nodes)
7. [Configuration Management](#configuration-management)
8. [Network & Authentication](#network--authentication)
9. [Resource Management](#resource-management)
10. [Verification & Troubleshooting](#verification--troubleshooting)

---

## Overview

Add compute capacity to an existing Slurm cluster on OpenShift via:
1. **OpenShift Worker Nodes**: Scale NodeSets to use new worker nodes
2. **External Physical Nodes**: Bare metal servers with native Slurm
3. **External Virtual Nodes**: VMs with native Slurm
4. **Containerized Nodes**: Scale existing NodeSets or create new ones

### Architecture Scenarios

#### Current Architecture (Existing Slurm on OpenShift)

```
┌─────────────────────────────────────────────────────────┐
│              OpenShift Cluster                          │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Slurm Controller (Pod)                   │   │
│  │  - slurmctld running in container                │   │
│  │  - Manages job scheduling                        │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│  ┌───────────────────────┴─────────────────────────┐    │
│  │                                                 │    │
│  │  ┌──────────────┐      ┌──────────────────┐     │    │
│  │  │ Slurm Compute│      │ Slurm Compute    │     │    │
│  │  │ Pod 0        │      │ Pod 1            │     │    │
│  │  │ (slurmd)     │      │ (slurmd)         │     │    │
│  │  └──────────────┘      └──────────────────┘     │    │
│  │                                                 │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

#### Target Architecture (After Adding Nodes)

```
┌─────────────────────────────────────────────────────────┐
│              OpenShift Cluster                          │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Slurm Controller (Pod)                   │   │
│  │  - slurmctld running in container                │   │
│  │  - Manages job scheduling                        │   │
│  └──────────────────────────────────────────────────┘   │
│                          │                              │
│  ┌───────────────────────┴──────────────────────────┐   │
│  │                                                  │   │
│  │  ┌──────────────┐      ┌──────────────────┐      │   │
│  │  │ Slurm Compute│      │ Slurm Compute    │      │   │
│  │  │ Pod 0        │      │ Pod 1            │      │   │
│  │  │ (slurmd)     │      │ (slurmd)         │      │   │
│  │  └──────────────┘      └──────────────────┘      │   │
│  │                                                  │   │
│  │  ┌──────────────┐      ┌──────────────────-┐     │   │
│  │  │ Slurm Compute│      │ NEW: Slurm Compute│     │   │
│  │  │ Pod 2        │      │ Pod 3             │     │   │
│  │  │ (slurmd)     │      │ (slurmd)          │     │   │
│  │  └──────────────┘      └──────────────────-┘     │   │
│  │                                                  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Network
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌──────────────┐                    ┌──────────────┐
│ External     │                    │ External     │
│ Physical     │                    │ Virtual      │
│ Node         │                    │ Node (VM)    │
│              │                    │              │
│ - slurmd     │                    │ - slurmd     │
│ - Native     │                    │ - Native     │
└──────────────┘                    └──────────────┘
```

---

## Prerequisites

### Verify Existing Cluster

```bash
# Check controller and compute nodes
oc get pods -n slurm -l app.kubernetes.io/name=slurmctld
oc get pods -n slurm -l app.kubernetes.io/name=slurmd

# Check cluster status
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo

# Get controller service (pattern: <controller-name>-controller)
CONTROLLER_NAME=$(oc get controller -n slurm -o jsonpath='{.items[0].metadata.name}')
CONTROLLER_SVC="${CONTROLLER_NAME}-controller"
CONTROLLER_IP=$(oc get svc -n slurm $CONTROLLER_SVC -o jsonpath='{.spec.clusterIP}')
echo "Service: $CONTROLLER_SVC, IP: $CONTROLLER_IP"
```

### Required Information

**Use the provided scripts to get this information:**

```bash
# 1. Controller address and port
./scripts/get-controller-address.sh

# 2. See which worker nodes host Slurm pods
./scripts/show-slurm-placement.sh

# 3. Munge key (extract from controller pod - see Step 3 for details)
# Note: Munge key extraction has permission issues - see Step 3 for solutions

# 4. Slurm configuration
oc get configmap slurm-config -n slurm -o jsonpath='{.data.slurm\.conf}'

# 5. Standard network ports:
#    - 6817: slurmctld (controller)
#    - 6818: slurmd (compute nodes)
#    - 28596: munge (authentication)
```

---

## Scenario 1: Adding OpenShift Worker Nodes

**Simplest method**: Scale existing NodeSets to use new OpenShift worker nodes.

### Steps

```bash
# 1. Add worker nodes to OpenShift (via OpenShift admin process)
oc get nodes

# 2. Scale NodeSet UP (add more compute nodes)
oc scale nodeset slurm-compute --replicas=4 -n slurm
# Or: oc edit nodeset slurm-compute -n slurm  # Change spec.replicas

# 3. Verify new pods
oc get pods -n slurm -l app.kubernetes.io/name=slurmd -w
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo -N -l

# 4. Test job
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- \
  sbatch --wrap="hostname && echo 'Running on new node'" --output=/tmp/test.out

# 5. Scale NodeSet DOWN (remove compute nodes)
oc scale nodeset slurm-compute --replicas=2 -n slurm
# Or: oc edit nodeset slurm-compute -n slurm  # Change spec.replicas to desired number

# Verify pods are being terminated
oc get pods -n slurm -l app.kubernetes.io/name=slurmd -w
```

### Optional: Node Selectors

```yaml
# Update NodeSet to target specific nodes
apiVersion: slinky.slurm.net/v1beta1
kind: NodeSet
metadata:
  name: slurm-compute
  namespace: slurm
spec:
  template:
    spec:
      nodeSelector:
        slurm-compute: "true"
        kubernetes.io/os: linux
```

---

## Scenario 2: Adding External Physical Nodes

**Add bare metal servers** (outside OpenShift) running native Slurm to the cluster.


### Step 1: Prepare Node

**Requirements**: Linux (RHEL 8/9, Ubuntu 20.04+), network access, root/sudo

```bash
# Install Slurm
sudo dnf install -y epel-release slurm-slurmd munge munge-libs  # RHEL
# or
sudo apt-get update && sudo apt-get install -y slurmd munge  # Ubuntu
```

### Step 2: Network & Firewall

**Important**: ClusterIP services are only accessible from within the OpenShift cluster network. External physical nodes need either a **LoadBalancer service** (provides external IP) or **port-forwarding** (temporary workaround).

**Standard Slurm Ports:**
- **6817**: slurmctld (controller)
- **6818**: slurmd (compute nodes)
- **28596**: munge (authentication)

**Option 1: LoadBalancer Service (Recommended - Production)**

LoadBalancer provides an external IP that works from any network location.

**Step 1: Check LoadBalancer Support**
```bash
./scripts/check-loadbalancer-support.sh
```

**Step 2: Create LoadBalancer Service**
```bash
./scripts/create-loadbalancer-service.sh
```

**Step 3: Get Controller Address**
```bash
./scripts/get-controller-address.sh
```

**Step 4: Test Connectivity (from physical node)**
```bash
# Test port connectivity
nc -zv <controller-address> 6817
# Or: timeout 3 bash -c "</dev/tcp/<controller-address>/6817" && echo "✓ Connected" || echo "✗ Failed"

# Configure firewall
sudo firewall-cmd --permanent --add-port={6817,6818,28596}/tcp && sudo firewall-cmd --reload
```

**Troubleshooting LoadBalancer:**
- **If LoadBalancer IP is pending**: Check for errors: `oc describe svc -n slurm <service-name>`
- **Common issue**: AWS security group configuration (requires cluster admin to fix)
  - Error: "Multiple tagged security groups found"
  - Solution: Contact cluster administrator to fix AWS security group tags

**Option 2: Port Forwarding (Temporary Workaround)**

Use this if LoadBalancer is not available or not working.

**Step 1: Start Port Forward (on your Mac/terminal with oc access)**
```bash
./scripts/port-forward-controller.sh
# Or manually:
oc port-forward --address 0.0.0.0 -n slurm svc/slurm-controller 6817:6817
```

**Step 2: Get Your Mac's IP Address**
```bash
./scripts/test-mac-connectivity.sh
# Or manually:
ipconfig getifaddr en0  # macOS Wi-Fi
```

**Step 3: Test Connectivity (from physical node)**
```bash
# Test connection to your Mac
nc -zv <your-mac-ip> 6817
```

**Step 4: Configure Physical Node**
Configure Slurm on physical node to connect to your Mac's IP:
```
ControlMachine=<your-mac-ip>
ControlAddr=<your-mac-ip>
SlurmctldPort=6817
```

**Important Notes:**
- Port-forward only works while the command is running (keep terminal open)
- Not suitable for production use
- Requires physical node to be able to reach your Mac

### Step 3: Munge Authentication

**Important**: The munge key **IS installed and working** in your Slurm cluster. The issue is only that we can't read it due to file permissions (the file is owned by `munge` user with 600 permissions, but the container runs as `slurm` user).

**Extract munge key:**

```bash
# Method 1: Try oc cp (simplest)
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
oc cp slurm/$CONTROLLER_POD:/etc/munge/munge.key /tmp/munge.key -c slurmctld

# Verify
if [ -f /tmp/munge.key ] && [ -s /tmp/munge.key ]; then
  echo "✓ Success! Munge key extracted to /tmp/munge.key"
else
  echo "✗ oc cp failed - you may need cluster admin privileges"
fi
```

**On Physical Node:**

```bash
# Copy the key to physical node (use scp or manual copy)
# Then on physical node:
sudo cp /path/to/munge.key /etc/munge/munge.key
sudo chown munge:munge /etc/munge/munge.key && sudo chmod 600 /etc/munge/munge.key
sudo systemctl enable --now munge
munge -n | unmunge  # Test
```

### Step 4: Configure Slurm

**On Controller (OpenShift):**
```bash
# Get current config
oc get configmap slurm-config -n slurm -o yaml > /tmp/slurm-config.yaml

# Edit to add node: NodeName=physical-node1 NodeAddr=<ip> CPUs=16 RealMemory=32768 State=UNKNOWN
# Update partition: PartitionName=all Nodes=ALL,physical-node1 Default=YES MaxTime=UNLIMITED State=UP

# Apply and reload
oc apply -f /tmp/slurm-config.yaml
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol reconfigure
```

**On Physical Node:**
```bash
# Get controller info
CONTROLLER_NAME=$(oc get controller -n slurm -o jsonpath='{.items[0].metadata.name}')
CONTROLLER_IP=$(oc get svc -n slurm ${CONTROLLER_NAME}-controller -o jsonpath='{.spec.clusterIP}')

# Edit /etc/slurm/slurm.conf
sudo vi /etc/slurm/slurm.conf
# Add:
# NodeName=physical-node1 NodeAddr=<node-ip> CPUs=16 RealMemory=32768 State=UNKNOWN
# ControlMachine=${CONTROLLER_NAME}.slurm.svc.cluster.local
# ControlAddr=${CONTROLLER_IP}
# SlurmctldPort=6817
# SlurmdPort=6818
```

### Step 5: Start & Verify

```bash
# Start slurmd
sudo mkdir -p /var/spool/slurm/slurmd /var/log/slurm
sudo chown -R slurm:slurm /var/spool/slurm /var/log/slurm
sudo systemctl enable --now slurmd
sudo journalctl -u slurmd -f  # Check logs

# Verify from controller
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo -N -l
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show node physical-node1

# Test job
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- \
  sbatch --nodelist=physical-node1 --wrap="hostname && lscpu" --output=/tmp/test.out
```

---

## Scenario 3: Adding External Virtual Nodes

**Similar to Scenario 2** but for VMs. Follow Scenario 2 steps with these considerations:

### VM-Specific Steps

```bash
# 1. Provision VM (minimum: 2 CPU, 4GB RAM, 20GB disk; recommended: 4+ CPU, 8GB+ RAM)
# Examples:
# AWS: aws ec2 run-instances --instance-type t3.medium ...
# OpenStack: openstack server create --flavor m1.medium ...

# 2. Configure network (ensure VM can reach OpenShift cluster)
# 3. Install Slurm & Munge (same as physical nodes)
# 4. Configure Slurm (same as physical nodes, adjust CPU/memory for VM specs)
# 5. Start & verify (same as physical nodes)
```

### VM Considerations

- **Resource Overcommit**: VMs may have overcommitted resources
- **Performance**: May be lower than physical nodes
- **Snapshots**: Avoid snapshots while jobs are running
- **Migration**: Live migration may affect running jobs

---

## Scenario 4: Scaling Containerized Nodes

Add more Slurm compute pods within OpenShift.

### Method 1: Scale Existing NodeSet

```bash
oc scale nodeset slurm-compute --replicas=6 -n slurm
# Or: oc edit nodeset slurm-compute -n slurm  # Change spec.replicas
```

### Method 2: Create Additional NodeSet

For different node types (e.g., GPU nodes):

```yaml
apiVersion: slinky.slurm.net/v1beta1
kind: NodeSet
metadata:
  name: slurm-compute-gpu
  namespace: slurm
spec:
  controllerRef:
    name: <controller-name>  # Get: oc get controller -n slurm -o jsonpath='{.items[0].metadata.name}'
    namespace: slurm
  replicas: 2
  partition:
    enabled: true
    name: gpu-partition
  slurmd:
    image: 'ghcr.io/slinkyproject/slurmd:25.11-ubuntu24.04'
    resources:
      requests:
        cpu: "4"
        memory: "8Gi"
        nvidia.com/gpu: 1
      limits:
        cpu: "8"
        memory: "16Gi"
        nvidia.com/gpu: 1
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-tesla-v100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

---

## Configuration Management

### Updating slurm.conf

```bash
# 1. Get current config
oc get configmap slurm-config -n slurm -o jsonpath='{.data.slurm\.conf}' > /tmp/slurm.conf

# 2. Edit to add nodes
vi /tmp/slurm.conf
# Add: NodeName=node1 NodeAddr=<ip> CPUs=16 RealMemory=32768 State=UNKNOWN
# Update partition: PartitionName=all Nodes=ALL,node1 Default=YES MaxTime=UNLIMITED State=UP

# 3. Update ConfigMap
oc create configmap slurm-config --from-file=slurm.conf=/tmp/slurm.conf \
  --dry-run=client -o yaml | oc apply -f - -n slurm

# 4. Reload
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol reconfigure
```

### Automation Script

```bash
#!/bin/bash
# add-node-to-slurm.sh <node-name> <node-addr> <cpus> <memory-mb> [namespace]

NODE_NAME=$1 NODE_ADDR=$2 CPUS=$3 MEMORY=$4 NAMESPACE=${5:-slurm}

[ -z "$NODE_NAME" ] && { echo "Usage: $0 <node-name> <node-addr> <cpus> <memory-mb> [namespace]"; exit 1; }

oc get configmap slurm-config -n $NAMESPACE -o jsonpath='{.data.slurm\.conf}' > /tmp/slurm.conf
echo "NodeName=$NODE_NAME NodeAddr=$NODE_ADDR CPUs=$CPUS RealMemory=$MEMORY State=UNKNOWN" >> /tmp/slurm.conf
sed -i "s/PartitionName=all Nodes=ALL/PartitionName=all Nodes=ALL,$NODE_NAME/" /tmp/slurm.conf

oc create configmap slurm-config --from-file=slurm.conf=/tmp/slurm.conf \
  --dry-run=client -o yaml | oc apply -f - -n $NAMESPACE

CONTROLLER_POD=$(oc get pods -n $NAMESPACE -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
oc exec -n $NAMESPACE $CONTROLLER_POD -c slurmctld -- scontrol reconfigure
echo "Node $NODE_NAME added successfully"
```

---

## Network & Authentication

### Network Configuration

**Required Ports**: 6817 (controller), 6818 (compute), 28596 (munge)

**Network Policies** (if using OpenShift NetworkPolicies):
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-slurm-external
  namespace: slurm
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: slurmctld
  policyTypes: [Ingress]
  ingress:
  - from:
    - ipBlock:
        cidr: 192.168.1.0/24  # External node network
    ports:
    - protocol: TCP
      port: 6817
    - protocol: TCP
      port: 6818
```

**Service Types for External Access:**

**Option 1: LoadBalancer Service** (Recommended - Works from any network)

**Use the provided script:**
```bash
# Check if LoadBalancer is supported
./scripts/check-loadbalancer-support.sh

# Create LoadBalancer service (provides external IP)
./scripts/create-loadbalancer-service.sh
```

**When to use LoadBalancer:**
- Physical node is on a different network/subnet
- Physical node cannot reach OpenShift worker node IPs
- You want a single external IP address
- Works from any network location

**Option 2: Port Forwarding** (Temporary Workaround)

**Use the provided script:**
```bash
# Start port forwarding (on Mac/terminal with oc access)
./scripts/port-forward-controller.sh

# Get your Mac's IP
./scripts/test-mac-connectivity.sh
```

**When to use Port Forwarding:**
- LoadBalancer is not available or not working
- Temporary development/testing
- Physical node can reach your Mac
- Not suitable for production (requires terminal to stay open)

**To see which worker nodes host Slurm pods:**
```bash
./scripts/show-slurm-placement.sh
```

**DNS Configuration** (Optional - for internal cluster DNS):
```bash
# Add to /etc/hosts on external nodes (if using ClusterIP internally)
# Note: External nodes typically use LoadBalancer or port-forwarding, not ClusterIP
CONTROLLER_NAME=$(oc get controller -n slurm -o jsonpath='{.items[0].metadata.name}')
EXTERNAL_IP=$(oc get svc -n slurm ${CONTROLLER_NAME}-controller-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
if [ -n "$EXTERNAL_IP" ]; then
  echo "$EXTERNAL_IP ${CONTROLLER_NAME}.slurm.svc.cluster.local" | sudo tee -a /etc/hosts
fi
```

## Resource Management

### Resource Allocation

**In slurm.conf**:
```ini
# Reserve resources for system/Kubernetes
DefMemPerNode=4096  # Reserve 4GB for system
MaxMemPerNode=32768  # Total 32GB
DefCpuPerNode=2  # Reserve 2 CPUs for system
```

**Resource Sharing** (nodes running both Slurm and Kubernetes):
- CPU: Use cgroups to partition
- Memory: Set limits in slurm.conf
- Storage: Ensure shared storage accessible
- Network: Both systems need access

### Partition Configuration

```ini
# Separate partitions by node type
PartitionName=container Nodes=slurm-compute-[0-5] Default=YES MaxTime=24:00:00 State=UP
PartitionName=physical Nodes=physical-node1,physical-node2 Default=NO MaxTime=INFINITE State=UP
PartitionName=virtual Nodes=vm-node1,vm-node2 Default=NO MaxTime=48:00:00 State=UP
```

**Job Submission**:
```bash
sbatch --partition=physical job.sh
sbatch --nodelist=physical-node1 job.sh
```

---

## Verification & Troubleshooting

### Verification

```bash
# 1. Verify node registration
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo -N -l
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show node <node-name>

# 2. Test connectivity (from external node)
telnet <controller-ip> 6817
munge -n | nc <controller-ip> 28596

# 3. Submit test jobs
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- \
  sbatch --wrap="hostname && echo 'Test'" --output=/tmp/test.out
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- squeue
```
