# Updated Deployment and Uninstallation Guide - Slurm on OpenShift

## Table of Contents

### Part 1: Deployment
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Method 1: Automated Deployment (Recommended)](#method-1-automated-deployment-recommended)
- [Method 2: Manual CLI Deployment](#method-2-manual-cli-deployment)
- [Method 3: OpenShift Web Console Deployment](#method-3-openshift-web-console-deployment)
- [Verification and Testing](#verification-and-testing)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)

### Part 2: Uninstallation
- [Uninstallation Overview](#uninstallation-overview)
- [Option 1: Remove Cluster Only (Keep Operator)](#option-1-remove-cluster-only-keep-operator)
- [Option 2: Complete Uninstallation (Remove Everything)](#option-2-complete-uninstallation-remove-everything)
- [Manual Uninstallation Steps](#manual-uninstallation-steps)
- [Troubleshooting Uninstallation](#troubleshooting-uninstallation)

---

# Part 1: Deployment

## Overview

This guide provides updated instructions for deploying Slurm on OpenShift using the Slinky operator. Three deployment methods are available:

1. **Automated Deployment** - Uses deployment script (fastest, recommended for most users)
2. **Manual CLI Deployment** - Step-by-step terminal commands (best for learning/troubleshooting)
3. **Web Console Deployment** - Browser-based UI method (best for visual learners)

### Architecture: Two Namespaces

| Namespace | Purpose | Components |
|-----------|---------|------------|
| **slinky** | Operator (management layer) | Slurm Operator CRDs, Operator pod |
| **slurm** | Workload (actual cluster) | Controller, NodeSets, compute pods, secrets, services |

Both namespaces are required. The operator in `slinky` watches and manages resources you create in `slurm`. Remove `slinky` and the operator stops; remove `slurm` and you have no Slurm cluster (only the operator).

---

## Prerequisites

Before starting deployment, ensure you have:

1. **OpenShift Cluster Access**
   ```bash
   oc whoami
   # Should show your username

   oc get nodes
   # Should list cluster nodes
   ```

2. **Required Tools**
   - `oc` CLI (OpenShift command-line tool)
   - `helm` v3.x (for Helm-based installation)
   - `openssl` (for generating auth keys)

3. **Cluster Resources**
   - Cluster admin privileges
   - Sufficient compute resources for Slurm pods
   - Storage available for persistent volumes (if persistence enabled)

4. **cert-manager** (Prerequisite)

   cert-manager is required for TLS certificate management. The Slurm operator requires TLS certificates for secure inter-component communication, and cert-manager automatically issues, renews, and manages these certificates. Without it, the operator cannot create required certificates and may fail to start.

   **Check if already installed:**
   ```bash
   oc get pods -n cert-manager
   # If you see cert-manager-xxx, cert-manager-cainjector-xxx, cert-manager-webhook-xxx
   # then cert-manager is already installed - SKIP to deployment!
   ```

   **If NOT installed, install via OperatorHub (recommended for OpenShift):**
   - Navigate to **Ecosystem** → **Software Catalog**
   - Search for "cert-manager Operator for Red Hat OpenShift"
   - Click **Install** → Select namespace `cert-manager` → Install

   **Or via Helm:**
   ```bash
   helm repo add jetstack https://charts.jetstack.io
   helm repo update
   helm install cert-manager jetstack/cert-manager \
     --namespace cert-manager \
     --create-namespace \
     --set installCRDs=true \
     --set 'crds.enabled=true' \
     --version v1.13.0 \
     --wait --timeout 5m

   # Verify
   oc get pods -n cert-manager
   ```

   **Expected Output:**
   ```
   NAME                                      READY   STATUS    RESTARTS   AGE
   cert-manager-xxx                          1/1     Running   0          2m
   cert-manager-cainjector-xxx               1/1     Running   0          2m
   cert-manager-webhook-xxx                  1/1     Running   0          2m
   ```

---

## Method 1: Automated Deployment (Recommended)

The deployment script automates the entire installation process.

### Quick Start

```bash
cd /path/to/slurm-on-ocp

# Full deployment (installs operator + cluster)
./scripts/deploy-slurm.sh

# Skip operator (if already installed via OperatorHub or previous run)
./scripts/deploy-slurm.sh --skip-operator
```

### Script Options

```bash
# Deploy with custom namespace
./scripts/deploy-slurm.sh --namespace my-slurm --skip-operator

# Deploy with custom values file
./scripts/deploy-slurm.sh --values-file configs/custom-values.yaml --skip-operator

# Deploy with specific image tags
./scripts/deploy-slurm.sh --controller-image ghcr.io/slinkyproject/slurmctld:25.11-ubuntu24.04 --skip-operator
```

### What the Script Does

1. Checks prerequisites (oc, helm, cert-manager)
2. Installs Slurm Operator CRDs (if not already installed)
3. Installs Slurm Operator (if not already installed)
4. Creates target namespace and configures security (anyuid SCC)
5. Generates and creates authentication secrets
6. Deploys Controller and NodeSet resources
7. Waits for pods to be ready
8. Runs basic verification tests

### Expected Output

```
✓ Prerequisites verified
✓ cert-manager running
✓ Slurm Operator CRDs installed
✓ Slurm Operator running
✓ Namespace 'slurm' created
✓ Secrets created
✓ Controller deployed
✓ NodeSet deployed
✓ All pods running
✓ Deployment successful!
```

After successful deployment, proceed to [Verification and Testing](#verification-and-testing).

---

## Method 2: Manual CLI Deployment

For step-by-step control, follow these manual CLI steps.

### Step 1: Install Slurm Operator CRDs

```bash
# Check if CRDs already exist
if oc get crd controllers.slinky.slurm.net &>/dev/null; then
  echo "CRDs already installed"
else
  helm install slurm-operator-crds \
    oci://ghcr.io/slinkyproject/charts/slurm-operator-crds \
    --namespace slinky \
    --create-namespace \
    --server-side=false

  # Wait for CRDs to register
  sleep 10
fi

# Verify CRDs
oc get crds | grep slurm
```

**Expected Output:**
```
controllers.slinky.slurm.net
nodesets.slinky.slurm.net
```

### Step 2: Install Slurm Operator

```bash
# Check if operator already exists
if oc get pods -n slinky -l app.kubernetes.io/name=slurm-operator 2>/dev/null | grep -q Running; then
  echo "Operator already installed"
else
  # --server-side=false avoids metadata.managedFields errors
  helm install slurm-operator \
    oci://ghcr.io/slinkyproject/charts/slurm-operator \
    --namespace slinky \
    --create-namespace \
    --server-side=false \
    --wait --timeout 5m
fi

# Verify operator is running
oc get pods -n slinky -l app.kubernetes.io/name=slurm-operator
```

**Expected Output:**
```
NAME                              READY   STATUS    RESTARTS   AGE
slurm-operator-xxx                1/1     Running   0          1m
```

### Step 3: Prepare Cluster Namespace

```bash
# Create namespace
oc create namespace slurm 2>/dev/null || true

# Grant anyuid SCC (required for Slurm to run with specific UID 401)
oc adm policy add-scc-to-user anyuid -z default -n slurm
```

### Step 4: Create Authentication Secrets

The Controller requires two secrets with specific names and keys. Uses `--dry-run=client` for idempotent creation (safe to re-run).

```bash
# Generate random keys
JWT_KEY=$(openssl rand -base64 32)
SLURM_KEY=$(openssl rand -base64 32)

# Create secrets (idempotent - safe to run multiple times)
oc create secret generic slurm-auth-jwths256 -n slurm \
  --from-literal=jwt_hs256.key="$JWT_KEY" \
  --dry-run=client -o yaml | oc apply -f -

oc create secret generic slurm-auth-slurm -n slurm \
  --from-literal=slurm.key="$SLURM_KEY" \
  --dry-run=client -o yaml | oc apply -f -

# Verify secrets
oc get secrets -n slurm | grep slurm-auth
```

### Step 5: Deploy Slurm Controller

```yaml
apiVersion: slinky.slurm.net/v1beta1
kind: Controller
metadata:
  name: slurm
  namespace: slurm
spec:
  jwtHs256KeyRef:
    name: slurm-auth-jwths256
    key: jwt_hs256.key
  slurmKeyRef:
    name: slurm-auth-slurm
    key: slurm.key
  slurmctld:
    image: 'ghcr.io/slinkyproject/slurmctld:25.11-ubuntu24.04'
    resources:
      requests:
        cpu: "2"
        memory: "4Gi"
      limits:
        cpu: "4"
        memory: "8Gi"
  persistence:
    enabled: true
    resources:
      requests:
        storage: 4Gi
    accessModes:
      - ReadWriteOnce
  reconfigure:
    image: 'ghcr.io/slinkyproject/slurmctld:25.11-ubuntu24.04'
    resources: {}
  logfile:
    image: 'docker.io/library/alpine:latest'
    resources: {}
  extraConf: |
    PartitionName=all Nodes=ALL Default=YES MaxTime=UNLIMITED State=UP
    MinJobAge=3600
```

**Apply via terminal:**
```bash
oc apply -f configs/slurm-cluster.yaml
```

Or apply inline:
```bash
oc apply -f - <<'EOF'
apiVersion: slinky.slurm.net/v1beta1
kind: Controller
metadata:
  name: slurm
  namespace: slurm
spec:
  jwtHs256KeyRef:
    name: slurm-auth-jwths256
    key: jwt_hs256.key
  slurmKeyRef:
    name: slurm-auth-slurm
    key: slurm.key
  slurmctld:
    image: 'ghcr.io/slinkyproject/slurmctld:25.11-ubuntu24.04'
    resources:
      requests:
        cpu: "2"
        memory: "4Gi"
      limits:
        cpu: "4"
        memory: "8Gi"
  persistence:
    enabled: true
    resources:
      requests:
        storage: 4Gi
    accessModes:
      - ReadWriteOnce
  reconfigure:
    image: 'ghcr.io/slinkyproject/slurmctld:25.11-ubuntu24.04'
    resources: {}
  logfile:
    image: 'docker.io/library/alpine:latest'
    resources: {}
  extraConf: |
    PartitionName=all Nodes=ALL Default=YES MaxTime=UNLIMITED State=UP
    MinJobAge=3600
EOF
```

**Note on `extraConf`:**
- `MinJobAge=3600` keeps completed jobs in Slurm's memory for 1 hour (default is 300s / 5 minutes), making it easier to check job history with `scontrol show job`.

### Step 6: Deploy NodeSet

```bash
oc apply -f - <<'EOF'
apiVersion: slinky.slurm.net/v1beta1
kind: NodeSet
metadata:
  name: slurm-worker-slinky
  namespace: slurm
spec:
  controllerRef:
    name: slurm
    namespace: slurm
  replicas: 2
  slurmd:
    image: 'ghcr.io/slinkyproject/slurmd:25.11-ubuntu24.04'
    resources:
      requests:
        cpu: "1"
        memory: "2Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
EOF
```

**Important:** The `image` tag for slurmd must match the slurmctld image version (both `25.11-ubuntu24.04`). Version mismatches cause "Insane message length" or connection errors.

### Step 7: Verify Deployment

```bash
# Check Controller and NodeSet resources
oc get controllers,nodesets -n slurm

# Check pods
oc get pods -n slurm

# Wait for all pods to be Running
oc get pods -n slurm -w
```

**Expected Pods:**
```
NAME                          READY   STATUS    RESTARTS   AGE
slurm-controller-0            1/1     Running   0          2m
slurm-worker-slinky-0         1/1     Running   0          2m
slurm-worker-slinky-1         1/1     Running   0          2m
```

Proceed to [Verification and Testing](#verification-and-testing).

---

## Method 3: OpenShift Web Console Deployment

For UI-based deployment, follow these steps in the OpenShift Web Console.

### Prerequisites via UI

1. **Access Web Console**
   - URL: `https://console-openshift-console.apps.<your-cluster-domain>`
   - Login with cluster admin credentials

2. **Verify cert-manager**
   - Go to **Ecosystem** → **Software Catalog**
   - Search for "cert-manager"
   - If not installed, click **Install** on "cert-manager Operator for Red Hat OpenShift"
   - Select namespace: `cert-manager` → **Install**

### Step 1: Install Slurm Operator via OperatorHub

1. **Navigate to OperatorHub**
   - **Ecosystem** → **Software Catalog**
   - Search for "slurm" or "slinky"

2. **Install Slurm Operator**
   - Click on "Slurm Operator"
   - Click **Install**
   - **Installation mode**: "A specific namespace on the cluster"
   - **Installed Namespace**: Create new namespace → Name: `slinky`
   - **Update channel**: Latest (e.g., "stable" or "alpha")
   - **Approval strategy**: "Automatic"
   - Click **Install**

3. **Verify Installation**
   - **Ecosystem** → **Installed Operators**
   - Filter by namespace: `slinky`
   - Verify "Slurm Operator" shows "Succeeded" status

### Step 2: Create Slurm Cluster Namespace

**Via Terminal (recommended):**
```bash
oc create namespace slurm
oc adm policy add-scc-to-user anyuid -z default -n slurm
```

**Via UI:**
- **Home** → **Projects** → **Create Project**
- Name: `slurm` → **Create**
- Then run SCC command via terminal: `oc adm policy add-scc-to-user anyuid -z default -n slurm`

### Step 3: Create Secrets

**Via Terminal (recommended):**
```bash
JWT_KEY=$(openssl rand -base64 32)
SLURM_KEY=$(openssl rand -base64 32)
oc create secret generic slurm-auth-jwths256 -n slurm \
  --from-literal=jwt_hs256.key="$JWT_KEY" \
  --dry-run=client -o yaml | oc apply -f -
oc create secret generic slurm-auth-slurm -n slurm \
  --from-literal=slurm.key="$SLURM_KEY" \
  --dry-run=client -o yaml | oc apply -f -
```

**Via UI:**
- **Workloads** → **Secrets** (namespace: `slurm`)
- **Create** → **Key/value secret**
- **First secret**: Name: `slurm-auth-jwths256`, Key: `jwt_hs256.key`, Value: (run `openssl rand -base64 32` to generate)
- **Second secret**: Name: `slurm-auth-slurm`, Key: `slurm.key`, Value: (run `openssl rand -base64 32` to generate)

### Step 4: Create Controller

Do NOT use "Create Deployment" page — it only accepts Deployment resources. Use the Operator UI's "Create Controller" option instead.

1. **Navigate to Operator**
   - **Ecosystem** → **Installed Operators** → **Slurm Operator**

2. **Create Controller**
   - Click **Controller** tab
   - Click **Create Controller**
   - Select **YAML view**

3. **Paste YAML:**
   ```yaml
   apiVersion: slinky.slurm.net/v1beta1
   kind: Controller
   metadata:
     name: slurm
     namespace: slurm
   spec:
     jwtHs256KeyRef:
       name: slurm-auth-jwths256
       key: jwt_hs256.key
     slurmKeyRef:
       name: slurm-auth-slurm
       key: slurm.key
     slurmctld:
       image: 'ghcr.io/slinkyproject/slurmctld:25.11-ubuntu24.04'
       resources:
         requests:
           cpu: "2"
           memory: "4Gi"
         limits:
           cpu: "4"
           memory: "8Gi"
     persistence:
       enabled: true
       resources:
         requests:
           storage: 4Gi
       accessModes:
         - ReadWriteOnce
     reconfigure:
       image: 'ghcr.io/slinkyproject/slurmctld:25.11-ubuntu24.04'
       resources: {}
     logfile:
       image: 'docker.io/library/alpine:latest'
       resources: {}
     extraConf: |
       PartitionName=all Nodes=ALL Default=YES MaxTime=UNLIMITED State=UP
       MinJobAge=3600
   ```

4. Click **Create**

**Alternative — Import YAML (if Operator UI doesn't show Create button):**
   - Go to any page with the **"+"** button in the top-right
   - Click **"+"** → **"Import YAML"** (NOT "Create Deployment")
   - Paste the YAML above → Click **Create**

### Step 5: Create NodeSet

1. **Navigate to Operator**
   - **Ecosystem** → **Installed Operators** → **Slurm Operator**

2. **Create NodeSet**
   - Click **NodeSet** tab
   - Click **Create NodeSet**
   - Select **YAML view**

3. **Paste YAML:**
   ```yaml
   apiVersion: slinky.slurm.net/v1beta1
   kind: NodeSet
   metadata:
     name: slurm-worker-slinky
     namespace: slurm
   spec:
     controllerRef:
       name: slurm
       namespace: slurm
     replicas: 2
     slurmd:
       image: 'ghcr.io/slinkyproject/slurmd:25.11-ubuntu24.04'
       resources:
         requests:
           cpu: "1"
           memory: "2Gi"
         limits:
           cpu: "2"
           memory: "4Gi"
   ```

4. Click **Create**

### Step 6: Monitor Deployment

**Via UI:**
- **Workloads** → **Pods** (namespace: `slurm`)
- Wait for all pods to show "Running" status (1-3 minutes)

**Via Terminal:**
```bash
oc get pods -n slurm -w
```

Proceed to [Verification and Testing](#verification-and-testing).

---

## Verification and Testing

After deployment via any method, verify the cluster is working correctly.

### Basic Verification

```bash
# Set controller pod variable (do this once per terminal session)
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
echo $CONTROLLER_POD
# Should show: slurm-controller-0

# Check cluster info
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo

# Check nodes
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show nodes
```

**Expected Output:**
```
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
all*         up   infinite      2   idle slinky-0,slinky-1
```

**Important:** The `-c slurmctld` flag is required — it specifies which container in the pod to execute in. Slurm commands are NOT installed on your local machine; they must run inside the pod.

### Test 1: Basic Job Submission

```bash
# Submit a test job
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch \
  --output=/tmp/test.out \
  --wrap="echo 'Hello from Slurm' && hostname && date"

# Output: Submitted batch job <JOB_ID>

# Check job queue
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- squeue

# Check job status (replace <JOB_ID>)
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job <JOB_ID>

# Find which node ran the job
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job <JOB_ID> | grep NodeList

# View job output (try both nodes)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/test.out 2>/dev/null || \
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- cat /tmp/test.out
```

**Expected Output:**
```
Hello from Slurm
slinky-0
Wed Jun 10 20:00:00 UTC 2026
```

### Test 2: Multiple Jobs and Queue Monitoring

```bash
# Submit 3 jobs
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch --output=/tmp/job1.out --wrap="sleep 5 && echo 'Job 1 done'"
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch --output=/tmp/job2.out --wrap="sleep 5 && echo 'Job 2 done'"
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch --output=/tmp/job3.out --wrap="sleep 5 && echo 'Job 3 done'"

# Monitor queue
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- squeue

# Check all job statuses
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job
```

### Test 3: Job with Resource Requirements

```bash
# Submit job requesting 1 CPU and 1GB memory
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch \
  --cpus-per-task=1 \
  --mem=1G \
  --output=/tmp/resource-job.out \
  --wrap="echo 'CPU: ' && nproc && echo 'Memory: ' && free -h && echo 'Resource job completed'"

# Check resource allocation
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job <JOB_ID> | grep -E "(ReqTRES|AllocTRES|NumCPUs|NodeList)"
```

### Test 4: Long-Running Job

```bash
# Submit a 30-second job
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch \
  --output=/tmp/long-job.out \
  --wrap='for i in $(seq 1 30); do echo "Iteration $i at $(date)"; sleep 1; done'

# Monitor while running
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- squeue

# View output after completion
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/long-job.out 2>/dev/null || \
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- cat /tmp/long-job.out
```

### Test 5: View All Output Files

```bash
# List all output files on both compute nodes
echo "=== Files on compute-0 ==="
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- ls -la /tmp/*.out 2>/dev/null || echo "No files"

echo "=== Files on compute-1 ==="
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- ls -la /tmp/*.out 2>/dev/null || echo "No files"

# View all outputs
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- sh -c 'for f in /tmp/*.out; do [ -f "$f" ] && echo "=== $f ===" && cat "$f" && echo ""; done' 2>/dev/null
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- sh -c 'for f in /tmp/*.out; do [ -f "$f" ] && echo "=== $f ===" && cat "$f" && echo ""; done' 2>/dev/null
```

### Test 6: Verify Job Retention (MinJobAge)

```bash
# Submit a quick job
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch --output=/tmp/retention-test.out --wrap="echo 'Testing job retention' && date"

# Wait for completion
sleep 5

# View all jobs in memory (retained for up to 1 hour with MinJobAge=3600)
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job

# Filter key information
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job | grep -E "(JobId|JobState|ExitCode|NodeList|StdOut|RunTime)"

# Count completed jobs
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job | grep "JobState=COMPLETED" | wc -l
```

### Test 7: Job Distribution Check

```bash
# See which nodes jobs ran on
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job | grep "NodeList=" | sort | uniq -c
```

From the output you can verify:
- Jobs are distributed across both compute nodes (`slinky-0` and `slinky-1`)
- All show `JobState=COMPLETED` with `ExitCode=0:0` (success)
- Different resource allocations per job

### Comprehensive Test Script

```bash
./scripts/test-slurm.sh --comprehensive --namespace slurm
```

---

## Monitoring and Observability

### Check Operator Logs

```bash
# Operator in slinky namespace (Helm install)
oc logs -n slinky -l app.kubernetes.io/name=slurm-operator --tail=100

# Operator in openshift-operators namespace (OperatorHub install)
oc logs -n openshift-operators -l app.kubernetes.io/name=slurm-operator --tail=100
```

### Check Controller Logs

```bash
oc logs -n slurm slurm-controller-0 -c slurmctld --tail=100
```

### Check Compute Node Logs

```bash
oc logs -n slurm slurm-worker-slinky-0 -c slurmd --tail=100
oc logs -n slurm slurm-worker-slinky-1 -c slurmd --tail=100
```

### Monitor Pod Status

```bash
# Watch pods in real-time
oc get pods -n slurm -w

# Check pod resource usage
oc top pods -n slurm

# Check events (sorted by time)
oc get events -n slurm --sort-by='.lastTimestamp'
```

### Monitoring via Web Console

1. **View Pod Logs**: Workloads → Pods → Click pod → Logs tab
2. **View Events**: Observe → Events → Filter by namespace: `slurm`
3. **View Metrics**: Observe → Metrics → Select namespace: `slurm`

---

## Troubleshooting

### "container not found (slurmctld)"

The controller pod may still be initializing (Init:0/2 or 0/1).

```bash
# Wait until pod is Running and Ready
oc get pods -n slurm -l app.kubernetes.io/name=slurmctld
# Must show 1/1 (or 3/3) Running before exec works
```

### "container not found (slurmd)"

The slurmd container may not be running yet.

```bash
# Check pod status
oc get pods -n slurm -l app.kubernetes.io/name=slurmd

# List container names in the pod
oc get pod slurm-worker-slinky-0 -n slurm -o jsonpath='{.spec.containers[*].name}'
# Use the actual container name with -c flag
```

### "Unable to contact slurm controller" / "Insane message length"

Usually a **version or protocol mismatch** between slurmctld and slurmd, or **auth key mismatch**.

**Fix:**
```bash
# 1. Verify images match
oc get controller slurm -n slurm -o jsonpath='{.spec.slurmctld.image}'
oc get nodeset slurm-worker-slinky -n slurm -o jsonpath='{.spec.slurmd.image}'
# Both should use the same version tag (e.g., 25.11-ubuntu24.04)

# 2. If NodeSet has no image set, add it explicitly
oc patch nodeset slurm-worker-slinky -n slurm --type=merge \
  -p '{"spec":{"slurmd":{"image":"ghcr.io/slinkyproject/slurmd:25.11-ubuntu24.04"}}}'

# 3. Clean restart - delete workers, let controller stabilize, workers recreate
oc delete pod -n slurm -l app.kubernetes.io/name=slurmd
# Wait for controller to show Ready, then check pods
oc get pods -n slurm -w
```

### Slurmd "connect failure" to controller

Compute nodes reach the controller at `slurm-controller.slurm:6817`.

```bash
# 1. Verify controller is Running and Ready
oc get pods -n slurm -l app.kubernetes.io/name=slurmctld

# 2. Check service exists with port 6817
oc get svc -n slurm

# 3. Test DNS from compute pod
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- getent hosts slurm-controller.slurm

# 4. Restart compute pods (they retry connection on startup)
oc delete pod -n slurm -l app.kubernetes.io/name=slurmd
```

### SSSD "No domains configured" / exit status 4

Expected when not using LDAP/AD. Slurmd and sshd still run; SSSD can be ignored for basic job execution. No change needed for POC.

### Image pull error (manifest unknown)

Override the image to one that exists:

```bash
# Check available tags at ghcr.io/slinkyproject
# Then update:
oc edit controller slurm -n slurm
# Change spec.slurmctld.image to a valid tag

oc edit nodeset slurm-worker-slinky -n slurm
# Change spec.slurmd.image to matching valid tag
```

### Pods not starting at all

```bash
# Check events for errors
oc get events -n slurm --sort-by='.lastTimestamp'

# Check operator logs
oc logs -n slinky -l app.kubernetes.io/name=slurm-operator --tail=50

# Verify anyuid SCC is applied
oc adm policy add-scc-to-user anyuid -z default -n slurm

# Check if secrets exist
oc get secret slurm-auth-jwths256 slurm-auth-slurm -n slurm
```

### "no matches for kind Controller"

The CRD may not support the API version you're using.

```bash
# Check supported versions
oc get crd controllers.slinky.slurm.net -o jsonpath='{.spec.versions[*].name}'
# Ensure your YAML uses apiVersion: slinky.slurm.net/v1beta1
```

---

# Part 2: Uninstallation

## Uninstallation Overview

There are two levels of uninstallation:

1. **Cluster Only** - Removes Slurm cluster (keeps operator for redeployment)
2. **Complete** - Removes everything (cluster + operator + CRDs)

Choose based on your needs:
- **Cluster only**: When you want to clean up but may redeploy soon
- **Complete**: When completely removing Slurm from the cluster

---

## Option 1: Remove Cluster Only (Keep Operator)

This removes the Slurm cluster from the target namespace but keeps the operator installed for future deployments.

### Using Cleanup Script (Recommended)

```bash
# Remove cluster in default namespace (slurm)
./scripts/cleanup-slurm.sh

# Remove cluster in custom namespace
./scripts/cleanup-slurm.sh my-slurm
```

### Manual Cluster Removal

```bash
NAMESPACE="slurm"

# 1. Delete Controller and NodeSet first (prevents operator from recreating)
oc delete controller --all -n $NAMESPACE --timeout=60s
oc delete nodeset --all -n $NAMESPACE --timeout=60s

# Wait for operator to clean up
sleep 10

# 2. Delete workloads
oc delete statefulset,deployment,replicaset,pod,job --all -n $NAMESPACE --timeout=60s

# 3. Delete services, configs, secrets
oc delete svc,configmap,secret,pvc --all -n $NAMESPACE --timeout=60s

# 4. Remove SCC from service account
oc adm policy remove-scc-from-user anyuid -z default -n $NAMESPACE

# 5. Delete namespace
oc delete namespace $NAMESPACE --timeout=120s
```

### Verification

```bash
# Check namespace is gone
oc get namespace slurm
# Should show: Error from server (NotFound)

# Verify operator still exists
oc get pods -n slinky -l app.kubernetes.io/name=slurm-operator
# Should show running operator pod
```

### Redeploy After Cluster-Only Removal

Since the operator is still installed, you can quickly redeploy:

```bash
./scripts/deploy-slurm.sh --skip-operator
```

---

## Option 2: Complete Uninstallation (Remove Everything)

This removes the entire Slurm installation including operator, CRDs, and all related resources.

### Using Cleanup Script (Recommended)

```bash
# Complete uninstallation
./scripts/cleanup-slurm.sh --remove-operator

# Or with custom namespace
./scripts/cleanup-slurm.sh my-slurm --remove-operator
```

### What Gets Removed

1. **Cluster namespace** (`slurm` or custom)
   - All Controller and NodeSet resources
   - All pods, services, configs, secrets, PVCs
   - SCC bindings
   - The namespace itself

2. **Operator namespace** (`slinky`)
   - Slurm Operator deployment
   - Operator pods and services
   - Helm releases (if installed via Helm)
   - The namespace itself

3. **OperatorHub resources** (if installed via OperatorHub)
   - Subscriptions
   - ClusterServiceVersions (CSVs)
   - Operator pods in `openshift-operators`

4. **CRDs** (Custom Resource Definitions)
   - `controllers.slinky.slurm.net`
   - `nodesets.slinky.slurm.net`
   - `loginsets.slinky.slurm.net`
   - `accountings.slinky.slurm.net`
   - `restapis.slinky.slurm.net`
   - `tokens.slinky.slurm.net`
   - `slurmclusters.slurm.schedmd.com`
   - `slurmjobs.slurm.schedmd.com`

### Expected Output

```
🧹 Slurm cleanup: namespace=slurm, remove_operator=true

[INFO] Cleaning cluster namespace: slurm
[INFO] Deleting Controller and NodeSet (custom resources)...
[INFO] Deleting workloads...
[INFO] Deleting Services, ConfigMaps, Secrets, PVCs...
[INFO] Removing SCC from default service account...
[INFO] Deleting namespace slurm...
[INFO] Removing Slurm operator (OperatorHub and/or Helm)...
[INFO] Removing OperatorHub subscription and CSV...
[INFO] Removing Helm releases in slinky (if any)...
[INFO] Deleting namespace slinky...
[INFO] Deleting Slurm/Slinky CRDs...
[INFO] ✅ Operator removal completed (OperatorHub + slinky + CRDs)

📋 Summary:
   ✅ Cluster namespace slurm cleaned
   ✅ Operator and CRDs removed (full uninstall)

Redeploy cluster: ./scripts/deploy-slurm.sh
```

### Verification

```bash
# Verify cluster namespace is gone
oc get namespace slurm
# Should show: Error from server (NotFound)

# Verify operator namespace is gone
oc get namespace slinky
# Should show: Error from server (NotFound)

# Verify CRDs are removed
oc get crds | grep slurm
# Should show no results

# Verify operator pods are gone
oc get pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator
# Should show: No resources found
```

---

## Manual Uninstallation Steps

If you prefer to uninstall manually or the script encounters issues, follow these steps.

### Manual Complete Uninstallation

#### Step 1: Remove Cluster Namespace Resources

```bash
NAMESPACE="slurm"

# Delete custom resources first (prevents operator from recreating)
oc delete controller --all -n $NAMESPACE --ignore-not-found --timeout=60s
oc delete nodeset --all -n $NAMESPACE --ignore-not-found --timeout=60s

# Wait for operator to clean up
sleep 10

# Delete workloads
oc delete statefulset --all -n $NAMESPACE --ignore-not-found --timeout=60s
oc delete deployment --all -n $NAMESPACE --ignore-not-found --timeout=60s
oc delete replicaset --all -n $NAMESPACE --ignore-not-found --timeout=30s
oc delete pod --all -n $NAMESPACE --ignore-not-found --timeout=30s
oc delete job --all -n $NAMESPACE --ignore-not-found --timeout=30s

# Delete configs and storage
oc delete svc --all -n $NAMESPACE --ignore-not-found --timeout=30s
oc delete configmap --all -n $NAMESPACE --ignore-not-found --timeout=30s
oc delete secret --all -n $NAMESPACE --ignore-not-found --timeout=30s
oc delete pvc --all -n $NAMESPACE --ignore-not-found --timeout=60s

# Remove SCC binding
oc adm policy remove-scc-from-user anyuid -z default -n $NAMESPACE

# Delete namespace
oc delete namespace $NAMESPACE --timeout=120s
```

#### Step 2: Remove Operator (OperatorHub)

If installed via OperatorHub:

```bash
# Find and delete subscription
oc get subscription -n openshift-operators -o name | grep -i slurm | \
  xargs -I {} oc delete -n openshift-operators {} --timeout=30s

# Find and delete CSV (ClusterServiceVersion)
oc get csv -n openshift-operators -o name | grep -i slurm | \
  xargs -I {} oc delete -n openshift-operators {} --timeout=60s

# Delete operator pods
oc delete pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator --timeout=30s
```

#### Step 3: Remove Operator (Helm)

If installed via Helm:

```bash
# Uninstall Helm releases
helm uninstall slurm-operator -n slinky
helm uninstall slurm-operator-crds -n slinky

# Clean up any remaining resources
oc delete deployment,statefulset,replicaset,pod,service,configmap,secret --all -n slinky --timeout=60s

# Delete slinky namespace
oc delete namespace slinky --timeout=120s
```

#### Step 4: Remove CRDs

```bash
oc delete crd controllers.slinky.slurm.net --ignore-not-found --timeout=60s
oc delete crd nodesets.slinky.slurm.net --ignore-not-found --timeout=60s
oc delete crd loginsets.slinky.slurm.net --ignore-not-found --timeout=60s
oc delete crd accountings.slinky.slurm.net --ignore-not-found --timeout=60s
oc delete crd restapis.slinky.slurm.net --ignore-not-found --timeout=60s
oc delete crd tokens.slinky.slurm.net --ignore-not-found --timeout=60s
oc delete crd slurmclusters.slurm.schedmd.com --ignore-not-found --timeout=60s
oc delete crd slurmjobs.slurm.schedmd.com --ignore-not-found --timeout=60s
```

#### Step 5: Verify Complete Removal

```bash
# Check no Slurm namespaces remain
oc get namespace | grep -E '(slurm|slinky)'

# Check no CRDs remain
oc get crds | grep slurm

# Check no operator pods remain
oc get pods -A | grep slurm

# Check no PVCs remain
oc get pvc -A | grep slurm
```

---

## Troubleshooting Uninstallation

### Namespace Stuck in Terminating

```bash
# Check namespace status
oc get namespace slurm -o yaml

# Remove finalizers if stuck
oc patch namespace slurm -p '{"metadata":{"finalizers":[]}}' --type=merge

# Force delete (use with caution)
oc delete namespace slurm --force --grace-period=0
```

### Resources Not Deleting

```bash
# Check for finalizers on resources
oc get controller slurm -n slurm -o yaml | grep finalizers

# Remove finalizers if stuck
oc patch controller slurm -n slurm -p '{"metadata":{"finalizers":[]}}' --type=merge

# Then delete
oc delete controller slurm -n slurm
```

### PVCs Not Deleting

```bash
# Check PVC status
oc get pvc -n slurm

# Delete pods using the PVC first
oc delete pod <pod-name> -n slurm --force --grace-period=0

# Then delete PVC
oc delete pvc <pvc-name> -n slurm
```

### Operator Pods Not Terminating

```bash
# Force delete operator pod
oc delete pods -n slinky -l app.kubernetes.io/name=slurm-operator --force --grace-period=0
```

### CRDs Not Deleting

Usually due to existing resources using the CRD:

```bash
# Check for remaining resources
oc get controllers --all-namespaces
oc get nodesets --all-namespaces

# Delete all instances first
oc delete controllers --all --all-namespaces
oc delete nodesets --all --all-namespaces

# Wait, then try deleting CRD again
sleep 10
oc delete crd controllers.slinky.slurm.net
```

### OperatorHub Subscription Won't Delete

```bash
# Force remove subscription
oc patch subscription <subscription-name> -n openshift-operators \
  -p '{"metadata":{"finalizers":[]}}' --type=merge
oc delete subscription <subscription-name> -n openshift-operators --force --grace-period=0

# Force remove CSV
oc patch csv <csv-name> -n openshift-operators \
  -p '{"metadata":{"finalizers":[]}}' --type=merge
oc delete csv <csv-name> -n openshift-operators --force --grace-period=0
```

### Clean Slate Verification

```bash
echo "Checking namespaces..."
oc get namespace | grep -E '(slurm|slinky)' || echo "✓ No Slurm namespaces"

echo "Checking CRDs..."
oc get crds | grep slurm || echo "✓ No Slurm CRDs"

echo "Checking pods..."
oc get pods -A | grep slurm || echo "✓ No Slurm pods"

echo "Checking PVCs..."
oc get pvc -A | grep slurm || echo "✓ No Slurm PVCs"

echo "Checking subscriptions..."
oc get subscription -n openshift-operators | grep -i slurm || echo "✓ No Slurm subscriptions"

echo "Complete removal verified!"
```

---

## Redeployment After Uninstallation

### After Cluster-Only Removal

```bash
# Operator still installed — skip it
./scripts/deploy-slurm.sh --skip-operator
```

### After Complete Uninstallation

```bash
# Full deployment (installs everything)
./scripts/deploy-slurm.sh
```

---

## Quick Reference

### Deployment Commands

```bash
# Full automated deployment
./scripts/deploy-slurm.sh

# Skip operator (already installed)
./scripts/deploy-slurm.sh --skip-operator

# Custom namespace
./scripts/deploy-slurm.sh --namespace my-slurm
```

### Cleanup Commands

```bash
# Cluster only
./scripts/cleanup-slurm.sh

# Complete uninstallation
./scripts/cleanup-slurm.sh --remove-operator

# Custom namespace
./scripts/cleanup-slurm.sh my-slurm --remove-operator
```

### Verification Commands

```bash
# Check cluster resources
oc get controllers,nodesets -n slurm

# Check pods
oc get pods -n slurm

# Test cluster
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- squeue

# Check operator
oc get pods -n slinky

# Run test script
./scripts/test-slurm.sh
```

---

## References

- [Original Deployment Guide](DEPLOYMENT_GUIDE.md) - Legacy detailed deployment instructions
- [DDP Test Guide](DDP_TEST_GUIDE.md) - PyTorch Distributed Data Parallel testing
- [Slinky Project Documentation](https://slinky.schedmd.com/)
- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [OpenShift Documentation](https://docs.openshift.com/)
- [Helm Documentation](https://helm.sh/docs/)
