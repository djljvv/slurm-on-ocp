# Step-by-Step Deployment Guide - Slurm on OpenShift

## Overview

This guide provides **two methods** to deploy Slurm on OpenShift for the POC:
1. **Terminal/CLI Method** (using `oc` and `helm` commands) - Recommended for automation
2. **Browser UI Method** (using OpenShift Web Console) - Easier for beginners

### Why two namespaces? (slinky vs slurm)

You will see two projects/namespaces. **Both are needed;** they have different roles:

| Namespace | Purpose | What runs there |
|-----------|---------|------------------|
| **slinky** | Operator (the "manager") | Slurm Operator CRDs, Slurm Operator pod. The operator watches the cluster and reconciles Controller/NodeSet resources. |
| **slurm** | Workload (your cluster) | Controller CR, NodeSet CR, Slurm controller pod (slurmctld), compute pods (slurmd), secrets, services. This is the actual Slurm cluster you run jobs on. |

**In short:** The operator lives in **slinky** and manages resources you create in **slurm**. You need both: remove slinky and the operator stops; remove slurm and you have no Slurm cluster (only the operator).

---

## Method 1: Terminal/CLI Deployment (Recommended)

### Prerequisites Check

```bash
# 1. Verify you're logged in to OpenShift
oc whoami
# Should show your username (e.g., admin or kube:admin)

# 2. Verify cluster access
oc get nodes
# Should list your cluster nodes

# 3. Check if helm is installed
helm version
# Should show Helm v3.x
```

### Step 1: Verify cert-manager (Prerequisite)

cert-manager is required for TLS certificates used by the Slurm operator. It automates certificate management for secure communication.

**Why cert-manager is needed:**
- The Slurm operator requires TLS certificates for secure inter-component communication
- cert-manager automatically issues, renews, and manages these certificates
- Without it, the operator cannot create required certificates and may fail to start

**Check if cert-manager is already installed:**

```bash
# Check if cert-manager pods exist
oc get pods -n cert-manager

# If you see pods like cert-manager-xxx, cert-manager-cainjector-xxx, cert-manager-webhook-xxx
# then cert-manager is already installed - SKIP to Step 2!
```

**If cert-manager is NOT installed, install it:**

**Option A: Via OpenShift UI (Recommended for OpenShift)**
1. Go to "Ecosystem" → "Software Catalog"
2. Search for "cert-manager Operator for Red Hat OpenShift"
3. Click "Install" → Select namespace → Install

**Option B: Via Helm (if not using OpenShift operator)**
```bash
# Add Jetstack Helm repository
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Install cert-manager
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true \
  --set 'crds.enabled=true' \
  --version v1.13.0 \
  --wait --timeout 5m

# Verify installation
oc get pods -n cert-manager
# Wait until all pods show "Running" status
```

**Expected Output:**
```
NAME                                      READY   STATUS    RESTARTS   AGE
cert-manager-xxx                          1/1     Running   0          2m
cert-manager-cainjector-xxx               1/1     Running   0          2m
cert-manager-webhook-xxx                  1/1     Running   0          2m
```

### Step 2: Install Slurm Operator CRDs

```bash
# Check if CRDs are already installed (skip if already installed)
if oc get crd controllers.slinky.slurm.net &>/dev/null; then
  echo "CRDs already installed, skipping..."
else
  # Install Slurm Operator CRDs
  # NOTE: --version is pinned deliberately. Chart 1.2.0+ bumps the operator to
  # Slurm app version 26.05, whose NodeSet reconciler requests `privileged: true`
  # plus BPF/NET_ADMIN/SYS_ADMIN capabilities on worker pods — no OpenShift SCC
  # (including anyuid) allows that, so worker pods get silently rejected at
  # admission and never appear. 1.1.1 is the last chart release on app version
  # 25.11, matching the image tags already pinned in configs/slurm-cluster.yaml.
  helm install slurm-operator-crds \
    oci://ghcr.io/slinkyproject/charts/slurm-operator-crds \
    --version 1.1.1 \
    --namespace slinky \
    --create-namespace \
    --server-side=false

  # Wait a few seconds for CRDs to be registered
  sleep 10
fi

# Verify CRDs are installed
oc get crds | grep slurm
```

**Troubleshooting:** If you get `Error: unknown flag: --server-side`, your Helm version (< 3.12) doesn't support this flag — simply remove `--server-side=false` from the command.

**Expected Output (Helm installs Slinky CRDs):**
```
controllers.slinky.slurm.net
nodesets.slinky.slurm.net
```
*(If you installed via OperatorHub you may see different CRD names; use Option B in Step 4 to deploy the cluster.)*

### Step 3: Install Slurm Operator

```bash
# Check if operator is already installed AND running (must verify a Running pod exists)
if oc get pods -n slinky -l app.kubernetes.io/name=slurm-operator -o jsonpath='{.items[0].status.phase}' 2>/dev/null | grep -q Running || \
   oc get pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator -o jsonpath='{.items[0].status.phase}' 2>/dev/null | grep -q Running; then
  echo "Slurm Operator already installed and running, skipping..."
else
  # Install Slurm Operator (--version pinned to match the CRDs chart in Step 2)
  helm install slurm-operator \
    oci://ghcr.io/slinkyproject/charts/slurm-operator \
    --version 1.1.1 \
    --namespace slinky \
    --create-namespace \
    --server-side=false \
    --wait --timeout 5m
fi

# Verify operator is running (namespace depends on installation method)
# Try OperatorHub first (most common), then Helm namespace
oc get pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator 2>/dev/null || \
oc get pods -n slinky -l app.kubernetes.io/name=slurm-operator
```

**Expected Output:**
```
NAME                               READY   STATUS    RESTARTS   AGE
slurm-operator-xxx                  1/1     Running   0          1m
```

**Troubleshooting:**
- **Operator shows "already installed" but no pods are Running** — The previous detection check may have been a false positive. Verify with: `oc get pods -n slinky` and `oc get pods -n openshift-operators | grep slurm`. If no Running pods exist, install the operator manually (run the `helm install` command above).
- **`Error: unknown flag: --server-side`** — Your Helm version (< 3.12) doesn't support this flag. Remove `--server-side=false` from the command.
- **To uninstall the operator later** (do NOT run this during installation): `helm uninstall slurm-operator -n slinky`

### Step 4: Deploy Slurm Cluster

If operator is installed via Software Catalog/OperatorHub, use Option B (Direct YAML). The repo uses the v1beta1 API for Controller and NodeSet.

#### Option A: Using Deployment Script (RECOMMENDED)

```bash
# The script automatically detects OperatorHub installation and uses direct YAML
./scripts/deploy-slurm.sh --skip-operator
```

This will:
- Detect existing CRDs and operator (skip installation)
- Create the `slurm-workload` service account and grant it the `anyuid` SCC (bootstraps `default` too, then narrows to whichever SA the operator actually assigns to the controller) **and** the `privileged` SCC (required — see note below)
- Create the `slurm-worker-scripts` ConfigMap (mounts `demos/ddp_test.py` onto every worker pod) and `slurm-gres-conf` ConfigMap (static GPU device mapping so `slurmd` can report its GPU at dynamic registration time)
- Deploy cluster using direct YAML (works with OperatorHub) — each worker runs an init container that installs PyTorch **before** `slurmd` starts, so a node can never be scheduled onto until it's actually ready (see Step 5 troubleshooting for what this looks like)
- Deploy the autoscaler (scales workers up/down automatically based on Slurm queue demand — pending/running jobs that need more nodes trigger scale-up; idle workers scale back down after 5 minutes)
- Handle all setup automatically

#### Option B: Using Direct YAML (Works with OperatorHub)

**Note:** `configs/slurm-cluster.yaml` runs worker pods under the `slurm-workload` service account and mounts two ConfigMaps (`slurm-worker-scripts`, `slurm-gres-conf`). If you skip creating them, worker pods will fail to start (`serviceaccount "slurm-workload" not found` or `configmap "..." not found`). `./scripts/deploy-slurm.sh` (Option A) creates all of this for you automatically — the steps below just show what it does manually.

**Note on the `privileged` SCC:** this is a genuine upstream requirement, not an OpenShift-specific workaround. Slurm's cgroup/v2 plugin manages per-job GPU/device access via an eBPF program that `slurmstepd` loads into the kernel at runtime (see [Slurm's cgroup v2 docs](https://slurm.schedmd.com/cgroup_v2.html)), which needs the `BPF`/`NET_ADMIN`/`SYS_ADMIN` capabilities and cgroup write access that only the `privileged` SCC provides — `anyuid` alone explicitly forbids both. Without it, worker pods are rejected at admission (`unable to validate against any security context constraint`) and never get created, even though the `Controller`/`NodeSet` objects and `slurmctld` come up fine.

```bash
# 1. Create namespace, service account, and configure security
oc create namespace slurm 2>/dev/null || true
oc create serviceaccount slurm-workload -n slurm 2>/dev/null || true
oc adm policy add-scc-to-user anyuid -z slurm-workload -n slurm
oc adm policy add-scc-to-user anyuid -z default -n slurm
oc adm policy add-scc-to-user privileged -z slurm-workload -n slurm

# 2. Create secrets (operator default names/keys so UI template works without editing refs)
JWT_KEY=$(openssl rand -base64 32)
SLURM_KEY=$(openssl rand -base64 32)
oc create secret generic slurm-auth-jwths256 -n slurm \
  --from-literal=jwt_hs256.key="$JWT_KEY" \
  --dry-run=client -o yaml | oc apply -f -
oc create secret generic slurm-auth-slurm -n slurm \
  --from-literal=slurm.key="$SLURM_KEY" \
  --dry-run=client -o yaml | oc apply -f -

# 3. Create ConfigMaps required by configs/slurm-cluster.yaml
#    - slurm-worker-scripts: mounts demos/ddp_test.py onto every worker pod
#      (including ones the autoscaler creates later)
#    - slurm-gres-conf: static gres.conf GPU device mapping. Needed because the
#      slurmd image has no NVML plugin, so AutoDetect=nvml would always report 0 GPUs
oc create configmap slurm-worker-scripts -n slurm \
  --from-file=ddp_test.py=demos/ddp_test.py \
  --dry-run=client -o yaml | oc apply -f -
oc create configmap slurm-gres-conf -n slurm \
  --from-literal=gres.conf="NodeName=slinky-[0-7] Name=gpu File=/dev/nvidia0" \
  --dry-run=client -o yaml | oc apply -f -

# 4. Deploy using the config file (uses v1beta1 API)
oc apply -f configs/slurm-cluster.yaml
```

#### Option C: Using Helm Chart (Only if operator installed via Helm)

This only works if the operator was installed via Helm (not OperatorHub).

```bash
# Create namespace and secrets first (operator default names/keys)
oc create namespace slurm 2>/dev/null || true
JWT_KEY=$(openssl rand -base64 32)
SLURM_KEY=$(openssl rand -base64 32)
oc create secret generic slurm-auth-jwths256 -n slurm --from-literal=jwt_hs256.key="$JWT_KEY"
oc create secret generic slurm-auth-slurm -n slurm --from-literal=slurm.key="$SLURM_KEY"

# Deploy with default settings (--server-side=false avoids metadata.managedFields errors)
helm upgrade --install slurm \
  oci://ghcr.io/slinkyproject/charts/slurm \
  --namespace slurm \
  --create-namespace \
  --server-side=false \
  --wait --timeout 10m
```

**Note:** The Helm chart typically creates a Controller whose name matches the release (e.g. `slurm`). Use `oc get controllers -n slurm` to see the exact name; then e.g. `oc describe controller slurm -n slurm`.

**If using a custom values file (`slurm-values.yaml`):** Verify it does not set `runAsNonRoot: true` in the security context — `slurmd` worker pods require the `anyuid` SCC to run as the Slurm user (UID 401).

**If you get version mismatch or "no matches for kind Controller" errors:**
- Check the CRD supports v1beta1: `oc get crd controllers.slinky.slurm.net -o jsonpath='{.spec.versions[*].name}'`
- Ensure your YAML uses `apiVersion: slinky.slurm.net/v1beta1` for both Controller and NodeSet.
- Or use Option B (Direct YAML) or the deployment script (Option A).

### Step 5: Verify Deployment

```bash
# Check all pods (namespace: slurm by default, or your custom namespace)
oc get pods -n slurm

# Check Controller and NodeSet resources (not SlurmCluster - that's the old API)
oc get controllers,nodesets -n slurm
# Use the controller name from the list above (e.g. slurm)
oc describe controller <controller-name> -n slurm

# Check services
oc get svc -n slurm
```

**Expected Output:**
```
NAME                          READY   STATUS    RESTARTS   AGE
slurm-controller-0            3/3     Running   0          2m
slurm-worker-slinky-0         2/2     Running   0          2m
slurm-worker-slinky-1         2/2     Running   0          2m
```

> **Note on READY counts:** The controller pod shows `3/3` because it has sidecar containers (slurmctld, reconfigure, logfile). Worker pods show `2/2` (slurmd + logfile sidecar). If you see `1/1`, you may be running a minimal configuration without sidecars — this is also fine.

**Troubleshooting Step 5:**
- **Pods stuck in `Pending`** — Check if the `anyuid` SCC was applied: `oc get pods -n slurm -o wide` and `oc describe pod <pod-name> -n slurm | grep -A5 Events`. If you see SCC-related errors, apply it: `oc adm policy add-scc-to-user anyuid -z default -n slurm`. Worker pods run as the `slurm-workload` service account (see `spec.template.spec.serviceAccountName` in `configs/slurm-cluster.yaml`), so also grant it there: `oc adm policy add-scc-to-user anyuid -z slurm-workload -n slurm`.
- **No worker pods ever appear at all (`oc get pods -n slurm` shows only `slurm-controller-0`, `Controller`/`NodeSet` both exist and look healthy)** — This means the operator is failing to even create the Pod objects, which won't show up as `Pending` because they never got past admission. Confirm by checking the operator's own logs: `oc logs -n slinky -l app.kubernetes.io/name=slurm-operator --tail=200 | grep -i nodeset`. If you see `unable to validate against any security context constraint` mentioning `privileged` or `capabilities.add` (`BPF`/`NET_ADMIN`/`SYS_ADMIN`), `anyuid` isn't enough — Slurm's cgroup/v2 plugin needs the `privileged` SCC too (it loads an eBPF program for per-job device/GPU cgroup control; see [Slurm's cgroup v2 docs](https://slurm.schedmd.com/cgroup_v2.html)). Fix with: `oc adm policy add-scc-to-user privileged -z slurm-workload -n slurm`, then nudge the NodeSet to reconcile (see next bullet). `./scripts/deploy-slurm.sh` grants this automatically as of this fix.
- **Worker pods not appearing after SCC fix** — The operator may need a nudge to reconcile. Force it by annotating the NodeSet:
  ```bash
  oc annotate nodeset slurm-worker-slinky -n slurm reconcile=$(date +%s) --overwrite
  ```
- **Pods in `Init:0/2` or `Init:CrashLoopBackOff`** — Init containers may be waiting for dependencies. Check init container logs: `oc logs <pod-name> -n slurm -c <init-container-name> --previous`
- **Worker pods sitting in `Init:0/1` for 1-2 minutes** — This is expected, not a failure. Every worker pod runs a `provision-pytorch` init container that installs PyTorch via `pip3` into a per-pod `emptyDir` **before** `slurmd` starts — the node intentionally won't register with (and can't be scheduled onto by) the Slurm controller until this finishes. This runs fresh on every new pod (first deploy, or any time the autoscaler scales up new replicas) — it's not cached across pods. Watch progress with: `oc logs <worker-pod> -n slurm -c provision-pytorch -f`. Once it prints `PyTorch install complete`, the pod moves on to starting `slurmd`.
- **`configmap "slurm-worker-scripts" not found` or `configmap "slurm-gres-conf" not found`** — These ConfigMaps are required by `configs/slurm-cluster.yaml` but aren't created by `oc apply` itself. Run `./scripts/deploy-slurm.sh` (which creates them automatically), or create them manually as shown in Step 4, Option B.

### Step 6: Deploy the Autoscaler (Required for Auto Scale-Down)

The Controller and NodeSet you just deployed run Slurm itself, but nothing yet watches the job queue to scale the `NodeSet` up or down automatically. `./scripts/deploy-slurm.sh` (Option A in Step 4) deploys this for you as its last step — but if you followed Option B or C above (or are walking through everything by hand), it's **easy to miss entirely**, since it's a separate piece from the Controller/NodeSet you just verified. Symptom if you skip this: workers and jobs run fine, but nothing ever scales up under load or back down when idle.

**Recommended (via script):**
```bash
./scripts/deploy-autoscale.sh
```

**Manual (what the script above does under the hood):**
```bash
# 1. Bake the autoscaler loop script into a ConfigMap
oc create configmap slurm-autoscaler-script -n slurm \
  --from-file=autoscaler.sh=scripts/autoscaler-loop.sh \
  --dry-run=client -o yaml | oc apply -f -

# 2. Apply the autoscaler's Deployment + RBAC (ServiceAccount, Role, RoleBinding)
oc apply -f configs/slurm-autoscaler.yaml

# 3. Grant the privileged SCC to the autoscaler's ServiceAccount. This is
#    unrelated to the privileged SCC granted to slurm-workload above — the
#    autoscaler pod itself runs fully unprivileged (runAsNonRoot, all
#    capabilities dropped); it needs this SCC only because `oc exec`/`kubectl
#    exec` into slurmctld (to run squeue) requires the CALLER's SA to hold an
#    SCC that can validate the TARGET container's security context, and
#    slurmctld runs with elevated capabilities of its own.
oc adm policy add-scc-to-user privileged -z slurm-autoscaler -n slurm

# Verify it's running
oc get pods -n slurm -l app.kubernetes.io/name=slurm-autoscaler
```

**Verify it's actually working:**
```bash
oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler -f --tail=15
```
You should see it logging `IDLE`/`BUSY`/`SCALING` messages every ~30 seconds. If the log shows repeated `WARNING: failed to query pending jobs` instead, re-check step 3 above — the SCC grant is almost always the missing piece.

### Step 7: Test Slurm Cluster

```bash
# Get controller pod name
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')

# Check cluster info
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo

# Submit a test job
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch --output=/tmp/test.out --wrap="echo 'Hello from Slurm' && hostname && date"

# Check job queue
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- squeue

# Check job status (get job ID from squeue output)
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job <JOB_ID>

# Find which node ran the job
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job <JOB_ID> | grep NodeList

# View job output (on the compute node that ran it)
# If NodeList=slinky-0, use slurm-worker-slinky-0; if slinky-1, use slurm-worker-slinky-1
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/test.out 2>/dev/null || \
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- cat /tmp/test.out
```

**Troubleshooting Step 7:**
- **"container not found (slurmctld)"** — The controller pod may still be in Init (0/1 or Init:0/2). Wait until the pod is **Running** and **1/1** (or 3/3). Check with: `oc get pods -n slurm -l app.kubernetes.io/name=slurmctld`.
- **"container not found (slurmd)"** — The slurmd container may not be running (e.g. pod still initializing or slurmd crashing). Check: (1) Pod status: `oc get pods -n slurm -l app.kubernetes.io/name=slurmd` — ensure **Running** and **Ready**. (2) List container names: `oc get pod <slurmd-pod> -n slurm -o jsonpath='{.spec.containers[*].name}'` — use that name with `-c`. (3) If the controller is in CrashLoopBackOff, fix the controller first; worker pods often wait or fail until the controller is Ready.
- **"Unable to contact slurm controller"** or **"Insane message length"** (controller logs: `on_data returned rc: Insane message length`) — Usually a **version or protocol mismatch** between slurmctld and the connecting slurmd, or **auth key mismatch** (slurm.key). Fix: (1) **Match images**: set `spec.slurmctld.image` and `spec.slurmd.image` to the same tag (e.g. `ghcr.io/slinkyproject/slurmctld:25.11-ubuntu24.04` and `ghcr.io/slinkyproject/slurmd:25.11-ubuntu24.04`) in the Controller and NodeSet CRs. (2) **Verify what's running**: `oc get controller slurm -n slurm -o jsonpath='{.spec.slurmctld.image}'` and `oc get nodeset slurm-worker-slinky -n slurm -o jsonpath='{.spec.slurmd.image}'`; if the NodeSet has no image, the operator default may differ — add the image to the NodeSet and re-apply. (3) **Secrets**: Controller and workers must use the same `slurm.key` (from `slurm-auth-slurm`); the operator copies from the Controller's `slurmKeyRef` to workers — do not recreate the secret without redeploying. (4) **Clean restart**: delete worker pods so they stop connecting, let the controller become Ready, then workers will be recreated and reconnect: `oc delete pod -n slurm -l app.kubernetes.io/name=slurmd`; wait for controller to show Ready, then check `oc get pods -n slurm`.
- **Slurmd: "Unable to contact slurm controller (connect failure)"** — Compute nodes reach the controller at `slurm-controller.slurm:6817`. Check: (1) Controller pod is **Running** and **Ready** first. (2) Service exists: `oc get svc -n slurm` (look for `slurm-controller` or similar with port 6817). (3) From a compute pod: `oc exec -n slurm <slurmd-pod> -c slurmd -- getent hosts slurm-controller.slurm` and test port 6817. If the service name is different (e.g. `slurm-controller-controller`), the image may expect a different hostname; fix by creating a Service that matches what slurmd uses, or restart compute pods after the controller is Ready so they retry. (4) Restart compute pods once the controller is Ready: `oc delete pod -n slurm -l app.kubernetes.io/name=slurmd`.
- **SSSD "No domains configured" / "exited (exit status 4)"** — Expected when not using LDAP/AD. Slurmd and sshd still run; you can ignore SSSD for basic job execution. No change needed for POC.
- **Image pull error (e.g. manifest unknown)** — Override the image in the Controller or NodeSet to one that exists (e.g. `ghcr.io/slinkyproject/slurmctld:25.11-ubuntu24.04`). Edit with: `oc edit controller slurm -n slurm` and set `spec.slurmctld.image`.

---

## Method 2: Browser UI Deployment (OpenShift Web Console)

### Prerequisites

1. Access to OpenShift Web Console: `console-openshift-console.apps.<your-cluster-domain>`
2. Cluster admin privileges
3. Browser with access to the cluster

### Step 1: Verify cert-manager (Prerequisite)

**Why cert-manager is needed:**
- The Slurm operator requires TLS certificates for secure communication
- cert-manager automatically manages certificate lifecycle (issuance, renewal)
- Without it, the operator cannot function properly

**Check if already installed:**

1. **Via Terminal:**
   ```bash
   oc get pods -n cert-manager
   # If you see running pods, cert-manager is already installed - SKIP installation!
   ```

2. **Via UI:**
   - Go to "Ecosystem" → "Software Catalog"
   - Search for "cert-manager"
   - If you see "cert-manager Operator for Red Hat OpenShift" with green "Installed" badge, it's already installed

**If NOT installed, install via OperatorHub:**

1. **Navigate to Ecosystem → Software Catalog**

2. **Search for cert-manager**
   - In the search box, type: `cert-manager`
   - **Choose**: "cert-manager Operator for Red Hat OpenShift" (Red Hat certified version)

3. **Install cert-manager**
   - Click "Install" button
   - **Installation mode**: Select "A specific namespace on the cluster"
   - **Installed Namespace**: Select "Create new namespace" → Name: `cert-manager`
   - **Update channel**: Select latest (e.g., "stable")
   - **Approval strategy**: Select "Automatic" (or "Manual" if preferred)
   - Click "Install"
   - Wait for installation to complete (status shows "Succeeded")

4. **Verify Installation**
   - Go to "Ecosystem" → "Installed Operators"
   - Filter by namespace: `cert-manager`
   - Verify cert-manager shows "Succeeded" status
   - Or check pods: "Workloads" → "Pods" → Filter: `cert-manager`

### Step 2: Install Slurm Operator

#### Option A: Install via OperatorHub (Recommended - UI Method)

1. **Navigate to Ecosystem → Software Catalog**

2. **Search for Slurm Operator**
   - In the search box, type: `slinky` or `slurm`
   - You should see "Slurm Operator" (provided by Red Hat HPC Community)
   - Click on the "Slurm Operator" card

3. **Install Slurm Operator**
   - Click "Install" button
   - **Installation mode**: Select "A specific namespace on the cluster"
   - **Installed Namespace**: Select "Create new namespace" → Name: `slinky`
   - **Update channel**: Select latest available (e.g., "stable" or "alpha")
   - **Approval strategy**: Select "Automatic" (or "Manual" if preferred)
   - Click "Install"
   - Wait for installation to complete (status shows "Succeeded")

4. **Verify Installation**
   - Go to "Ecosystem" → "Installed Operators"
   - Filter by namespace: `slinky`
   - Verify "Slurm Operator" shows "Succeeded" status
   - Or check pods: "Workloads" → "Pods" → Filter: `slinky`
   - You should see `slurm-operator-xxx` pod in Running state

#### Option B: Install via Helm (Alternative - Terminal Method)

If you prefer using Helm or the OperatorHub installation doesn't work:

```bash
# Check if CRDs are already installed (skip if already installed)
if oc get crd controllers.slinky.slurm.net &>/dev/null; then
  echo "CRDs already installed, skipping..."
else
  # Install Slurm Operator CRDs
  # NOTE: --version is pinned deliberately. Chart 1.2.0+ bumps the operator to
  # Slurm app version 26.05, whose NodeSet reconciler requests `privileged: true`
  # plus BPF/NET_ADMIN/SYS_ADMIN capabilities on worker pods — no OpenShift SCC
  # (including anyuid) allows that, so worker pods get silently rejected at
  # admission and never appear. 1.1.1 is the last chart release on app version
  # 25.11, matching the image tags already pinned in configs/slurm-cluster.yaml.
  helm install slurm-operator-crds \
    oci://ghcr.io/slinkyproject/charts/slurm-operator-crds \
    --version 1.1.1 \
    --namespace slinky \
    --create-namespace \
    --server-side=false

  # Wait a few seconds for CRDs to be registered
  sleep 10
fi

# Verify CRDs are installed
oc get crds | grep slurm

# Check if operator is already installed AND running (must verify a Running pod exists)
if oc get pods -n slinky -l app.kubernetes.io/name=slurm-operator -o jsonpath='{.items[0].status.phase}' 2>/dev/null | grep -q Running || \
   oc get pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator -o jsonpath='{.items[0].status.phase}' 2>/dev/null | grep -q Running; then
  echo "Slurm Operator already installed and running, skipping..."
else
  # Install Slurm Operator (--version pinned to match the CRDs chart above)
  helm install slurm-operator \
    oci://ghcr.io/slinkyproject/charts/slurm-operator \
    --version 1.1.1 \
    --namespace slinky \
    --create-namespace \
    --server-side=false \
    --wait --timeout 5m
fi
```

**Verify in UI**:
   - Go to "Workloads" → "Pods"
   - Filter by namespace: `slinky`
   - Verify `slurm-operator` pod is Running

### Step 3: Deploy Slurm Cluster

Now that the operator is installed, you can deploy a Slurm cluster. Follow these steps **IN ORDER**:

Do NOT use "Create Deployment" page - it only accepts Deployment resources. Use the Operator UI's "Create Controller" and "Create NodeSet" options instead.

#### Step 3.1: Create Namespace and Configure Security (FIRST STEP)

**Via Terminal (Recommended):**
```bash
# Create namespace
oc create namespace slurm

# Grant anyuid SCC (allows pods to run with the UID required by Slurm - UID 401)
oc adm policy add-scc-to-user anyuid -z default -n slurm

# Grant privileged SCC (required — Slurm's cgroup/v2 plugin loads an eBPF
# program into the kernel for per-job device/cgroup control, which needs
# BPF/NET_ADMIN/SYS_ADMIN capabilities that anyuid alone does not grant;
# see https://slurm.schedmd.com/cgroup_v2.html). Without this, worker pods
# are rejected at admission and never get created.
oc adm policy add-scc-to-user privileged -z default -n slurm
```

> **Why anyuid?** Slurm daemons need to run as specific UIDs (e.g., UID 401 for the `slurm` user). OpenShift's default SCC restricts pods to a narrow UID range. The `anyuid` SCC grants the exception.

**Via UI:**
- Go to "Home" → "Projects"
- Click "Create Project"
- Name: `slurm`
- Click "Create"
- **Then via terminal**, grant the SCCs:
  ```bash
  oc adm policy add-scc-to-user anyuid -z default -n slurm
  oc adm policy add-scc-to-user privileged -z default -n slurm
  ```

#### Step 3.2: Create Required Secrets (SECOND STEP)

The Controller requires JWT and Slurm keys. Create them **before** creating the Controller. Use the **operator default** secret names and key names so the Operator UI template works without editing secret references.

**Secret structure (operator default):**
- **Secret 1**: name `slurm-auth-jwths256`, key inside secret: `jwt_hs256.key` (random value)
- **Secret 2**: name `slurm-auth-slurm`, key inside secret: `slurm.key` (random value)

**Via Terminal (Recommended):**
```bash
# Generate keys and create secrets (operator default names/keys)
JWT_KEY=$(openssl rand -base64 32)
SLURM_KEY=$(openssl rand -base64 32)
oc create secret generic slurm-auth-jwths256 -n slurm \
  --from-literal=jwt_hs256.key="$JWT_KEY"
oc create secret generic slurm-auth-slurm -n slurm \
  --from-literal=slurm.key="$SLURM_KEY"

# Verify secrets were created
oc get secret slurm-auth-jwths256 slurm-auth-slurm -n slurm
```

   **Via UI:**
   - Go to "Workloads" → "Secrets", namespace `slurm`
   - **First secret:** Create → Key/value secret. **Name**: `slurm-auth-jwths256`. Add key `jwt_hs256.key`, value = random base64 (e.g. `openssl rand -base64 32`). Create.
   - **Second secret:** Create → Key/value secret. **Name**: `slurm-auth-slurm`. Add key `slurm.key`, value = random base64. Create.
   
   **If pods fail to start because the secret is missing**, delete the pods so they restart:
   ```bash
   oc delete pod -n slurm -l app.kubernetes.io/name=slurmctld
   oc delete pod -n slurm -l app.kubernetes.io/name=slurmd
   ```

#### Step 3.3: Create Controller (THIRD STEP)

**Via Operator UI:**

1. **Navigate to the Slurm Operator**
   - Go to **"Ecosystem"** → **"Installed Operators"**
   - Filter by namespace: `openshift-operators`
   - Click on **"Slurm Operator"**

2. **Find the Controller Resource Type**
   - Look for a section showing "Provided APIs" or "Resource Types"
   - You should see **"Controller"** listed
   - Click on **"Controller"**, OR
   - Look for "Create Instance" or "Create Controller" button
   - If you see "All instances" tab, click it, then click "+ Create"

3. **Create Controller**
   - You should be on the **"Create Controller"** page
   - Select **"YAML view"** (not Form view)
   - You'll see a default YAML template - **MODIFY it** with the following changes:

**Complete YAML (with all defaults + our changes):**

```yaml
apiVersion: slinky.slurm.net/v1beta1
kind: Controller
metadata:
  name: slurm
  namespace: slurm
spec:
  accountingRef:
    name: slurm
    namespace: slurm
  jwtHs256KeyRef:
    key: jwt_hs256.key
    name: slurm-auth-jwths256
  slurmKeyRef:
    key: slurm.key
    name: slurm-auth-slurm
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
    # Increase job retention time in memory (default is 300 seconds = 5 minutes)
    # Set to 3600 seconds (1 hour) - adjust as needed
    MinJobAge=3600
```

**Or use the minimal version (if you want to keep it simple):**

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
```

1. **Click "Create"**
   - Wait for the Controller to be created (status should show as Ready)

**Via Terminal (Alternative):**
```bash
oc apply -f - <<EOF
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
EOF
```

#### Step 3.4: Create NodeSet (FOURTH STEP - After Controller is Ready)

**Important:** Resources in NodeSet go under `spec.slurmd.resources`, NOT directly under `spec.resources`.

**Note on PyTorch/DDP demo:** The NodeSet YAML below is a minimal example for basic `sinfo`/`sbatch` testing — it does **not** include the `provision-pytorch` init container, GPU `gres.conf`, or worker-scripts ConfigMap that `configs/slurm-cluster.yaml` (used by Method 1 / `./scripts/deploy-slurm.sh`) sets up automatically. If you plan to run `python demos/ddp_test.py --launch` (see the [DDP Test Guide](DDP_TEST_GUIDE.md)), deploy via Method 1 instead (or apply `configs/slurm-cluster.yaml` directly) so workers come up with PyTorch already provisioned — there's no runtime fallback that installs it for you anymore.

**Via Operator UI:**

1. **Navigate back to the Slurm Operator**
   - Go to **"Ecosystem"** → **"Installed Operators"** → **"Slurm Operator"**

2. **Find the NodeSet Resource Type**
   - Look for **"NodeSet"** in the resource types
   - Click on **"NodeSet"**
   - Click **"Create NodeSet"**

3. **Create NodeSet**
   - Select **"YAML view"**
   - You'll see a default YAML template - **MODIFY it** with the following changes:



**Complete YAML (with defaults + our changes):**

Use the OCP default NodeSet name `slurm-worker-slinky` so you don't have to change it.

```yaml
apiVersion: slinky.slurm.net/v1beta1
kind: NodeSet
metadata:
  name: slurm-worker-slinky
  namespace: slurm
  labels:
    nodeset.slinky.slurm.net/name: slurm-worker-slinky
spec:
  controllerRef:
    name: slurm
    namespace: slurm
  replicas: 2
  partition:
    enabled: true
  slurmd:
    image: 'ghcr.io/slinkyproject/slurmd:25.11-ubuntu24.04'
    resources:
      requests:
        cpu: "1"
        memory: "2Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
    env:
      - name: POD_CPUS
        value: '0'
      - name: POD_MEMORY
        value: '0'
    volumeMounts:
      - name: dshm
        mountPath: /dev/shm
  logfile:
    image: 'docker.io/library/alpine:latest'
    resources: {}
  template:
    metadata:
      labels:
        nodeset.slinky.slurm.net/name: slurm-worker-slinky
    spec:
      affinity: {}
      hostname: slinky-
      imagePullSecrets: null
      initContainers: []
      nodeSelector:
        kubernetes.io/os: linux
      priorityClassName: null
      tolerations: []
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 2Gi
  updateStrategy:
    rollingUpdate:
      maxUnavailable: 100%
    type: RollingUpdate
```

**Or use the minimal version (OCP default name):**

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

1. **Click "Create"**

**Via Terminal (Alternative, OCP default name):**
```bash
oc apply -f - <<EOF
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

**Alternative: Generic YAML Import (if Operator UI doesn't show Create button)**
   - Go to **"Workloads"** → **"Topology"** (or any page with "+" button)
   - Click **"+"** in top right
   - Select **"Import YAML"** (NOT "Create Deployment")
   - Paste the YAML above
   - Click "Create"

**Note**: If you're looking at the operator Deployment details page (like the Actions menu), that's the operator itself, not where you create the cluster. Navigate back to the Operator page to create Controller and NodeSet.

### Step 4: Monitor Deployment via UI

**Via Terminal:**
```bash
# Check Controller and NodeSet resources (namespace: slurm by default, or your custom namespace)
oc get controllers -n slurm
oc get nodesets -n slurm
oc describe controller <controller-name> -n slurm
oc describe nodeset <nodeset-name> -n slurm

# Check pods being created
oc get pods -n slurm

# Watch pods (they may take a few minutes to start)
oc get pods -n slurm -w
```

**Via UI:**

1. **Check Pods**:
   - Go to "Workloads" → "Pods"
   - Filter by namespace: `slurm`
   - You should see:
     - `slurm-controller-0` (Running)
     - `slurm-worker-slinky-0` (Running)
     - `slurm-worker-slinky-1` (Running)
   - Note: Pods may take 1-3 minutes to start

2. **Check Controller and NodeSet Resources**:
   - Go to "Operators" → "Installed Operators"
   - Click on "Slurm Operator"
   - Click "Controller" tab to see your controller
   - Click "NodeSet" tab to see your compute nodes
   - Or check via terminal: `oc get controllers,nodesets -n slurm`

3. **Check Services**:
   - Go to "Networking" → "Services"
   - Filter by namespace: `slurm`
   - Verify Slurm services are created

### Step 5: Access Slurm via Terminal Pod

1. **Open Terminal in Pod**:
   - Go to "Workloads" → "Pods"
   - Find `slurm-controller-xxx` pod
   - Click on the pod name
   - Click "Terminal" tab

2. **Test Slurm Commands**:
   
   **If using UI Terminal (inside the pod):**
   ```bash
   # When you're inside the pod terminal via UI, you can run commands directly:
   sinfo
   scontrol show nodes
   sbatch --output=/tmp/test.out --wrap="echo 'Hello from Slurm' && hostname && date"
   squeue
   scontrol show job <JOB_ID>
   ```
   
   **If using your local terminal (via oc exec):**
   ```bash
   # First, set the controller pod variable
   CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')
   
   # Then run commands with oc exec (note: -c slurmctld is REQUIRED)
   oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo
   oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show nodes
   oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch --output=/tmp/test.out --wrap="echo 'Hello from Slurm' && hostname && date"
   oc exec -n slurm $CONTROLLER_POD -c slurmctld -- squeue
   oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job <JOB_ID>
   
   # Find which node ran the job
   oc exec -n slurm $CONTROLLER_POD -c slurmctld -- scontrol show job <JOB_ID> | grep NodeList
   
   # View job output (on the compute node that ran it)
   # If NodeList=slinky-0, use slurm-worker-slinky-0; if slinky-1, use slurm-worker-slinky-1
   oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/test.out 2>/dev/null || \
   oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- cat /tmp/test.out
   ```
   
   **Important:** If you're running commands from your local terminal (not inside the pod), you MUST use `oc exec` with the `-c slurmctld` flag. Slurm commands are NOT installed on your local machine.

### Step 6: Deploy the Autoscaler (Required for Auto Scale-Down)

Everything above gets Slurm itself running, but nothing yet watches the job queue to scale the `NodeSet` up or down automatically — that's a separate piece, easy to miss when following this UI-driven method since there's no operator screen for it. See Method 1's [Step 6](#step-6-deploy-the-autoscaler-required-for-auto-scale-down) for the full explanation of what each command does and why the SCC grant is needed; the commands themselves are identical regardless of which method you used to get here:

```bash
# Recommended:
./scripts/deploy-autoscale.sh

# Or manually:
oc create configmap slurm-autoscaler-script -n slurm \
  --from-file=autoscaler.sh=scripts/autoscaler-loop.sh \
  --dry-run=client -o yaml | oc apply -f -
oc apply -f configs/slurm-autoscaler.yaml
oc adm policy add-scc-to-user privileged -z slurm-autoscaler -n slurm

# Verify:
oc get pods -n slurm -l app.kubernetes.io/name=slurm-autoscaler
oc logs -n slurm -l app.kubernetes.io/name=slurm-autoscaler -f --tail=15
```

#### Complete Terminal Method (All Steps in Order)

If you prefer to do everything via terminal, here's the complete sequence:

```bash
# Step 1: Create namespace and configure security
oc create namespace slurm

# Grant anyuid SCC to allow Slurm to run with required UID (401)
oc adm policy add-scc-to-user anyuid -z default -n slurm

# Grant privileged SCC (required for Slurm's eBPF-based cgroup/v2 device
# control; see the "Why anyuid?" note in Step 3.1 above). Without this,
# worker pods are rejected at admission and never get created.
oc adm policy add-scc-to-user privileged -z default -n slurm

# Step 2: Create required secrets (operator default names/keys)
JWT_KEY=$(openssl rand -base64 32)
SLURM_KEY=$(openssl rand -base64 32)
oc create secret generic slurm-auth-jwths256 -n slurm --from-literal=jwt_hs256.key="$JWT_KEY"
oc create secret generic slurm-auth-slurm -n slurm --from-literal=slurm.key="$SLURM_KEY"

# Step 3: Create Controller (use name: slurm; do not change name after creation)
oc apply -f - <<EOF
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
EOF

# Step 4: Wait for Controller to be ready (optional but recommended)
oc wait --for=condition=Ready controller/slurm -n slurm --timeout=300s 2>/dev/null || true
# If wait fails (e.g. condition not supported), just wait for pods: oc get pods -n slurm -w

# Step 5: Create NodeSet (OCP default name: slurm-worker-slinky; controllerRef.name must match Controller: slurm)
# IMPORTANT: Include the image field — operator may not set it automatically
oc apply -f - <<EOF
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

# Step 6: Verify resources were created
oc get controllers -n slurm
oc get nodesets -n slurm
oc get pods -n slurm

# Step 7: Deploy the autoscaler (easy to forget — see Step 6 above for why
# each command below is needed)
oc create configmap slurm-autoscaler-script -n slurm \
  --from-file=autoscaler.sh=scripts/autoscaler-loop.sh \
  --dry-run=client -o yaml | oc apply -f -
oc apply -f configs/slurm-autoscaler.yaml
oc adm policy add-scc-to-user privileged -z slurm-autoscaler -n slurm
```

---

## Step-by-Step POC Execution (Hybrid Approach)

### Phase 1: Deployment (Terminal)

```bash
# 1. Login to OpenShift
oc login https://api.<your-cluster-domain>:6443

# 2. Run deployment script (from your repo root)
cd /path/to/slurm-on-ocp
./scripts/deploy-slurm.sh --skip-operator

# 3. Wait for deployment (script handles this)
```

### Phase 2: Verification (Browser UI)

1. Open browser: `console-openshift-console.apps.<your-cluster-domain>`
2. Navigate to "Workloads" → "Pods" → Filter: `slurm`
3. Verify all pods are "Running"
4. Check "Operators" → "Installed Operators" → "Slurm Operator"

### Phase 3: Testing (Terminal)

```bash
# Get controller pod
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')

# Run test jobs
./scripts/test-slurm.sh --comprehensive --namespace slurm

# Or manually test (namespace: slurm by default, or your custom namespace)
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch --wrap="echo 'Test job' && sleep 10"
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- squeue
```

### Phase 4: Monitoring (Browser UI)

1. View pod logs: "Workloads" → "Pods" → Click pod → "Logs" tab
2. View events: "Observe" → "Events" → Filter by namespace: `slurm`
3. View resource usage: "Observe" → "Metrics" → Select namespace: `slurm`

---

## Step 6: Test the Slurm Cluster

Now that your cluster is deployed and running, let's test it by submitting jobs and viewing the output.

**⚠️ Important: Slurm commands must run INSIDE the pod**

Slurm commands (`sbatch`, `sinfo`, `squeue`, etc.) are **NOT** installed on your local machine. You must use `oc exec` to run them inside the controller pod.

**Quick Setup (do this first):**
```bash
# Set the controller pod variable (do this once per terminal session)
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')

# Verify it's set
echo $CONTROLLER_POD
# Should show: slurm-controller-0
```

**All Slurm commands must use this format:**
```bash
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- <slurm-command>
```

**Note:** The `-c slurmctld` flag is **REQUIRED** - it specifies which container in the pod to use.

### Test 1: Basic Job Submission and Output

**Objective**: Submit a simple job and verify it completes successfully.

**Via Terminal:**

```bash
# 1. Check cluster status
oc exec -n slurm slurm-controller-0 -c slurmctld -- sinfo

# 2. Check available nodes
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show nodes

# 3. Submit a simple test job
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch --output=/tmp/test-job.out --wrap="echo 'Hello from Slurm!' && hostname && date && echo 'Job completed successfully'"

# Output will show: Submitted batch job <JOB_ID>
# Note the JOB_ID (e.g., 1, 2, 3, etc.)
```

**Check Job Status:**

```bash
# Wait a few seconds, then check if job is in queue
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue

# If queue is empty, job completed quickly. Check detailed status:
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job <JOB_ID>

# Look for:
# - JobState=COMPLETED
# - ExitCode=0:0 (success)
# - StdOut=/tmp/test-job.out (output file path)
# - NodeList=slinky-1 (which compute node ran it)
```

**View Job Output:**

```bash
# Step 1: Find which compute node ran the job
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job <JOB_ID> | grep NodeList

# Step 2: Map NodeList to compute pod
# NodeList=slinky-0 → slurm-worker-slinky-0
# NodeList=slinky-1 → slurm-worker-slinky-1

# Step 3: View output on the correct compute node
# If NodeList shows slinky-0:
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/test-job.out

# If NodeList shows slinky-1:
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- cat /tmp/test-job.out

# Step 4: Or try both nodes (if you're not sure - RECOMMENDED)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/test-job.out 2>/dev/null || \
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- cat /tmp/test-job.out

# Expected output:
# Hello from Slurm!
# slinky-0 (or slinky-1, depending on which node ran it)
# Mon Nov 24 19:28:16 UTC 2025
# Job completed successfully
```

**Troubleshooting Test 1:**

- **`event not found` error when using `!` in `sbatch --wrap`** — Bash interprets `!` inside double quotes as history expansion. Use single quotes instead:
  ```bash
  # WRONG (will fail with "event not found"):
  oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch --wrap="echo 'Hello!' && hostname"

  # CORRECT (use single quotes or avoid ! in double quotes):
  oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch --wrap='echo Hello && hostname && date'
  ```
- **`sacct` returns "Slurm accounting storage is disabled"** — `sacct` requires `slurmdbd` (Slurm Database Daemon), which is **not deployed by default**. Use `scontrol show job <JOB_ID>` instead to check job status. Jobs remain visible via `scontrol` for the duration set by `MinJobAge` (default: 300 seconds, set to 3600 in this guide).
- **Job output file not found** — Remember that job output files are written **on the worker node** that ran the job, not on the controller. Always check which node ran the job first with `scontrol show job <JOB_ID> | grep NodeList`.
- **`--export=ALL` for environment propagation** — If your job script relies on environment variables from the submitting shell, add `--export=ALL` to the `sbatch` command:
  ```bash
  oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch --export=ALL --output=/tmp/test.out --wrap="env | head -20"
  ```

### Test 2: Multiple Jobs and Queue Monitoring

**Objective**: Submit multiple jobs and monitor the queue.

```bash
# Submit 3 jobs
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch --output=/tmp/job1.out --wrap="sleep 5 && echo 'Job 1 done'"
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch --output=/tmp/job2.out --wrap="sleep 5 && echo 'Job 2 done'"
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch --output=/tmp/job3.out --wrap="sleep 5 && echo 'Job 3 done'"

# Monitor queue in real-time
watch -n 2 "oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue"
# macOS (no watch): while true; do clear; oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue; sleep 2; done

# Or check once
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue

# Check all job statuses
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job
```

### Test 3: Job with Resource Requirements

**Objective**: Submit a job requesting specific CPU and memory.

```bash
# Submit job requesting 1 CPU and 1GB memory
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch \
  --cpus-per-task=1 \
  --mem=1G \
  --output=/tmp/resource-job.out \
  --wrap="echo 'CPU: ' && nproc && echo 'Memory: ' && free -h && echo 'Job with resource requirements completed'"

# Check job details to verify resources were allocated
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job <JOB_ID> | grep -E "(ReqTRES|AllocTRES|NumCPUs|NodeList)"

# Find which node ran the job and view output
NODE=$(oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job <JOB_ID> | grep NodeList | awk '{print $1}' | cut -d= -f2)
echo "Job ran on: $NODE"

# View output (try both nodes if unsure)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/resource-job.out 2>/dev/null || \
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- cat /tmp/resource-job.out
```

### Test 4: Long-Running Job

**Objective**: Submit a job that runs for a longer duration to test monitoring.

```bash
# Submit a job that runs for 30 seconds
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch --output=/tmp/long-job.out --wrap='for i in $(seq 1 30); do echo "Iteration $i at $(date)"; sleep 1; done'

# Monitor the job while it's running
oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue

# Check job status and find which node ran it
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job <JOB_ID> | grep -E "(JobState|ExitCode|NodeList|StdOut)"

# View output (try both nodes)
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- cat /tmp/long-job.out 2>/dev/null || \
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- cat /tmp/long-job.out
```

### Test 5: Check All Output Files

**Objective**: View all job outputs from recent test jobs.

```bash
# List all output files on both compute nodes
echo "=== Files on compute-0 ==="
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- ls -la /tmp/*.out 2>/dev/null || echo "No files found on compute-0"

echo "=== Files on compute-1 ==="
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- ls -la /tmp/*.out 2>/dev/null || echo "No files found on compute-1"

# View all outputs from compute-0
oc exec -n slurm slurm-worker-slinky-0 -c slurmd -- sh -c 'for f in /tmp/*.out 2>/dev/null; do [ -f "$f" ] && echo "=== $f (compute-0) ===" && cat "$f" && echo ""; done'

# View all outputs from compute-1
oc exec -n slurm slurm-worker-slinky-1 -c slurmd -- sh -c 'for f in /tmp/*.out 2>/dev/null; do [ -f "$f" ] && echo "=== $f (compute-1) ===" && cat "$f" && echo ""; done'
```

### Test 6: Verify Job Retention

**Objective**: Verify that completed jobs remain accessible (thanks to MinJobAge=3600).

```bash
# Submit a job
oc exec -n slurm slurm-controller-0 -c slurmctld -- sbatch --output=/tmp/retention-test.out --wrap="echo 'Testing job retention' && date"

# Wait for it to complete
sleep 5

# Check job status immediately
JOB_ID=$(oc exec -n slurm slurm-controller-0 -c slurmctld -- squeue -h -o "%i" 2>/dev/null | head -1 || echo "")
if [ -z "$JOB_ID" ]; then
  # Get the latest job ID from scontrol
  JOB_ID=$(oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job 2>/dev/null | grep JobId | head -1 | awk -F= '{print $2}' | awk '{print $1}')
fi

echo "Checking job $JOB_ID"
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job $JOB_ID

# This should work for up to 1 hour after job completion (MinJobAge=3600)
```

### Test 7: View All Jobs in Memory

**Objective**: See all completed jobs that are still retained in Slurm's memory.

```bash
# View all jobs currently in memory (up to MinJobAge duration)
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job

# This will show all jobs with details like:
# - JobId, JobState (COMPLETED, RUNNING, etc.)
# - ExitCode (0:0 = success)
# - NodeList (which compute node ran it: slinky-0 or slinky-1)
# - StdOut (output file path)
# - RunTime, StartTime, EndTime
# - Resource allocation (ReqTRES, AllocTRES)

# Filter for specific information
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job | grep -E "(JobId|JobState|ExitCode|NodeList|StdOut|RunTime)"

# Count completed jobs
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job | grep "JobState=COMPLETED" | wc -l

# See which nodes jobs ran on
oc exec -n slurm slurm-controller-0 -c slurmctld -- scontrol show job | grep "NodeList=" | sort | uniq -c
```

**Example Output Analysis:**

From the output, you can see:
- **Job Distribution**: Jobs are distributed across both compute nodes (`slinky-0` and `slinky-1`)
- **Job States**: All showing `JobState=COMPLETED` with `ExitCode=0:0` (success)
- **Resource Usage**: Different jobs requested different resources:
  - Job 8: Requested 2 CPUs (`ReqTRES=cpu=2`)
  - Job 9: Requested 1 CPU and 2G memory (`ReqTRES=cpu=1,mem=2G`)
- **Job Timing**: Various runtimes from seconds to minutes
- **Output Files**: Each job has its output file path in `StdOut` field

### Quick Test Script

A ready-to-use test script is available: `scripts/test-slurm.sh`

```bash
# Run the test script
./scripts/test-slurm.sh
```

This script will:
1. Check cluster status
2. Show available nodes
3. Submit a test job
4. Monitor job completion
5. Display job status and output

Make it executable and run:
```bash
chmod +x scripts/test-slurm.sh
./scripts/test-slurm.sh
```

---

## Monitoring and Observability

### Check Operator Logs
```bash
# If operator was installed via Helm (in slinky namespace):
oc logs -n slinky -l app.kubernetes.io/name=slurm-operator --tail=100

# If operator was installed via OperatorHub (in openshift-operators namespace):
oc logs -n openshift-operators -l app.kubernetes.io/name=slurm-operator --tail=100

# Not sure which namespace? Check both:
oc get pods -n slinky -l app.kubernetes.io/name=slurm-operator 2>/dev/null || \
oc get pods -n openshift-operators -l app.kubernetes.io/name=slurm-operator
```

### Check Controller Logs
```bash
oc logs -n slurm slurm-controller-0 -c slurmctld --tail=100
```

### Check Compute Node Logs
```bash
oc logs -n slurm slurm-worker-slinky-0 -c slurmd --tail=100
```

### Monitor Pod Status
```bash
# Watch pods in real-time
oc get pods -n slurm -w

# Check pod resource usage
oc top pods -n slurm
```

### Check Compute Node Logs
```bash
oc logs -n slurm -l app.kubernetes.io/component=compute --tail=100
```

### Monitoring via UI

1. **View Pod Logs**:
   - Go to "Workloads" → "Pods"
   - Click on pod name
   - Click "Logs" tab

2. **View Events**:
   - Go to "Observe" → "Events"
   - Filter by namespace: `slurm` or `slinky`

3. **View Metrics**:
   - Go to "Observe" → "Metrics"
   - Select namespace: `slurm`
   - View CPU, memory, and other metrics

---

## Quick Reference

### Cleanup and Fresh Start

**To delete everything and start fresh:**
```bash
# Clean up all Slurm resources
./scripts/cleanup-slurm.sh slurm

# Or manually:
oc delete -f configs/slurm-autoscaler.yaml --ignore-not-found
oc delete controller,nodeset --all -n slurm
oc delete statefulset,deployment,pods,svc,pvc --all -n slurm
oc delete configmap --all -n slurm
oc delete secret slurm-auth-jwths256 slurm-auth-slurm -n slurm
oc adm policy remove-scc-from-user anyuid -z slurm-workload -n slurm
oc adm policy remove-scc-from-user privileged -z slurm-workload -n slurm
oc adm policy remove-scc-from-user anyuid -z default -n slurm
oc delete namespace slurm
```

**To deploy from scratch:**
```bash
# Complete deployment (cluster + autoscaler) in correct order
./scripts/deploy-slurm.sh

# Run DDP training (auto-detects cluster, scales, submits, monitors, retrieves results):
python demos/ddp_test.py --launch
```


## References

- [Slurm Operator Installation Guide](https://slinky.schedmd.com/projects/slurm-operator/en/release-0.4/installation.html)
- [Slurm Operator Configuration](https://slinky.schedmd.com/projects/slurm-operator/en/release-0.4/configuration.html)
- [Slurm Documentation](https://slurm.schedmd.com/documentation.html)
- [Slinky Project](https://slinky.schedmd.com/)

