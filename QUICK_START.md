# Quick Start Guide - Slurm on OpenShift

Get Slurm running on OpenShift in a few steps. Use this if you already have an OpenShift cluster and `oc`/`helm` configured.

## Prerequisites

- OpenShift 4.x cluster with admin access
- `oc` CLI installed and logged in
- `helm` 3.x installed

## Option A: Automated Deployment (Recommended)

```bash
# From this repository
chmod +x scripts/deploy-slurm.sh
./scripts/deploy-slurm.sh

# With custom namespace
./scripts/deploy-slurm.sh --namespace my-slurm

# Skip cert-manager if already installed
./scripts/deploy-slurm.sh --skip-cert-manager
```

The script will:
1. Check prerequisites (`oc`, `helm`)
2. Install cert-manager (unless `--skip-cert-manager`)
3. Install Slurm Operator CRDs and operator (unless already present)
4. Create the `slurm` namespace and auth secrets
5. Deploy the Slurm cluster (Controller + NodeSet)
6. Deploy the scale-down autoscaler watchdog
7. Verify deployment

## Option B: Manual Steps

### Step 1: Install cert-manager

cert-manager is required for the Slurm operator TLS certificates.

```bash
# Check if already installed
oc get pods -n cert-manager

# If not, install via Helm
helm repo add jetstack https://charts.jetstack.io
helm repo update
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager --create-namespace \
  --set installCRDs=true --set 'crds.enabled=true' \
  --version v1.13.0 --wait --timeout 5m
```

### Step 2: Install Slurm Operator CRDs and Operator

```bash
# CRDs
helm install slurm-operator-crds \
  oci://ghcr.io/slinkyproject/charts/slurm-operator-crds \
  --namespace slinky --create-namespace --server-side=false
sleep 10

# Operator
helm install slurm-operator \
  oci://ghcr.io/slinkyproject/charts/slurm-operator \
  --namespace slinky --create-namespace --server-side=false --wait --timeout 5m

oc get pods -n slinky
```

### Step 3: Create Slurm namespace and secrets

```bash
oc create namespace slurm
JWT_KEY=$(openssl rand -base64 32)
SLURM_KEY=$(openssl rand -base64 32)
oc create secret generic slurm-auth-jwths256 -n slurm --from-literal=jwt_hs256.key="$JWT_KEY"
oc create secret generic slurm-auth-slurm -n slurm --from-literal=slurm.key="$SLURM_KEY"
```

### Step 4: Deploy Slurm cluster

```bash
oc apply -f configs/slurm-cluster.yaml
```

### Step 5: Verify

```bash
oc get pods -n slurm
oc get controllers,nodesets -n slurm
```

## Testing

```bash
# Get controller pod name
CONTROLLER_POD=$(oc get pods -n slurm -l app.kubernetes.io/name=slurmctld -o jsonpath='{.items[0].metadata.name}')

# Run sinfo
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sinfo

# Submit a test job
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- sbatch --output=/tmp/test.out --wrap="echo 'Hello from Slurm' && hostname && date"
oc exec -n slurm $CONTROLLER_POD -c slurmctld -- squeue
```

Or use the test script:

```bash
./scripts/test-slurm.sh
```

## Run DDP Training

Once the cluster is up, run distributed PyTorch training (auto-detects cluster, scales nodes, provisions workers):

```bash
python demos/ddp_test.py --launch
```

## Next Steps

- Full walkthrough (CLI and UI): [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)
- Add more nodes: [docs/ADD_NODES.md](docs/ADD_NODES.md)
- Clean up: `./scripts/cleanup-slurm.sh`

## Reference

- [Slinky Project](https://slinky.schedmd.com/)
- [Slurm Documentation](https://slurm.schedmd.com/)
