#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install OpenSandbox on nemo-dev-blue with both crun and kata-qemu server profiles.
#
# Prerequisites:
#   - kubectl context: nv-prd-nemo.teleport.sh-nemo-dev-blue
#   - RuntimeClass/kata-qemu present (install-kata-qemu.sh already done)
#   - Local checkout of OpenSandbox with charts under kubernetes/charts/
#
# Usage:
#   ./packages/nhx_customization_common/values/opensandbox/install.sh
#   OPENSANDBOX_DIR=/path/to/OpenSandbox \
#     ./packages/nhx_customization_common/values/opensandbox/install.sh
set -exuo pipefail

VALUES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# values/opensandbox -> nhx_customization_common -> packages -> nemo-helix -> work
WORK_ROOT="$(cd "${VALUES_DIR}/../../../../.." && pwd)"
OPENSANDBOX_DIR="${OPENSANDBOX_DIR:-${WORK_ROOT}/OpenSandbox}"
CONTROLLER_CHART="${OPENSANDBOX_DIR}/kubernetes/charts/opensandbox-controller"
SERVER_CHART="${OPENSANDBOX_DIR}/kubernetes/charts/opensandbox-server"

for path in "${CONTROLLER_CHART}" "${SERVER_CHART}"; do
  if [[ ! -d "${path}" ]]; then
    echo "missing chart: ${path}" >&2
    echo "set OPENSANDBOX_DIR to your OpenSandbox checkout" >&2
    exit 1
  fi
done

echo "=== context ==="
kubectl config current-context
kubectl get runtimeclass kata-qemu >/dev/null
kubectl get nodes -l katacontainers.io/kata-runtime=true --no-headers

echo "=== namespaces ==="
for ns in opensandbox-system opensandbox-crun opensandbox-kata; do
  kubectl create namespace "${ns}" --dry-run=client -o yaml | kubectl apply -f -
done

echo "=== BatchSandbox templates ==="
kubectl apply -f "${VALUES_DIR}/batchsandbox-template-crun.yaml"
kubectl apply -f "${VALUES_DIR}/batchsandbox-template-kata-qemu.yaml"

echo "=== API key Secrets ==="
# Create only if missing so re-runs do not rotate keys. Override with
# CRUN_API_KEY / KATA_API_KEY env vars, or create the Secrets yourself first.
ensure_api_key_secret() {
  local name="$1"
  local env_var="$2"
  if kubectl get secret "${name}" -n opensandbox-system >/dev/null 2>&1; then
    echo "Secret ${name} already exists — keeping"
    return
  fi
  local key="${!env_var:-}"
  if [[ -z "${key}" ]]; then
    key="$(openssl rand -hex 32)"
    echo "Generated ${name} (set ${env_var}=... to supply your own)"
  else
    echo "Creating ${name} from \$${env_var}"
  fi
  kubectl create secret generic "${name}" \
    -n opensandbox-system \
    --from-literal=api-key="${key}"
}
ensure_api_key_secret opensandbox-server-crun-api-key CRUN_API_KEY
ensure_api_key_secret opensandbox-server-kata-api-key KATA_API_KEY

echo "=== controller ==="
helm upgrade --install opensandbox-controller "${CONTROLLER_CHART}" \
  --namespace opensandbox-system \
  -f "${VALUES_DIR}/opensandbox-controller.yaml"

echo "=== server (crun / shared-kernel) ==="
helm upgrade --install opensandbox-server-crun "${SERVER_CHART}" \
  --namespace opensandbox-system \
  -f "${VALUES_DIR}/opensandbox-server-crun.yaml"

echo "=== server (kata-qemu) ==="
helm upgrade --install opensandbox-server-kata "${SERVER_CHART}" \
  --namespace opensandbox-system \
  -f "${VALUES_DIR}/opensandbox-server-kata-qemu.yaml"

echo "=== wait ==="
kubectl rollout status deploy/opensandbox-controller-manager -n opensandbox-system --timeout=180s
kubectl rollout status deploy/opensandbox-server-crun -n opensandbox-system --timeout=180s
kubectl rollout status deploy/opensandbox-server-kata -n opensandbox-system --timeout=180s

echo "=== done ==="
echo "crun  Service: http://opensandbox-server-crun.opensandbox-system.svc.cluster.local"
echo "kata  Service: http://opensandbox-server-kata.opensandbox-system.svc.cluster.local"
echo "API keys: Secrets opensandbox-server-crun-api-key / opensandbox-server-kata-api-key (key: api-key)"
echo "  kubectl get secret -n opensandbox-system opensandbox-server-crun-api-key -o jsonpath='{.data.api-key}' | base64 -d; echo"
echo "  kubectl get secret -n opensandbox-system opensandbox-server-kata-api-key -o jsonpath='{.data.api-key}' | base64 -d; echo"
echo "Verify runtimes:"
echo "  ${VALUES_DIR}/verify/crun.sh"
echo "  ${VALUES_DIR}/verify/kata-qemu.sh"
echo "  ${VALUES_DIR}/verify/all.sh"
