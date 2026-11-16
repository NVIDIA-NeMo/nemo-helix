#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Let every kind node pull from the builder's development registry (builder.devRegistry), so a sandbox pod can run
# an image the builder built.
#
# Images are named <release>-builder-registry.<namespace>.svc.cluster.local:5000/..., which a node's containerd can't
# resolve (it doesn't use cluster DNS) and would reach over HTTPS. This writes a certs.d hosts.toml on each node
# that sends that host to the registry Service's ClusterIP over plain HTTP, after turning certs.d on (one containerd
# restart). Pull credentials still come from the pod's imagePullSecrets.
#
# Usage: KIND_CLUSTER_NAME=<name> KUBE_NAMESPACE=nemo-helix e2e/k8s/scripts/kind_dev_registry_mirror.sh
#
# Environment:
#   KIND_CLUSTER_NAME  kind cluster (default: nhx-e2e)
#   KUBE_NAMESPACE     Helm release namespace, where the registry runs (default: nemo-helix)
#   REGISTRY_SERVICE   The registry's Service (default: nemo-helix-builder-registry)
set -euo pipefail

KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-nhx-e2e}"
KUBE_NAMESPACE="${KUBE_NAMESPACE:-nemo-helix}"
REGISTRY_SERVICE="${REGISTRY_SERVICE:-nemo-helix-builder-registry}"
CONTEXT="kind-${KIND_CLUSTER_NAME}"
HOST="${REGISTRY_SERVICE}.${KUBE_NAMESPACE}.svc.cluster.local:5000"

cluster_ip="$(kubectl --context "${CONTEXT}" -n "${KUBE_NAMESPACE}" get service "${REGISTRY_SERVICE}" \
  -o jsonpath='{.spec.clusterIP}')"
if [ -z "${cluster_ip}" ]; then
    echo "Service ${KUBE_NAMESPACE}/${REGISTRY_SERVICE} has no ClusterIP; install the platform with builder.devRegistry first" >&2
    exit 1
fi

mapfile -t nodes < <(kind get nodes --name "${KIND_CLUSTER_NAME}")
if [ "${#nodes[@]}" -eq 0 ]; then
    echo "No kind nodes found for cluster ${KIND_CLUSTER_NAME}" >&2
    exit 1
fi

# kind's node images don't point containerd at certs.d: add the registry table that does, and restart containerd.
# Running containers outlive the restart (their shims do), and the kubelet reconnects.
enable_certs_d() {
    local node="$1" config=/etc/containerd/config.toml table
    if docker exec "${node}" grep -Eq "config_path *= *[\"']/etc/containerd/certs.d[\"']" "${config}"; then
        return
    fi
    if docker exec "${node}" grep -q 'io.containerd.cri.v1.images' "${config}"; then
        table='plugins."io.containerd.cri.v1.images".registry'  # containerd 2, config version 3
    else
        table='plugins."io.containerd.grpc.v1.cri".registry'  # containerd 1.x, config version 2
    fi
    if docker exec "${node}" grep -qF "[${table}]" "${config}"; then
        echo "${node}: ${config} already has [${table}] without config_path; not editing it" >&2
        exit 1
    fi
    printf '\n[%s]\n  config_path = "/etc/containerd/certs.d"\n' "${table}" \
        | docker exec -i "${node}" sh -c "cat >> '${config}'"
    docker exec "${node}" systemctl restart containerd
    for _ in $(seq 30); do
        docker exec "${node}" crictl info >/dev/null 2>&1 && break
        sleep 2
    done
    docker exec "${node}" crictl info >/dev/null || {
        echo "${node}: containerd did not become ready after its restart" >&2
        exit 1
    }
    echo "${node}: containerd reads /etc/containerd/certs.d"
}

for node in "${nodes[@]}"; do
    enable_certs_d "${node}"
    docker exec -i "${node}" sh -c "mkdir -p '/etc/containerd/certs.d/${HOST}' && cat > '/etc/containerd/certs.d/${HOST}/hosts.toml'" <<EOF
server = "http://${HOST}"

[host."http://${cluster_ip}:5000"]
  capabilities = ["pull", "resolve"]
EOF
    echo "${node}: ${HOST} -> http://${cluster_ip}:5000"
done
