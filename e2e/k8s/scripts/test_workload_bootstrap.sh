#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run against a disposable, prepared Kind cluster (also run by CI).
# The generic chart must bootstrap ALL controllers with only platform.base_url.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
source "${SCRIPT_DIR}/lib.sh"
NAMESPACE="${NAMESPACE:-nemo-helix}"
: "${NHX_E2E_REGISTRY:?set the platform image registry}"
: "${NHX_E2E_TAG:?set the platform image tag}"

# A pre-existing release would test upgrade readiness rather than bootstrap.
existing_releases="$(helm list -n "${NAMESPACE}" --filter '^workload-bootstrap$' -q)"
if [[ -n "${existing_releases}" ]]; then
    echo 'workload-bootstrap must be a fresh install' >&2
    exit 1
fi

tmp_dir="$(mktemp -d)"
finish() {
    local status=$?
    if [[ ${status} -ne 0 ]]; then
        kubectl -n "${NAMESPACE}" get pods || true
        kubectl -n "${NAMESPACE}" describe pods || true
        kubectl -n "${NAMESPACE}" logs deployment/workload-bootstrap-core-controller --tail=150 || true
    fi
    rm -rf "${tmp_dir}"
    exit "${status}"
}
trap finish EXIT

create_platform_secrets "${NAMESPACE}"
openssl req -x509 -newkey rsa:2048 -nodes -days 1 \
    -subj /CN=workload-tls-gateway -addext subjectAltName=DNS:workload-tls-gateway \
    -keyout "${tmp_dir}/tls.key" -out "${tmp_dir}/tls.crt"
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "${tmp_dir}/signing.pem"
# Preserve public roots as well as trusting the test gateway.
cat "$(uv run --frozen python -m certifi)" "${tmp_dir}/tls.crt" > "${tmp_dir}/ca.crt"
kubectl -n "${NAMESPACE}" create secret tls workload-bootstrap-tls \
    --cert="${tmp_dir}/tls.crt" --key="${tmp_dir}/tls.key"
kubectl -n "${NAMESPACE}" create secret generic workload-bootstrap-signing \
    --from-file="private-key.pem=${tmp_dir}/signing.pem"
kubectl -n "${NAMESPACE}" create configmap workload-bootstrap-ca --from-file="ca.crt=${tmp_dir}/ca.crt"

# Resolve the API Service at request time; Helm creates it after the gateway.
dns_ip="$(kubectl -n kube-system get service kube-dns -o jsonpath='{.spec.clusterIP}')"
cat > "${tmp_dir}/nginx.conf" <<EOF
pid /tmp/nginx.pid;
events {}
http {
  client_body_temp_path /tmp/client;
  proxy_temp_path /tmp/proxy;
  server {
    listen 8443 ssl;
    ssl_certificate /tls/tls.crt;
    ssl_certificate_key /tls/tls.key;
    location / {
      resolver ${dns_ip};
      set \$backend workload-bootstrap-api.${NAMESPACE}.svc.cluster.local:8080;
      proxy_pass http://\$backend;
    }
  }
}
EOF
kubectl -n "${NAMESPACE}" create configmap workload-bootstrap-gateway --from-file="nginx.conf=${tmp_dir}/nginx.conf"
kubectl -n "${NAMESPACE}" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: workload-tls-gateway
spec:
  replicas: 1
  selector:
    matchLabels: {app: workload-tls-gateway}
  template:
    metadata:
      labels: {app: workload-tls-gateway}
    spec:
      containers:
        - name: gateway
          image: nginx:1.28-alpine
          ports:
            - {containerPort: 8443}
          volumeMounts:
            - {name: tls, mountPath: /tls, readOnly: true}
            - {name: config, mountPath: /etc/nginx/nginx.conf, subPath: nginx.conf, readOnly: true}
      volumes:
        - name: tls
          secret: {secretName: workload-bootstrap-tls}
        - name: config
          configMap: {name: workload-bootstrap-gateway}
---
apiVersion: v1
kind: Service
metadata:
  name: workload-tls-gateway
spec:
  selector: {app: workload-tls-gateway}
  ports:
    - {port: 443, targetPort: 8443}
EOF
kubectl -n "${NAMESPACE}" rollout status deployment/workload-tls-gateway --timeout=120s
helm dependency build "${REPO_ROOT}/k8s/helm"
helm upgrade --install workload-bootstrap "${REPO_ROOT}/k8s/helm" \
    --namespace "${NAMESPACE}" \
    -f "${REPO_ROOT}/e2e/k8s/values/kind.yaml" \
    -f "${REPO_ROOT}/e2e/k8s/values/workload-bootstrap.yaml" \
    --set-string "api.image.repository=${NHX_E2E_REGISTRY}/nhx-api" \
    --set-string "core.image.repository=${NHX_E2E_REGISTRY}/nhx-api" \
    --set-string "api.image.tag=${NHX_E2E_TAG}" \
    --set-string "core.image.tag=${NHX_E2E_TAG}" \
    --set-string "platformConfig.platform.image_registry=${NHX_E2E_REGISTRY}" \
    --set-string "platformConfig.platform.image_tag=${NHX_E2E_TAG}" \
    --wait --timeout "${HELM_TIMEOUT:-15m}"
kubectl -n "${NAMESPACE}" rollout status deployment/workload-bootstrap-core-controller --timeout=60s
printf 'PASS: generic all-controller bootstrap with HTTPS platform.base_url and advertised URL fallback\n'
