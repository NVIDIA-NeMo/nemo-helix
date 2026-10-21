#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install regression for controller bootstrap with workload token exchange on.
# Requires a disposable cluster, helm, kubectl, openssl, and pullable chart images.
# Additional arguments are passed to Helm, e.g. --set api.image.tag=... .
# Uses a test-only TLS gateway and CA; does not exercise an external OIDC provider.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
NAMESPACE="${NAMESPACE:-nhx-workload-tls-${RANDOM}}"
RELEASE=nemo-workload-tls
TLS_HOST=workload-tls-gateway
TMP_DIR="$(mktemp -d)"
CREATED_NAMESPACE=false

cleanup() {
    local status=$?
    if [[ ${status} -ne 0 && ${CREATED_NAMESPACE} == true ]]; then
        kubectl -n "${NAMESPACE}" get pods || true
        kubectl -n "${NAMESPACE}" logs "deployment/${RELEASE}-core-controller" --tail=100 || true
    fi
    rm -rf "${TMP_DIR}"
    if [[ ${CREATED_NAMESPACE} == true && ${KEEP_NAMESPACE:-false} != true ]]; then
        kubectl delete namespace "${NAMESPACE}" --wait=false || true
    fi
    exit "${status}"
}
trap cleanup EXIT

# Refuse to reuse a namespace: this script owns and removes all its resources.
kubectl create namespace "${NAMESPACE}"
CREATED_NAMESPACE=true
source "${SCRIPT_DIR}/lib.sh"
create_platform_secrets "${NAMESPACE}"

openssl req -x509 -newkey rsa:2048 -nodes -days 1 \
    -subj "/CN=${TLS_HOST}" -addext "subjectAltName=DNS:${TLS_HOST}" \
    -keyout "${TMP_DIR}/tls.key" -out "${TMP_DIR}/tls.crt"
openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "${TMP_DIR}/signing.pem"
kubectl -n "${NAMESPACE}" create secret tls workload-tls \
    --cert="${TMP_DIR}/tls.crt" --key="${TMP_DIR}/tls.key"
kubectl -n "${NAMESPACE}" create secret generic workload-signing \
    --from-file="private-key.pem=${TMP_DIR}/signing.pem"
kubectl -n "${NAMESPACE}" create configmap workload-ca --from-file="ca.crt=${TMP_DIR}/tls.crt"

# Resolve the upstream at request time: the API Service is created by Helm later.
cat > "${TMP_DIR}/nginx.conf" <<EOF
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
      resolver $(kubectl -n kube-system get service kube-dns -o jsonpath='{.spec.clusterIP}');
      set \$backend ${RELEASE}-api.${NAMESPACE}.svc.cluster.local:8080;
      proxy_pass http://\$backend;
    }
  }
}
EOF
kubectl -n "${NAMESPACE}" create configmap workload-gateway --from-file="nginx.conf=${TMP_DIR}/nginx.conf"
kubectl -n "${NAMESPACE}" apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${TLS_HOST}
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
          secret: {secretName: workload-tls}
        - name: config
          configMap: {name: workload-gateway}
---
apiVersion: v1
kind: Service
metadata:
  name: ${TLS_HOST}
spec:
  selector: {app: workload-tls-gateway}
  ports:
    - {port: 443, targetPort: 8443}
EOF

cat > "${TMP_DIR}/values.yaml" <<EOF
fullnameOverride: ${RELEASE}
api:
  internalBaseUrl: https://${TLS_HOST}
  extraVolumes: &volumes
    - name: workload-signing
      secret: {secretName: workload-signing}
    - name: workload-ca
      configMap: {name: workload-ca}
  extraVolumeMounts: &mounts
    - {name: workload-signing, mountPath: /etc/workload-signing, readOnly: true}
    - {name: workload-ca, mountPath: /etc/workload-ca, readOnly: true}
core:
  controller:
    extraVolumes: *volumes
    extraVolumeMounts: *mounts
platformSeedJob:
  enabled: false
env:
  NHX_CLIENT_SSL_CERT_FILE: /etc/workload-ca/ca.crt
platformConfig:
  auth:
    enabled: true
    oidc:
      workload:
        client_id: install-regression
        token_issuer: https://${TLS_HOST}
        token_key_id: install-regression
        token_private_key_file: /etc/workload-signing/private-key.pem
EOF

kubectl -n "${NAMESPACE}" rollout status "deployment/${TLS_HOST}" --timeout=120s
helm upgrade --install "${RELEASE}" "${REPO_ROOT}/k8s/helm" \
    --namespace "${NAMESPACE}" -f "${REPO_ROOT}/e2e/k8s/values/kind.yaml" \
    "$@" -f "${TMP_DIR}/values.yaml" --wait --timeout "${HELM_TIMEOUT:-10m}"
kubectl -n "${NAMESPACE}" rollout status "deployment/${RELEASE}-core-controller" --timeout=60s
printf 'PASS: controller is Ready with workload token exchange and verified HTTPS transport.\n'
