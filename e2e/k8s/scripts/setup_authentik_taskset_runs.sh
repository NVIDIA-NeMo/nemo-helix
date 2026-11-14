#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The Authentik demo on kind, with taskset runs: real OIDC users and workload token exchange, so Jobs run as the
# nemo-helix-jobs ServiceAccount for their submitter. For e2e/test_taskset_runs.py in the nightly's auth-workload
# row, and locally. Idempotent: rerun it to reapply the values after a code or image change.
#
#   1. contrib/auth/authentik/run.sh up k8s: the cluster (created, or reused), Authentik and the platform
#   2. with NHX_E2E_LOAD_IMAGES=1, load the step, builder and kaniko images from the local Docker into kind
#   3. OpenSandbox, with sandboxes pulling task images from the builder's registry
#   4. helm upgrade --reuse-values -f e2e/k8s/values/authentik-taskset-runs.yaml: evals, builder, controllers
#   5. the node pulls from the in-cluster builder registry
#   6. the gateway on https://127.0.0.1:18082, kept up
#   7. nemo-setup becomes the platform admin and creates the builder's secrets in `system`
#
# Inputs:
#   NHX_E2E_IMAGE_REGISTRY, NHX_E2E_IMAGE_TAG   the platform images (nhx-api, -tasks, -builder-tasks, -kaniko)
#   KIND_CLUSTER_NAME, NHX_E2E_STATE_DIR        see authentik_lib.sh
#   NHX_E2E_LOAD_IMAGES=1                       load the images from the local Docker (local builds). CI instead
#                                               pre-creates the cluster with its ghcr-pull secret and sets
#                                               NHX_AUTHENTIK_K8S_REUSE_CLUSTER, _IMAGE_PULL_SECRET, _SKIP_IMAGE_LOAD
#   BUILDER_REGISTRY_PASSWORD_FILE              the builder registry's password (default: generated in the state dir)
#
# Exports for the steps after it, to GITHUB_ENV in Actions and to ${NHX_E2E_STATE_DIR}/env:
#   KUBECONFIG, KUBE_CONTEXT, NHX_E2E_KUBE_CONTEXT, NHX_E2E_KUBE_NAMESPACE, NHX_E2E_CLUSTER_URL,
#   NHX_CLIENT_SSL_CERT_FILE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=authentik_lib.sh
source "${SCRIPT_DIR}/authentik_lib.sh"

: "${NHX_E2E_IMAGE_REGISTRY:?}" "${NHX_E2E_IMAGE_TAG:?}"
image() { printf '%s/%s:%s' "${NHX_E2E_IMAGE_REGISTRY}" "$1" "${NHX_E2E_IMAGE_TAG}"; }
mkdir -p "${STATE_DIR}" && chmod 700 "${STATE_DIR}"
password_file="${BUILDER_REGISTRY_PASSWORD_FILE:-${STATE_DIR}/builder-registry-password}"
[[ -s "${password_file}" ]] || (umask 077; openssl rand -hex 16 | tr -d '\n' >"${password_file}")
cd "${REPO_ROOT}"

# 1. Cluster, Authentik and the platform. run.sh's helm upgrade drops step 4's values, so step 4 always reapplies them.
log "run.sh up k8s on ${CLUSTER} with $(image nhx-api)"
helm repo add authentik https://charts.goauthentik.io --force-update >/dev/null
helm dependency build k8s/helm >/dev/null
helm dependency build contrib/auth/authentik/helm >/dev/null
NHX_AUTHENTIK_K8S_CLUSTER_NAME="${CLUSTER}" contrib/auth/authentik/run.sh up k8s --runtime kind --image "$(image nhx-api)"
[[ -f "${KUBECONFIG}" ]] || die "run.sh wrote no kubeconfig at ${KUBECONFIG}"

# 2. Local builds: the Job images by the tag the platform config names. CI pre-pulls them onto the node instead.
if [[ "${NHX_E2E_LOAD_IMAGES:-0}" == "1" ]]; then
    for name in nhx-tasks nhx-builder-tasks nhx-kaniko; do
        log "loading $(image "${name}")"
        kind load docker-image "$(image "${name}")" --name "${CLUSTER}" >/dev/null
    done
fi

# 3. OpenSandbox.
kubectl -n "${NAMESPACE}" create secret docker-registry builder-registry-pull \
    --docker-server="nemo-helix-builder-registry.${NAMESPACE}.svc.cluster.local:5000" \
    --docker-username=builder --docker-password="$(cat "${password_file}")" \
    --dry-run=client -o yaml | kubectl -n "${NAMESPACE}" apply -f - >/dev/null
if kubectl -n opensandbox-system get deploy opensandbox-server >/dev/null 2>&1; then
    log "OpenSandbox already installed"
else
    log "installing OpenSandbox"
    KUBE_NAMESPACE="${NAMESPACE}" BATCHSANDBOX_TEMPLATE=e2e/k8s/values/batchsandbox-kind-template.yaml \
        SANDBOX_PULL_SECRET=builder-registry-pull e2e/k8s/scripts/install_opensandbox_minikube.sh
fi

# 4. Evals, builder and taskset runs. No --wait: a hook can hold it for its whole timeout with every pod ready.
log "helm upgrade with e2e/k8s/values/authentik-taskset-runs.yaml"
helm -n "${NAMESPACE}" upgrade "${RELEASE}" contrib/auth/authentik/helm --reuse-values \
    -f e2e/k8s/values/authentik-taskset-runs.yaml \
    --set-file nemo-helix.builder.devRegistry.password="${password_file}" >/dev/null
kubectl -n "${NAMESPACE}" rollout status deploy/nemo-helix-api --timeout=10m
kubectl -n "${NAMESPACE}" rollout status deploy/nemo-helix-core-controller --timeout=5m

# 5. The node pulls task images from the in-cluster registry by its service name.
KIND_CLUSTER_NAME="${CLUSTER}" KUBE_NAMESPACE="${NAMESPACE}" e2e/k8s/scripts/kind_dev_registry_mirror.sh

# 6. Gateway.
gateway_keepalive
status="$(curl -sS --cacert "${CLUSTER_DIR}/ca.crt" "${GATEWAY_URL}/status")"
for service in evals builder; do
    [[ "${status}" == *"\"${service}\""* ]] || die "${service} is not running: ${status}"
done

# 7. The platform admin (no demo user has auth.admin_email) and the builder's push secrets in `system`.
authentik_admin set-email nemo-setup "${PLATFORM_ADMIN_EMAIL}"
key="${STATE_DIR}/builder-signing-key.pem"
[[ -s "${key}" ]] || (umask 077; openssl ecparam -name prime256v1 -genkey -noout | openssl pkcs8 -topk8 -nocrypt -out "${key}")
"${SCRIPT_DIR}/authentik_user_token.sh" nemo-setup "${STATE_DIR}/nemo-setup.token"
(
    export NHX_BASE_URL="${GATEWAY_URL}" NHX_CLIENT_SSL_CERT_FILE="${CLUSTER_DIR}/ca.crt" _TYPER_FORCE_DISABLE_TERMINAL=1
    NHX_ACCESS_TOKEN="$(cat "${STATE_DIR}/nemo-setup.token")"
    export NHX_ACCESS_TOKEN
    # Create, or update one a rerun left, so it always matches the password Helm gave the registry.
    put() {
        local name="$1" out
        shift
        if out="$(uv run --frozen nemo secrets create "${name}" --workspace system "$@" 2>&1)"; then
            log "created system/${name}"
        elif [[ "${out}" == *409* || "${out}" == *"already exists"* ]]; then
            out="$(uv run --frozen nemo secrets update "${name}" --workspace system "$@" 2>&1)" \
                || die "updating system/${name} failed: $(tail -3 <<<"${out}")"
            log "updated system/${name}"
        else
            die "creating system/${name} failed: $(tail -3 <<<"${out}")"
        fi
    }
    put builder-registry-username --value builder
    put builder-registry-password --from-file "${password_file}"
    put builder-signing-key --from-file "${key}"
)

exports=(
    "KUBECONFIG=${KUBECONFIG}"
    "KUBE_CONTEXT=${KUBE_CONTEXT}"
    "NHX_E2E_KUBE_CONTEXT=${KUBE_CONTEXT}"
    "NHX_E2E_KUBE_NAMESPACE=${NAMESPACE}"
    "NHX_E2E_CLUSTER_URL=${GATEWAY_URL}"
    "NHX_CLIENT_SSL_CERT_FILE=${CLUSTER_DIR}/ca.crt"
)
printf '%s\n' "${exports[@]}" >"${STATE_DIR}/env"
[[ -n "${GITHUB_ENV:-}" ]] && printf '%s\n' "${exports[@]}" >>"${GITHUB_ENV}"
log "ready on ${GATEWAY_URL}; for a shell: set -a; source ${STATE_DIR}/env; set +a"
