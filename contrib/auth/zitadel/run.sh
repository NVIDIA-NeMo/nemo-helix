#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ZITADEL_ROOT="${SCRIPT_DIR}"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

ACTION=""
TARGET=""
DRY_RUN="false"
INSTANCE_KEY=""
IMAGE_SELECTED="false"
IMAGE_REGISTRY="${IMAGE_REGISTRY:-my-registry}"
BAKE_TAG="${BAKE_TAG:-local}"
REUSE_SET="false"
TEST_PLATFORM=""
TEST_PLATFORM_SET="false"
TEST_DOCKER_TARGET="nmp-api-docker"
REUSE_K8S_CLUSTER_NAME="${NMP_ZITADEL_K8S_REUSE_CLUSTER_NAME:-}"
DEFAULT_K8S_GATEWAY_PORT="18084"
K8S_GATEWAY_PORT="${NMP_ZITADEL_K8S_GATEWAY_PORT:-}"
K8S_JUNIT_XML="${NMP_ZITADEL_K8S_JUNIT_XML:-report-auth-idp-zitadel-kubernetes.xml}"
HELM_NAMESPACE="${HELM_NAMESPACE:-${NMP_ZITADEL_K8S_NAMESPACE:-nemo-zitadel}}"
HELM_RELEASE="${HELM_RELEASE:-${NMP_ZITADEL_K8S_HELM_RELEASE:-zitadel-demo}}"
HELM_WAIT_TIMEOUT="${HELM_WAIT_TIMEOUT:-${NMP_ZITADEL_K8S_HELM_WAIT_TIMEOUT:-20m}}"
K8S_CLUSTER_NAME="${NMP_ZITADEL_K8S_CLUSTER_NAME:-}"
K8S_RUNTIME="${NMP_ZITADEL_K8S_RUNTIME:-kind}"
K8S_RUNTIME_SET="false"
K8S_KEEP_CLUSTER="${NMP_ZITADEL_K8S_KEEP_CLUSTER:-0}"
K8S_REUSE_CLUSTER="${NMP_ZITADEL_K8S_REUSE_CLUSTER:-0}"
K8S_SKIP_IMAGE_LOAD="${NMP_ZITADEL_K8S_SKIP_IMAGE_LOAD:-0}"
K8S_SKIP_IMAGE_LOAD_SET="false"
K8S_EXPORT_KUBECONFIG="${NMP_ZITADEL_K8S_EXPORT_KUBECONFIG:-0}"
K8S_EXPORT_KUBECONFIG_SET="false"
K8S_NGC_EXISTING_SECRET="${NMP_ZITADEL_K8S_NGC_EXISTING_SECRET:-}"
K8S_IMAGE_PULL_SECRET="${NMP_ZITADEL_K8S_IMAGE_PULL_SECRET:-}"
ZITADEL_WORKSPACE="${NMP_ZITADEL_WORKSPACE:-zitadel-demo}"
ZITADEL_PROVIDER_IMAGES=(
    "ghcr.io/zitadel/zitadel:v4.15.3"
    "docker.io/alpine/k8s:1.32.2"
)

diagnostics_dir() {
    local mode="$1"
    local configured="${E2E_SERVICES_LOG_DIR:-}"
    local timestamp

    if [[ -n "${configured}" ]]; then
        if [[ "${configured}" = /* ]]; then
            printf "%s" "${configured}"
        else
            printf "%s/%s" "${REPO_ROOT}" "${configured#./}"
        fi
        return
    fi

    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    printf "%s/docker/logs/zitadel-%s/%s" "${REPO_ROOT}" "${mode}" "${timestamp}"
}

prepare_diagnostics_dir() {
    local mode="$1"
    local output

    output="$(diagnostics_dir "${mode}")"
    if [[ "${DRY_RUN}" != "true" ]]; then
        mkdir -p "${output}"
    fi
    printf "%s" "${output}"
}

write_diagnostics_metadata() {
    local mode="$1"
    local output="$2"

    if [[ "${DRY_RUN}" == "true" ]]; then
        return
    fi

    {
        printf "mode=%s\n" "${mode}"
        printf "timestamp_utc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf "image=%s\n" "$(image_ref)"
        printf "k8s_runtime=%s\n" "${K8S_RUNTIME}"
        printf "k8s_cluster_name=%s\n" "${K8S_CLUSTER_NAME}"
        printf "k8s_gateway_port=%s\n" "${K8S_GATEWAY_PORT}"
        printf "k8s_namespace=%s\n" "${HELM_NAMESPACE}"
        printf "k8s_release=%s\n" "${HELM_RELEASE}"
    } >"${output}/run-metadata.txt"
}

choose_free_tcp_port() {
    python3 -c 'import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])'
}

validate_k8s_gateway_port() {
    if [[ ! "${K8S_GATEWAY_PORT}" =~ ^[0-9]+$ ]]; then
        die "NMP_ZITADEL_K8S_GATEWAY_PORT must be an integer TCP port"
    fi
    if ((K8S_GATEWAY_PORT < 1 || K8S_GATEWAY_PORT > 65535)); then
        die "NMP_ZITADEL_K8S_GATEWAY_PORT must be between 1 and 65535"
    fi
}

configure_k8s_gateway_port() {
    if [[ "${TARGET}" != "k8s" ]]; then
        return
    fi
    if [[ "${ACTION}" != "up" && "${ACTION}" != "test" ]]; then
        return
    fi
    if [[ -z "${K8S_GATEWAY_PORT}" ]]; then
        case "${ACTION}" in
            up)
                if [[ -n "${INSTANCE_KEY}" ]]; then
                    K8S_GATEWAY_PORT="$(choose_free_tcp_port)"
                else
                    K8S_GATEWAY_PORT="${DEFAULT_K8S_GATEWAY_PORT}"
                fi
                ;;
            test)
                K8S_GATEWAY_PORT="$(choose_free_tcp_port)"
                ;;
        esac
    fi
    validate_k8s_gateway_port
}

usage() {
    cat <<'EOF'
Usage:
  contrib/auth/zitadel/run.sh up k8s [options]
  contrib/auth/zitadel/run.sh test k8s [options]
  contrib/auth/zitadel/run.sh down [k8s|all] [options]
  contrib/auth/zitadel/run.sh clean [k8s|all] [options]

Runs the local Kubernetes-only ZITADEL reference example or its auth-idp test suite.

Actions:
  up k8s               Create or reuse the Kubernetes auth-idp cluster, install
                       the Helm chart, start a local gateway port-forward, and
                       skip tests.
  test k8s             Run the Helm-based kind/k3d Kubernetes E2E test.
  down k8s             Remove managed ZITADEL Kubernetes instances and contexts.
  down all             Alias for down k8s. Bare "down" defaults to "down k8s".
  clean                Remove stale recorded instances using down semantics.

Image options:
  --image IMAGE        Use an existing nmp-api image.
                       Expected format: <registry>/nmp-api:<tag>

Test options:
  --reuse              Reuse a deterministic test environment.
                       For k8s, use cluster nmp-zitadel-reuse with the selected
                       runtime, creating it if needed and keeping it after the
                       run. The up action uses this reusable resource by default.
                       The k8s up action uses gateway port 18084 by default.
                       The k8s test action chooses a free local port by default.
                       Override either with NMP_ZITADEL_K8S_GATEWAY_PORT.
  --platform PLATFORM  Platform for the default local test image build.
                       Default: current machine architecture.
  --runtime RUNTIME    Kubernetes runtime for k8s/down: kind or k3d.
                       Default: kind.
  --skip-image-load    Do not load the nmp-api image into the reused cluster.
                       For a fresh cluster, use only with an explicit pullable
                       --image.
  --export-kubeconfig  Also merge and switch the Kubernetes context into the
                       default kubeconfig. By default, k8s up writes an
                       isolated kubeconfig under .generated only.

Instance options:
  --key KEY            Manage a named up/down instance. The default instance
                       uses context zitadel-k8s. A key derives names such as
                       zitadel-k8s-KEY and matching local resources.

Other options:
  --dry-run            Print commands without running them.
  -h, --help           Show this help.

Examples:
  contrib/auth/zitadel/run.sh up k8s
  contrib/auth/zitadel/run.sh up k8s --key dev
  contrib/auth/zitadel/run.sh test k8s
  contrib/auth/zitadel/run.sh test k8s --reuse
  contrib/auth/zitadel/run.sh down k8s
  contrib/auth/zitadel/run.sh down k8s --key dev
  contrib/auth/zitadel/run.sh clean
  contrib/auth/zitadel/run.sh down
EOF
}

die() {
    echo "error: $*" >&2
    echo >&2
    usage >&2
    exit 2
}

fail() {
    echo "error: $*" >&2
    exit 1
}

image_ref() {
    printf "%s/nmp-api:%s" "${IMAGE_REGISTRY}" "${BAKE_TAG}"
}

parse_image() {
    local image="$1"

    if [[ "${image}" != */nmp-api:* ]]; then
        die "--image must use the form <registry>/nmp-api:<tag>"
    fi

    IMAGE_REGISTRY="${image%/nmp-api:*}"
    BAKE_TAG="${image##*:}"
    IMAGE_SELECTED="true"

    if [[ -z "${IMAGE_REGISTRY}" || -z "${BAKE_TAG}" ]]; then
        die "--image must include both a registry path and a tag"
    fi
}

validate_instance_key() {
    if [[ -z "${INSTANCE_KEY}" ]]; then
        return
    fi
    if [[ ! "${INSTANCE_KEY}" =~ ^[a-z0-9]([a-z0-9-]{0,30}[a-z0-9])?$ ]]; then
        die "--key must use 1-32 lowercase letters, digits, and hyphens, and start/end with a letter or digit"
    fi
}

instance_suffix() {
    if [[ -n "${INSTANCE_KEY}" ]]; then
        printf -- "-%s" "${INSTANCE_KEY}"
    fi
}

k8s_context_name() {
    printf "zitadel-k8s%s" "$(instance_suffix)"
}

key_option_suffix() {
    if [[ -n "${INSTANCE_KEY}" ]]; then
        printf " --key %s" "${INSTANCE_KEY}"
    fi
}

state_matches_instance_key() {
    local state_file="$1"
    local state_key

    if [[ -z "${INSTANCE_KEY}" ]]; then
        return 0
    fi

    state_key="$(read_lifecycle_field "${state_file}" instance_key)"
    [[ "${state_key}" == "${INSTANCE_KEY}" ]]
}

configure_instance_defaults() {
    validate_instance_key

    if [[ -z "${REUSE_K8S_CLUSTER_NAME}" ]]; then
        if [[ -n "${INSTANCE_KEY}" ]]; then
            REUSE_K8S_CLUSTER_NAME="nmp-zitadel-${INSTANCE_KEY}"
        else
            REUSE_K8S_CLUSTER_NAME="nmp-zitadel-reuse"
        fi
    fi
}

host_platform() {
    case "$(uname -m)" in
        x86_64 | amd64)
            printf "linux/amd64"
            ;;
        arm64 | aarch64)
            printf "linux/arm64"
            ;;
        *)
            die "unsupported host architecture for test image build: $(uname -m). Pass --platform explicitly."
            ;;
    esac
}

validate_k8s_runtime() {
    case "${K8S_RUNTIME}" in
        kind | k3d)
            ;;
        *)
            die "--runtime must be kind or k3d"
            ;;
    esac
}

quote_args() {
    local arg

    for arg in "$@"; do
        printf "%q " "${arg}"
    done
}

print_command_in_dir() {
    local dir="$1"
    shift

    printf "+ cd %q && " "${dir}"
    quote_args "$@"
    printf "\n"
}

print_command() {
    printf "+ "
    quote_args "$@"
    printf "\n"
}

workload_token_private_key_file() {
    printf "%s/.generated/workload-token-private-key.pem" "${ZITADEL_ROOT}"
}

lifecycle_state_dir() {
    local configured="${NEMO_ZITADEL_STATE_DIR:-./.generated/instances}"
    if [[ "${configured}" = /* ]]; then
        printf "%s" "${configured}"
    else
        printf "%s/%s" "${ZITADEL_ROOT}" "${configured#./}"
    fi
}

safe_state_id() {
    printf "%s" "$1" | tr -c '[:alnum:]_.-' '_'
}

lifecycle_state_file() {
    local target="$1"
    local instance_id="$2"

    printf "%s/%s-%s.json" "$(lifecycle_state_dir)" "${target}" "$(safe_state_id "${instance_id}")"
}

write_lifecycle_state() {
    local target="$1"
    local instance_id="$2"
    local state_file
    shift 2

    state_file="$(lifecycle_state_file "${target}" "${instance_id}")"
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf "+ mkdir -p %q\n" "$(dirname -- "${state_file}")"
        printf "+ write lifecycle state %q\n" "${state_file}"
        return
    fi

    mkdir -p "$(dirname -- "${state_file}")"
    python3 - "${state_file}" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone

path = sys.argv[1]
data = {
    "version": 1,
    "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}
for item in sys.argv[2:]:
    key, value = item.split("=", 1)
    data[key] = value

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

read_lifecycle_field() {
    local state_file="$1"
    local field="$2"

    python3 - "${state_file}" "${field}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    value = json.load(f).get(sys.argv[2], "")
if value is None:
    value = ""
print(value)
PY
}

remove_lifecycle_state_file() {
    local state_file="$1"

    if [[ "${DRY_RUN}" == "true" ]]; then
        printf "+ rm -f %q\n" "${state_file}"
        return
    fi

    rm -f "${state_file}"
}

ensure_workload_token_private_key() {
    local output
    local output_dir

    output="$(workload_token_private_key_file)"
    output_dir="$(dirname -- "${output}")"

    if [[ -f "${output}" ]]; then
        echo "Using workload token signing key: ${output}"
        return
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        printf "+ mkdir -p %q\n" "${output_dir}"
        printf "+ openssl genrsa -out %q 2048\n" "${output}"
        printf "+ chmod 600 %q\n" "${output}"
        return
    fi

    if ! command -v openssl >/dev/null 2>&1; then
        fail "openssl is required to generate the workload token signing key"
    fi

    mkdir -p "${output_dir}"
    openssl genrsa -out "${output}" 2048 >/dev/null 2>&1
    chmod 600 "${output}"
    echo "Generated workload token signing key: ${output}"
}

run_in_repo() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        print_command_in_dir "${REPO_ROOT}" "$@"
        return
    fi

    (cd "${REPO_ROOT}" && "$@")
}

run_command() {
    if [[ "${DRY_RUN}" == "true" ]]; then
        print_command "$@"
        return
    fi

    "$@"
}

register_nemo_context() {
    local context_name="$1"
    local gateway_url="$2"
    local ca_bundle="$3"

    echo "Registering NeMo context ${context_name}: ${gateway_url}"
    run_in_repo uv run --frozen nemo config set \
        --context "${context_name}" \
        --base-url "${gateway_url}" \
        --certificate-authority "${ca_bundle}" \
        --workspace "${ZITADEL_WORKSPACE}"
}

delete_nemo_context() {
    local context_name="$1"
    local status

    if [[ -z "${context_name}" ]]; then
        return
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        run_in_repo uv run --frozen nemo config delete-context "${context_name}" --prune-orphans
        return
    fi

    set +e
    (cd "${REPO_ROOT}" && uv run --frozen nemo config delete-context "${context_name}" --prune-orphans)
    status="$?"
    set -e
    if [[ "${status}" -ne 0 ]]; then
        echo "Warning: failed to delete NeMo context ${context_name}" >&2
    fi
}

wait_for_https_ready() {
    local url="$1"
    local ca_cert="$2"
    local timeout_seconds="${3:-180}"
    local started="${SECONDS}"

    if [[ "${DRY_RUN}" == "true" ]]; then
        print_command curl -fsS --cacert "${ca_cert}" "${url}"
        return
    fi

    if ! command -v curl >/dev/null 2>&1; then
        fail "curl is required to wait for ${url}"
    fi

    until curl -fsS --cacert "${ca_cert}" "${url}" >/dev/null 2>&1; do
        if ((SECONDS - started >= timeout_seconds)); then
            echo "timed out waiting for ${url}" >&2
            return 1
        fi
        sleep 2
    done
}

k8s_state_dir_for_cluster() {
    local cluster_name="$1"
    printf "%s/.generated/k8s/%s" "${ZITADEL_ROOT}" "${cluster_name}"
}

k8s_kubeconfig_file_for_cluster() {
    local cluster_name="$1"
    printf "%s/kubeconfig.yaml" "$(k8s_state_dir_for_cluster "${cluster_name}")"
}

k8s_ca_bundle_file_for_cluster() {
    local cluster_name="$1"
    printf "%s/ca.crt" "$(k8s_state_dir_for_cluster "${cluster_name}")"
}

k8s_port_forward_pid_file_for_cluster() {
    local cluster_name="$1"
    printf "%s/port-forward.pid" "$(k8s_state_dir_for_cluster "${cluster_name}")"
}

k8s_port_forward_log_file_for_cluster() {
    local cluster_name="$1"
    printf "%s/port-forward.log" "$(k8s_state_dir_for_cluster "${cluster_name}")"
}

k8s_context_for_cluster() {
    local runtime="$1"
    local cluster_name="$2"

    case "${runtime}" in
        kind)
            printf "kind-%s" "${cluster_name}"
            ;;
        k3d)
            printf "k3d-%s" "${cluster_name}"
            ;;
        *)
            die "--runtime must be kind or k3d"
            ;;
    esac
}

ensure_k8s_state_dir() {
    local cluster_name="$1"
    local state_dir

    state_dir="$(k8s_state_dir_for_cluster "${cluster_name}")"
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf "+ mkdir -p %q\n" "${state_dir}"
        return
    fi

    mkdir -p "${state_dir}"
}

k8s_write_existing_kubeconfig() {
    local cluster_name="$1"
    local kubeconfig="$2"

    case "${K8S_RUNTIME}" in
        kind)
            kind get kubeconfig --name "${cluster_name}" >"${kubeconfig}"
            ;;
        k3d)
            k3d kubeconfig get "${cluster_name}" >"${kubeconfig}"
            ;;
    esac
}

k8s_create_cluster() {
    local cluster_name="$1"
    local kubeconfig="$2"

    case "${K8S_RUNTIME}" in
        kind)
            run_command kind create cluster --name "${cluster_name}" --kubeconfig "${kubeconfig}" --wait 180s
            ;;
        k3d)
            run_command k3d cluster create "${cluster_name}" \
                --wait \
                --agents 0 \
                --k3s-arg "--disable=traefik@server:0" \
                --kubeconfig-update-default=false
            if [[ "${DRY_RUN}" == "true" ]]; then
                print_command k3d kubeconfig get "${cluster_name}"
                printf "+ write %q\n" "${kubeconfig}"
            else
                k3d kubeconfig get "${cluster_name}" >"${kubeconfig}"
            fi
            ;;
    esac
}

k8s_update_default_kubeconfig() {
    local cluster_name="$1"
    local context="$2"

    case "${K8S_RUNTIME}" in
        kind)
            run_command kind export kubeconfig --name "${cluster_name}"
            ;;
        k3d)
            run_command k3d kubeconfig merge "${cluster_name}" \
                --kubeconfig-merge-default \
                --kubeconfig-switch-context
            ;;
    esac
    run_command kubectl config use-context "${context}"
}

k8s_ensure_cluster() {
    local cluster_name="$1"
    local kubeconfig="$2"

    ensure_k8s_state_dir "${cluster_name}"
    if [[ "${DRY_RUN}" == "true" ]]; then
        case "${K8S_RUNTIME}" in
            kind)
                print_command kind get kubeconfig --name "${cluster_name}"
                ;;
            k3d)
                print_command k3d kubeconfig get "${cluster_name}"
                ;;
        esac
        printf "+ if cluster is missing:\n"
        k8s_create_cluster "${cluster_name}" "${kubeconfig}"
        return
    fi

    if k8s_write_existing_kubeconfig "${cluster_name}" "${kubeconfig}" 2>/dev/null; then
        echo "Using existing ${K8S_RUNTIME} cluster: ${cluster_name}"
        return
    fi

    echo "Creating ${K8S_RUNTIME} cluster: ${cluster_name}"
    k8s_create_cluster "${cluster_name}" "${kubeconfig}"
}

k8s_load_image_ref() {
    local cluster_name="$1"
    local image="$2"

    case "${K8S_RUNTIME}" in
        kind)
            run_command kind load docker-image "${image}" --name "${cluster_name}"
            ;;
        k3d)
            run_command k3d image import "${image}" -c "${cluster_name}"
            ;;
    esac
}

k8s_load_image() {
    local cluster_name="$1"

    if [[ "${K8S_SKIP_IMAGE_LOAD}" == "1" ]]; then
        echo "Skipping Kubernetes image load for: $(image_ref)"
        return
    fi

    k8s_load_image_ref "${cluster_name}" "$(image_ref)"
}

k8s_load_provider_image_if_present() {
    local cluster_name="$1"
    local image="$2"

    if [[ "${K8S_SKIP_IMAGE_LOAD}" == "1" ]]; then
        echo "Skipping Kubernetes provider image load for: ${image}"
        return
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        print_command docker image inspect "${image}"
        printf "+ if provider image exists locally:\n"
        k8s_load_image_ref "${cluster_name}" "${image}"
        return
    fi

    if docker image inspect "${image}" >/dev/null 2>&1; then
        k8s_load_image_ref "${cluster_name}" "${image}"
    fi
}

k8s_load_provider_images() {
    local cluster_name="$1"
    local image

    for image in "${ZITADEL_PROVIDER_IMAGES[@]}"; do
        k8s_load_provider_image_if_present "${cluster_name}" "${image}"
    done
}

k8s_helm_command() {
    local context="$1"
    local kubeconfig="$2"
    shift 2

    run_in_repo helm --kubeconfig "${kubeconfig}" --kube-context "${context}" "$@"
}

k8s_kubectl_command() {
    local context="$1"
    local kubeconfig="$2"
    shift 2

    run_command kubectl --kubeconfig "${kubeconfig}" --context "${context}" "$@"
}

k8s_helm_install() {
    local cluster_name="$1"
    local context="$2"
    local kubeconfig="$3"
    local image
    local registry
    local tag
    local workload_token_private_key
    local -a args

    image="$(image_ref)"
    registry="${image%/nmp-api:*}"
    tag="${image##*:}"
    workload_token_private_key="$(workload_token_private_key_file)"

    run_in_repo helm repo add nvidia https://helm.ngc.nvidia.com/nvidia --force-update
    run_in_repo helm repo add zitadel https://charts.zitadel.com --force-update
    run_in_repo helm dependency build k8s/helm
    run_in_repo helm dependency build contrib/auth/zitadel/helm

    args=(
        upgrade
        --install
        "${HELM_RELEASE}"
        "${ZITADEL_ROOT}/helm"
        --namespace
        "${HELM_NAMESPACE}"
        --create-namespace
        --wait
        --wait-for-jobs
        --timeout
        "${HELM_WAIT_TIMEOUT}"
        --set
        "nemo-platform.api.image.repository=${registry}/nmp-api"
        --set
        "nemo-platform.api.image.tag=${tag}"
        --set
        "nemo-platform.core.image.repository=${registry}/nmp-api"
        --set
        "nemo-platform.core.image.tag=${tag}"
        --set-string
        "nemo-platform.platformConfig.platform.image_registry=${registry}"
        --set-string
        "nemo-platform.platformConfig.platform.image_tag=${tag}"
        --set-string
        "nemo-platform.platformConfig.auth.access_keys.enabled=true"
        --set-string
        "nemo-platform.zitadelPublicGateway.port=${K8S_GATEWAY_PORT}"
        --set-file
        "workloadTokenSigningKey.privateKeyPem=${workload_token_private_key}"
    )
    if [[ -n "${K8S_NGC_EXISTING_SECRET}" ]]; then
        args+=(--set-string "nemo-platform.existingSecret=${K8S_NGC_EXISTING_SECRET}")
    fi
    if [[ -n "${K8S_IMAGE_PULL_SECRET}" ]]; then
        args+=(--set-string "nemo-platform.imagePullSecrets[0].name=${K8S_IMAGE_PULL_SECRET}")
    fi

    echo "Installing ZITADEL Kubernetes demo into ${cluster_name}/${HELM_NAMESPACE}"
    k8s_helm_command "${context}" "${kubeconfig}" "${args[@]}"
}

k8s_wait_for_zitadel() {
    local context="$1"
    local kubeconfig="$2"
    local deployment

    for deployment in zitadel nemo-platform-api nemo-platform-core-controller nemo-platform-envoy; do
        k8s_kubectl_command "${context}" "${kubeconfig}" \
            -n "${HELM_NAMESPACE}" rollout status "deploy/${deployment}" --timeout=240s
    done
}

k8s_write_ca_bundle() {
    local context="$1"
    local kubeconfig="$2"
    local ca_bundle="$3"
    local encoded

    if [[ "${DRY_RUN}" == "true" ]]; then
        print_command kubectl --kubeconfig "${kubeconfig}" --context "${context}" \
            -n "${HELM_NAMESPACE}" get secret nemo-platform-envoy-tls -o "jsonpath={.data.ca\\.crt}"
        printf "+ write %q\n" "${ca_bundle}"
        return
    fi

    encoded="$(
        kubectl --kubeconfig "${kubeconfig}" --context "${context}" \
            -n "${HELM_NAMESPACE}" get secret nemo-platform-envoy-tls -o "jsonpath={.data.ca\\.crt}"
    )"
    if [[ -z "${encoded}" ]]; then
        fail "secret nemo-platform-envoy-tls in ${HELM_NAMESPACE} has no ca.crt entry"
    fi
    if printf "%s" "${encoded}" | base64 --decode >"${ca_bundle}" 2>/dev/null; then
        return
    fi
    if printf "%s" "${encoded}" | base64 -D >"${ca_bundle}" 2>/dev/null; then
        return
    fi
    fail "failed to decode Kubernetes gateway CA bundle"
}

stop_k8s_port_forward_for_cluster() {
    local cluster_name="$1"
    local pid_file
    local pid

    pid_file="$(k8s_port_forward_pid_file_for_cluster "${cluster_name}")"
    if [[ "${DRY_RUN}" == "true" ]]; then
        printf "+ stop Kubernetes gateway port-forward recorded in %q if running\n" "${pid_file}"
        return
    fi
    if [[ ! -f "${pid_file}" ]]; then
        return
    fi

    pid="$(<"${pid_file}")"
    if k8s_port_forward_pid_is_running "${pid}"; then
        echo "Stopping Kubernetes gateway port-forward: ${pid}"
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
}

k8s_port_forward_pid_is_running() {
    local pid="$1"
    local expected_port="${2:-}"
    local args

    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    args="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    [[ "${args}" == *"kubectl"* ]] || return 1
    [[ "${args}" == *"port-forward"* ]] || return 1
    [[ "${args}" == *"svc/nemo-platform-envoy"* ]] || return 1
    if [[ -n "${expected_port}" ]]; then
        [[ " ${args} " == *" ${expected_port}:8080 "* ]] || return 1
    fi
}

show_k8s_port_forward_log_tail() {
    local log_file="$1"

    if [[ ! -f "${log_file}" ]]; then
        echo "Kubernetes gateway port-forward log not found: ${log_file}" >&2
        return
    fi
    if [[ ! -s "${log_file}" ]]; then
        echo "Kubernetes gateway port-forward log is empty: ${log_file}" >&2
        return
    fi

    echo "Kubernetes gateway port-forward log tail (${log_file}):" >&2
    tail -n 40 "${log_file}" >&2 || true
}

k8s_wait_for_port_forward_ready() {
    local gateway_url="$1"
    local ca_bundle="$2"
    local log_file="$3"

    if wait_for_https_ready "${gateway_url}/health/gateway/ready" "${ca_bundle}" 30; then
        return
    fi
    show_k8s_port_forward_log_tail "${log_file}"
    fail "timed out waiting for Kubernetes gateway port-forward readiness"
}

k8s_start_port_forward() {
    local cluster_name="$1"
    local context="$2"
    local kubeconfig="$3"
    local ca_bundle="$4"
    local pid_file
    local log_file
    local pid=""
    local gateway_url="https://127.0.0.1:${K8S_GATEWAY_PORT}"

    pid_file="$(k8s_port_forward_pid_file_for_cluster "${cluster_name}")"
    log_file="$(k8s_port_forward_log_file_for_cluster "${cluster_name}")"

    if [[ "${DRY_RUN}" == "true" ]]; then
        printf "+ nohup "
        quote_args kubectl --kubeconfig "${kubeconfig}" --context "${context}" \
            -n "${HELM_NAMESPACE}" port-forward svc/nemo-platform-envoy "${K8S_GATEWAY_PORT}:8080"
        printf "> %q 2>&1 &\n" "${log_file}"
        printf "+ write %q\n" "${pid_file}"
        wait_for_https_ready "${gateway_url}/health/gateway/ready" "${ca_bundle}" 30
        return
    fi

    if [[ -f "${pid_file}" ]]; then
        pid="$(<"${pid_file}")"
    fi
    if k8s_port_forward_pid_is_running "${pid}" "${K8S_GATEWAY_PORT}"; then
        echo "Using existing Kubernetes gateway port-forward: ${pid}"
        k8s_wait_for_port_forward_ready "${gateway_url}" "${ca_bundle}" "${log_file}"
        return
    fi
    if [[ -n "${pid}" ]]; then
        stop_k8s_port_forward_for_cluster "${cluster_name}"
    fi

    nohup kubectl --kubeconfig "${kubeconfig}" --context "${context}" \
        -n "${HELM_NAMESPACE}" port-forward svc/nemo-platform-envoy "${K8S_GATEWAY_PORT}:8080" \
        >"${log_file}" 2>&1 &
    printf "%s\n" "$!" >"${pid_file}"
    k8s_wait_for_port_forward_ready "${gateway_url}" "${ca_bundle}" "${log_file}"
}

k8s_up() {
    local cluster_name
    local context
    local nemo_context
    local kubeconfig
    local ca_bundle
    local gateway_url="https://127.0.0.1:${K8S_GATEWAY_PORT}"

    validate_k8s_runtime
    ensure_workload_token_private_key
    cluster_name="${K8S_CLUSTER_NAME:-${REUSE_K8S_CLUSTER_NAME}}"
    K8S_CLUSTER_NAME="${cluster_name}"
    context="$(k8s_context_for_cluster "${K8S_RUNTIME}" "${cluster_name}")"
    nemo_context="$(k8s_context_name)"
    kubeconfig="$(k8s_kubeconfig_file_for_cluster "${cluster_name}")"
    ca_bundle="$(k8s_ca_bundle_file_for_cluster "${cluster_name}")"

    if [[ "${IMAGE_SELECTED}" == "true" ]]; then
        echo "Using prebuilt auth-idp Kubernetes image: $(image_ref)"
    elif [[ "${K8S_SKIP_IMAGE_LOAD}" == "1" ]]; then
        echo "Reusing Kubernetes cluster without rebuilding or loading image: $(image_ref)"
    else
        build_default_test_image
    fi

    k8s_ensure_cluster "${cluster_name}" "${kubeconfig}"
    if [[ "${K8S_EXPORT_KUBECONFIG}" == "1" ]]; then
        k8s_update_default_kubeconfig "${cluster_name}" "${context}"
    fi
    k8s_load_image "${cluster_name}"
    k8s_load_provider_images "${cluster_name}"
    k8s_helm_install "${cluster_name}" "${context}" "${kubeconfig}"
    k8s_wait_for_zitadel "${context}" "${kubeconfig}"
    k8s_write_ca_bundle "${context}" "${kubeconfig}" "${ca_bundle}"
    k8s_start_port_forward "${cluster_name}" "${context}" "${kubeconfig}" "${ca_bundle}"
    register_nemo_context "${nemo_context}" "${gateway_url}" "${ca_bundle}"
    write_lifecycle_state k8s "${K8S_RUNTIME}-${cluster_name}" \
        "target=k8s" \
        "instance_key=${INSTANCE_KEY}" \
        "context_name=${nemo_context}" \
        "gateway_url=${gateway_url}" \
        "certificate_authority=${ca_bundle}" \
        "workspace=${ZITADEL_WORKSPACE}" \
        "runtime=${K8S_RUNTIME}" \
        "cluster_name=${cluster_name}" \
        "kubernetes_context=${context}" \
        "kubeconfig=${kubeconfig}" \
        "namespace=${HELM_NAMESPACE}" \
        "helm_release=${HELM_RELEASE}" \
        "gateway_port=${K8S_GATEWAY_PORT}" \
        "port_forward_pid_file=$(k8s_port_forward_pid_file_for_cluster "${cluster_name}")"

    echo "ZITADEL Kubernetes is ready: ${gateway_url}"
    echo "NeMo context: ${nemo_context}"
    echo "Cluster: ${cluster_name}"
    echo "Kubernetes context: ${context}"
    echo "Kubeconfig: ${kubeconfig}"
    echo "Use it with: KUBECONFIG=${kubeconfig} kubectl -n ${HELM_NAMESPACE} get pods"
    echo "Gateway CA bundle: ${ca_bundle}"
    echo "Lifecycle state: $(lifecycle_state_file k8s "${K8S_RUNTIME}-${cluster_name}")"
    echo "Stop it with: ${SCRIPT_DIR}/run.sh down k8s$(key_option_suffix)"
}

delete_reuse_k8s_cluster() {
    local cluster_name="${K8S_CLUSTER_NAME:-${REUSE_K8S_CLUSTER_NAME}}"
    local -a delete_command
    local status

    validate_k8s_runtime

    case "${K8S_RUNTIME}" in
        kind)
            delete_command=(kind delete cluster --name "${cluster_name}")
            ;;
        k3d)
            delete_command=(k3d cluster delete "${cluster_name}")
            ;;
    esac

    if [[ "${DRY_RUN}" == "true" ]]; then
        print_command "${delete_command[@]}"
        return
    fi

    if ! command -v "${delete_command[0]}" >/dev/null 2>&1; then
        echo "Skipping reused Kubernetes test cluster cleanup: ${delete_command[0]} not found"
        return
    fi

    set +e
    "${delete_command[@]}"
    status="$?"
    set -e
    if [[ "${status}" -ne 0 ]]; then
        echo "Warning: failed to delete reused Kubernetes test cluster ${cluster_name} with ${K8S_RUNTIME}" >&2
    fi
}

k8s_down() {
    local status=0
    local state_dir
    local state_file
    local found_state="false"
    local context_name
    local original_runtime="${K8S_RUNTIME}"
    local original_cluster_name="${K8S_CLUSTER_NAME}"

    state_dir="$(lifecycle_state_dir)"
    for state_file in "${state_dir}"/k8s-*.json; do
        [[ -e "${state_file}" ]] || continue
        state_matches_instance_key "${state_file}" || continue
        found_state="true"
        context_name="$(read_lifecycle_field "${state_file}" context_name)"
        K8S_RUNTIME="$(read_lifecycle_field "${state_file}" runtime)"
        K8S_CLUSTER_NAME="$(read_lifecycle_field "${state_file}" cluster_name)"

        echo "Stopping ZITADEL Kubernetes instance: ${K8S_RUNTIME}/${K8S_CLUSTER_NAME}"
        stop_k8s_port_forward_for_cluster "${K8S_CLUSTER_NAME}" || status="$?"
        delete_reuse_k8s_cluster || status="$?"
        delete_nemo_context "${context_name}"
        remove_lifecycle_state_file "${state_file}"
    done

    if [[ "${found_state}" != "true" ]]; then
        K8S_CLUSTER_NAME="${K8S_CLUSTER_NAME:-${REUSE_K8S_CLUSTER_NAME}}"
        stop_k8s_port_forward_for_cluster "${K8S_CLUSTER_NAME}" || status="$?"
        delete_reuse_k8s_cluster || status="$?"
        delete_nemo_context "$(k8s_context_name)"
    fi

    K8S_RUNTIME="${original_runtime}"
    K8S_CLUSTER_NAME="${original_cluster_name}"
    return "${status}"
}

down() {
    local target="${1:-k8s}"

    case "${target}" in
        k8s | all)
            k8s_down
            ;;
        *)
            die "down target must be k8s or all"
            ;;
    esac
}

build_default_test_image() {
    local platform="${TEST_PLATFORM:-$(host_platform)}"
    local -a args

    echo "Building auth-idp test image for ${platform}: $(image_ref)"
    args=(docker buildx bake)
    if [[ -n "${DOCKER_BAKE_ALLOW_FS_READ:-}" ]]; then
        args+=("--allow=fs.read=${DOCKER_BAKE_ALLOW_FS_READ}")
    fi
    args+=(
        -f
        docker-bake.hcl
        "${TEST_DOCKER_TARGET}"
        --set
        "*.platform=${platform}"
        --load
    )
    run_in_repo env "IMAGE_REGISTRY=${IMAGE_REGISTRY}" "BAKE_TAG=${BAKE_TAG}" "${args[@]}"
}

run_pytest_with_diagnostics() {
    local diagnostics="$1"
    local status
    shift

    if [[ "${DRY_RUN}" == "true" ]]; then
        print_command_in_dir "${REPO_ROOT}" "$@"
        return
    fi

    set +e
    (
        cd "${REPO_ROOT}"
        set +e
        "$@" 2>&1 | tee "${diagnostics}/pytest.log"
        status="${PIPESTATUS[0]}"
        set -e
        exit "${status}"
    )
    status="$?"
    set -e
    return "${status}"
}

run_k8s_tests() {
    local diagnostics
    local k8s_diagnostics
    local workload_token_private_key
    local status

    validate_k8s_runtime
    ensure_workload_token_private_key
    workload_token_private_key="$(workload_token_private_key_file)"
    diagnostics="$(prepare_diagnostics_dir kubernetes)"
    write_diagnostics_metadata kubernetes "${diagnostics}"
    echo "Auth-idp Kubernetes diagnostics: ${diagnostics}"

    if [[ "${IMAGE_SELECTED}" == "true" ]]; then
        echo "Using prebuilt auth-idp Kubernetes test image: $(image_ref)"
    elif [[ "${K8S_REUSE_CLUSTER}" == "1" && "${K8S_SKIP_IMAGE_LOAD}" == "1" ]]; then
        echo "Reusing Kubernetes cluster without rebuilding or loading image: $(image_ref)"
    else
        build_default_test_image
    fi

    k8s_diagnostics="${diagnostics}/kubernetes"
    if [[ "${DRY_RUN}" != "true" ]]; then
        mkdir -p "${k8s_diagnostics}"
    fi

    if [[ "${DRY_RUN}" == "true" ]]; then
        run_pytest_with_diagnostics "${diagnostics}" \
            env "IMAGE_REGISTRY=${IMAGE_REGISTRY}" "BAKE_TAG=${BAKE_TAG}" \
            "E2E_SERVICES_LOG_DIR=${diagnostics}" \
            "NMP_ZITADEL_K8S_LOG_DIR=${k8s_diagnostics}" \
            "NMP_ZITADEL_K8S_HELM_RELEASE=${HELM_RELEASE}" \
            "NMP_ZITADEL_K8S_HELM_WAIT_TIMEOUT=${HELM_WAIT_TIMEOUT}" \
            "NMP_ZITADEL_K8S_NAMESPACE=${HELM_NAMESPACE}" \
            "NMP_ZITADEL_K8S_RUNTIME=${K8S_RUNTIME}" \
            "NMP_ZITADEL_K8S_CLUSTER_NAME=${K8S_CLUSTER_NAME}" \
            "NMP_ZITADEL_K8S_GATEWAY_PORT=${K8S_GATEWAY_PORT}" \
            "NMP_ZITADEL_K8S_KEEP_CLUSTER=${K8S_KEEP_CLUSTER}" \
            "NMP_ZITADEL_K8S_REUSE_CLUSTER=${K8S_REUSE_CLUSTER}" \
            "NMP_ZITADEL_K8S_SKIP_IMAGE_LOAD=${K8S_SKIP_IMAGE_LOAD}" \
            "NMP_ZITADEL_K8S_NGC_EXISTING_SECRET=${K8S_NGC_EXISTING_SECRET}" \
            "NMP_ZITADEL_K8S_IMAGE_PULL_SECRET=${K8S_IMAGE_PULL_SECRET}" \
            "NMP_ZITADEL_K8S_WORKLOAD_TOKEN_PRIVATE_KEY_FILE=${workload_token_private_key}" \
            uv run --frozen pytest tests/auth_idp/contracts -v --auth-idp-runtime zitadel-kubernetes -m auth_idp_runtime --junitxml="${K8S_JUNIT_XML}"
        return
    fi

    echo "Writing ZITADEL Kubernetes diagnostics to: ${diagnostics}"
    run_pytest_with_diagnostics "${diagnostics}" \
        env "IMAGE_REGISTRY=${IMAGE_REGISTRY}" "BAKE_TAG=${BAKE_TAG}" \
        "E2E_SERVICES_LOG_DIR=${diagnostics}" \
        "NMP_ZITADEL_K8S_LOG_DIR=${k8s_diagnostics}" \
        "NMP_ZITADEL_K8S_HELM_RELEASE=${HELM_RELEASE}" \
        "NMP_ZITADEL_K8S_HELM_WAIT_TIMEOUT=${HELM_WAIT_TIMEOUT}" \
        "NMP_ZITADEL_K8S_NAMESPACE=${HELM_NAMESPACE}" \
        "NMP_ZITADEL_K8S_RUNTIME=${K8S_RUNTIME}" \
        "NMP_ZITADEL_K8S_CLUSTER_NAME=${K8S_CLUSTER_NAME}" \
        "NMP_ZITADEL_K8S_GATEWAY_PORT=${K8S_GATEWAY_PORT}" \
        "NMP_ZITADEL_K8S_KEEP_CLUSTER=${K8S_KEEP_CLUSTER}" \
        "NMP_ZITADEL_K8S_REUSE_CLUSTER=${K8S_REUSE_CLUSTER}" \
        "NMP_ZITADEL_K8S_SKIP_IMAGE_LOAD=${K8S_SKIP_IMAGE_LOAD}" \
        "NMP_ZITADEL_K8S_NGC_EXISTING_SECRET=${K8S_NGC_EXISTING_SECRET}" \
        "NMP_ZITADEL_K8S_IMAGE_PULL_SECRET=${K8S_IMAGE_PULL_SECRET}" \
        "NMP_ZITADEL_K8S_WORKLOAD_TOKEN_PRIVATE_KEY_FILE=${workload_token_private_key}" \
        uv run --frozen pytest tests/auth_idp/contracts -v --auth-idp-runtime zitadel-kubernetes -m auth_idp_runtime --junitxml="${K8S_JUNIT_XML}"
    status="$?"
    echo "Auth-idp Kubernetes diagnostics: ${diagnostics}"
    return "${status}"
}

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        up | test)
            if [[ -n "${ACTION}" ]]; then
                die "only one action can be specified"
            fi
            ACTION="$1"
            shift
            [[ $# -ge 1 ]] || die "${ACTION} requires a target: k8s"
            case "$1" in
                k8s)
                    TARGET="$1"
                    shift
                    ;;
                *)
                    die "${ACTION} target must be k8s"
                    ;;
            esac
            ;;
        down | clean)
            if [[ -n "${ACTION}" ]]; then
                die "only one action can be specified"
            fi
            ACTION="$1"
            TARGET="k8s"
            shift
            if [[ $# -gt 0 ]]; then
                case "$1" in
                    k8s | all)
                        TARGET="$1"
                        shift
                        ;;
                esac
            fi
            ;;
        --image)
            [[ $# -ge 2 ]] || die "--image requires a value"
            parse_image "$2"
            shift 2
            ;;
        --platform)
            [[ $# -ge 2 ]] || die "--platform requires a value"
            TEST_PLATFORM="$2"
            TEST_PLATFORM_SET="true"
            shift 2
            ;;
        --runtime)
            [[ $# -ge 2 ]] || die "--runtime requires a value"
            K8S_RUNTIME="$2"
            K8S_RUNTIME_SET="true"
            shift 2
            ;;
        --runtime=*)
            K8S_RUNTIME="${1#*=}"
            K8S_RUNTIME_SET="true"
            shift
            ;;
        --reuse)
            REUSE_SET="true"
            shift
            ;;
        --skip-image-load)
            K8S_SKIP_IMAGE_LOAD="1"
            K8S_SKIP_IMAGE_LOAD_SET="true"
            shift
            ;;
        --export-kubeconfig)
            K8S_EXPORT_KUBECONFIG="1"
            K8S_EXPORT_KUBECONFIG_SET="true"
            shift
            ;;
        --key)
            [[ $# -ge 2 ]] || die "--key requires a value"
            INSTANCE_KEY="$2"
            shift 2
            ;;
        --key=*)
            INSTANCE_KEY="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

if [[ -z "${ACTION}" ]]; then
    die "missing action"
fi

if [[ -n "${INSTANCE_KEY}" &&
    "${ACTION}" != "up" &&
    "${ACTION}" != "down" &&
    "${ACTION}" != "clean" ]]; then
    die "--key is only valid with up, down, or clean"
fi

configure_instance_defaults

if [[ "${REUSE_SET}" == "true" ]]; then
    case "${TARGET}" in
        k8s)
            K8S_REUSE_CLUSTER="1"
            K8S_KEEP_CLUSTER="1"
            if [[ -z "${K8S_CLUSTER_NAME}" ]]; then
                K8S_CLUSTER_NAME="${REUSE_K8S_CLUSTER_NAME}"
            fi
            ;;
        *)
            die "--reuse is only valid with k8s"
            ;;
    esac
fi

if [[ "${ACTION}" == "up" ]]; then
    case "${TARGET}" in
        k8s)
            K8S_REUSE_CLUSTER="1"
            K8S_KEEP_CLUSTER="1"
            if [[ -z "${K8S_CLUSTER_NAME}" ]]; then
                K8S_CLUSTER_NAME="${REUSE_K8S_CLUSTER_NAME}"
            fi
            ;;
    esac
fi

if [[ "${TARGET}" == "k8s" && "${K8S_REUSE_CLUSTER}" == "1" && -z "${K8S_CLUSTER_NAME}" ]]; then
    K8S_CLUSTER_NAME="${REUSE_K8S_CLUSTER_NAME}"
fi

if [[ -z "${IMAGE_REGISTRY}" || -z "${BAKE_TAG}" ]]; then
    die "image registry and tag must be non-empty"
fi

if [[ "${TEST_PLATFORM_SET}" == "true" && "${TARGET}" != "k8s" ]]; then
    die "--platform is only valid with the k8s action"
fi

if [[ "${K8S_RUNTIME_SET}" == "true" && "${TARGET}" != "k8s" && "${ACTION}" != "down" && "${ACTION}" != "clean" ]]; then
    die "--runtime is only valid with k8s, down, or clean"
fi

if [[ "${K8S_SKIP_IMAGE_LOAD_SET}" == "true" && "${TARGET}" != "k8s" ]]; then
    die "--skip-image-load is only valid with k8s"
fi

if [[ "${K8S_EXPORT_KUBECONFIG_SET}" == "true" ]]; then
    if [[ "${ACTION}" != "up" || "${TARGET}" != "k8s" ]]; then
        die "--export-kubeconfig is only valid with up k8s"
    fi
fi

if [[ "${TARGET}" == "k8s" &&
    "${K8S_SKIP_IMAGE_LOAD}" == "1" &&
    "${K8S_REUSE_CLUSTER}" != "1" &&
    "${IMAGE_SELECTED}" != "true" ]]; then
    die "--skip-image-load with a fresh k8s cluster requires a pullable image selected with" \
        "--image, or a reused cluster via --reuse"
fi

configure_k8s_gateway_port

case "${ACTION}" in
    up)
        k8s_up
        ;;
    test)
        run_k8s_tests
        ;;
    down | clean)
        down "${TARGET}"
        ;;
esac
