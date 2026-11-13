#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install shared-kernel OpenSandbox on a kind cluster made by setup_local_kind_cpu.sh, for the
# sandboxed Gym e2e (the gym_sandbox_e2e tests in e2e/test_evaluator_plugin.py).
#
# A wrapper around install_opensandbox_minikube.sh, which is cluster-agnostic apart from its
# defaults. This one names the kind context and the kind BatchSandbox template, pulls OpenSandbox's
# own images onto the node first, and only warns about a missing pull Secret: on kind the sandbox
# images are loaded onto the node, so the template's ghcr-pull Secret is a fallback.
#
# Run it before installing the platform with e2e/k8s/values/gym-sandbox.yaml, so the API-key Secret
# is in the release namespace when the first job starts.
#
# Local run, from the repository root. Every step must use the same namespace: the cluster setup
# puts the platform Secrets and the Gateway there, OpenSandbox runs sandboxes there, and the tests
# look for the Gym host there. CI uses nemo-helix.
#
#   export KIND_CLUSTER_NAME=nhx-e2e KUBE_NAMESPACE=nemo-helix   # and NGC_API_KEY, GITHUB_TOKEN
#   ./e2e/k8s/scripts/setup_local_kind_cpu.sh
#   ./e2e/k8s/scripts/install_opensandbox_kind.sh
#   NAMESPACE="${KUBE_NAMESPACE}" NHX_E2E_REGISTRY=<registry> NHX_E2E_TAG=<tag> \
#     HELM_VALUES=e2e/k8s/values/kind.yaml HELM_EXTRA_ARGS="-f e2e/k8s/values/gym-sandbox.yaml" \
#     ./e2e/k8s/scripts/install_helm_e2e.sh
#   # Optional, as CI does; needs bash 4.4 or later, and for a private registry KIND_IMAGE_PULL_USER
#   # and KIND_IMAGE_PULL_TOKEN. It keeps the Gym host image pull out of the tests' 15 minutes.
#   ./e2e/k8s/scripts/prepull_kind_images.sh <registry>/nhx-gym-host:<tag>
#   NHX_E2E_SANDBOXED_GYM=1 NHX_E2E_KUBE_CONTEXT="kind-${KIND_CLUSTER_NAME}" \
#     NHX_E2E_KUBE_NAMESPACE="${KUBE_NAMESPACE}" \
#     NHX_E2E_INTERNAL_BASE_URL="http://nemo-helix-api.${KUBE_NAMESPACE}.svc.cluster.local:8080" \
#     uv run --frozen pytest e2e/test_evaluator_plugin.py --kubernetes \
#     --cluster-url=<the Gateway URL the Helm install logs> -m gym_sandbox_e2e -v -rs --no-cov
#
# Environment:
#   KIND_CLUSTER_NAME   kind cluster name (default: nhx-e2e). The kubectl/helm context is always
#                       kind-<name>, so the images are pulled onto the cluster that is installed on.
#   KUBE_CONTEXT        Leave unset. Set to anything other than kind-<name>, it stops the install.
#   KUBE_NAMESPACE      Job / Helm-release namespace (default: default, as in
#                       setup_local_kind_cpu.sh and install_helm_e2e.sh). NHX_NAMESPACE, an alias
#                       install_opensandbox_minikube.sh reads, is ignored here, with a warning when
#                       it names another namespace.
#   BATCHSANDBOX_TEMPLATE
#                       BatchSandbox template ConfigMap to apply (default:
#                       e2e/k8s/values/batchsandbox-kind-template.yaml)
#   PULL_SECRET_CHECK   As in install_opensandbox_minikube.sh, but the default is warn.
#   SKIP_PREPULL        1 or true: do not pull the images onto the node first. Also what happens,
#                       with a warning, under bash older than 4.4, which prepull_kind_images.sh
#                       cannot run on.
#   SKIP_VERIFY         1 or true: skip the closing verification sandbox.
#                       Both switches take 0, false or empty for off; any other value stops here.
#   SANDBOX_IMAGE       Image of the verification sandbox. Unset, it is the default in
#                       k8s/helm/examples/opensandbox/verify/lib.sh, pulled here with the others.
#                       Set, it is not pulled here: the kubelet pulls it, with the template's pull
#                       Secret if it needs one.
#   NGC_API_KEY, KIND_IMAGE_PULL_USER, KIND_IMAGE_PULL_TOKEN
#                       Credentials prepull_kind_images.sh pulls with: NGC_API_KEY for nvcr.io; the
#                       other two, when both are set, for any other registry except Docker Hub,
#                       whose images it pulls anonymously. Needed only when the values move an
#                       OpenSandbox image off Docker Hub.
#   Everything else install_opensandbox_minikube.sh reads, such as HELM_TIMEOUT and the chart
#   versions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
EXAMPLES="${REPO_ROOT}/k8s/helm/examples/opensandbox"

# Read once, as 1 or empty, so this script and the installer it runs make the same decision, and a
# value neither understands is not taken as off.
for flag in SKIP_PREPULL SKIP_VERIFY; do
    case "${!flag:-}" in
        1|true) printf -v "${flag}" '%s' 1 ;;
        ""|0|false) printf -v "${flag}" '%s' "" ;;
        *)
            log_error "${flag} must be 1 or true to skip, or 0, false or empty not to, got: ${!flag}"
            exit 1
            ;;
    esac
done
export SKIP_VERIFY

export KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-nhx-e2e}"
# Not overridable: the pre-pull below finds the node by cluster name, and the install must land on
# that same cluster. A caller's KUBE_CONTEXT that names another cluster stops here, not silently.
if [ -n "${KUBE_CONTEXT:-}" ] && [ "${KUBE_CONTEXT}" != "kind-${KIND_CLUSTER_NAME}" ]; then
    log_error "KUBE_CONTEXT is ${KUBE_CONTEXT}, but this script installs on kind-${KIND_CLUSTER_NAME}."
    log_error "Set KIND_CLUSTER_NAME to the kind cluster's name and leave KUBE_CONTEXT unset."
    exit 1
fi
export KUBE_CONTEXT="kind-${KIND_CLUSTER_NAME}"
# The rule setup_local_kind_cpu.sh and install_helm_e2e.sh use, which do not read NHX_NAMESPACE.
# Exported, so install_opensandbox_minikube.sh does not fall back to it either.
export KUBE_NAMESPACE="${KUBE_NAMESPACE:-default}"
if [ -n "${NHX_NAMESPACE:-}" ] && [ "${NHX_NAMESPACE}" != "${KUBE_NAMESPACE}" ]; then
    log_warn "Ignoring NHX_NAMESPACE=${NHX_NAMESPACE}: installing for KUBE_NAMESPACE=${KUBE_NAMESPACE}, which the other kind scripts use."
fi
export BATCHSANDBOX_TEMPLATE="${BATCHSANDBOX_TEMPLATE:-${REPO_ROOT}/e2e/k8s/values/batchsandbox-kind-template.yaml}"
export PULL_SECRET_CHECK="${PULL_SECRET_CHECK:-warn}"

# Prints "<component> <image>" for OpenSandbox's own images, read from the values this install
# applies, so a version bump there needs no second edit here. A component it cannot find prints
# nothing; the caller checks each by name.
opensandbox_images() {
    # controller and server: repository and tag from each chart's values.
    awk -v sq="'" '
        function unquote(s) { gsub(/"/, "", s); gsub(sq, "", s); return s }
        FNR == 1 { repository = "" }
        /^[[:space:]]*repository:/ { repository = unquote($2) }
        /^[[:space:]]*tag:/ && repository != "" {
            tag = unquote($2)
            if (tag != "") print component, repository ":" tag
            repository = ""
        }
    ' component=controller "${EXAMPLES}/opensandbox-controller.yaml" \
        component=server "${EXAMPLES}/opensandbox-server.yaml" || return 1
    # execd and egress: by key, from the uncommented lines of the server's configToml, so any
    # registry and any digest are kept as the server will ask for them.
    awk -v sq="'" '
        function value(line) {
            sub(/^[^=]*=[[:space:]]*/, "", line)
            if (substr(line, 1, 1) == "\"") { line = substr(line, 2); sub(/".*/, "", line) }
            else if (substr(line, 1, 1) == sq) { line = substr(line, 2); sub(sq ".*", "", line) }
            else sub(/[[:space:]#].*/, "", line)
            return line
        }
        /^configToml:/ { in_toml = 1; next }
        /^[^[:space:]]/ { in_toml = 0 }
        !in_toml || /^[[:space:]]*#/ { next }
        /^[[:space:]]*\[/ {
            section = $0
            sub(/#.*/, "", section)
            gsub(/[[:space:]]/, "", section)
            next
        }
        /^[[:space:]]*execd_image[[:space:]]*=/ { print "execd", value($0) }
        section == "[egress]" && /^[[:space:]]*image[[:space:]]*=/ { print "egress", value($0) }
    ' "${EXAMPLES}/opensandbox-server.yaml" || return 1
}

# The name to pull a reference under. containerd's CRI drops the tag of a reference that also has a
# digest and looks the image up as name@digest, so an image pulled as name:tag@digest would not be
# found, and the kubelet would go to the registry again.
pull_reference() {
    local name="${1%%@*}"
    case "$1" in
        *@*) ;;
        *)
            printf '%s\n' "$1"
            return
            ;;
    esac
    # A colon after the last slash starts a tag; one before it is a registry port.
    case "${name##*/}" in
        *:*) name="${name%:*}" ;;
    esac
    printf '%s@%s\n' "${name}" "${1#*@}"
}

# One pull attempt, run with the bash that passed the version check below. Without a terminal, as
# in CI, and where `timeout` exists (not always on macOS), the attempt has a deadline, so a pull
# that stalls fails this step by name instead of holding the job until its own limit. From a
# terminal it has none: timeout puts the attempt in its own process group, where Ctrl-C does not
# reach it, and the run would go on to install. /dev/tty opens only with a controlling terminal.
pull_images() {
    if command -v timeout >/dev/null 2>&1 && ! { : </dev/tty; } 2>/dev/null; then
        timeout 300 "${BASH}" "${SCRIPT_DIR}/prepull_kind_images.sh" "$@"
    else
        "${BASH}" "${SCRIPT_DIR}/prepull_kind_images.sh" "$@"
    fi
}

# prepull_kind_images.sh uses mapfile and expands an empty array under `set -u`, which bash before
# 4.4 (stock macOS has 3.2) cannot do. There the kubelet pulls the images instead, as it did before.
if [ "${SKIP_PREPULL}" != "1" ] &&
    { [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 4 ]; }; }; then
    log_warn "Not pulling the images onto the node first: that needs bash 4.4 or later, and this is ${BASH_VERSION}."
    SKIP_PREPULL=1
fi

# These come from Docker Hub without credentials. The retries absorb transient errors; the point of
# pulling here is that a failure that outlasts them, a rate limit included, fails this step by name.
# Left to the kubelet, it would fail the Helm wait, or land inside a sandbox's create wait: the
# verification's below, or for the egress sidecar the first Gym host's, in a test.
if [ "${SKIP_PREPULL}" != "1" ]; then
    # Checked once, so a missing tool or cluster is reported as that, not retried as a failed pull.
    for tool in kind docker; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            log_error "${tool} is not installed. It pulls the images onto the node; SKIP_PREPULL=1 leaves that to the kubelet."
            exit 1
        fi
    done
    if ! kind_nodes="$(kind get nodes --name "${KIND_CLUSTER_NAME}")"; then
        log_error "Could not list the nodes of kind cluster ${KIND_CLUSTER_NAME}"
        exit 1
    fi
    if [ -z "${kind_nodes}" ]; then
        log_error "No kind cluster named ${KIND_CLUSTER_NAME}. Create it with setup_local_kind_cpu.sh, or set KIND_CLUSTER_NAME to its name."
        exit 1
    fi

    if ! image_list="$(opensandbox_images)"; then
        log_error "Could not read the OpenSandbox images from ${EXAMPLES}/opensandbox-{controller,server}.yaml"
        exit 1
    fi
    IMAGES=()
    found=" "
    while IFS= read -r line; do
        component="${line%% *}"
        image="${line#* }"
        if [ -z "${line}" ] || [ -z "${image}" ]; then
            continue
        fi
        case "${image}" in
            *[[:space:]\"\'\$]*)
                log_error "The OpenSandbox ${component} image in ${EXAMPLES} is not a plain image reference: ${image}"
                exit 1
                ;;
        esac
        found="${found}${component} "
        image="$(pull_reference "${image}")"
        case " ${IMAGES[*]} " in
            *" ${image} "*) ;;
            *) IMAGES+=("${image}") ;;
        esac
    done <<<"${image_list}"
    # Each by name: a values file that changes shape would otherwise shorten this list unnoticed.
    for component in controller server execd egress; do
        case "${found}" in
            *" ${component} "*) ;;
            *)
                log_error "Found no OpenSandbox ${component} image in ${EXAMPLES}/opensandbox-{controller,server}.yaml"
                exit 1
                ;;
        esac
    done

    # The verification's sandbox image. Only its default is pulled here, read from the verification
    # and exported, so what is pulled is what it uses. An image the caller chose may need
    # credentials or a short-name lookup that only the kubelet has.
    if [ "${SKIP_VERIFY}" != "1" ]; then
        if [ -n "${SANDBOX_IMAGE:-}" ]; then
            log_info "Not pulling SANDBOX_IMAGE=${SANDBOX_IMAGE} here: the kubelet pulls it for the verification, with the template's pull Secret if it needs one."
        else
            SANDBOX_IMAGE="$(sed -nE 's/^SANDBOX_IMAGE="\$\{SANDBOX_IMAGE:-([^}]*)\}"([[:space:]]+#.*)?[[:space:]]*$/\1/p' "${EXAMPLES}/verify/lib.sh")"
            case "${SANDBOX_IMAGE}" in
                ""|*[[:space:]\"\'\$]*)
                    log_error "Could not read a plain verification image default from ${EXAMPLES}/verify/lib.sh, got: '${SANDBOX_IMAGE}'"
                    exit 1
                    ;;
            esac
            export SANDBOX_IMAGE
            IMAGES+=("$(pull_reference "${SANDBOX_IMAGE}")")
        fi
    fi

    log_info "Pulling OpenSandbox images onto the kind node: ${IMAGES[*]}"
    attempt=1
    until pull_images "${IMAGES[@]}"; do
        status=$?
        if [ "${status}" -eq 124 ]; then
            log_warn "The pull did not finish within its 300s deadline"
        fi
        if [ "${attempt}" -ge 3 ]; then
            log_error "Could not pull the OpenSandbox images after ${attempt} attempts"
            exit 1
        fi
        delay=$((attempt == 1 ? 15 : 60))
        log_warn "Pulling the OpenSandbox images failed; retrying in ${delay}s (${attempt}/3)"
        attempt=$((attempt + 1))
        sleep "${delay}"
    done
fi

exec "${SCRIPT_DIR}/install_opensandbox_minikube.sh"
