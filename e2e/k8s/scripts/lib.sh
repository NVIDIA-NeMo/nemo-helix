#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared utilities for e2e K8s setup scripts.

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log_info()  { echo -e "\033[0;32m[INFO]\033[0m $*"; }
log_warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $*"; }

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

# _dockerconfigjson SERVER USERNAME PASSWORD
#
# Prints a dockerconfigjson blob for a single registry.
_dockerconfigjson() {
    printf '{"auths":{"%s":{"username":"%s","password":"%s","auth":"%s"}}}' \
      "$1" "$2" "$3" "$(printf '%s:%s' "$2" "$3" | base64 | tr -d '\n')"
}

# create_platform_secrets NAMESPACE
#
# Creates the standard set of platform secrets in the given namespace.
# NGC secrets are always created (with a placeholder if NGC_API_KEY is unset)
# because the helm chart expects the ngc-api secret to exist. Other secrets
# are created only when their corresponding env var is set:
#   - ghcr-pull: when GITHUB_TOKEN is set
#   - huggingface-token: when HF_TOKEN is set
#
# Secret values are passed via process substitution to keep them out of
# process arguments visible in /proc.
create_platform_secrets() {
    local namespace="${1:?namespace is required}"
    local kubectl_ns=(kubectl -n "${namespace}")
    local ngc_key="${NGC_API_KEY:-placeholder}"

    if [ -z "${NGC_API_KEY:-}" ]; then
        log_warn "NGC_API_KEY not set, creating NGC secrets with placeholder"
    fi

    log_info "Creating NGC API secret..."
    "${kubectl_ns[@]}" create secret generic ngc-api \
      --from-file=NGC_API_KEY=<(printf '%s' "${ngc_key}") \
      --dry-run=client -o yaml | "${kubectl_ns[@]}" apply -f -

    log_info "Creating NGC image pull secret..."
    "${kubectl_ns[@]}" create secret generic nvcrimagepullsecret \
      --type=kubernetes.io/dockerconfigjson \
      --from-file=.dockerconfigjson=<(_dockerconfigjson nvcr.io '$oauthtoken' "${ngc_key}") \
      --dry-run=client -o yaml | "${kubectl_ns[@]}" apply -f -

    if [ -n "${GITHUB_TOKEN:-}" ]; then
        log_info "Creating GHCR image pull secret..."
        "${kubectl_ns[@]}" create secret generic ghcr-pull \
          --type=kubernetes.io/dockerconfigjson \
          --from-file=.dockerconfigjson=<(_dockerconfigjson ghcr.io x-access-token "${GITHUB_TOKEN}") \
          --dry-run=client -o yaml | "${kubectl_ns[@]}" apply -f -
    else
        log_warn "GITHUB_TOKEN not set, skipping GHCR image pull secret"
    fi

    if [ -n "${HF_TOKEN:-}" ]; then
        log_info "Creating HuggingFace token secret..."
        "${kubectl_ns[@]}" create secret generic huggingface-token \
          --from-file=HF_TOKEN=<(printf '%s' "${HF_TOKEN}") \
          --dry-run=client -o yaml | "${kubectl_ns[@]}" apply -f -
    else
        log_warn "HF_TOKEN not set, skipping HuggingFace token secret"
    fi
}

# Waits for Calico when the cluster runs it. When MINIKUBE_CNI=calico was requested
# but the profile lacks it, warns that NetworkPolicy resources (networkPolicies.*
# chart values) will be accepted but not enforced. minikube's default docker-driver
# CNI ignores NetworkPolicy, and --cni only takes effect when the profile is created.
check_minikube_network_policy_cni() {
    local profile="$1" cni="$2"
    if kubectl -n kube-system get daemonset calico-node >/dev/null 2>&1; then
        log_info "Waiting for Calico (NetworkPolicy enforcement)..."
        kubectl -n kube-system rollout status daemonset/calico-node --timeout=5m
        kubectl -n kube-system rollout status deployment/calico-kube-controllers --timeout=5m
    elif [ "${cni}" = "calico" ]; then
        log_warn "Profile ${profile} is running without Calico, so NetworkPolicy is not enforced."
        log_warn "Recreate it to enforce: minikube delete -p ${profile} && rerun this script"
    fi
}
