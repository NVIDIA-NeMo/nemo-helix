#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pull images directly into every kind node's containerd store.

set -euo pipefail

KIND_CLUSTER_NAME="${KIND_CLUSTER_NAME:-nhx-e2e}"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 IMAGE [IMAGE...]" >&2
    exit 1
fi

if ! command -v kind >/dev/null 2>&1; then
    echo "kind is not installed" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed" >&2
    exit 1
fi

mapfile -t KIND_NODES < <(kind get nodes --name "${KIND_CLUSTER_NAME}")
if [ "${#KIND_NODES[@]}" -eq 0 ]; then
    echo "No kind nodes found for cluster ${KIND_CLUSTER_NAME}" >&2
    exit 1
fi

pull_image_on_node() {
    local node="$1"
    local image="$2"
    local auth_args=()

    if [[ "${image}" == nvcr.io/* ]]; then
        if [ -z "${NGC_API_KEY:-}" ]; then
            echo "NGC_API_KEY is required to pull ${image}" >&2
            return 1
        fi
        auth_args=(--user "\$oauthtoken:${NGC_API_KEY}")
    elif [[ "${image}" == docker.io/* || "${image}" == registry-1.docker.io/* ]]; then
        # Public Docker Hub images are pulled anonymously. Passing the GHCR
        # KIND_IMAGE_PULL_* credentials here makes auth.docker.io reject the
        # request with a 401, so leave auth_args empty.
        auth_args=()
    elif [ -n "${KIND_IMAGE_PULL_USER:-}" ] && [ -n "${KIND_IMAGE_PULL_TOKEN:-}" ]; then
        auth_args=(--user "${KIND_IMAGE_PULL_USER}:${KIND_IMAGE_PULL_TOKEN}")
    fi

    echo "Pulling ${image} on ${node}..."
    docker exec "${node}" ctr -n=k8s.io images pull "${auth_args[@]}" "${image}"
}

for node in "${KIND_NODES[@]}"; do
    for image in "$@"; do
        pull_image_on_node "${node}" "${image}"
    done
done
