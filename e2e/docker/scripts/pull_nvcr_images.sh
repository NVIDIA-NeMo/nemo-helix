#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pull NVCR images with retries.
#
# Usage:
#   pull_nvcr_images.sh [--minikube-profile PROFILE] IMAGE [IMAGE...]
#
# Environment:
#   NGC_API_KEY  NGC API key for nvcr.io ($oauthtoken login). Required unless
#                already logged in to nvcr.io in the active Docker context.
#
# Can also be sourced to compose custom pull steps:
#   source ./e2e/docker/scripts/pull_nvcr_images.sh
#   use_minikube_docker "${MINIKUBE_PROFILE}"
#   pull_image "${NHX_E2E_REGISTRY}/nhx-tasks:${NHX_E2E_TAG}"

set -euo pipefail

nvcr_login() {
  if [[ -z "${NGC_API_KEY:-}" ]]; then
    echo "NGC_API_KEY is not set; assuming nvcr.io credentials are already configured"
    return 0
  fi
  echo "Logging in to nvcr.io..."
  echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
}

pull_image() {
  local image="$1"
  local attempt

  for attempt in $(seq 1 12); do
    if docker pull "${image}"; then
      echo "Pulled ${image}"
      return 0
    fi

    if [[ "${attempt}" -eq 12 ]]; then
      echo "Failed to pull ${image} after ${attempt} attempts"
      return 1
    fi

    echo "Pull failed for ${image} (attempt ${attempt}/12); re-authenticating and retrying..."
    nvcr_login
    sleep "$((attempt * 15))"
  done
}

use_minikube_docker() {
  local profile="$1"
  local docker_env
  docker_env="$(minikube docker-env -p "${profile}")" || return "$?"
  eval "${docker_env}"
  nvcr_login
}

pull_nvcr_images_main() {
  local minikube_profile=""
  local args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --minikube-profile)
        minikube_profile="$2"
        shift 2
        ;;
      *)
        args+=("$1")
        shift
        ;;
    esac
  done

  if [[ ${#args[@]} -eq 0 ]]; then
    echo "usage: $0 [--minikube-profile PROFILE] IMAGE [IMAGE...]" >&2
    exit 1
  fi

  if [[ -n "${minikube_profile}" ]]; then
    use_minikube_docker "${minikube_profile}"
  else
    nvcr_login
  fi

  local image
  for image in "${args[@]}"; do
    pull_image "${image}"
  done
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  pull_nvcr_images_main "$@"
fi
