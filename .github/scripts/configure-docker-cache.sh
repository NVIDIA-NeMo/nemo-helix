#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Separate packages keep parallel image jobs from overwriting shared base-layer
# caches and keep caches out of the release/CI-image cleanup package list.
case "${CACHE_SCOPE}" in
  cpu|garak|safe-synthesizer) ;;
  *)
    echo "Unsupported Docker cache scope: ${CACHE_SCOPE}" >&2
    exit 1
    ;;
esac

repository="$(printf '%s' "${GITHUB_REPOSITORY}" | tr '[:upper:]' '[:lower:]')"
cache_registry="ghcr.io/${repository}/ci-cache/${CACHE_SCOPE}"
cache_target=""
# PRs, merge groups, and manual runs only read the trusted main cache.
# CI's workflow concurrency serializes main pushes, so each scope has one writer.
if [ "${GITHUB_EVENT_NAME}" = "push" ] && [ "${GITHUB_REF}" = "refs/heads/main" ]; then
  cache_target="main"
fi

{
  printf 'CACHE_REGISTRY=%s\n' "${cache_registry}"
  printf 'CACHE_IGNORE_ERROR=true\n'
  printf 'BAKE_CACHE_SOURCE_BRANCH=main\n'
  printf 'BAKE_CACHE_TARGET_BRANCH=%s\n' "${cache_target}"
} >> "${GITHUB_ENV}"

{
  echo "### Docker layer cache (${CACHE_SCOPE})"
  echo
  echo "Registry: \`${cache_registry}\`"
  echo "Import: \`cache-main-linux-<arch>\`"
  if [ -n "${cache_target}" ]; then
    echo "Export: \`cache-main-linux-<arch>\` (trusted main push)"
  else
    echo "Export: disabled (read-only consumer)"
  fi
} >> "${GITHUB_STEP_SUMMARY}"
