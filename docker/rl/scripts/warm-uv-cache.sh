#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Warm /opt/uv_cache. Each sync is exact, so the venv keeps only the last one.
#
# The Dockerfile passes two variables:
#   UV_SYNC_MODE          --frozen, or empty to relock.
#   NO_INSTALL_PACKAGES   JSON array of package names to leave out of every sync.
set -euo pipefail

umask 022
export UV_LINK_MODE=symlink
stamp=$(chmod-newer-than.sh mark)

# Turn the JSON array into one --no-install-package flag per package.
skip_args=()
while IFS= read -r package; do
    skip_args+=(--no-install-package "${package}")
done < <(jq -r '.[]' <<<"${NO_INSTALL_PACKAGES}")

uv venv --seed

uv sync ${UV_SYNC_MODE-} --no-install-project "${skip_args[@]}"

uv sync ${UV_SYNC_MODE-} --extra vllm --no-install-project "${skip_args[@]}"

uv sync ${UV_SYNC_MODE-} --extra automodel --no-install-project "${skip_args[@]}"
uv sync ${UV_SYNC_MODE-} --extra mcore --no-install-project "${skip_args[@]}"
uv sync ${UV_SYNC_MODE-} --extra modelopt --no-install-project "${skip_args[@]}"
uv sync ${UV_SYNC_MODE-} --extra nemo_gym --no-install-project "${skip_args[@]}"
uv sync ${UV_SYNC_MODE-} --extra mcore --extra nemo_gym --no-install-project "${skip_args[@]}"
uv sync ${UV_SYNC_MODE-} --extra vllm --extra nemo_gym --no-install-project "${skip_args[@]}"
uv sync ${UV_SYNC_MODE-} --all-groups --no-install-project "${skip_args[@]}"

# Drop Ray's vendored aiohttp, yarl, and jars, and PyNvVideoCodec's bundled FFmpeg.
find "${UV_CACHE_DIR}" -type d -path "*ray/_private/runtime_env/agent/thirdparty_files/aiohttp*" -prune -exec rm -rf {} + || true
find "${UV_CACHE_DIR}" -type d -path "*ray/_private/runtime_env/agent/thirdparty_files/yarl*" -prune -exec rm -rf {} + || true
find "${UV_CACHE_DIR}" -type d -path "*/ray/jars" -exec rm -rf {} + || true
find "${UV_CACHE_DIR}" -type d \
    \( -name PyNvVideoCodec -o -name 'pynvvideocodec-*.data' \) -exec rm -rf {} + || true

# Each sync above is exact, so an intermediate resolve (torch 2.15 and its CUDA
# stack) stays in archive-v0 after the venv has moved on. Delete it in this layer.
prune-unreferenced-uv-archives.sh
chmod-newer-than.sh "${stamp}" /opt/nemo_rl_venv
