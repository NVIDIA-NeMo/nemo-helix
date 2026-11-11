#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Warm /opt/uv_cache. Each sync is exact, so the venv keeps only the last one.
# UV_SYNC_MODE is passed by the Dockerfile (--frozen, or empty to relock).
set -euo pipefail

umask 022
export UV_LINK_MODE=symlink
stamp=$(chmod-newer-than.sh mark)
SKIP=(
    --no-install-package causal-conv1d
    --no-install-package mamba-ssm
    --no-install-package nv-grouped-gemm
    --no-install-package fast-hadamard-transform
    --no-install-package flash-mla
    --no-install-package deep-ep
    --no-install-package deep-gemm
    --no-install-package flash-attn
    --no-install-package transformer-engine-torch
    # Optional transfer backend unused by RL; its wheel bundles Go/etcd libraries.
    --no-install-package mooncake-transfer-engine
    --no-install-package mooncake-transfer-engine-cuda13
    --no-install-package nemo-sandboxed-gym
)
uv venv --seed

uv sync ${UV_SYNC_MODE-} --no-install-project "${SKIP[@]}"

uv sync ${UV_SYNC_MODE-} --extra vllm --no-install-project "${SKIP[@]}"

uv sync ${UV_SYNC_MODE-} --extra automodel --no-install-project "${SKIP[@]}"
uv sync ${UV_SYNC_MODE-} --extra mcore --no-install-project "${SKIP[@]}"
uv sync ${UV_SYNC_MODE-} --extra modelopt --no-install-project "${SKIP[@]}"
uv sync ${UV_SYNC_MODE-} --extra nemo_gym --no-install-project "${SKIP[@]}"
uv sync ${UV_SYNC_MODE-} --extra mcore --extra nemo_gym --no-install-project "${SKIP[@]}"
uv sync ${UV_SYNC_MODE-} --extra vllm --extra nemo_gym --no-install-project "${SKIP[@]}"
uv sync ${UV_SYNC_MODE-} --all-groups --no-install-project "${SKIP[@]}"
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
