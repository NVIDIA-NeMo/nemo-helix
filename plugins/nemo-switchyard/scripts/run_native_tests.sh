#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
VENV="${SWITCHYARD_NATIVE_VENV:-${TMPDIR:-/tmp}/nhx-switchyard-native-venv}"

uv venv --clear "$VENV"
uv pip install --python "$VENV/bin/python" \
  "nemo-switchyard==0.3.0" \
  pytest \
  pytest-asyncio \
  httpx \
  "jinja2>=3.1.6" \
  pydantic

export PYTHONPATH="${ROOT}/plugins/nemo-switchyard/src:${ROOT}/packages/nemo_helix_plugin/src:${PYTHONPATH:-}"
cd "$ROOT"
"$VENV/bin/python" -m pytest plugins/nemo-switchyard/tests/test_native_libsy.py \
  --noconftest \
  -p no:cacheprovider \
  -o addopts= \
  -o "markers=switchyard_native: native rust integration" \
  -v
