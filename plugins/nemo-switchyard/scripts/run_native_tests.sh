#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Isolated native Switchyard tests. Do not install into the worktree .venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TAG="${SWITCHYARD_NATIVE_TAG:-v0.3.0-rc.1}"
VENV="${SWITCHYARD_NATIVE_VENV:-${TMPDIR:-/tmp}/nmp-switchyard-native-venv}"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip
python -m pip install "git+https://github.com/NVIDIA-NeMo/Switchyard.git@${TAG}"
python -m pip install pytest pytest-asyncio httpx
# Plugin + IGW types without the May vendor:
export PYTHONPATH="${ROOT}/plugins/nemo-switchyard/src:${ROOT}/packages/nemo_platform_plugin/src:${PYTHONPATH:-}"
python -c "import switchyard_rust; import switchyard_rust.libsy"
cd "$ROOT"
python -m pytest plugins/nemo-switchyard/tests/test_native_libsy.py \
  -c pytest.ini -m switchyard_native -v --strict-markers
