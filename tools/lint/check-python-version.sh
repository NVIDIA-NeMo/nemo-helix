#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

make_python_version="$(sed -n 's/^PYTHON_VERSION[[:space:]]*?=[[:space:]]*\([^[:space:]#]*\).*/\1/p' Makefile)"
flox_python_version="$(yq -r '.vars.UV_PYTHON' tools/python/.flox/env/manifest.toml)"

if [[ -z "${make_python_version}" ]]; then
  echo "Could not find the default PYTHON_VERSION in Makefile." >&2
  exit 1
fi

if [[ -z "${flox_python_version}" || "${flox_python_version}" == "null" ]]; then
  echo "Could not find UV_PYTHON in tools/python/.flox/env/manifest.toml." >&2
  exit 1
fi

if [[ "${make_python_version}" != "${flox_python_version}" ]]; then
  echo "Makefile Python version ${make_python_version} does not match Flox UV_PYTHON ${flox_python_version}." >&2
  exit 1
fi
