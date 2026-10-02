#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

required_uv_version="0.10.10"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to update uv.lock." >&2
  echo "Install uv ${required_uv_version}: curl -LsSf https://astral.sh/uv/${required_uv_version}/install.sh | sh" >&2
  exit 1
fi

uv_version_output="$(uv --version)"
actual_uv_version="$(printf '%s\n' "${uv_version_output}" | awk '{print $2}')"

if [[ "${actual_uv_version}" != "${required_uv_version}" ]]; then
  echo "uv.lock must be checked or updated with uv ${required_uv_version}, matching platform containers and CI." >&2
  echo "Current uv: ${uv_version_output}" >&2
  echo "Install uv ${required_uv_version}: curl -LsSf https://astral.sh/uv/${required_uv_version}/install.sh | sh" >&2
  exit 1
fi

exec uv lock "$@"
