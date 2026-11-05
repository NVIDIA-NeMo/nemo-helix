#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

required_uv_version="0.13.4"

shim="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/uv"

uv_version_output="$("${shim}" --version)"
actual_uv_version="$(printf '%s\n' "${uv_version_output}" | awk '{print $2}')"

if [[ "${actual_uv_version}" != "${required_uv_version}" ]]; then
  echo "uv.lock must be checked or updated with version ${required_uv_version}, matching platform containers and CI." >&2
  echo "Current version: ${uv_version_output}" >&2
  echo "Commit from an activated Flox environment ('flox -q activate'), or install ${required_uv_version} on PATH and rerun with TOOLCHAIN=system." >&2
  exit 1
fi

exec "${shim}" lock "$@"
