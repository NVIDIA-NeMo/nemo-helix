#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

required_uv_version="0.13.4"

shim="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/uv"

uv_version_output="$("${shim}" --version)"
# Anchor to the uv line: a Flox activate profile can write banners to stdout
# around the version output, so field 2 of every line is not a version.
actual_uv_version="$(printf '%s\n' "${uv_version_output}" | awk '/^uv [0-9]/{print $2; exit}')"

if [[ "${actual_uv_version}" != "${required_uv_version}" ]]; then
  echo "uv.lock must be checked or updated with uv ${required_uv_version}, matching platform containers and CI." >&2
  echo "Current uv: ${actual_uv_version:-${uv_version_output}}" >&2
  echo "Commit from an activated Flox environment ('flox -q activate'), or install ${required_uv_version} on PATH and rerun with TOOLCHAIN=system." >&2
  exit 1
fi

exec "${shim}" lock "$@"
