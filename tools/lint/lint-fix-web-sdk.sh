#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail
# Regenerate the TypeScript web SDK (web/packages/sdk) from the current OpenAPI spec.
# Skips locally when pnpm is absent (mirrors lint-web-sdk.sh); hard-fails under CI.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CI_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}/web" || exit 1

if ! command -v pnpm >/dev/null 2>&1; then
  if [[ -n "${CI:-}" ]]; then
    echo "pnpm required in CI but not found on PATH" >&2
    exit 1
  fi
  echo "pnpm not installed locally — skipping web SDK regeneration (see web/README.md for setup)"
  exit 0
fi

pnpm_path="$(command -v pnpm)"
exec "${pnpm_path}" gen
