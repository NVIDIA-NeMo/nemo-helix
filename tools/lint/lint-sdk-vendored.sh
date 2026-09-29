#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
# Verify wrapper wheel metadata is in sync (make vendor leaves no uncommitted changes).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CI_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}"
export PATH="$HOME/.local/bin:$PATH"

make vendor
mapfile -t BUNDLE_PACKAGE_PYPROJECTS < <(
  git grep -l '^\[tool\.bundle-package\]' -- '**/pyproject.toml' || true
)

VENDORED_CHECK_PATHS=()
for pyproject in "${BUNDLE_PACKAGE_PYPROJECTS[@]}"; do
  VENDORED_CHECK_PATHS+=("${PROJECT_ROOT}/${pyproject}")
done

git add "${VENDORED_CHECK_PATHS[@]}"
git diff --cached --exit-code "${VENDORED_CHECK_PATHS[@]}" > "${PROJECT_ROOT}/diff.txt" || {
  echo "Run 'make vendor' to sync the wrapper wheel metadata."
  exit 1
}

make generate-cli-reference-docs
git add "${PROJECT_ROOT}/docs/cli"
git diff --cached --exit-code "${PROJECT_ROOT}/docs/cli" > "${PROJECT_ROOT}/diff.txt" || {
  echo "Run 'make generate-cli-reference-docs' to sync cli docs."
  exit 1
}
