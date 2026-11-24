#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../web"

if ! command -v pnpm >/dev/null 2>&1; then
  echo "pnpm is required to check the web lockfile; activate the repository's Flox environment." >&2
  exit 1
fi

# Use pnpm's own frozen-install validation, without installing node_modules or
# running lifecycle scripts. In particular, this validates the workspace catalog.
if ! pnpm install --lockfile-only --frozen-lockfile --ignore-scripts; then
  echo "Web lockfile validation failed. For stale manifests/catalog entries, regenerate and stage it:" >&2
  echo "  cd web && pnpm install --lockfile-only --ignore-scripts" >&2
  echo "  git add pnpm-lock.yaml" >&2
  exit 1
fi
