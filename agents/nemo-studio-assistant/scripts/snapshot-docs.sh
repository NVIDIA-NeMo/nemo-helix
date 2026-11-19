#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copy the repository's Markdown documentation into docs-snapshot/ so
# `nemo agents package` bundles it into the image for the search_docs tool.
set -euo pipefail

agent_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "${agent_dir}/../.." && pwd)"
dest="${agent_dir}/docs-snapshot"

rm -rf "${dest}"
mkdir -p "${dest}"
(
  cd "${repo_root}/docs"
  find . -type f \( -name '*.md' -o -name '*.mdx' \) \
    -not -path './_scripts/*' -not -path './fern/*' -not -path './assets/*' \
    -not -path './notebooks/*' -not -path './contributing/*' \
    -not -name 'AGENTS.md' -not -name 'eula.mdx' \
    -exec sh -c 'mkdir -p "$1/$(dirname "$2")" && cp "$2" "$1/$2"' _ "${dest}" {} \;
)

echo "Snapshot $(find "${dest}" -type f | wc -l | tr -d ' ') docs into ${dest}"
