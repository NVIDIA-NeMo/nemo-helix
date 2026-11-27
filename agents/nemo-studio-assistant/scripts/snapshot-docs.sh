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
  # A plain loop, not find -exec: a failed mkdir or cp must stop the script under set -e.
  # The file list goes to a manifest so a failed find also stops it.
  manifest="$(mktemp)"
  trap 'rm -f "${manifest}"' EXIT
  find . -type f \( -name '*.md' -o -name '*.mdx' \) \
    -not -path './_scripts/*' -not -path './fern/*' -not -path './assets/*' \
    -not -path './notebooks/*' -not -path './contributing/*' \
    -not -name 'AGENTS.md' -not -name 'eula.mdx' \
    -print0 > "${manifest}"
  while IFS= read -r -d '' doc; do
    mkdir -p "${dest}/$(dirname "${doc}")"
    cp "${doc}" "${dest}/${doc}"
  done < "${manifest}"
)

echo "Snapshot $(find "${dest}" -type f | wc -l | tr -d ' ') docs into ${dest}"
