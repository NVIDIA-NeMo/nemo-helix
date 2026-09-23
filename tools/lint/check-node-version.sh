#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

node_version="$(yq '.install.nodejs.version' tools/nodejs/.flox/env/manifest.toml)"
pnpm_version="$(yq -r '.vars.PNPM_VERSION' tools/nodejs/.flox/env/manifest.toml)"
make_node_version="$(sed -n 's/^NODE_VERSION[[:space:]]*:=[[:space:]]*\([^[:space:]#]*\).*/\1/p' Makefile)"
make_pnpm_version="$(sed -n 's/^PNPM_VERSION[[:space:]]*:=[[:space:]]*\([^[:space:]#]*\).*/\1/p' Makefile)"
nvmrc_version="$(tr -d '[:space:]' < .nvmrc)"
package_manager="$(yq -r '.packageManager' web/package.json)"

if [[ "${node_version}" != "${nvmrc_version}" ]]; then
  echo "Flox Node.js version ${node_version} does not match .nvmrc ${nvmrc_version}." >&2
  exit 1
fi

if [[ -z "${make_node_version}" || "${node_version}" != "${make_node_version}" ]]; then
  echo "Makefile Node.js version ${make_node_version:-missing} does not match Flox Node.js version ${node_version}." >&2
  exit 1
fi

if [[ "${package_manager}" != "pnpm@${pnpm_version}" ]]; then
  echo "Corepack pnpm version ${pnpm_version} does not match web/package.json ${package_manager}." >&2
  exit 1
fi

if [[ -z "${make_pnpm_version}" || "${pnpm_version}" != "${make_pnpm_version}" ]]; then
  echo "Makefile pnpm version ${make_pnpm_version:-missing} does not match Flox Corepack pnpm version ${pnpm_version}." >&2
  exit 1
fi

assert_setup_node_uses_nvmrc() {
  local workflow="$1"
  local version_file="$2"
  local setup_node_count
  local version_file_count

  setup_node_count="$(grep -c 'uses: actions/setup-node@' "${workflow}" || true)"
  version_file_count="$(grep -F -c "node-version-file: ${version_file}" "${workflow}" || true)"

  if [[ "${setup_node_count}" -eq 0 || "${setup_node_count}" != "${version_file_count}" ]]; then
    echo "Every setup-node step in ${workflow} must use ${version_file}." >&2
    exit 1
  fi
}

assert_setup_node_uses_nvmrc .github/workflows/ci.yaml .nvmrc
assert_setup_node_uses_nvmrc .github/actions/build-nemo-helix-wheel/action.yaml '${{ inputs.source-root }}/.nvmrc'
