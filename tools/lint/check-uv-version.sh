#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

_flox_ver="$(yq '.install.uv.version // .vars.UV_VERSION' tools/python/.flox/env/manifest.toml)"
_make_ver="$(sed -n 's/^UV_VERSION[[:space:]]*:=[[:space:]]*\([^[:space:]#]*\).*/\1/p' Makefile)"
_pyproject_minimum="$(sed -n 's/^required-version = ">=\([^"]*\)"/\1/p' pyproject.toml)"

if [[ ! "${_flox_ver}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  printf "Flox must install an exact uv version, found: %s\n" "${_flox_ver}" >&2
  exit 1
fi

if [[ -z "${_make_ver}" || "${_make_ver}" != "${_flox_ver}" ]]; then
  printf "Makefile UV_VERSION %s does not match Flox uv version %s.\n" "${_make_ver:-missing}" "${_flox_ver}" >&2
  exit 1
fi

if [[ -z "${_pyproject_minimum}" || "${_make_ver}" != "${_pyproject_minimum}" ]]; then
  printf "Makefile UV_VERSION %s does not match pyproject.toml required-version minimum %s.\n" "${_make_ver}" "${_pyproject_minimum:-missing}" >&2
  exit 1
fi

_workflows=()
while IFS= read -r _workflow; do
  _workflows+=("${_workflow}")
done < <(find .github/workflows -type f \( -name '*.yaml' -o -name '*.yml' \) -exec grep -l 'astral-sh/setup-uv' {} +)

if [[ "${#_workflows[@]}" -eq 0 ]]; then
  echo "No workflows using astral-sh/setup-uv were found." >&2
  exit 1
fi

for _workflow in "${_workflows[@]}"; do
  _ci_ver="$(yq '.env.UV_VERSION' "${_workflow}")"

  if [[ "${_ci_ver}" != "${_flox_ver}" ]]; then
    printf "CI and Flox uv versions differ:\n\n" >&2
    printf "workflow flox-env\n%s %s\n" "${_workflow}:${_ci_ver}" "${_flox_ver}" | column -t >&2
    exit 1
  fi
done
