#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Delete /opt/uv_cache/archive-v0 entries that no venv symlink points at.
#
# Call this at the end of the RUN that downloaded the archives. A later layer
# can only whiteout those files, which does not shrink the pulled image.
# Referenced archives stay. Symlink targets in the cache are not rewritten.
set -euo pipefail

cache_root="${UV_CACHE_DIR:-/opt/uv_cache}"
archives="${cache_root}/archive-v0"
if [[ ! -d "${archives}" ]]; then
    exit 0
fi

referenced="$(mktemp)"
trap 'rm -f "${referenced}"' EXIT

roots=()
for root in /opt/nemo_rl_venv /opt/ray_venvs /opt/gym_venvs; do
    if [[ -d "${root}" ]]; then
        roots+=("${root}")
    fi
done
if [[ ${#roots[@]} -gt 0 ]]; then
    find "${roots[@]}" -type l -printf '%l\n' \
        | sed -n 's|.*archive-v0/\([^/]*\).*|\1|p' \
        | sort -u > "${referenced}"
fi

shopt -s nullglob
for archive in "${archives}"/*/; do
    [[ -d "${archive}" ]] || continue
    name="$(basename "${archive}")"
    if grep -qxF -e "${name}" "${referenced}"; then
        continue
    fi
    for index in "${cache_root}"/wheels-v*/; do
        if [[ -d "${index}" ]]; then
            find "${index}" -type l -lname "*${name}*" -delete
        fi
    done
    rm -rf "${archive}"
done
