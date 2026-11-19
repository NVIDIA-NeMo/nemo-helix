#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Fail when the runtime user cannot read the image trees.
#
# The smoke stage runs as root, and access(2) then succeeds even for mode 000, so this checks
# other-permission bits: directories need r+x, files need r, and anything with an execute bit
# needs x. Symlinks are not followed. Worker venvs point at /opt/uv_cache, which is one of the
# roots, and following those links would stat the cache once per venv.
set -euo pipefail

roots=(
    /opt/cpython
    /opt/uv_cache
    /opt/nemo_rl_venv
    /opt/nemo-rl
    /opt/gym_venvs
    /opt/ray_venvs
)

matches=$(mktemp)
trap 'rm -f "${matches}"' EXIT

# -005: other read+execute. -004: other read. /111: any execute bit. -001: other execute.
find -P "${roots[@]}" \
    \( \
        \( -type d ! -perm -005 \) \
        -o \( -type f ! -perm -004 \) \
        -o \( -type f -perm /111 ! -perm -001 \) \
    \) -print > "${matches}"

if [[ -s "${matches}" ]]; then
    head -n 20 "${matches}" >&2
    printf '%s entries are not readable or traversable by other users\n' "$(wc -l < "${matches}")" >&2
    exit 1
fi
