#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# chmod a+rX on files and directories created after a mark.
#
#   stamp=$(chmod-newer-than.sh mark)
#   ... create files ...
#   chmod-newer-than.sh "$stamp" /opt/nemo_rl_venv /opt/ray_venvs
#
# Regular files and directories only. Symlinks are not followed, so venv links
# into /opt/uv_cache stay out of this and a later layer does not copy them up.
set -euo pipefail

if [[ "${1:-}" == "mark" ]]; then
    stamp=$(mktemp)
    touch "${stamp}"
    printf '%s\n' "${stamp}"
    exit 0
fi

stamp=${1:?usage: chmod-newer-than.sh mark | chmod-newer-than.sh STAMP PATH...}
shift
if [[ $# -eq 0 ]]; then
    echo "chmod-newer-than.sh: at least one path is required" >&2
    exit 2
fi
find "$@" \( -type d -o -type f \) -newer "${stamp}" -exec chmod a+rX {} +
rm -f "${stamp}"
