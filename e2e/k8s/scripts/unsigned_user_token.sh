#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# unsigned_user_token.sh <email> <out>: an unsigned JWT (alg=none) for <email>, valid 4 hours, for a platform with
# auth.allow_unsigned_jwt (the nightly's auth-on-exchange-false row). Writes it to <out> (0600), never to the log.

set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 <email> <out>" >&2; exit 2; }
email="$1"
out="$2"
mkdir -p "$(dirname "${out}")"
(
    umask 077
    uv run --frozen python -c '
import sys
from nemo_helix_ext.auth.helpers import generate_unsigned_jwt
print(generate_unsigned_jwt(sys.argv[1], email=sys.argv[1], expires_in_seconds=4 * 3600))
' "${email}" >"${out}"
)
chmod 600 "${out}"
