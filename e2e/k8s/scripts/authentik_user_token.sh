#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Usage: authentik_user_token.sh <user> <out-file>
#
# Write a 1 h OIDC access token for an Authentik demo user to out-file (0600), through the gateway. Prints nothing
# secret. Run after setup_authentik_taskset_runs.sh, with the same KIND_CLUSTER_NAME and NHX_E2E_STATE_DIR.
#   nemo-setup  the demo's setup user, with its generated password (the platform admin once the setup has run)
#   any other   an app password, minted once through the Authentik API and kept in NHX_E2E_STATE_DIR: the
#               password grant rejects a user's own password

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=authentik_lib.sh
source "${SCRIPT_DIR}/authentik_lib.sh"

[[ $# -eq 2 ]] || die "usage: authentik_user_token.sh <user> <out-file>"
user="$1" out="$2"
mkdir -p "${STATE_DIR}" && chmod 700 "${STATE_DIR}"

password_file=""
if [[ "${user}" != "nemo-setup" ]]; then
    password_file="${STATE_DIR}/${user}.app-password"
    [[ -s "${password_file}" ]] || authentik_admin app-password "${user}" "${password_file}"
fi

(
    set -a
    # shellcheck disable=SC1091
    source "${GENERATED_DIR}/auth-idp-e2e.env"
    set +a
    if [[ -n "${password_file}" ]]; then password="$(cat "${password_file}")"; else password="${AUTHENTIK_SETUP_PASSWORD}"; fi
    curl -sS --cacert "${CLUSTER_DIR}/ca.crt" "${GATEWAY_URL}/application/o/token/" \
        -d grant_type=password -d client_id=nemo-helix \
        --data-urlencode "client_secret=${AUTHENTIK_SETUP_CLIENT_SECRET}" \
        -d username="${user}" --data-urlencode "password=${password}" -d scope="openid email groups"
) | python3 -c '
import json, os, sys
body = json.load(sys.stdin)
token = body.get("access_token")
if not token:
    sys.exit("token request failed: " + json.dumps({k: v for k, v in body.items() if "token" not in k}))
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.fdopen(fd, "w").write(token)
' "${out}"
log "token for ${user} written to ${out}"
