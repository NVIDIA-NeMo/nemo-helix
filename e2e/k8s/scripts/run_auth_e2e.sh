#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run a fast auth-enabled smoke check, then the existing auth E2E suite.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
MINIKUBE_PROFILE="${MINIKUBE_PROFILE:-minikube-auth}"
BASE_URL="${NHX_E2E_CLUSTER_URL:-http://localhost:30080}"
PRINCIPAL_ID="${NHX_E2E_PRINCIPAL_ID:-e2e-test-user@example.com}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_PROJECT="${NHX_E2E_UV_PROJECT:-${REPO_ROOT}}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

if [ ! -f "${UV_PROJECT}/pyproject.toml" ]; then
    echo "Could not find pyproject.toml at ${UV_PROJECT}. Set NHX_E2E_UV_PROJECT to a checkout with pyproject.toml." >&2
    exit 1
fi

if [ "$#" -gt 0 ]; then
    PYTEST_TARGETS=("$@")
else
    PYTEST_TARGETS=()
fi

wait_for_url() {
    local url="$1"
    local attempts="${2:-60}"
    local sleep_seconds="${3:-2}"
    local i
    for ((i=1; i<=attempts; i++)); do
        if curl -fsS "${url}" >/dev/null 2>&1; then
            return 0
        fi
        sleep "${sleep_seconds}"
    done
    return 1
}

echo "Using minikube profile: ${MINIKUBE_PROFILE}"
echo "Using base URL: ${BASE_URL}"

if ! wait_for_url "${BASE_URL}/status"; then
    echo "Platform did not become ready at ${BASE_URL}/status" >&2
    exit 1
fi

discovery_json="$(curl -fsS "${BASE_URL}/apis/auth/discovery")"
DISCOVERY_JSON="${discovery_json}" "${PYTHON_BIN}" - <<'PY'
import json
import os
payload = json.loads(os.environ["DISCOVERY_JSON"])
if payload.get("auth_enabled") is not True:
    raise SystemExit(f"Expected auth_enabled=true, got: {payload}")
print("Auth discovery check passed")
PY

unauth_code="$(curl -s -o /dev/null -w '%{http_code}' "${BASE_URL}/apis/entities/v2/workspaces")"
if [ "${unauth_code}" != "401" ]; then
    echo "Expected unauthenticated workspace list to return 401, got ${unauth_code}" >&2
    exit 1
fi
echo "Unauthenticated request check passed"

auth_code="$(curl -s -o /dev/null -w '%{http_code}' -H "X-NHX-Principal-Id: ${PRINCIPAL_ID}" "${BASE_URL}/apis/entities/v2/workspaces")"
if [ "${auth_code}" != "200" ]; then
    echo "Expected authenticated workspace list to return 200, got ${auth_code}" >&2
    exit 1
fi
echo "Authenticated request check passed"

export NHX_E2E_CLUSTER_URL="${BASE_URL}"
export NHX_BASE_URL="${BASE_URL}"

cd "${REPO_ROOT}"
if [ "${#PYTEST_TARGETS[@]}" -gt 0 ]; then
    uv run --project "${UV_PROJECT}" --frozen pytest "${PYTEST_TARGETS[@]}" --run-e2e --feature auth -v
fi
