#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
clickhouse_version="$(tr -d '[:space:]' < "${script_dir}/../../.clickhouse-version")"

# Preserve the script's historical credential overrides while delegating all
# lifecycle, identity, port allocation, and readiness logic to Intake's Python
# provisioner.
export NHX_INTAKE_CLICKHOUSE_USER="${NHX_INTAKE_CLICKHOUSE_USER:-${CLICKHOUSE_USER:-default}}"
export NHX_INTAKE_CLICKHOUSE_PASSWORD="${NHX_INTAKE_CLICKHOUSE_PASSWORD:-${CLICKHOUSE_PASSWORD:-}}"
export NHX_INTAKE_CLICKHOUSE_IMAGE="${NHX_INTAKE_CLICKHOUSE_IMAGE:-${CLICKHOUSE_IMAGE:-clickhouse/clickhouse-server:${clickhouse_version}}}"
export NHX_INTAKE_CLICKHOUSE_DATA_DIR="${NHX_INTAKE_CLICKHOUSE_DATA_DIR:-${CLICKHOUSE_DATA_DIR:-${repo_root}/tmp/intake-clickhouse}}"

cd "${repo_root}"
exec uv run python -m nhx.intake.local_clickhouse --legacy-script-mode "$@"
