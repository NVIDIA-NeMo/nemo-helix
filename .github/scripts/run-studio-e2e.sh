#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dependencies and Studio/plugin bundles must already be installed/built.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export NHX_BASE_URL=http://127.0.0.1:8080
export VSERVICE_URL_STUDIO_UI="${NHX_BASE_URL}/studio/"
export VITE_NHX_BASE_URL="$NHX_BASE_URL"
export CI=1
LOG_DIR="${STUDIO_E2E_LOG_DIR:-$(mktemp -d)}"
mkdir -p "$LOG_DIR"

if (echo > /dev/tcp/127.0.0.1/8080) 2>/dev/null; then
  echo 'Port 8080 is already in use; refusing to test against an unrelated platform.' >&2
  exit 1
fi

# Never read or modify a developer's existing platform state.
export NHX_DATA_DIR
NHX_DATA_DIR="$(mktemp -d)"
export NHX_FILES_DEFAULT_STORAGE_CONFIG="{\"type\":\"local\",\"path\":\"$NHX_DATA_DIR/files\"}"
platform_pid=''
cleanup() {
  if [[ -n "$platform_pid" ]]; then
    kill "$platform_pid" 2>/dev/null || true
    wait "$platform_pid" 2>/dev/null || true
  fi
  rm -rf "$NHX_DATA_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

uv run --frozen nemo services run \
  --services entities,auth,studio,agent-hardener,agents,files,jobs,models,secrets,inference-gateway \
  --config e2e/configs/studio.yaml \
  > "$LOG_DIR/platform.log" 2>&1 &
platform_pid=$!

ready=false
for ((attempt = 0; attempt < 120; attempt++)); do
  if ! kill -0 "$platform_pid" 2>/dev/null; then
    break
  fi
  if curl --fail --silent --max-time 2 "$NHX_BASE_URL/health/ready" > /dev/null; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]] || ! curl --fail --silent --max-time 10 "$VSERVICE_URL_STUDIO_UI" > /dev/null; then
  echo 'Studio/platform failed readiness checks:' >&2
  tail -100 "$LOG_DIR/platform.log" >&2
  exit 1
fi

cd web
pnpm --filter nemo-studio-ui test:e2e "$@"
