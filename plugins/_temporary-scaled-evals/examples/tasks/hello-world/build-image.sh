#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build and push the hello-world task environment image for remote K8s sandboxes.
#
# sandbox-k8s does not build Harbor task Dockerfiles on-cluster; without TASK_IMAGE
# it falls back to its default image rather than exercising this task image.
#
# Usage:
#   ./build-image.sh <target>      # reads TASK_IMAGE from targets/<target>.env
#   TASK_IMAGE=... ./build-image.sh # explicit image
#   TARGET_PLATFORM=linux/arm64 ... # override the remote cluster platform
set -euo pipefail
cd "$(dirname "$0")"

TARGET="${1:-}"
if [[ -n "$TARGET" ]]; then
  ENV_FILE="../agent-sandbox/targets/${TARGET}.env"
  if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
      [[ "$line" =~ ^[[:space:]]*# ]] && continue
      [[ "$line" =~ ^[[:space:]]*$ ]] && continue
      if [[ "$line" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
        export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
      fi
    done < "$ENV_FILE"
  fi
fi

: "${TASK_IMAGE:?set TASK_IMAGE (e.g. in examples/agent-sandbox/targets/<target>.env)}"
TARGET_PLATFORM="${TARGET_PLATFORM:-linux/amd64}"

TASK_DIR="$(cd task && pwd)"
docker buildx build \
  --platform "$TARGET_PLATFORM" \
  -f "${TASK_DIR}/environment/Dockerfile" \
  -t "$TASK_IMAGE" \
  --push \
  "${TASK_DIR}/environment"
echo "pushed $TASK_IMAGE"
