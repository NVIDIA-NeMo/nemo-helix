#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -e

ROOT_DIR=$(git rev-parse --show-toplevel)

# Port for this instance
PORT=${PORT:-8002}

export NHX_CONFIG_FILE_PATH=${ROOT_DIR}/packages/nhx_platform/config/local.yaml
export DATABASE_DIALECT=sqlite
export DATABASE_PATH=$HOME/.local/share/nemo/nhx-platform.db
export UVICORN_RELOAD=true

# Override URLs to point to this instance
# Use localhost (not 0.0.0.0) for client connections
# Note: Platform config uses NHX_ prefix (see config/base.py)
export NHX_BASE_URL="http://localhost:${PORT}"
export NHX_MODELS_URL="http://localhost:${PORT}"
export NHX_SECRETS_URL="http://localhost:${PORT}"
export NHX_FILES_URL="http://localhost:${PORT}"
export NHX_ENTITYSTORE_URL="http://localhost:${PORT}"

# Debug: print the URLs being used
echo "=== Platform URLs ==="
echo "BASE_URL: ${NHX_BASE_URL}"
echo "MODELS_URL: ${NHX_MODELS_URL}"
echo "===================="

uv run nemo services run \
	--services entities,secrets,files,models,inference-gateway \
	--controllers models \
	--host=0.0.0.0 \
	--port=${PORT}
