#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Check if Python version argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <python_version>"
    echo "Example: $0 3.10"
    exit 1
fi

PYTHON_VERSION="$1"

# Get the repository base directory
REPO_ROOT=$(git rev-parse --show-toplevel)

cd "$REPO_ROOT/sdk/python/nemo-helix"
uv venv --python "$PYTHON_VERSION" ".venv-$PYTHON_VERSION"
source ".venv-${PYTHON_VERSION}/bin/activate"
uv pip install -e . --all-extras --group dev -r pyproject.toml
pytest tests/vendored/