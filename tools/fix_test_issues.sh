#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fix common test issues for pytest

set -e

echo "=========================================="
echo "Fixing NeMo Helix Test Issues"
echo "=========================================="
echo ""

echo "1. Fixing test file name conflicts..."
# Fix test_utils.py conflict
if [ -f "packages/data_designer/tests/engine/sampling_gen/test_utils.py" ]; then
    echo "   - Renaming data_designer test_utils.py to test_sampling_utils.py"
    mv packages/data_designer/tests/engine/sampling_gen/test_utils.py \
       packages/data_designer/tests/engine/sampling_gen/test_sampling_utils.py
fi

echo ""
echo "2. Cleaning Python cache files..."
echo "   - Removing __pycache__ directories"
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "   - Removing .pyc files"
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo ""
echo "3. Cleaning pytest cache..."
rm -rf .pytest_cache

echo ""
echo "=========================================="
echo "✓ Fixes applied successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run unit tests: uv run pytest -m 'not e2e and not integration'"
echo "  2. Or use helper:   uv run python tools/run_all_tests.py"
echo "  3. See TEST_FAILURE_ANALYSIS.md for details"
echo ""

