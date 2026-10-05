#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Shared test runner for Harbor evals.
# This file is sourced by test.sh scripts via: source /app/tests/agentic-use/shared/verify-tests.sh
# Note: curl, uv, and Python are already installed in the nhx-agentic-base base image.

# This is installed in the nhx-agentic-base base image at /app/tests/agentic-use/shared/verify-tests.sh

# Add shared utilities (trace_reader, etc.) to Python path
export PYTHONPATH="/app/tests/agentic-use/shared:${PYTHONPATH}"

/app/.venv/bin/python -m pytest /tests/test_outputs.py -rA
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
