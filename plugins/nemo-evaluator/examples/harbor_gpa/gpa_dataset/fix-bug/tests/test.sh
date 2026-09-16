#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Outcome verifier: the check passes and check.py itself was not edited.
mkdir -p /logs/verifier

if ! grep -q 'cases = \[(\[2, 4, 6\], 4.0), (\[1\], 1.0), (\[1.5, 2.5\], 2.0), (\[10, 20, 30, 40\], 25.0)\]' /app/check.py; then
  echo 0 > /logs/verifier/reward.txt
elif (cd /app && python3 check.py > /logs/verifier/check.log 2>&1); then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
