#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Never decides this task's outcome: solution/solve.sh sleeps past the 1s agent
# timeout, so Harbor records exception_info before the verifier matters. Kept
# byte-identical to hello-world's so the task shape matches for Harbor discovery.
mkdir -p /logs/verifier

if [ -f /app/hello.txt ] && [ "$(tr -d '\n' < /app/hello.txt)" = "Hello, world!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
