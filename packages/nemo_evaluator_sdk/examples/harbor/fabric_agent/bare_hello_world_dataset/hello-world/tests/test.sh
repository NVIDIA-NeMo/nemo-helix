#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Self-contained verifier (no network/pip): pass iff /app/hello.txt says
# "Hello, world!". Harbor reads the reward from /logs/verifier/reward.txt.
mkdir -p /logs/verifier

# `$(cat ...)` strips trailing newlines but keeps interior ones, so "Hello,\n world!" fails here
# while a file written with or without a trailing newline passes -- the instruction asks for neither.
if [ -f /app/hello.txt ] && [ "$(cat /app/hello.txt)" = "Hello, world!" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
