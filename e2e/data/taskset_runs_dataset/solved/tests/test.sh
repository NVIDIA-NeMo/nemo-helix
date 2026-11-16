#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Both tasks emit the same keys: the merge rejects inconsistent metric sets. format_ok 1 proves the artifact
# reached the verifier: a missing file scores reward 0 too, but with format_ok 0.
mkdir -p /logs/verifier
reward=0.0
format_ok=0.0
if [ -f /app/artifacts/output.txt ]; then
  format_ok=1.0
  [ "$(cat /app/artifacts/output.txt)" = "Hello, universe!" ] && reward=1.0
fi
printf '{"reward": %s, "format_ok": %s}\n' "$reward" "$format_ok" > /logs/verifier/reward.json
