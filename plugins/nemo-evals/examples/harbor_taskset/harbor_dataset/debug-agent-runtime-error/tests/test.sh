#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The first multi-step agent phase is expected to fail, but Harbor still runs
# this shared verifier. Its zero reward plus the step-level exception reproduces
# the dangerous result shape: a consumer that reads only top-level
# exception_info will misclassify the infrastructure failure as a real score.
# Keep the same metric keys as the healthy validation tasks so aggregation can
# still describe the failed attempt.

set -uo pipefail

mkdir -p /logs/verifier
printf '{"reward": 0.0, "format_ok": 0.0}\n' > /logs/verifier/reward.json
cat /logs/verifier/reward.json
