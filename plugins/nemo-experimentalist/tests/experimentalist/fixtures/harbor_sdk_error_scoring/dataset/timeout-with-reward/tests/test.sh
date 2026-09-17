#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

mkdir -p /logs/verifier
printf '{"reward": 0.8}\n' > /logs/verifier/reward.json
cat /logs/verifier/reward.json
