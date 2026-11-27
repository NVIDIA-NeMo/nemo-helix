#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

git ls-files -z -- 'go.mod' '**/go.mod' |
  jq -Rsc 'split("\u0000") | map(select(length > 0) | if . == "go.mod" then "." else rtrimstr("/go.mod") end)'
