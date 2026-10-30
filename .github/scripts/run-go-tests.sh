#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Use tracked modules, not dependency/toolchain caches or other worktrees.
while IFS= read -r -d '' module; do
    printf '\nTesting Go module: %s\n' "$module"
    (
        cd "$(dirname "$module")"
        go test -race ./...
        go vet ./...
    )
done < <(git ls-files -z -- ':(glob)**/go.mod')
