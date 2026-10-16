#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
# Run Python type checker in CI.
#
# The rules below are globally suppressed for CI only because the existing
# codebase still contains pre-existing violations left over. 
# They should be fixed incrementally and re-enabled as they are fixed.
# Generate error reports by running `make reports` in script/ty_issues_report.
# Counts reflect the violation count at the time of suppression.
ci_ignored_rules=(
  invalid-argument-type    # 204
  unresolved-attribute     # 72
  # Re-enable last: while the rules above are ignored, the ty: ignore comments that suppress
  # their (still real) violations would all report as unused.
  unused-ignore-comment    # 14
)

ignore_args=()
for rule in "${ci_ignored_rules[@]}"; do
  ignore_args+=(--ignore "$rule")
done

uv run --frozen --group insights --group typecheck ty check --exit-zero-on-warning "${ignore_args[@]}"
