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
  unused-ignore-comment    # 14
  unresolved-attribute     # 72
  not-subscriptable        # 31
  invalid-assignment       # 30
  invalid-return-type      # 18
  invalid-method-override  # 18
  no-matching-overload     # 2
  unsupported-operator     # 2
)

ignore_args=()
for rule in "${ci_ignored_rules[@]}"; do
  ignore_args+=(--ignore "$rule")
done

uv run --frozen --group insights --group typecheck --group scaled-evals ty check --exit-zero-on-warning "${ignore_args[@]}"
