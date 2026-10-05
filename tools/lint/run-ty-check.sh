#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Get the list of changed Python files
files=$(git diff --cached --name-only --diff-filter=ACMR | grep '\.py$' || true)

if [ -z "$files" ]; then
	echo "No Python files to check"
	exit 0
fi

# Filter out files excluded in pyproject.toml's [tool.ty.src].exclude.
filtered_files=()
while IFS= read -r filtered_file; do
	[ -n "$filtered_file" ] || continue
	filtered_files+=("$filtered_file")
done < <(printf '%s\n' "$files" | uv run --frozen tools/lint/filter_ty_exclusions.py)

if [ ${#filtered_files[@]} -eq 0 ]; then
	echo "No Python files to check (all files are excluded)"
	exit 0
fi

# Match the CI type-check policy: the repository still has known violations in
# these categories, so pre-commit should not block broad mechanical changes on
# diagnostics that CI intentionally suppresses.
ci_ignored_rules=(
	invalid-argument-type
	unused-ignore-comment
	unresolved-attribute
	not-subscriptable
	invalid-assignment
	invalid-return-type
	invalid-method-override
	no-matching-overload
	unsupported-operator
)

ignore_args=()
for rule in "${ci_ignored_rules[@]}"; do
	ignore_args+=(--ignore "$rule")
done

uv run --frozen --group typecheck ty check --exit-zero-on-warning "${ignore_args[@]}" "${filtered_files[@]}"
