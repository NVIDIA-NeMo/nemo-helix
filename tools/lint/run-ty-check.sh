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

uv run --frozen --group insights --group typecheck --group scaled-evals ty check --exit-zero-on-warning "${filtered_files[@]}"
