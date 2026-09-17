#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -uo pipefail
# Run all lint scripts serially, report summary, exit with failure if any failed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CI_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}" || exit 1

declare -a scripts=(
  "check-uv-version-consistency:tools/lint/check-uv-version.sh"
  "check-python-version-consistency:tools/lint/check-python-version.sh"
  "check-node-version-consistency:tools/lint/check-node-version.sh"
  "check-flox-lock-consistency:tools/lint/check-flox-locks.sh"
  "lint-licenses:tools/lint/lint-licenses.sh"
  "lint-openapi:tools/lint/lint-openapi.sh"
  "lint-config-reference-docs:tools/lint/lint-config-reference-docs.sh"
  "lint-python-style:tools/lint/lint-python-style.sh"
  "lint-python-types:tools/lint/lint-python-types.sh"
  "lint-sdk-vendored:tools/lint/lint-sdk-vendored.sh"
  "lint-web-sdk:tools/lint/lint-web-sdk.sh"
  "lint-cli:tools/lint/lint-cli.sh"
  "lint-auth-config:tools/lint/lint-auth-config.sh"
  "lint-merge-conflict:tools/lint/lint-merge-conflict.sh"
  "lint-copyright-headers:tools/lint/lint-copyright-headers.sh"
  "lint-no-nmp-common-in-plugins:tools/lint/lint-no-nmp-common-in-plugins.sh"
  "lint-pre-commit-all:tools/lint/lint-pre-commit-all.sh"
)

is_no_fix_lint() {
  local lint_name="$1"
  case "${lint_name}" in
    lint-python-types|lint-merge-conflict|lint-no-nmp-common-in-plugins)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

declare -a failed=()
declare -a timing_lines=()
for entry in "${scripts[@]}"; do
  name="${entry%%:*}"
  path="${entry#*:}"
  start=$(date +%s)
  if bash "${path}"; then
    echo "[PASS] ${name}"
    result="PASS"
  else
    echo "[FAIL] ${name}"
    failed+=("${name}")
    result="FAIL"
  fi
  elapsed=$(( $(date +%s) - start ))
  timing_lines+=("${name}:${result} ${elapsed}s")
done

echo ""
echo "--- Lint summary ---"
echo "Passed: $((${#scripts[@]} - ${#failed[@]}))"
echo "Failed: ${#failed[@]}"
echo ""
echo "Timings:"
for line in "${timing_lines[@]}"; do
  name="${line%%:*}"
  details="${line#*:}"
  printf "  %-40s %s\n" "${name}" "${details}"
done
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed lints: ${failed[*]}"
  echo ""
  # Check if any failed lint has an auto-fix command
  has_fix=false
  for name in "${failed[@]}"; do
    if ! is_no_fix_lint "${name}"; then
      has_fix=true
      break
    fi
  done
  if [[ "${has_fix}" == "true" && "${LINT_AFTER_FIX:-}" != "1" ]]; then
    echo "To fix auto-fixable issues, run:"
    echo "  make lint-fix"
    echo ""
  elif [[ "${LINT_AFTER_FIX:-}" == "1" ]]; then
    echo "Auto-fix completed; remaining issues require manual fixes (see output above)."
    echo ""
  fi
  for name in "${failed[@]}"; do
    if is_no_fix_lint "${name}"; then
      echo "  ${name}: (see output above for manual fix)"
    fi
  done
  exit 1
fi
exit 0
