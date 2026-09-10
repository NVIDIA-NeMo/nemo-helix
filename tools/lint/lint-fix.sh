#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
# Run all auto-fix commands in dependency order:
#   1. OpenAPI spec regeneration (other steps depend on this)
#   2. Web SDK regeneration (Orval reads openapi/ga/individual/platform.openapi.yaml)
#   3. Stainless sync (pulls updated Python SDK from Stainless; openapi already done in step 1)
#   4. Python style (ruff; run before vendoring so generated files aren't re-linted)
#   5. CLI command generation (the vendoring and docs are handled by the next step)
#   6. Vendor all packages (covers nemo_platform_ext too) + CLI reference docs
#   7. Copyright headers (after generated files are in place)
#   8. License update (may change after vendoring)
#   9. Config reference docs (independent, but run after structural changes)
#  10. Auth docs (regenerate permissions reference from static-authz.yaml)
#  11. Verification (optional) — run the same checks as CI (tools/lint/lint-all.sh)
#      Enable with LINT_FIX_VERIFY=1.
#
# Note: update-sdk = build-policy + refresh-openapi + stainless + update-cli, so we use
# stainless directly here to avoid re-running refresh-openapi and update-cli redundantly.
# Note: update-cli = vendor-nemo-platform-ext + generate-cli-reference-docs, but
# vendor-nemo-platform-ext is a subset of make vendor, so we let make vendor cover
# all vendoring and then regenerate the CLI reference docs once.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${CI_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${PROJECT_ROOT}" || exit 1

verify=false
if [[ "${LINT_FIX_VERIFY:-}" == "1" ]]; then
  verify=true
fi

declare -a steps=(
  "refresh-openapi:make refresh-openapi"
  "web-sdk:bash tools/lint/lint-fix-web-sdk.sh"
  "stainless:uv run --frozen nemo-platform-sdk-tools is-up-to-date --output-dir \"${TMPDIR:-/tmp}/nmp-sdk-lint\" || make stainless"
  "python-style:uv run ruff format && uv run ruff check --fix"
  "vendor+cli-reference-docs:make vendor && make generate-cli-reference-docs"
  "copyright-headers:make update-copyright-headers"
  "update-licenses:bash tools/lint/lint-fix-licenses.sh"
  "auth-config:uv run python services/core/auth/scripts/auth-tools.py update"
  "generate-config-docs:uv run generate-config-docs"
  "generate-auth-docs:uv run python services/core/auth/scripts/auth-tools.py generate-docs"
)

declare -a failed=()
declare -a timing_rows=()
for entry in "${steps[@]}"; do
  name="${entry%%:*}"
  cmd="${entry#*:}"
  echo ">>> ${name}: ${cmd}"
  start=$(date +%s)
  if eval "${cmd}"; then
    echo "[DONE] ${name}"
    result="DONE"
  else
    echo "[FAIL] ${name}"
    failed+=("${name}")
    result="FAIL"
  fi
  elapsed=$(( $(date +%s) - start ))
  timing_rows+=("$(printf '%-40s %s' "${name}" "${result} ${elapsed}s")")
  echo ""
done

echo "--- Fix summary ---"
echo "Completed: $((${#steps[@]} - ${#failed[@]}))"
echo "Failed: ${#failed[@]}"
echo ""
echo "Timings:"
for row in "${timing_rows[@]}"; do
  printf '  %s\n' "${row}"
done
if [[ ${#failed[@]} -gt 0 ]]; then
  echo ""
  echo "Failed fix steps: ${failed[*]}"
fi

if [[ "${verify}" == "true" ]]; then
  echo ""
  echo ">>> verification: bash tools/lint/lint-all.sh"
  verify_start=$(date +%s)
  if LINT_AFTER_FIX=1 bash tools/lint/lint-all.sh; then
    verify_result="PASS"
  else
    verify_result="FAIL"
    failed+=("verification")
  fi
  verify_elapsed=$(( $(date +%s) - verify_start ))
  printf '  %-40s %s\n' "verification (lint-all)" "${verify_result} ${verify_elapsed}s"
fi

echo ""
if [[ "${verify}" == "true" ]]; then
  echo "--- Overall summary ---"
  if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All fix steps and CI lint checks passed."
    exit 0
  fi
  if [[ ${#failed[@]} -eq 1 && " ${failed[*]} " == *" verification "* ]]; then
    echo "Auto-fix completed; remaining lint issues require manual fixes (see output above)."
  else
    echo "Some fix steps and/or lint checks failed: ${failed[*]}"
  fi
  exit 1
fi

if [[ ${#failed[@]} -eq 0 ]]; then
  echo "All fix steps completed."
  exit 0
fi
echo "Some fix steps failed: ${failed[*]}"
echo "Run with LINT_FIX_VERIFY=1 to check CI lint after fixing."
exit 1
