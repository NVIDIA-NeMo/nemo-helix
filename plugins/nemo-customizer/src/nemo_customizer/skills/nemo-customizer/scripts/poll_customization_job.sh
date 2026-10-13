#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Poll customization job until top-level status is terminal.
# Usage: poll_customization_job.sh <plugin>-<job-id> [interval_seconds]
# Requires: NHX_BASE_URL; run from nemo-helix root.
# Resolves `nemo` on PATH, else `uv run nemo` (see SKILL.md Pre-flight).
# Exit 0 on completed; exit 1 on error, cancelled, or get-status failure.

set -euo pipefail

run_nemo() {
  if command -v nemo >/dev/null 2>&1; then
    nemo "$@"
  elif command -v uv >/dev/null 2>&1 && uv run nemo --help >/dev/null 2>&1; then
    uv run nemo "$@"
  else
    echo "NeMo CLI not found. Install NeMo Helix (nemo-setup: make bootstrap && nemo setup)." >&2
    return 127
  fi
}

JOB="${1:?usage: poll_customization_job.sh <plugin>-<id> [interval_seconds]}"
INTERVAL="${2:-15}"

while true; do
  JSON=$(run_nemo jobs get-status "$JOB" 2>/dev/null) || {
    echo "get-status failed for $JOB" >&2
    exit 1
  }
  read -r STATUS PHASE <<<"$(printf '%s' "$JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['status'], d.get('status_details', {}).get('phase', ''))
")"
  echo "$(date +%H:%M:%S) status=$STATUS phase=$PHASE"
  case "$STATUS" in
    completed)
      printf '%s\n' "$JSON" | python3 -m json.tool
      exit 0
      ;;
    error|cancelled)
      printf '%s\n' "$JSON" | python3 -m json.tool >&2
      exit 1
      ;;
  esac
  sleep "$INTERVAL"
done
