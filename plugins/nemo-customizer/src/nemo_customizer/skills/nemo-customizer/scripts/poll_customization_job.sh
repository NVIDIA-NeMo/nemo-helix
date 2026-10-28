#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Poll a customization job until its top-level status is terminal, printing training health on each poll.
# Usage: poll_customization_job.sh <plugin>-<job-id> [interval_seconds] [--continue-on-alert]
# Resolves `nemo` on PATH, else `uv run nemo` (see SKILL.md Pre-flight).
# Exit 0 on completed; 1 on error, cancelled, or get-status failure;
# 3 when job_health.py raises an ALERT while the job is still running (the job keeps running:
# ask the user, then cancel it or re-run with --continue-on-alert).

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

JOB="${1:?usage: poll_customization_job.sh <plugin>-<id> [interval_seconds] [--continue-on-alert]}"
shift
INTERVAL=15
CONTINUE_ON_ALERT=false
for arg in "$@"; do
  case "$arg" in
    --continue-on-alert) CONTINUE_ON_ALERT=true ;;
    *) INTERVAL="$arg" ;;
  esac
done

while true; do
  JSON=$(run_nemo jobs get-status "$JOB" -f json 2>/dev/null) || {
    echo "get-status failed for $JOB" >&2
    exit 1
  }
  STATUS=$(printf '%s' "$JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
  HEALTH=$(printf '%s' "$JSON" | python3 "$HERE/job_health.py")
  echo "$(date +%H:%M:%S) $HEALTH"
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
  if [[ "$CONTINUE_ON_ALERT" != true ]] && grep -q '^ALERT:' <<<"$HEALTH"; then
    exit 3
  fi
  sleep "$INTERVAL"
done
