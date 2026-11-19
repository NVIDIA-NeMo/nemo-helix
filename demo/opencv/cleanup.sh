#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

case "${1:-}" in
    "") apply=false ;;
    --apply) apply=true ;;
    *) echo "Usage: $0 [--apply]" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Usage: $0 [--apply]" >&2; exit 2; }

# This demo runs against the local Files and Jobs services.
export NHX_BASE_URL=http://127.0.0.1:8080
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
nemo="$repo_root/.venv/bin/nemo"

list_jobs() {
    "$nemo" jobs list --workspace default --filter.source opencv-demo --all-pages -f json |
        jq -r '.data[] | select(.source == "opencv-demo") | [.name, .status] | @tsv'
}

list_filesets() {
    "$nemo" files filesets list --workspace default --all-pages -f json |
        jq -r '.data[].name | select(test("^(images|sutro-tower-images-[0-9a-f]{10}|job-fileset-opencv-[0-9a-f]{10}|opencv-[0-9a-f]{10}-(images|edges|montage|captions))$"))'
}

jobs=$(list_jobs)
filesets=$(list_filesets)
printf 'OpenCV Jobs in local/default:\n%s\nFileSets (including the shared demo input):\n%s\n' "${jobs:-(none)}" "${filesets:-(none)}"
if ! "$apply"; then
    echo "Preview only. Run again with --apply to delete these resources."
    exit 0
fi

# Job deletion is asynchronous. Let the controller drain running steps before
# removing the FileSets they use, including when resuming an interrupted cleanup.
while IFS=$'\t' read -r job state; do
    [[ -n "$job" ]] || continue
    deadline=$((SECONDS + 300))
    requested=""
    while [[ -n "$state" ]]; do
        case "$state" in
            deletion_pending|cancelling) ;;
            completed|error|cancelled|creation_pending)
                if [[ "$requested" != delete ]]; then
                    "$nemo" jobs delete "$job" --workspace default
                    requested=delete
                fi ;;
            *)
                if [[ -z "$requested" ]]; then
                    "$nemo" jobs cancel "$job" --workspace default
                    requested=cancel
                fi ;;
        esac
        if (( SECONDS >= deadline )); then
            echo "Timed out waiting for $job (last status: $state). Rerun cleanup after checking the Jobs controller." >&2
            exit 1
        fi
        sleep 2
        state=$(list_jobs | awk -v name="$job" '$1 == name {print $2}')
    done
done <<< "$jobs"

# The controller may already have reclaimed a Job's artifact FileSet.
filesets=$(list_filesets)
while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    if ! "$nemo" files filesets delete "$name" --workspace default; then
        remaining=$(list_filesets)
        if grep -Fxq "$name" <<< "$remaining"; then exit 1; fi
    fi
done <<< "$filesets"

jobs=$(list_jobs)
filesets=$(list_filesets)
if [[ -n "$jobs" || -n "$filesets" ]]; then
    echo "OpenCV resources still present. Rerun the preview to inspect them." >&2
    exit 1
fi
echo "Verified: no matching OpenCV Jobs or FileSets remain in local/default."
