#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

source_lock="docker/locks/nhx-gym-host/uv.lock"
bundled_lock="plugins/nemo-evals/src/nemo_evals/gym_registered_agent/nhx-gym-host.uv.lock"

# Same byte-for-byte invariant as test_the_bundled_host_lock_is_the_gym_host_images_lock,
# without loading the evals plugin's Python dependencies just to compare two files.
if ! cmp -s "${source_lock}" "${bundled_lock}"; then
  echo "Bundled Gym host lock is missing or differs from ${source_lock}." >&2
  echo "After regenerating the host lock, sync and stage the bundled copy:" >&2
  echo "  cp ${source_lock} ${bundled_lock}" >&2
  echo "  git add ${source_lock} ${bundled_lock}" >&2
  exit 1
fi
