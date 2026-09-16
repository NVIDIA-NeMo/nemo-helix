#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

legacy_product_pattern="nemo[ _-]?plat""form"
legacy_acronym_upper="NM""P"
legacy_acronym_title="Nm""p"
legacy_acronym_lower="nm""p"
expected_image_prefix="nhx-"
failed=0

if rg -n -I -i --hidden --glob '!.git' --glob '!.git/**' "$legacy_product_pattern" .; then
  echo "Legacy product names remain in tracked file contents." >&2
  failed=1
fi

if rg -n -I -F --hidden --glob '!.git' --glob '!.git/**' \
  -e "$legacy_acronym_upper" -e "$legacy_acronym_title" -e "$legacy_acronym_lower" .; then
  echo "Legacy acronym references remain in tracked file contents." >&2
  failed=1
fi

while IFS= read -r path; do
  [[ -e "$path" || -L "$path" ]] || continue
  if [[ "${path,,}" =~ $legacy_product_pattern \
    || "$path" == *"$legacy_acronym_upper"* \
    || "$path" == *"$legacy_acronym_title"* \
    || "$path" == *"$legacy_acronym_lower"* ]]; then
    echo "Legacy name remains in tracked path: $path" >&2
    failed=1
  fi
done < <(git ls-files --cached --others --exclude-standard)

while IFS= read -r image; do
  if [[ "$image" != "$expected_image_prefix"* ]]; then
    echo "First-party published image lacks the $expected_image_prefix prefix: $image" >&2
    failed=1
  fi
done < <(
  grep -Eo '(sha_and_maybe_latest_tags|base_tags)\("[^"]+"\)' docker-bake.hcl \
    | sed -E 's/^[^(]+\("([^"]+)"\)$/\1/' \
    | sort -u
)

if ((failed)); then
  exit 1
fi

echo "No legacy product, package, acronym, path, or first-party image names remain."
