#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

old_product="NeMo Plat""form"
old_product_lower_platform="NeMo plat""form"
old_product_compact="NeMoPlat""form"
old_product_kebab="NeMo-Plat""form"
old_product_kebab_lower_platform="NeMo-plat""form"
old_title_product="Nemo Plat""form"
old_title_compact="NemoPlat""form"
old_title_kebab="Nemo-Plat""form"
old_title_kebab_lower_platform="Nemo-plat""form"
old_lower_product="nemo plat""form"
old_lower_compact="nemoplat""form"
old_lower_camel="nemoPlat""form"
old_slug="nemo-plat""form"
old_module="nemo_plat""form"
old_upper_product="NEMO PLAT""FORM"
old_upper_title_product="NEMO Plat""form"
old_upper_slug="NEMO-PLAT""FORM"
old_upper_module="NEMO_PLAT""FORM"
old_acronym_upper="NM""P"
old_acronym_title="Nm""p"
old_acronym_lower="nm""p"

new_product="NeMo Helix"
new_product_lower_platform="NeMo Helix"
new_product_compact="NeMoHelix"
new_product_kebab="NeMo-Helix"
new_product_kebab_lower_platform="NeMo-Helix"
new_title_product="Nemo Helix"
new_title_compact="NemoHelix"
new_title_kebab="Nemo-Helix"
new_title_kebab_lower_platform="Nemo-Helix"
new_lower_product="nemo helix"
new_lower_compact="nemohelix"
new_lower_camel="nemoHelix"
new_slug="nemo-helix"
new_module="nemo_helix"
new_upper_product="NEMO HELIX"
new_upper_title_product="NEMO Helix"
new_upper_slug="NEMO-HELIX"
new_upper_module="NEMO_HELIX"
new_acronym_upper="NHX"
new_acronym_title="Nhx"
new_acronym_lower="nhx"

old_values=(
  "$old_product"
  "$old_product_lower_platform"
  "$old_product_compact"
  "$old_product_kebab"
  "$old_product_kebab_lower_platform"
  "$old_title_product"
  "$old_title_compact"
  "$old_title_kebab"
  "$old_title_kebab_lower_platform"
  "$old_lower_product"
  "$old_lower_compact"
  "$old_lower_camel"
  "$old_slug"
  "$old_module"
  "$old_upper_product"
  "$old_upper_title_product"
  "$old_upper_slug"
  "$old_upper_module"
  "$old_acronym_upper"
  "$old_acronym_title"
  "$old_acronym_lower"
)

new_values=(
  "$new_product"
  "$new_product_lower_platform"
  "$new_product_compact"
  "$new_product_kebab"
  "$new_product_kebab_lower_platform"
  "$new_title_product"
  "$new_title_compact"
  "$new_title_kebab"
  "$new_title_kebab_lower_platform"
  "$new_lower_product"
  "$new_lower_compact"
  "$new_lower_camel"
  "$new_slug"
  "$new_module"
  "$new_upper_product"
  "$new_upper_title_product"
  "$new_upper_slug"
  "$new_upper_module"
  "$new_acronym_upper"
  "$new_acronym_title"
  "$new_acronym_lower"
)

# These are first-party published image names that predate the common prefix.
unprefixed_images=(
  "auditor-tasks"
  "guardrails-callout-mock-llm"
  "guardrails-callout"
  "safe-synthesizer-tasks"
)

renamed_path() {
  local renamed="$1"
  local index image
  for index in "${!old_values[@]}"; do
    renamed="${renamed//${old_values[$index]}/${new_values[$index]}}"
  done
  for image in "${unprefixed_images[@]}"; do
    if [[ "$renamed" != *"$new_acronym_lower-$image"* ]]; then
      renamed="${renamed//$image/$new_acronym_lower-$image}"
    fi
  done
  printf '%s\n' "$renamed"
}

usage() {
  echo "Usage: $0 [--dry-run|--continue]"
}

inventory() {
  echo "Legacy content categories:"
  for index in "${!old_values[@]}"; do
    local old_value="${old_values[$index]}"
    local new_value="${new_values[$index]}"
    local count
    count="$({ git grep -IohF "$old_value" -- . 2>/dev/null || true; } | wc -l)"
    printf '  %-24s -> %-24s %8d matching lines\n' "$old_value" "$new_value" "$count"
  done

  echo
  echo "Legacy tracked paths:"
  git ls-files | while IFS= read -r path; do
    local renamed
    renamed="$(renamed_path "$path")"
    if [[ "$renamed" != "$path" ]]; then
      printf '  %s -> %s\n' "$path" "$renamed"
    fi
  done

  echo
  echo "First-party published image renames:"
  grep -Eo '(sha_and_maybe_latest_tags|base_tags)\("[^"]+"\)' docker-bake.hcl \
    | sed -E 's/^[^(]+\("([^"]+)"\)$/\1/' \
    | sort -u \
    | while IFS= read -r image; do
        local renamed_image="$image"
        for index in "${!old_values[@]}"; do
          renamed_image="${renamed_image//${old_values[$index]}/${new_values[$index]}}"
        done
        if [[ "$renamed_image" != "$new_acronym_lower-"* ]]; then
          renamed_image="$new_acronym_lower-$renamed_image"
        fi
        if [[ "$renamed_image" != "$image" ]]; then
          printf '  %s -> %s\n' "$image" "$renamed_image"
        fi
      done
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

mode="${1:-}"
if [[ "$mode" == "--dry-run" ]]; then
  inventory
  exit 0
fi

if [[ $# -eq 1 && "$mode" != "--continue" ]]; then
  usage >&2
  exit 2
fi

if [[ "$mode" != "--continue" && -n $(git status --short) ]]; then
  echo "The worktree must be clean before running the rename." >&2
  exit 1
fi

export PAIR_COUNT="${#old_values[@]}"
for index in "${!old_values[@]}"; do
  printf -v suffix '%02d' "$index"
  export "OLD_$suffix=${old_values[$index]}"
  export "NEW_$suffix=${new_values[$index]}"
done

grep_args=()
for old_value in "${old_values[@]}"; do
  grep_args+=("-e" "$old_value")
done
mapfile -d '' content_files < <(
  rg -l -0 -I -F --hidden --glob '!.git' --glob '!.git/**' "${grep_args[@]}" . || true
)

for path in "${content_files[@]}"; do
  [[ -L "$path" ]] && continue
  perl -0pi -e '
    for my $index (0 .. $ENV{PAIR_COUNT} - 1) {
      my $old = $ENV{sprintf("OLD_%02d", $index)};
      my $new = $ENV{sprintf("NEW_%02d", $index)};
      s/\Q$old\E/$new/g;
    }
  ' "$path"
done

# Normalize published image names which did not previously carry the project prefix.
image_grep_args=()
for image in "${unprefixed_images[@]}"; do
  image_grep_args+=("-e" "$image")
done
mapfile -d '' image_files < <(
  rg -l -0 -I -F --hidden --glob '!.git' --glob '!.git/**' \
    --glob '!scripts/rename-to-nemo-helix.sh' "${image_grep_args[@]}" . || true
)
export IMAGE_PREFIX="$new_acronym_lower-"
export IMAGE_NAMES="$(printf '%s\n' "${unprefixed_images[@]}")"
for path in "${image_files[@]}"; do
  [[ -L "$path" ]] && continue
  perl -0pi -e '
    BEGIN {
      @images = split /\n/, $ENV{IMAGE_NAMES};
      $pattern = join "|", map { quotemeta } @images;
    }
    s/(?<![A-Za-z0-9-])($pattern)/$ENV{IMAGE_PREFIX}$1/g;
  ' "$path"
done

# Rename tracked and newly-created files without traversing ignored build environments.
while IFS= read -r -d '' path; do
  [[ -e "$path" || -L "$path" ]] || continue
  destination="$(renamed_path "$path")"
  [[ "$destination" == "$path" ]] && continue
  if [[ -e "$destination" || -L "$destination" ]]; then
    echo "Cannot rename $path: destination already exists: $destination" >&2
    exit 1
  fi
  mkdir -p "${destination%/*}"
  mv "$path" "$destination"
done < <(git ls-files -z --cached --others --exclude-standard)

echo "Rename complete. Run scripts/verify-nemo-helix-rename.sh before committing."
