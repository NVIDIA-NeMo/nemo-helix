#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: collect-git-evidence.sh <previous-ref> <release-ref> <main-ref> <output-dir>

Collect deterministic Git evidence for the release-test-scope skill. The output
is for analysis, not a user-facing artifact.
USAGE
}

if [ "$#" -ne 4 ]; then
  usage
  exit 2
fi

previous_ref="$1"
release_ref="$2"
main_ref="$3"
out_dir="$4"

mkdir -p "$out_dir"

previous_sha="$(git rev-parse --verify "${previous_ref}^{commit}")"
release_sha="$(git rev-parse --verify "${release_ref}^{commit}")"
main_sha="$(git rev-parse --verify "${main_ref}^{commit}")"

git merge-base --is-ancestor "$previous_sha" "$release_sha"
git merge-base --is-ancestor "$previous_sha" "$main_sha"
merge_base="$(git merge-base "$release_sha" "$main_sha")"

{
  printf 'name\tinput\tsha\n'
  printf 'previous\t%s\t%s\n' "$previous_ref" "$previous_sha"
  printf 'release\t%s\t%s\n' "$release_ref" "$release_sha"
  printf 'main\t%s\t%s\n' "$main_ref" "$main_sha"
  printf 'release_main_merge_base\t%s...%s\t%s\n' "$release_ref" "$main_ref" "$merge_base"
} >"$out_dir/refs.tsv"

git log --format='%H%x09%P%x09%cs%x09%s' \
  "$previous_sha..$release_sha" >"$out_dir/release-commits.tsv"

git log --left-right --cherry-mark --no-merges \
  --format='%m%x09%H%x09%cs%x09%s' \
  "$release_sha...$main_sha" >"$out_dir/release-main-cherry.tsv"

git diff --name-status "$previous_sha..$release_sha" -- docs/ \
  >"$out_dir/docs-name-status.tsv"

git diff --find-renames "$previous_sha..$release_sha" -- docs/ \
  >"$out_dir/docs.diff"

git diff --find-renames "$previous_sha..$release_sha" -- \
  docs/about/release-notes/ >"$out_dir/release-notes.diff"

git log --format='%H%x09%cs%x09%s' \
  "$previous_sha..$release_sha" -- docs/about/release-notes/ \
  >"$out_dir/release-notes-history.tsv"

git log --right-only --first-parent --merges \
  --format='%H%x09%P%x09%cs%x09%s' \
  "$release_sha...$main_sha" >"$out_dir/main-first-parent-merges.tsv"

if git cat-file -e "$release_sha:docs/fern/versions/latest.yml" 2>/dev/null; then
  git show "$release_sha:docs/fern/versions/latest.yml" \
    >"$out_dir/release-fern-latest.yml"
fi

printf 'wrote deterministic release evidence to %s\n' "$out_dir"
