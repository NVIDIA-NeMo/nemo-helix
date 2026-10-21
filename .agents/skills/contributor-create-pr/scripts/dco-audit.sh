#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Audit every commit in NHX_BASE_REF..HEAD for a Signed-off-by trailer whose email
# matches the commit author's email, case-insensitively.
#
# Usage: NHX_BASE_REF=<ref> NHX_PUSH_REMOTE=<remote> dco-audit.sh

set -euo pipefail

: "${NHX_BASE_REF:?set NHX_BASE_REF to the refreshed canonical upstream default branch}"
: "${NHX_PUSH_REMOTE:?set NHX_PUSH_REMOTE to the validated push remote}"

if [ "$(git rev-parse --is-shallow-repository)" = true ]; then
  git fetch --unshallow "$NHX_PUSH_REMOTE"
fi
if [ "$(git rev-parse --is-shallow-repository)" = true ] \
  || ! git merge-base "$NHX_BASE_REF" HEAD >/dev/null; then
  printf 'DCO audit requires complete feature-branch ancestry\n' >&2
  exit 1
fi

NHX_DCO_FAILED=0
NHX_COMMIT_COUNT=0

while IFS= read -r NHX_SHA; do
  NHX_COMMIT_COUNT=$((NHX_COMMIT_COUNT + 1))
  NHX_AUTHOR_EMAIL="$(git show -s --format=%ae "$NHX_SHA")"
  NHX_SIGNOFF_EMAILS="$(
    git show -s --format=%B "$NHX_SHA" \
      | git interpret-trailers --parse --no-divider \
      | awk -F '[<>]' 'tolower($1) ~ /^signed-off-by:[[:space:]]+[^[:space:]<][^<]*[[:space:]]+$/ && NF >= 3 { print $2 }'
  )"

  if ! printf '%s\n' "$NHX_SIGNOFF_EMAILS" \
    | awk -v want="$NHX_AUTHOR_EMAIL" 'tolower($0) == tolower(want) { found=1 } END { exit(found ? 0 : 1) }'; then
    printf 'DCO FAIL %s: no Signed-off-by trailer matches author email %s\n' "$NHX_SHA" "$NHX_AUTHOR_EMAIL" >&2
    NHX_DCO_FAILED=1
  else
    printf 'DCO OK   %s\n' "$NHX_SHA"
  fi
done < <(git rev-list --reverse "$NHX_BASE_REF..HEAD")

test "$NHX_COMMIT_COUNT" -gt 0
test "$NHX_DCO_FAILED" -eq 0
