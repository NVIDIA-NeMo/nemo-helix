#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if ! [[ "${STALE_HOURS}" =~ ^[0-9]+$ ]]; then
  echo "stale_hours must be a positive integer, got: ${STALE_HOURS}" >&2
  exit 1
fi

if [ "${STALE_HOURS}" -lt 1 ]; then
  echo "stale_hours must be at least 1, got: ${STALE_HOURS}" >&2
  exit 1
fi

repo="${GITHUB_REPOSITORY#*/}"
repo="${repo,,}"
repo_root="$(git rev-parse --show-toplevel)"

date_utc() {
  if command -v gdate >/dev/null 2>&1; then
    gdate -u -d "$1" '+%Y-%m-%dT%H:%M:%SZ'
    return
  fi

  date -u -d "$1" '+%Y-%m-%dT%H:%M:%SZ'
}

cutoff="$(date_utc "${STALE_HOURS} hours ago")"

# GitHub's package version API exposes updated_at, not a separate last-pulled timestamp.
# Match RC artifacts and the exact historical release formats still in GHCR.
# Other timestamped tags keep their existing CI retention policy.
release_tag_pattern='^((rc|nightly)-[0-9]{14}|[0-9]+[.][0-9]+[.][0-9]+-(rc|night|nightly)-[0-9]{14})$'

# NGC metadata is required for each released container and chart. Its filename
# is also the corresponding GHCR package name.
if ! package_id_lines="$(
  find "${repo_root}/.github/assets/ngc/containers" \
    "${repo_root}/.github/assets/ngc/charts" \
    -maxdepth 1 -type f -name '*.md' -exec basename {} .md \; | sort -u
)"; then
  echo "Failed to discover NGC metadata files." >&2
  exit 1
fi

package_ids=()
if [ -n "${package_id_lines}" ]; then
  while IFS= read -r package_id; do
    package_ids+=("${package_id}")
  done <<< "${package_id_lines}"
fi

case "${CLEANUP_SCOPE:-ci}" in
  ci)
    tag_filter="select(((.metadata.container.tags // []) | any(test(\"${release_tag_pattern}\"))) | not)"
    ;;
  release)
    tag_filter="select((.metadata.container.tags // []) | any(test(\"${release_tag_pattern}\")))"
    ;;
  *)
    echo "cleanup_scope must be ci or release, got: ${CLEANUP_SCOPE}" >&2
    exit 1
    ;;
esac

echo "Scanning ${CLEANUP_SCOPE:-ci} package versions last updated before ${cutoff}"
echo "dry_run=${DRY_RUN}"

total_deleted=0
total_primary_candidates=0
total_child_candidates=0

url_encode() {
  jq -nr --arg value "$1" '$value|@uri'
}

find_package_versions() {
  local endpoint="$1"

  gh api --paginate "${endpoint}/versions?per_page=100" --jq '.[]'
}

find_stale_versions() {
  local package_versions="$1"
  local jq_filter
  jq_filter="
    select(.updated_at != null and .updated_at < \"${cutoff}\") |
    select((.metadata.container.tags // []) | length > 0) |
    select(((.metadata.container.tags // []) | index(\"latest\")) | not) |
    ${tag_filter} |
    [.id, .name, .updated_at, ((.metadata.container.tags // []) | join(\",\"))] |
    @tsv
  "

  jq -r "${jq_filter}" <<< "${package_versions}"
}

find_tagged_versions() {
  local package_versions="$1"

  jq -r '
    select((.metadata.container.tags // []) | length > 0) |
    [.id, .name] |
    @tsv
  ' <<< "${package_versions}"
}

find_manifest_children() {
  local package_ref="$1"
  local parent_versions="$2"
  local parent_id
  local parent_digest

  while IFS=$'\t' read -r parent_id parent_digest; do
    if [ -z "${parent_id}" ]; then
      continue
    fi

    if ! docker manifest inspect "${package_ref}@${parent_digest}" |
      jq -r --arg parent_id "${parent_id}" '
        .manifests[]? |
        [$parent_id, .digest] |
        @tsv
      '; then
      return 1
    fi
  done <<< "${parent_versions}"
}

find_child_versions() {
  local package_versions="$1"
  local child_digests="$2"
  local child_digest
  local child_version
  local child_id
  local child_tag_count

  while IFS= read -r child_digest; do
    if [ -z "${child_digest}" ]; then
      continue
    fi

    child_version="$(
      jq -r --arg digest "${child_digest}" '
        select(.name == $digest) |
        [.id, ((.metadata.container.tags // []) | length)] |
        @tsv
      ' <<< "${package_versions}"
    )"
    if [ -z "${child_version}" ]; then
      echo "Could not find package version for child manifest ${child_digest}." >&2
      return 1
    fi

    IFS=$'\t' read -r child_id child_tag_count <<< "${child_version}"
    if [ "${child_tag_count}" -ne 0 ]; then
      echo "Keeping child manifest ${child_digest}; it has tags." >&2
      continue
    fi

    printf '%s\t%s\n' "${child_id}" "${child_digest}"
  done <<< "${child_digests}"
}

for package_id in "${package_ids[@]}"; do
  package_name="${repo}/${package_id}"
  encoded_package_name="$(url_encode "${package_name}")"
  endpoint="/orgs/${GITHUB_REPOSITORY_OWNER}/packages/container/${encoded_package_name}"

  echo "::group::${package_name}"

  api_error="$(mktemp)"
  if ! package_versions="$(find_package_versions "${endpoint}" 2>"${api_error}")"; then
    if grep -q "HTTP 404" "${api_error}"; then
      echo "Package ${package_name} was not found; skipping."
      echo "::endgroup::"
      continue
    fi

    cat "${api_error}" >&2
    exit 1
  fi

  stale_versions="$(find_stale_versions "${package_versions}")"
  if [ -z "${stale_versions}" ]; then
    echo "No stale package versions found."
    echo "::endgroup::"
    continue
  fi

  tagged_versions="$(find_tagged_versions "${package_versions}")"
  candidate_ids="$(cut -f1 <<< "${stale_versions}")"
  candidate_parents="$(cut -f1,2 <<< "${stale_versions}")"
  retained_parents="$(
    awk -F $'\t' '
      NR == FNR { candidate[$1] = 1; next }
      !candidate[$1]
    ' <(printf '%s\n' "${candidate_ids}") <<< "${tagged_versions}"
  )"
  package_ref="ghcr.io/${GITHUB_REPOSITORY,,}/${package_id}"

  if ! candidate_children="$(find_manifest_children "${package_ref}" "${candidate_parents}")"; then
    echo "Could not inspect candidate manifests; no package versions were deleted." >&2
    exit 1
  fi

  candidate_child_digests="$(cut -f2 <<< "${candidate_children}" | sort -u)"
  retained_children=""
  if [ -n "${candidate_child_digests}" ] && [ -n "${retained_parents}" ]; then
    if ! retained_children="$(find_manifest_children "${package_ref}" "${retained_parents}")"; then
      echo "Could not inspect retained manifests; no package versions were deleted." >&2
      exit 1
    fi
  fi

  child_digests_to_delete="$(
    awk -F $'\t' '
      NR == FNR { retained[$2] = 1; next }
      !retained[$1] { print $1 }
    ' <(printf '%s\n' "${retained_children}") <(printf '%s\n' "${candidate_child_digests}")
  )"
  if ! child_versions="$(find_child_versions "${package_versions}" "${child_digests_to_delete}")"; then
    echo "Could not resolve child manifest versions; no package versions were deleted." >&2
    exit 1
  fi

  while IFS=$'\t' read -r version_id digest updated_at tags; do
    if [ -z "${version_id}" ]; then
      continue
    fi

    total_primary_candidates=$((total_primary_candidates + 1))
    tag_summary="${tags:-<untagged>}"
    echo "Candidate ${package_name}@${digest} updated_at=${updated_at} tags=${tag_summary}"
  done <<< "${stale_versions}"

  while IFS=$'\t' read -r child_id child_digest; do
    if [ -z "${child_id}" ]; then
      continue
    fi

    total_child_candidates=$((total_child_candidates + 1))
    echo "Child candidate ${package_name}@${child_digest}"
  done <<< "${child_versions}"

  if [ "${DRY_RUN}" = "true" ]; then
    echo "::endgroup::"
    continue
  fi

  while IFS=$'\t' read -r version_id digest updated_at tags; do
    if [ -z "${version_id}" ]; then
      continue
    fi

    gh api -X DELETE "${endpoint}/versions/${version_id}" >/dev/null
    total_deleted=$((total_deleted + 1))
    echo "Deleted package version ${version_id}"
  done <<< "${stale_versions}"

  while IFS=$'\t' read -r child_id child_digest; do
    if [ -z "${child_id}" ]; then
      continue
    fi

    gh api -X DELETE "${endpoint}/versions/${child_id}" >/dev/null
    total_deleted=$((total_deleted + 1))
    echo "Deleted child package version ${child_id}"
  done <<< "${child_versions}"

  echo "::endgroup::"
done

echo "Stale primary candidates: ${total_primary_candidates}"
echo "Child manifest candidates: ${total_child_candidates}"
echo "Deleted versions: ${total_deleted}"
