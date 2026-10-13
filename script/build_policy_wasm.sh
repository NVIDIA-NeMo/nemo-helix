#!/usr/bin/env sh
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds policy.wasm from OPA Rego policy sources using a pinned OPA version.
#
# Used in all contexts: local dev, SDK wheel builds, and Docker image builds.
# The pinned version ensures reproducible output regardless of what (if anything)
# the developer has installed locally.
#
# Environment variables:
#   OUTPUT_DIR     - Directory to write policy.wasm into.
#                    Default: services/core/auth/src/nhx/core/auth/assets
#   REPO_ROOT      - Repository root. Default: auto-detected via git.
#   OPA_VERSION    - OPA release to use. Default: v1.8.0
#   OPA_BIN        - Optional explicit OPA binary path. Must match OPA_VERSION.
#   OPA_CACHE_DIR  - Directory for downloaded OPA binaries. Default: .cache/opa
#   OPA_DOWNLOAD_BASE_URLS
#                  - Space-separated OPA download roots to try in order.
#                    Default: official OPA downloads, then GitHub releases.
#   OPA_DOWNLOAD_BASE_URL
#                  - Legacy single download root override.
set -eu

# --- Configuration ---
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
OPA_VERSION="${OPA_VERSION:-v1.8.0}"
OPA_VERSION_NO_V="${OPA_VERSION#v}"
OPA_CACHE_DIR="${OPA_CACHE_DIR:-${REPO_ROOT}/.cache/opa}"
if [ -n "${OPA_DOWNLOAD_BASE_URLS:-}" ]; then
  RESOLVED_OPA_DOWNLOAD_BASE_URLS="${OPA_DOWNLOAD_BASE_URLS}"
elif [ -n "${OPA_DOWNLOAD_BASE_URL:-}" ]; then
  RESOLVED_OPA_DOWNLOAD_BASE_URLS="${OPA_DOWNLOAD_BASE_URL}"
else
  RESOLVED_OPA_DOWNLOAD_BASE_URLS="https://openpolicyagent.org/downloads https://github.com/open-policy-agent/opa/releases/download"
fi

POLICY_DIR="${REPO_ROOT}/services/core/auth/src/nhx/core/auth/app/policies"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/services/core/auth/src/nhx/core/auth/assets}"
ENTRYPOINTS="-e authz/allow -e authz/has_permissions -e authz/has_role"

detect_opa_asset() {
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"

  case "${arch}" in
    x86_64 | amd64) arch="amd64" ;;
    aarch64 | arm64) arch="arm64" ;;
    *)
      echo "Unsupported architecture for OPA download: ${arch}" >&2
      exit 1
      ;;
  esac

  case "${os}" in
    linux | darwin) ;;
    *)
      echo "Unsupported OS for OPA download: ${os}" >&2
      exit 1
      ;;
  esac

  echo "opa_${os}_${arch}_static"
}

print_opa_help() {
  asset="$1"
  cache_dir="${OPA_CACHE_DIR}/${OPA_VERSION}"
  candidate="${cache_dir}/${asset}"

  cat >&2 <<EOF

Unable to prepare OPA ${OPA_VERSION}.

Offline options:
  1. Provide a local OPA ${OPA_VERSION} binary:
       OPA_BIN=/path/to/${asset} ./script/build_policy_wasm.sh

  2. Seed the script cache and rerun:
       mkdir -p ${cache_dir}
       cp /path/to/${asset} ${candidate}
       chmod +x ${candidate}
       ./script/build_policy_wasm.sh

The binary must report "Version: ${OPA_VERSION_NO_V}" from:
  /path/to/${asset} version
EOF
}

opa_version_matches() {
  candidate="$1"
  version="$("${candidate}" version 2>/dev/null | awk '/^Version:/ {print $2; exit}')"
  test "${version}" = "${OPA_VERSION_NO_V}"
}

sha256_file() {
  file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${file}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${file}" | awk '{print $1}'
  else
    echo "Neither sha256sum nor shasum is available for OPA checksum verification." >&2
    exit 1
  fi
}

download_opa() {
  asset="$(detect_opa_asset)"
  cache_dir="${OPA_CACHE_DIR}/${OPA_VERSION}"
  candidate="${cache_dir}/${asset}"

  if [ -x "${candidate}" ] && opa_version_matches "${candidate}"; then
    echo "${candidate}"
    return
  fi

  mkdir -p "${cache_dir}"
  tmp_bin="$(mktemp)"
  tmp_sha="$(mktemp)"
  cleanup_download() {
    rm -f "${tmp_bin}" "${tmp_sha}"
  }
  trap cleanup_download EXIT

  # Bound both a single attempt (--max-time) and the whole retry window (--retry-max-time).
  # --max-time RESETS on each retry, so without --retry-max-time the aggregate is unbounded. That
  # matters because callers impose their own ceiling — embedded_pdp/policy_wasm.py runs this script
  # with DEFAULT_BUILD_TIMEOUT_SECONDS=120 — and would kill the build mid-retry. Keep each download
  # root bounded so the default official+GitHub fallback still leaves headroom for `opa build`.
  downloaded=0
  for base_url in ${RESOLVED_OPA_DOWNLOAD_BASE_URLS}; do
    url="${base_url%/}/${OPA_VERSION}/${asset}"
    sha_url="${url}.sha256"
    echo "Downloading OPA ${OPA_VERSION} from ${url}..." >&2

    if curl -fsSL --retry 3 --retry-delay 2 --retry-all-errors \
        --connect-timeout 10 --max-time 30 --retry-max-time 45 "${url}" -o "${tmp_bin}" && \
      curl -fsSL --retry 3 --retry-delay 2 --retry-all-errors \
        --connect-timeout 10 --max-time 10 --retry-max-time 15 "${sha_url}" -o "${tmp_sha}"; then
      downloaded=1
      break
    fi

    echo "Failed to download OPA from ${base_url}; trying next source if configured." >&2
    rm -f "${tmp_bin}" "${tmp_sha}"
  done

  if [ "${downloaded}" != "1" ]; then
    print_opa_help "${asset}"
    exit 1
  fi

  expected="$(awk '{print $1; exit}' "${tmp_sha}")"
  actual="$(sha256_file "${tmp_bin}")"
  if [ "${expected}" != "${actual}" ]; then
    echo "Checksum verification failed for ${asset}." >&2
    echo "expected: ${expected}" >&2
    echo "actual:   ${actual}" >&2
    print_opa_help "${asset}"
    exit 1
  fi

  chmod +x "${tmp_bin}"
  mv "${tmp_bin}" "${candidate}"
  trap - EXIT
  rm -f "${tmp_sha}"

  echo "${candidate}"
}

resolve_opa() {
  asset="$(detect_opa_asset)"

  if [ -n "${OPA_BIN:-}" ]; then
    if [ ! -x "${OPA_BIN}" ]; then
      echo "OPA_BIN is not executable: ${OPA_BIN}" >&2
      print_opa_help "${asset}"
      exit 1
    fi
    if ! opa_version_matches "${OPA_BIN}"; then
      echo "OPA_BIN must be OPA ${OPA_VERSION}; got:" >&2
      "${OPA_BIN}" version >&2 || true
      print_opa_help "${asset}"
      exit 1
    fi
    echo "${OPA_BIN}"
    return
  fi

  if command -v opa >/dev/null 2>&1; then
    path="$(command -v opa)"
    if opa_version_matches "${path}"; then
      echo "${path}"
      return
    fi
    echo "Found ${path}, but it is not OPA ${OPA_VERSION}; using pinned cached binary." >&2
  fi

  download_opa
}

OPA="$(resolve_opa)"

echo "###############################"
"${OPA}" version
echo "###############################"
echo ""

# --- Build policy.wasm ---
echo "Building policy.wasm from ${POLICY_DIR}..."
BUNDLE_TMP="$(mktemp -d)"
echo "Bundle temp dir: ${BUNDLE_TMP}"

cleanup() { rm -rf "${BUNDLE_TMP}"; }
trap cleanup EXIT

# Build in a subshell so the cd doesn't affect OUTPUT_DIR resolution.
# Using relative paths (./*.rego) ensures the wasm output is path-independent.
# shellcheck disable=SC2086

(cd "${POLICY_DIR}" && "${OPA}" build -t wasm ${ENTRYPOINTS} -o "${BUNDLE_TMP}/bundle.tar.gz" ./*.rego)

ls -1 "${BUNDLE_TMP}"

# --- Extract WASM ---
mkdir -p "${OUTPUT_DIR}"
POLICY_WASM_MEMBER="$(tar -tzf "${BUNDLE_TMP}/bundle.tar.gz" | awk '$0 == "/policy.wasm" || $0 == "policy.wasm" {print; exit}')"
if [ -z "${POLICY_WASM_MEMBER}" ]; then
  echo "policy.wasm not found in OPA bundle." >&2
  tar -tzf "${BUNDLE_TMP}/bundle.tar.gz" >&2
  exit 1
fi
tar -C "${OUTPUT_DIR}" -zxvf "${BUNDLE_TMP}/bundle.tar.gz" "${POLICY_WASM_MEMBER}"
echo "policy.wasm written to ${OUTPUT_DIR}/policy.wasm"

ls -lh "${OUTPUT_DIR}/policy.wasm"
