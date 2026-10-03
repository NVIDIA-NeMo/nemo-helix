#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script: publish_assets_to_s3.sh
# Description: Off-CI publisher for customizer E2E assets. Reads
#              e2e/customizer/assets_manifest.json for the dataset sources,
#              generation sizes, and model list; generates datasets, downloads
#              base model snapshots from Hugging Face, and uploads everything
#              (including the manifest) to the S3 asset bucket CI syncs from.
#              Run from a developer/maintainer machine whenever assets need
#              refreshing. CI must never pull from HF.
#
# Usage:
#   ./e2e/customizer/publish_assets_to_s3.sh
#
# Layout produced in the bucket:
#   s3://<bucket>/assets_manifest.json
#   s3://<bucket>/datasets/{chat_format,prompt_completion,dpo}/*.jsonl
#   s3://<bucket>/models/<model-folder>/   (full HF snapshot)
#
# Environment Variables (optional overrides):
#   S3_BUCKET            Target bucket (default from manifest: aire-e2e-assets)
#   S3_ENDPOINT_URL      Endpoint URL for S3-compatible stores (e.g. RustFS/MinIO)
#   MODEL_STAGING_DIR    Persistent dir for downloaded snapshots (enables re-use)
#   HF_TOKEN             Hugging Face token for gated repos
#   DRY_RUN              Set to "true" to print aws uploads without writing
#
# Prerequisites: aws CLI (configured with write access to the bucket) and uv.

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
CUSTOMIZER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
MANIFEST="${CUSTOMIZER_DIR}/assets_manifest.json"

GENERATOR="${CUSTOMIZER_DIR}/generate_customizer_data.py"
MODEL_DOWNLOADER="${CUSTOMIZER_DIR}/download_hf_model_snapshot.py"
MANIFEST_READER="${CUSTOMIZER_DIR}/assets_manifest.py"

S3_ENDPOINT_URL="${S3_ENDPOINT_URL:-}"
DRY_RUN="${DRY_RUN:-false}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

if [ ! -f "${MANIFEST}" ]; then
    log_error "Manifest not found: ${MANIFEST}"
    exit 1
fi

log_info "Validating environment..."
for tool in aws uv git; do
    if ! command -v "${tool}" &> /dev/null; then
        log_error "${tool} is not installed. Please install it first."
        exit 1
    fi
done

if [ ! -f "${MODEL_DOWNLOADER}" ]; then
    log_error "Model downloader not found: ${MODEL_DOWNLOADER}"
    exit 1
fi

# Read manifest fields for publishing.
PUBLISH_SETTINGS="$(uv run python "${MANIFEST_READER}" publish-settings "${REPO_ROOT}")"
read -r S3_BUCKET DATASET_SRC_DIR MODEL_IGNORE_GLOBS <<< "${PUBLISH_SETTINGS}"
if [ -z "${S3_BUCKET}" ] || [ -z "${DATASET_SRC_DIR}" ]; then
    log_error "Failed to read publish settings from manifest"
    exit 1
fi

# Build shared aws-cli args (endpoint override for S3-compatible stores).
AWS_ARGS=()
[ -n "${S3_ENDPOINT_URL}" ] && AWS_ARGS+=(--endpoint-url "${S3_ENDPOINT_URL}")

s3_upload_file() {
    local src_file="$1"
    local s3_uri="$2"

    log_info "Uploading ${src_file} -> ${s3_uri}"
    if [ "${DRY_RUN}" = "true" ]; then
        aws ${AWS_ARGS[@]+"${AWS_ARGS[@]}"} s3 cp "${src_file}" "${s3_uri}" --dryrun
    else
        aws ${AWS_ARGS[@]+"${AWS_ARGS[@]}"} s3 cp "${src_file}" "${s3_uri}"
    fi
}

# Upload a local directory tree to s3://<bucket>/<prefix>/ via aws s3 cp --recursive.
s3_upload_dir() {
    local src_dir="$1"
    local s3_uri="$2"
    shift 2
    local -a extra_args=("$@")

    log_info "Uploading ${src_dir} -> ${s3_uri}"
    if [ "${DRY_RUN}" = "true" ]; then
        aws ${AWS_ARGS[@]+"${AWS_ARGS[@]}"} s3 cp "${src_dir}" "${s3_uri}" \
            --recursive --dryrun ${extra_args[@]+"${extra_args[@]}"}
    else
        aws ${AWS_ARGS[@]+"${AWS_ARGS[@]}"} s3 cp "${src_dir}" "${s3_uri}" \
            --recursive ${extra_args[@]+"${extra_args[@]}"}
    fi
}

# Staging dir for model snapshots. Persist if the caller supplied one (fast re-runs).
CLEANUP_STAGING=false
if [ -n "${MODEL_STAGING_DIR:-}" ]; then
    mkdir -p "${MODEL_STAGING_DIR}"
else
    MODEL_STAGING_DIR="$(mktemp -d)"
    CLEANUP_STAGING=true
fi

cleanup() {
    if ${CLEANUP_STAGING} && [ -d "${MODEL_STAGING_DIR}" ]; then
        rm -rf "${MODEL_STAGING_DIR}"
    fi
}
trap cleanup EXIT

log_info "Manifest:      ${MANIFEST}"
log_info "Target bucket: s3://${S3_BUCKET}"
[ -n "${S3_ENDPOINT_URL}" ] && log_info "Endpoint URL:  ${S3_ENDPOINT_URL}"
[ "${DRY_RUN}" = "true" ] && log_warn "Dry run: no objects will be written."

# --------------------------------------------------------------------------- #
# Datasets (sources + sizes from manifest)
# --------------------------------------------------------------------------- #
log_info "Generating customizer datasets via ${GENERATOR}..."
uv run --with datasets python "${GENERATOR}"

if [ ! -d "${DATASET_SRC_DIR}" ]; then
    log_error "Dataset directory not found: ${DATASET_SRC_DIR}"
    exit 1
fi

s3_upload_dir "${DATASET_SRC_DIR}/" "s3://${S3_BUCKET}/datasets/" \
    --exclude "*" --include "*.jsonl"

# --------------------------------------------------------------------------- #
# Models (list from manifest)
# --------------------------------------------------------------------------- #
while IFS=$'\t' read -r s3_folder hf_repo; do
    [ -z "${s3_folder}" ] && continue
    local_dir="${MODEL_STAGING_DIR}/${s3_folder}"
    log_info "Downloading snapshot ${hf_repo} -> ${local_dir}"
    HF_REPO="${hf_repo}" LOCAL_DIR="${local_dir}" MODEL_IGNORE_GLOBS="${MODEL_IGNORE_GLOBS}" \
        uv run --with huggingface_hub python "${MODEL_DOWNLOADER}"

    s3_upload_dir "${local_dir}/" "s3://${S3_BUCKET}/models/${s3_folder}/" \
        --exclude ".cache/*" --exclude ".gitattributes"
done < <(uv run python "${MANIFEST_READER}" models)

# --------------------------------------------------------------------------- #
# Manifest (for CI / staging scripts)
# --------------------------------------------------------------------------- #
s3_upload_file "${MANIFEST}" "s3://${S3_BUCKET}/assets_manifest.json"

log_info "Done."
