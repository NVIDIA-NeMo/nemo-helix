#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -e

# Color codes for output
RED='\033[31m'
GREEN='\033[32m'
NC='\033[0m' # No Color

# Constants
SDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SDK_DIR}/.." && pwd)"

OPENAPI_SPEC="${REPO_ROOT}/openapi/openapi.yaml"
STAINLESS_CONFIG_PATH="${SDK_DIR}/stainless.yaml"
STAINLESS_PROJECT_NAME="nemo-helix"
SDK_PATH="sdk/python/nemo-helix"

# Repo from which we fetch the SDK code.
# - STAINLESS_CODE_REPO can be set explicitly to a full git remote URL.
# - Otherwise, STAINLESS_GIT_PROTOCOL controls which default URL to use: ssh (default), https, auto.
DEFAULT_STAINLESS_CODE_REPO_HTTPS="https://github.com/stainless-sdks/${STAINLESS_PROJECT_NAME}-python.git"
DEFAULT_STAINLESS_CODE_REPO_SSH="git@github.com:stainless-sdks/${STAINLESS_PROJECT_NAME}-python.git"

resolve_stainless_code_repo() {
    if [ -n "${STAINLESS_CODE_REPO}" ]; then
        echo "${STAINLESS_CODE_REPO}"
        return 0
    fi

    local protocol="${STAINLESS_GIT_PROTOCOL:-ssh}"
    case "${protocol}" in
        https)
            echo "${DEFAULT_STAINLESS_CODE_REPO_HTTPS}"
            ;;
        ssh)
            echo "${DEFAULT_STAINLESS_CODE_REPO_SSH}"
            ;;
        auto)
            local origin_url
            origin_url="$(cd "${REPO_ROOT}" && git remote get-url origin 2>/dev/null || true)"
            if [[ "${origin_url}" == git@github.com:* || "${origin_url}" == ssh://git@github.com/* ]]; then
                echo "${DEFAULT_STAINLESS_CODE_REPO_SSH}"
            else
                echo "${DEFAULT_STAINLESS_CODE_REPO_HTTPS}"
            fi
            ;;
        *)
            echo "Error: Unsupported STAINLESS_GIT_PROTOCOL='${protocol}'. Use one of: https, ssh, auto." >&2
            exit 1
            ;;
    esac
}

STAINLESS_CODE_REPO="$(resolve_stainless_code_repo)"

if [ ! -f "${OPENAPI_SPEC}" ]; then
    echo "Error: OpenAPI spec not found at ${OPENAPI_SPEC}"
    exit 1
fi

echo "Configuration:"
echo "  OPENAPI_SPEC: ${OPENAPI_SPEC}"
echo "  STAINLESS_CONFIG_PATH: ${STAINLESS_CONFIG_PATH}"
echo "  STAINLESS_PROJECT_NAME: ${STAINLESS_PROJECT_NAME}"
echo "  SDK_PATH: ${SDK_PATH}"
echo "  STAINLESS_CODE_REPO: ${STAINLESS_CODE_REPO}"
echo ""

# Use STAINLESS_BRANCH env var if set, otherwise use current git branch
if [ -z "${STAINLESS_BRANCH}" ]; then
    STAINLESS_BRANCH="$(cd "${REPO_ROOT}" && git rev-parse --abbrev-ref HEAD)"
    
    # Append a short version of the commit SHA to ensure uniqueness
    COMMIT_SHA="$(cd "${REPO_ROOT}" && git rev-parse --short HEAD)"
    STAINLESS_BRANCH="${STAINLESS_BRANCH}-${COMMIT_SHA}"
fi
echo "Using branch: ${STAINLESS_BRANCH}"
# Name of the remote that will be added to the NeMo Helix repo
STAINLESS_CODE_REMOTE_NAME="stainless-${STAINLESS_PROJECT_NAME}-python"

# Save original directory
ORIGINAL_DIR="$(pwd)"

# Create temp directory for intermediate files
TEMP_DIR="$(mktemp -d)"

# Cleanup function
cleanup() {
    cd "${ORIGINAL_DIR}"
    rm -rf "${TEMP_DIR}"
}

# Set up trap to cleanup on exit
trap cleanup EXIT INT TERM

# Check if STAINLESS_API_KEY is set
check_api_key() {
    if [ -z "${STAINLESS_API_KEY}" ]; then
        echo "Error: STAINLESS_API_KEY environment variable is not set."
        echo "Please create an API key in https://app.stainless.com/nvidia/settings and set it in your environment."
        exit 1
    fi
}

# Check if stl CLI is installed
check_stl_installed() {
    if ! command -v stl >/dev/null 2>&1; then
        echo "Error: stl CLI is not installed."
        echo "brew install stainless-api/tap/stl"
        exit 1
    fi
}

# Check if the remote exists, add it if it doesn't
ensure_remote() {
    (
        # Run in subshell to avoid changing the working directory
        cd "${REPO_ROOT}"

        # Remote for the code
        if ! git remote get-url "${STAINLESS_CODE_REMOTE_NAME}" > /dev/null 2>&1; then
            echo "Adding ${STAINLESS_CODE_REMOTE_NAME} remote..."
            git remote add "${STAINLESS_CODE_REMOTE_NAME}" "${STAINLESS_CODE_REPO}"
            return 0
        fi

        local existing_remote_url
        existing_remote_url="$(git remote get-url "${STAINLESS_CODE_REMOTE_NAME}")"
        if [ "${existing_remote_url}" != "${STAINLESS_CODE_REPO}" ]; then
            echo "Updating ${STAINLESS_CODE_REMOTE_NAME} remote URL..."
            git remote set-url "${STAINLESS_CODE_REMOTE_NAME}" "${STAINLESS_CODE_REPO}"
        fi
    )
}

git_with_error_handling() {
    # This function wraps git commands and prints a progress message with DONE/FAILED status.
    # It returns the exit code of the git command (which will terminate the script with '-e' if non-zero).
    local message="$1"; shift
    local temp_file="${TEMP_DIR}/git_output"

    # Print the progress message
    echo -n "${message}"

    # Run the git command and capture both stdout and stderr
    if git "$@" > "$temp_file" 2>&1; then
        echo -e " ${GREEN}DONE${NC}"
        return 0
    else
        local rc=$?
        echo -e " ${RED}FAILED${NC}" >&2
        echo "  Git command failed: git $*" >&2
        echo "  Output:" >&2
        sed 's/^/    /' "$temp_file" >&2
        return $rc
    fi
}

# Push the config and OpenAPI spec using the stl CLI
push_config() {
    check_api_key
    check_stl_installed

    # URL-encode the branch name by replacing forward slashes with %2F
    URL_ENCODED_BRANCH="${STAINLESS_BRANCH//\//%2F}"

    echo "Pushing Stainless config and OpenAPI spec..."
    echo "You can view the progress at: https://app.stainless.com/nvidia/${STAINLESS_PROJECT_NAME}/studio?language=python&branch=${URL_ENCODED_BRANCH}"
    echo ""

    # Wait for commit phase only (codegen) - don't wait for lint/test/build
    # --allow-empty: don't fail if there are no changes to commit
    if ! stl builds create \
        --project "${STAINLESS_PROJECT_NAME}" \
        --branch "${STAINLESS_BRANCH}" \
        --config "${STAINLESS_CONFIG_PATH}" \
        --openapi-spec "${OPENAPI_SPEC}" \
        --wait commit \
        --allow-empty; then
        echo -e "${RED}Error: stl builds create failed${NC}"
        exit 1
    fi

    echo ""
    echo "Build completed."
}

# Pull the latest code changes
pull_changes() {
    ensure_remote

    # Pull the code
    (
        # Run in subshell to avoid changing the working directory
        cd "${REPO_ROOT}"

        git_with_error_handling "Fetching latest code changes..." \
          fetch "${STAINLESS_CODE_REMOTE_NAME}" "${STAINLESS_BRANCH}"

        # Remove the existing sdk code
        echo "Removing existing SDK code..."
        rm -rf "${SDK_PATH}"

        git_with_error_handling "Pulling latest changes from ${STAINLESS_CODE_REMOTE_NAME}/${STAINLESS_BRANCH}..." \
          read-tree --prefix=tmp-sdk -u "${STAINLESS_CODE_REMOTE_NAME}/${STAINLESS_BRANCH}"

        # Move the SDK code to the correct location
        echo "Moving SDK code to correct location..."
        mv tmp-sdk "${SDK_PATH}"

        # Reset the working directory for the SDK folder
        git_with_error_handling "Resetting working directory for SDK folder..." \
          reset -- tmp-sdk

        echo "Code pull completed."

        # Make sure to update the pyproject first, before any uv command that will update the uv.lock
        uv run --frozen nemo-helix-sdk-tools post-generation update-pyproject

        # Vendor internal packages into the SDK
        make vendor

        # Sync after vendoring to update the uv.lock, as we added deps from vendored packages
        uv sync --inexact

        # Run all post-generation updates (license, readme, etc.)
        uv run --frozen nemo-helix-sdk-tools post-generation update-all

        # Add all files that were pulled or vendored
        git add "${SDK_PATH}"
    )
}

# Update the Stainless config if needed
update_stainless_config() {
    echo "Updating Stainless config..."
    (
        cd "${REPO_ROOT}"

        echo "Syncing endpoints from OpenAPI spec with Stainless methods..."
        uv run --frozen nemo-helix-sdk-tools openapi-stainless sync-methods \
          --openapi-spec-path "${OPENAPI_SPEC}" \
          --stainless-config-path "${STAINLESS_CONFIG_PATH}" \
          --output-path "${STAINLESS_CONFIG_PATH}"

        echo "Syncing schemas from OpenAPI spec with Stainless models..."
        uv run --frozen nemo-helix-sdk-tools openapi-stainless sync-models \
          --openapi-spec-path "${OPENAPI_SPEC}" \
          --stainless-config-path "${STAINLESS_CONFIG_PATH}" \
          --output-path "${STAINLESS_CONFIG_PATH}"
    )
    echo "Stainless config update completed."
}

# Sync: push, then pull
sync_changes() {
    update_stainless_config

    check_api_key
    ensure_remote

    # Push config and wait for commit phase to complete
    push_config

    # Always pull to ensure local SDK matches remote
    pull_changes
}

# Main function
main() {
    case "$1" in
        push)
            push_config
            ;;
        pull)
            pull_changes
            ;;
        sync)
            sync_changes
            ;;
        *)
            echo "Usage: $0 {push|pull|sync}"
            echo ""
            echo "  push  - Push the config and OpenAPI spec to Stainless"
            echo "  pull  - Pull the latest code changes from Stainless"
            echo "  sync  - Push config, wait for new commit, then pull changes"
            exit 1
            ;;
    esac
}

main "$@"
