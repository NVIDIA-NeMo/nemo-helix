#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reproduce the CPU Docker E2E setup failures from the GitHub Actions workflow.
#
# By default this intentionally does not forward NGC_API_KEY into pytest, matching
# the CPU Docker E2E jobs. It still uses the token for NVCR login and private job
# image pulls.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PLATFORM_ROOT="${PLATFORM_ROOT:-${REPO_ROOT}/../Platform}"

REGISTRY="${NHX_E2E_REGISTRY:-nvcr.io/0921617854601259/nemo-helix-dev}"
TAG="${NHX_E2E_TAG:-}"
PR_NUMBER=""
MODE="both"
FULL_SUITE=0
FORWARD_NGC=0
BUILD_LOCAL=0
DO_LOGIN=1
DO_PULL=1
TOKEN_FILE="${TOKEN_FILE:-${HOME}/.api_tokens.sh}"
OUTPUT_DIR=""
BUILD_ARCH="${BUILD_ARCH:-linux/amd64}"
BASE_REGISTRY="${BASE_REGISTRY:-ghcr.io/nvidia-nemo/nemo-helix}"

usage() {
  cat <<'EOF'
Usage: e2e/docker/scripts/repro_cpu_e2e_setup.sh [options]

Options:
  --mode default|auth|both   Which CPU Docker E2E path to run (default: both).
  --full                     Run the full CI suite for the selected mode(s).
  --pr NUMBER                Use the PR head SHA as the image tag, via gh.
  --tag TAG                  Image tag to test (default: NHX_E2E_TAG or git HEAD).
  --registry REGISTRY        Image registry (default: nemo-helix-dev NVCR registry).
  --platform-root PATH       Platform checkout to use as the uv project (default: ../Platform).
  --token-file PATH          Token file to source (default: ~/.api_tokens.sh).
  --build-local              Build CPU images locally from the Platform checkout
                             instead of pulling PR images.
  --base-registry REGISTRY   Base image registry for --build-local
                             (default: ghcr.io/nvidia-nemo/nemo-helix).
  --with-ngc                 Forward NGC_API_KEY into pytest; useful to verify the fix.
  --no-login                 Skip docker login.
  --no-pull                  Skip pre-pulling images.
  --output-dir PATH          Directory for junit/debug/docker logs.
  -h, --help                 Show this help.

Examples:
  # Reproduce PR 56 CPU setup failures using the PR images.
  e2e/docker/scripts/repro_cpu_e2e_setup.sh --pr 56

  # Run just the default CPU Docker setup path.
  e2e/docker/scripts/repro_cpu_e2e_setup.sh --pr 56 --mode default

  # Check whether forwarding NGC_API_KEY fixes the setup failure.
  e2e/docker/scripts/repro_cpu_e2e_setup.sh --pr 56 --with-ngc

  # No access to NVCR layers: build locally and run the same CPU setup path.
  e2e/docker/scripts/repro_cpu_e2e_setup.sh --build-local --registry local/nhx-e2e --tag local-cpu
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="${2:?--mode requires a value}"
      shift 2
      ;;
    --full)
      FULL_SUITE=1
      shift
      ;;
    --pr)
      PR_NUMBER="${2:?--pr requires a number}"
      shift 2
      ;;
    --tag)
      TAG="${2:?--tag requires a value}"
      shift 2
      ;;
    --registry)
      REGISTRY="${2:?--registry requires a value}"
      shift 2
      ;;
    --platform-root)
      PLATFORM_ROOT="${2:?--platform-root requires a path}"
      shift 2
      ;;
    --token-file)
      TOKEN_FILE="${2:?--token-file requires a path}"
      shift 2
      ;;
    --with-ngc)
      FORWARD_NGC=1
      shift
      ;;
    --build-local)
      BUILD_LOCAL=1
      DO_LOGIN=0
      DO_PULL=0
      shift
      ;;
    --base-registry)
      BASE_REGISTRY="${2:?--base-registry requires a value}"
      shift 2
      ;;
    --no-login)
      DO_LOGIN=0
      shift
      ;;
    --no-pull)
      DO_PULL=0
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a path}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${MODE}" in
  default|auth|both) ;;
  *)
    echo "--mode must be one of: default, auth, both" >&2
    exit 2
    ;;
esac

if [ ! -d "${PLATFORM_ROOT}" ]; then
  echo "Platform checkout not found at: ${PLATFORM_ROOT}" >&2
  echo "Set PLATFORM_ROOT or pass --platform-root." >&2
  exit 1
fi
PLATFORM_ROOT="$(cd "${PLATFORM_ROOT}" && pwd)"

if [ -f "${TOKEN_FILE}" ]; then
  # Export names from the token file while sourcing so they can be read below.
  set -a
  set +u
  # shellcheck disable=SC1090
  source "${TOKEN_FILE}"
  set -u
  set +a
fi

NVCR_TOKEN="${NVCR_TOKEN:-${AIRE_NVCR_GITHUB:-${NGC_API_KEY:-}}}"

if [ -z "${TAG}" ] && [ -n "${PR_NUMBER}" ]; then
  TAG="$(gh pr view "${PR_NUMBER}" --json headRefOid --jq .headRefOid)"
fi
if [ -z "${TAG}" ]; then
  TAG="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
fi

if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="${REPO_ROOT}/cpu-e2e-artifacts/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "${OUTPUT_DIR}"

if [ -z "${NVCR_TOKEN}" ] && { [ "${DO_LOGIN}" -eq 1 ] || [ "${DO_PULL}" -eq 1 ]; }; then
  echo "No NVCR token found. Put NGC_API_KEY in ${TOKEN_FILE}, export NVCR_TOKEN, or pass --no-login --no-pull." >&2
  exit 1
fi

echo "Repo:          ${REPO_ROOT}"
echo "Platform:      ${PLATFORM_ROOT}"
echo "Registry:      ${REGISTRY}"
echo "Tag:           ${TAG}"
echo "Mode:          ${MODE}"
echo "Forward NGC:   ${FORWARD_NGC}"
echo "Build local:   ${BUILD_LOCAL}"
echo "Output:        ${OUTPUT_DIR}"

if [ "${BUILD_LOCAL}" -eq 1 ]; then
  echo "Building local CPU E2E images..."
  IMAGE_REGISTRY="${REGISTRY}" \
  BAKE_TAG="${TAG}" \
  CI_COMMIT_SHA="${TAG}" \
  BUILD_ARCH="${BUILD_ARCH}" \
  BASE_REGISTRY="${BASE_REGISTRY}" \
  docker buildx bake \
    -f "${PLATFORM_ROOT}/docker-bake.hcl" \
    --allow="fs.read=${PLATFORM_ROOT}" \
    docker-cpu docker-auditor --load
fi

if [ "${DO_LOGIN}" -eq 1 ]; then
  echo "Logging in to nvcr.io..."
  echo "${NVCR_TOKEN}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
fi

if [ "${DO_PULL}" -eq 1 ]; then
  echo "Pre-pulling CPU E2E images..."
  docker pull "${REGISTRY}/nhx-api:${TAG}"
  docker pull "${REGISTRY}/nhx-tasks:${TAG}"
  docker pull "${REGISTRY}/nhx-auditor-tasks:${TAG}"
fi

run_pytest() {
  local name="$1"
  shift

  local run_dir="${OUTPUT_DIR}/${name}"
  mkdir -p "${run_dir}"

  echo
  echo "=== Running ${name} CPU Docker E2E repro ==="
  echo "Artifacts: ${run_dir}"

  local env_args=(
    "NHX_E2E_REGISTRY=${REGISTRY}"
    "NHX_E2E_TAG=${TAG}"
    "NEMO_JOBS_IMAGE_REGISTRY=nvcr.io"
    "NEMO_JOBS_IMAGE_REGISTRY_USER_NAME=\$oauthtoken"
    "NEMO_JOBS_IMAGE_REGISTRY_PASSWORD=${NVCR_TOKEN}"
    "PYTEST_ADDOPTS=--log-cli-level=INFO --log-file=${run_dir}/pytest-debug.log --log-file-level=DEBUG"
    "JOB_LOGS_DIR=${run_dir}/docker-logs"
  )

  local exit_code=0
  set +e
  if [ "${FORWARD_NGC}" -eq 1 ]; then
    env "${env_args[@]}" "NGC_API_KEY=${NVCR_TOKEN}" \
      uv run --project "${PLATFORM_ROOT}" --frozen pytest "$@" \
      -v --junitxml="${run_dir}/report.xml"
    exit_code=$?
  else
    env -u NGC_API_KEY "${env_args[@]}" \
      uv run --project "${PLATFORM_ROOT}" --frozen pytest "$@" \
      -v --junitxml="${run_dir}/report.xml"
    exit_code=$?
  fi
  set -e

  echo "${exit_code}" > "${run_dir}/exit-code.txt"
  echo "Exit code for ${name}: ${exit_code}"
  return "${exit_code}"
}

failed=0

run_default() {
  if [ "${FULL_SUITE}" -eq 1 ]; then
    run_pytest default e2e --docker || failed=1
  else
    run_pytest default e2e/test_secrets.py::test_secret_create_and_list --docker || failed=1
  fi
}

run_auth() {
  if [ "${FULL_SUITE}" -eq 1 ]; then
    run_pytest auth e2e --docker --feature auth || failed=1
  else
    run_pytest auth e2e/test_workspaces.py::TestWorkspaceCRUD::test_create_workspace --docker --feature auth || failed=1
  fi
}

cd "${REPO_ROOT}"
case "${MODE}" in
  default)
    run_default
    ;;
  auth)
    run_auth
    ;;
  both)
    run_default
    run_auth
    ;;
esac

echo
echo "Repro artifacts written to: ${OUTPUT_DIR}"
echo "Useful grep:"
echo "  rg -n 'Container did not become healthy|NGC API key not found|Service is ready|Uvicorn running' '${OUTPUT_DIR}'"

exit "${failed}"
