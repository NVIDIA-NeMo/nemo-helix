#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check that a locally loaded image, built with NHX_COLLECT_SOURCES=1, carries the source of every
# third-party component syft finds in it.
#
# Usage: check-image-source-coverage.sh <image-ref> <image-name>
#   image-name names /usr/share/nemo-helix/source-distributions/<image-name> inside the image.

set -euo pipefail

image_ref="$1"
image_name="$2"
syft_image="${SYFT_IMAGE:-anchore/syft:v1.18.1@sha256:b8c170b8e51bfc4779ec3ef4399942c57290f5ce76a9c3af564c9d00d4946a6b}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work_dir="$(mktemp -d)"
container=""
cleanup() {
    [[ -n "${container}" ]] && docker rm -f "${container}" >/dev/null 2>&1
    rm -rf "${work_dir}"
}
trap cleanup EXIT

docker run --rm -v /var/run/docker.sock:/var/run/docker.sock "${syft_image}" \
    "docker:${image_ref}" -o syft-json -q > "${work_dir}/sbom.json"

container="$(docker create "${image_ref}" true)"
docker cp -q "${container}:/usr/share/nemo-helix/source-distributions/${image_name}" "${work_dir}/sources"

python3 "${repo_root}/docker/scripts/check-source-coverage.py" \
    --sbom "${work_dir}/sbom.json" \
    --sources "${work_dir}/sources" \
    --policy "${repo_root}/docker/sources/source-coverage-policy.json"
echo "${image_ref}: every component syft found has collected source"
