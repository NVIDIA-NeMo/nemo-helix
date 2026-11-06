#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check that a locally loaded image, built with NHX_COLLECT_SOURCES=1, carries the source of every
# third-party component syft finds in it, and in each archive its com.nvidia.nemo-helix.source-scan-archives
# label names (space-separated paths): code an image ships packed, which syft doesn't look inside.
#
# Usage: check-image-source-coverage.sh <image-ref> <image-name>
#   image-name names /usr/share/nemo-helix/source-distributions/<image-name> inside the image, or
#   /kaniko/source-distributions/<image-name> in the kaniko image, whose builds replace everything
#   outside /kaniko.

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
for source_root in /usr/share/nemo-helix/source-distributions /kaniko/source-distributions; do
    if docker cp -q "${container}:${source_root}/${image_name}" "${work_dir}/sources" 2>/dev/null; then
        break
    fi
done
if [[ ! -d "${work_dir}/sources" ]]; then
    echo "${image_ref} has no source-distributions/${image_name} directory" >&2
    exit 1
fi

sboms=(--sbom "${work_dir}/sbom.json")
archives="$(docker image inspect "${image_ref}" \
    --format '{{ index .Config.Labels "com.nvidia.nemo-helix.source-scan-archives" }}')"
index=0
for archive in ${archives}; do
    index=$((index + 1))
    unpacked="${work_dir}/archive-${index}"
    mkdir -p "${unpacked}"
    docker cp -q "${container}:${archive}" "${work_dir}/archive-${index}.packed"
    tar -xf "${work_dir}/archive-${index}.packed" -C "${unpacked}"
    docker run --rm -v "${unpacked}:/scan:ro" "${syft_image}" dir:/scan -o syft-json -q \
        > "${work_dir}/archive-${index}.json"
    sboms+=(--sbom "${work_dir}/archive-${index}.json")
done

python3 "${repo_root}/docker/scripts/check-source-coverage.py" \
    "${sboms[@]}" \
    --sources "${work_dir}/sources" \
    --policy "${repo_root}/docker/sources/source-coverage-policy.json"
echo "${image_ref}: every component syft found has collected source"
