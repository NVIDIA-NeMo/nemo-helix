#!/usr/bin/env bash
# Fallback for tutorial step 7a, on the GPU host. Step 7a normally pulls the
# published images from nvcr.io/nvidia/nemo-helix (public since 2026-10-11).
# Use this script only when there is no published tag for your Helix version or
# the machine can't reach nvcr.io. It builds the two post-training plugin images
# the Docker job backend needs, from a Helix checkout at the matching release tag,
# using Helix's own bake file. Produces my-registry/nhx-customizer-tasks:local and
# my-registry/nhx-unsloth-training:local, which match the platform's default
# image_registry/image_tag.
#
#   HELIX_DIR=~/nemo-helix bash scripts/gpu-build-images.sh   # 30 to 60 min the first time (CUDA wheels)
set -uo pipefail
HELIX_DIR="${HELIX_DIR:-$HOME/nemo-helix}"
cd "$HELIX_DIR" || exit 1
if [ -n "${NGC_API_KEY:-}" ]; then
  echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin 2>&1 | tail -1
fi
docker buildx inspect >/dev/null 2>&1 || docker buildx create --use

echo "== $(date '+%F %T') CPU tasks image"
docker buildx bake -f docker-bake.hcl nhx-customizer-tasks --load --set "*.platform=linux/amd64" 2>&1 | tail -3

echo "== $(date '+%F %T') CUDA extension wheels (mamba-ssm, causal-conv1d) for the hybrid Nemotron models"
USE_LOCAL_WHEELS=1 docker buildx bake -f docker-bake.hcl nhx-automodel-gpu-wheels --load --set "*.platform=linux/amd64" 2>&1 | tail -3

echo "== $(date '+%F %T') Unsloth training image"
USE_LOCAL_WHEELS=1 docker buildx bake -f docker-bake.hcl nhx-unsloth-training --load --set "*.platform=linux/amd64" 2>&1 | tail -3

docker pull docker.io/vllm/vllm-openai:v0.22.1 2>&1 | tail -1
docker images --format "{{.Repository}}:{{.Tag}}\t{{.Size}}" | grep -E "nhx-|vllm"
df -h / | tail -1
echo "== $(date '+%F %T') images done"
