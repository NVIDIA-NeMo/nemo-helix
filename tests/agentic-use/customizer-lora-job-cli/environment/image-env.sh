#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Platform and nhx-automodel image registry/tag defaults for agentic GPU evals.
#
# Host operators typically bake images with:
#   export BAKE_TAG=$(git rev-parse --short HEAD)
#   export BASE_TAG_AUTOMODEL=$BAKE_TAG
#   export NHX_IMAGE_TAG=$BAKE_TAG
#   export NHX_IMAGE_REGISTRY=my-registry/nemo-helix-dev
#
# Override any variable before starting the eval container when testing a
# different tag or registry.

if [ -z "${BAKE_TAG:-}" ]; then
  if command -v git >/dev/null 2>&1 && git -C /app rev-parse --short HEAD >/dev/null 2>&1; then
    BAKE_TAG="$(git -C /app rev-parse --short HEAD)"
  else
    BAKE_TAG="${NHX_IMAGE_TAG:-local}"
  fi
fi

export BAKE_TAG
export BASE_TAG_AUTOMODEL="${BASE_TAG_AUTOMODEL:-$BAKE_TAG}"
export NHX_IMAGE_TAG="${NHX_IMAGE_TAG:-$BAKE_TAG}"
export NHX_IMAGE_REGISTRY="${NHX_IMAGE_REGISTRY:-my-registry/nemo-helix-dev}"
export NHX_AUTOMODEL_IMAGE_REGISTRY="${NHX_AUTOMODEL_IMAGE_REGISTRY:-$NHX_IMAGE_REGISTRY}"
