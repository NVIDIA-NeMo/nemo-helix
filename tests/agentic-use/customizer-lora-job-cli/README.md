<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# LoRA Customization Job (CLI, GPU)

Tests the agent's ability to set up and submit a real LoRA fine-tuning job through the **nemo-customizer** plugin with the **nhx-automodel** backend.

## What This Tests

- Creating workspaces and filesets via NeMo Helix CLI
- Preparing and uploading SFT training data in JSONL format
- Submitting a LoRA job with `nemo customization automodel submit`
- Monitoring job progress via the jobs API

## Prerequisites

Build and push platform/automodel images before running this eval. From the repo root:

```bash
export BAKE_TAG=$(git rev-parse --short HEAD)
export BASE_TAG_AUTOMODEL=$BAKE_TAG
export NHX_IMAGE_TAG=$BAKE_TAG
export NHX_IMAGE_REGISTRY=my-registry/nemo-helix-dev
echo "$NHX_IMAGE_TAG"

# Bake/push nhx-automodel images (and platform task images as needed)
docker buildx bake -f docker-bake.hcl --push
```

The eval container sources `environment/image-env.sh`, which defaults to the same registry and derives `NHX_IMAGE_TAG` from `git rev-parse --short HEAD` when unset.

## GPU Requirements

This eval requires **1 GPU** allocated to the Harbor container (via `environment/docker-compose.yaml`). The customization job performs actual LoRA fine-tuning on the GPU using `nhx-automodel-training`.

## Flow Reference

Implements flow **17a: Basic LoRA Customization Job** from `../agentic_flows/customizer.md`.
