#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

source /app/image-env.sh

echo '=== Pre-pulling nhx-automodel job images ==='
if command -v docker &> /dev/null && [ -S /var/run/docker.sock ]; then
    TRAINING_IMAGE="${NHX_IMAGE_REGISTRY}/nhx-automodel-training:${NHX_IMAGE_TAG}"
    TASKS_IMAGE="${NHX_IMAGE_REGISTRY}/nhx-customizer-tasks:${NHX_IMAGE_TAG}"

    docker pull "$TRAINING_IMAGE" && echo "Pulled ${TRAINING_IMAGE}" || \
        echo "WARNING: Failed to pull ${TRAINING_IMAGE} — bake and push with BASE_TAG_AUTOMODEL=${BASE_TAG_AUTOMODEL}"
    docker pull "$TASKS_IMAGE" && echo "Pulled ${TASKS_IMAGE}" || \
        echo "WARNING: Failed to pull ${TASKS_IMAGE} — bake and push with BASE_TAG_AUTOMODEL=${BASE_TAG_AUTOMODEL}"
else
    echo 'WARNING: Docker not available for image pre-pull'
fi

echo '=== Creating workspace ==='
/app/.venv/bin/nemo workspaces create --name lora-training-workspace || echo 'Workspace may already exist'

echo '=== Registering model weights fileset and entity ==='
/app/.venv/bin/nemo files filesets create smollm-135m-weights \
    --workspace lora-training-workspace \
    --purpose model \
    --exist-ok \
    --storage '{"type":"huggingface","repo_id":"HuggingFaceTB/SmolLM-135M","repo_type":"model","revision":"main"}' \
    2>&1 || echo 'Weights fileset may already exist'

/app/.venv/bin/nemo models create smollm-135m \
    --workspace lora-training-workspace \
    --exist-ok \
    --input-data '{
        "name": "smollm-135m",
        "fileset": "lora-training-workspace/smollm-135m-weights",
        "custom_fields": {
            "hf_model_id": "HuggingFaceTB/SmolLM-135M"
        }
    }' 2>&1 || echo 'Model may already exist'

echo '=== Environment setup complete ==='
