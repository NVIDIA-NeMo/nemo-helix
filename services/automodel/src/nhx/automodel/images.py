# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Docker image resolution for nhx-automodel job steps."""

from __future__ import annotations

from nhx.automodel.config import config
from nhx.customization_common.service.images import (
    CUSTOMIZER_PYTHON_ENTRYPOINT,
    get_customizer_tasks_image,
    resolve_qualified_image,
)

BASE_IMAGE_NAME = "nhx-automodel-base"
TRAINING_IMAGE_NAME = "nhx-automodel-training"

# Alias for backward compatibility in compiler imports.
AUTOMODEL_PYTHON_ENTRYPOINT = CUSTOMIZER_PYTHON_ENTRYPOINT

FILE_IO_TASK_COMMAND = [
    "-m",
    "nhx.customization_common.tasks.file_io",
    "--service-source",
    "automodel",
    "--service-name",
    "customizer",
]
MODEL_ENTITY_TASK_COMMAND = [
    "-m",
    "nhx.customization_common.tasks.model_entity",
    "--service-name",
    "customizer",
]


def get_automodel_qualified_image(name: str, override: str | None = None) -> str:
    """Resolve a job step image reference (see ``resolve_qualified_image``)."""
    return resolve_qualified_image(name, override, config.image_registry)


def get_tasks_image() -> str:
    """CPU task steps (file_io, model_entity) — shared ``nhx-customizer-tasks`` image."""
    return get_customizer_tasks_image(backend_override=config.tasks_image, image_registry=config.image_registry)


def get_training_image() -> str:
    """GPU training step."""
    return get_automodel_qualified_image(TRAINING_IMAGE_NAME, config.training_image)
