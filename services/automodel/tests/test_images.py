# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import nemo_helix_plugin.jobs.image as platform_image
import nhx.automodel.images as automodel_images
import nhx.customization_common.service.images as shared_images
import pytest
from nhx.automodel.config import AutomodelConfig
from nhx.automodel.images import (
    TRAINING_IMAGE_NAME,
    get_automodel_qualified_image,
    get_tasks_image,
    get_training_image,
)
from nhx.customization_common.service.images import CUSTOMIZER_TASKS_IMAGE_NAME


@pytest.fixture
def platform_config(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    config = SimpleNamespace(image_registry="registry.example.com/nemo", image_tag="test-tag")
    monkeypatch.setattr(shared_images, "get_platform_config", lambda: config)
    monkeypatch.setattr(platform_image, "get_platform_config", lambda: config)
    return config


def test_default_automodel_images_use_platform_registry(monkeypatch, platform_config):
    monkeypatch.setattr(automodel_images, "config", AutomodelConfig())

    tasks = get_tasks_image()
    training = get_training_image()

    assert tasks == f"{platform_config.image_registry}/{CUSTOMIZER_TASKS_IMAGE_NAME}:{platform_config.image_tag}"
    assert training == f"{platform_config.image_registry}/{TRAINING_IMAGE_NAME}:{platform_config.image_tag}"
    assert CUSTOMIZER_TASKS_IMAGE_NAME.count("/") == 0  # single repo segment, no nested paths


def test_automodel_image_registry_override(monkeypatch, platform_config):
    monkeypatch.setattr(
        automodel_images,
        "config",
        AutomodelConfig(image_registry="my-registry/other-registry"),
    )

    assert (
        get_automodel_qualified_image(TRAINING_IMAGE_NAME)
        == f"my-registry/other-registry/{TRAINING_IMAGE_NAME}:{platform_config.image_tag}"
    )


def test_automodel_full_image_override(monkeypatch, platform_config):
    monkeypatch.setattr(
        automodel_images,
        "config",
        AutomodelConfig(
            tasks_image="my-registry/nemo-helix-dev/nhx-customizer-tasks:dev",
        ),
    )

    assert get_tasks_image() == "my-registry/nemo-helix-dev/nhx-customizer-tasks:dev"
