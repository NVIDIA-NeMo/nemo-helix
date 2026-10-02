# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import nemo_helix_plugin.jobs.image as platform_image
import nhx.customization_common.service.images as shared_images
import nhx.unsloth.images as unsloth_images
import pytest
from nhx.customization_common.service.images import CUSTOMIZER_TASKS_IMAGE_NAME
from nhx.unsloth.config import UnslothConfig
from nhx.unsloth.images import (
    TRAINING_IMAGE_NAME,
    get_tasks_image,
    get_training_image,
    get_unsloth_qualified_image,
)


@pytest.fixture
def platform_config(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    config = SimpleNamespace(image_registry="registry.example.com/nemo", image_tag="test-tag")
    monkeypatch.setattr(shared_images, "get_platform_config", lambda: config)
    monkeypatch.setattr(platform_image, "get_platform_config", lambda: config)
    return config


def test_default_unsloth_images_use_platform_registry(monkeypatch, platform_config):
    monkeypatch.setattr(unsloth_images, "config", UnslothConfig())

    training = get_training_image()
    tasks = get_tasks_image()

    expected_training = f"{platform_config.image_registry}/{TRAINING_IMAGE_NAME}:{platform_config.image_tag}"
    expected_tasks = f"{platform_config.image_registry}/{CUSTOMIZER_TASKS_IMAGE_NAME}:{platform_config.image_tag}"
    assert training == expected_training
    assert tasks == expected_tasks
    assert CUSTOMIZER_TASKS_IMAGE_NAME.count("/") == 0  # single repo segment, no nested paths


def test_unsloth_image_registry_override(monkeypatch, platform_config):
    monkeypatch.setattr(
        unsloth_images,
        "config",
        UnslothConfig(image_registry="my-registry/other-registry"),
    )

    assert (
        get_unsloth_qualified_image(TRAINING_IMAGE_NAME)
        == f"my-registry/other-registry/{TRAINING_IMAGE_NAME}:{platform_config.image_tag}"
    )


def test_unsloth_tasks_image_override(monkeypatch, platform_config):
    monkeypatch.setattr(
        unsloth_images,
        "config",
        UnslothConfig(
            tasks_image="my-registry/nemo-helix-dev/nhx-customizer-tasks:dev",
            training_image="my-registry/nemo-helix-dev/nhx-unsloth-training:dev",
        ),
    )

    assert get_tasks_image() == "my-registry/nemo-helix-dev/nhx-customizer-tasks:dev"
    assert get_training_image() == "my-registry/nemo-helix-dev/nhx-unsloth-training:dev"
