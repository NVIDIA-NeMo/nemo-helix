# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image helper utilities for NeMo Helix jobs."""

from collections.abc import Callable

from nemo_helix_plugin.config import get_platform_config


def get_qualified_image(name: str, tag: str | None = None, registry: str | None = None) -> str:
    """Get a fully qualified Docker image name.

    Builds an image reference in the format: {registry}/{name}:{tag}
    """
    config = get_platform_config()

    effective_registry = registry if registry is not None else config.image_registry
    effective_tag = tag if tag is not None else config.image_tag

    return f"{effective_registry}/{name}:{effective_tag}"


def image_builder(registry: str | None = None, tag: str | None = None) -> Callable[[str], str]:
    """Create a function that builds qualified image names with preset registry and tag."""
    config = get_platform_config()

    effective_registry = registry if registry is not None else config.image_registry
    effective_tag = tag if tag is not None else config.image_tag

    def _build(name: str) -> str:
        return f"{effective_registry}/{name}:{effective_tag}"

    return _build
