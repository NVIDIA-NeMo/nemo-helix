# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the backend-agnostic generic-engine compiler."""

import pytest
from nhx.core.models.controllers.backends import generic_compiler
from nhx.core.models.controllers.backends.common import DeploymentConfigView


def _view(**kwargs) -> DeploymentConfigView:
    return DeploymentConfigView(**kwargs)


# ---------------------------------------------------------------------------
# Image resolution
# ---------------------------------------------------------------------------


def test_resolve_generic_image_uses_config_image_and_tag():
    view = _view(gpu=0, image_name="nvcr.io/nim/nvidia/nemoguard-jailbreak-detect", image_tag="1.10.1")
    name, tag = generic_compiler.resolve_generic_image(view)
    assert name == "nvcr.io/nim/nvidia/nemoguard-jailbreak-detect"
    assert tag == "1.10.1"


def test_resolve_generic_image_defaults_tag_to_latest():
    view = _view(gpu=0, image_name="my/container", image_tag=None)
    name, tag = generic_compiler.resolve_generic_image(view)
    assert name == "my/container"
    assert tag == "latest"


def test_resolve_generic_image_requires_image_name():
    """There is no platform default for a generic image, so image_name is required."""
    view = _view(gpu=0, image_name=None)
    with pytest.raises(ValueError, match="image_name"):
        generic_compiler.resolve_generic_image(view)


def test_resolve_generic_image_rejects_blank_image_name():
    view = _view(gpu=0, image_name="   ")
    with pytest.raises(ValueError, match="image_name"):
        generic_compiler.resolve_generic_image(view)


def test_resolve_generic_image_trims_whitespace():
    """Defensive: surrounding whitespace is stripped from name and tag."""
    view = _view(gpu=0, image_name="  my/image  ", image_tag="  v1  ")
    name, tag = generic_compiler.resolve_generic_image(view)
    assert name == "my/image"
    assert tag == "v1"


def test_resolve_generic_image_blank_tag_falls_back_to_latest():
    view = _view(gpu=0, image_name="my/image", image_tag="   ")
    _, tag = generic_compiler.resolve_generic_image(view)
    assert tag == "latest"


# ---------------------------------------------------------------------------
# Args + env passthrough (the platform synthesizes nothing for generic)
# ---------------------------------------------------------------------------


def test_compile_generic_args_passthrough():
    view = _view(gpu=0, image_name="x", additional_args=["--port", "9000", "--foo"])
    assert generic_compiler.compile_generic_args(view) == ["--port", "9000", "--foo"]


def test_compile_generic_args_empty_when_none():
    view = _view(gpu=0, image_name="x", additional_args=None)
    assert generic_compiler.compile_generic_args(view) == []


def test_compile_generic_env_passthrough_stringifies():
    view = _view(gpu=0, image_name="x", additional_envs={"A": "1", "B": 2})
    assert generic_compiler.compile_generic_env_vars(view) == {"A": "1", "B": "2"}


def test_compile_generic_env_empty_when_none():
    view = _view(gpu=0, image_name="x", additional_envs=None)
    assert generic_compiler.compile_generic_env_vars(view) == {}
