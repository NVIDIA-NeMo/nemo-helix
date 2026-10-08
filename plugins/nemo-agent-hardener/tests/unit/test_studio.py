# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_agent_hardener_plugin.studio."""

from __future__ import annotations

from nemo_agent_hardener_plugin.studio import get_studio_spec
from nemo_platform_plugin.interface import StudioSpec


def test_get_studio_spec_points_at_the_packaged_bundle():
    spec = get_studio_spec()
    assert isinstance(spec, StudioSpec)
    assert spec.name == "agent-hardener"
    assert spec.bundle_path is not None
    assert spec.bundle_path.name == "index.js"
    assert spec.bundle_path.parts[-3:] == ("web", "dist", "index.js")


def test_bundle_ships_with_the_package():
    """The wheel is only useful if the built bundle is committed alongside it."""
    bundle_path = get_studio_spec().bundle_path
    assert bundle_path is not None
    assert bundle_path.exists()
